from __future__ import annotations

import math
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Hashable, Literal

import networkx as nx

from .busqueda_indices import build_search_encoding
from .cargar_grafos import cargar_objeto_gexf
from .cargar_soluciones_iniciales import cargar_solucion_inicial
from .costo_solucion import costo_solucion
from .instances import load_instances
from .metaheuristicas_utils import (
    generar_reporte_detallado,
    guardar_resultado_csv,
    solucion_legible_humana,
)
from .vecindarios import MovimientoVecindario, OPERADORES_POPULARES, generar_vecino

__all__ = [
    "RecocidoSimuladoResult",
    "recocido_simulado",
    "recocido_simulado_desde_instancia",
]


@dataclass(frozen=True, slots=True)
class RecocidoSimuladoResult:
    """
    Resultado completo del recocido simulado clásico.

    - ``mejor_solucion`` y ``mejor_costo``: mejor hallazgo del algoritmo.
    - ``solucion_inicial_referencia`` y ``costo_inicial_referencia``: mejor solución detectada
      automáticamente dentro del objeto inicial (lista/dict/estructura anidada).
    - ``gap_porcentaje``: ((mejor_costo - costo_inicial_referencia) / costo_inicial_referencia) * 100.
      Si es negativo, hubo mejora.
    - ``tiempo_segundos``: tiempo total medido con ``perf_counter``.
    """

    mejor_solucion: list[list[str]]
    mejor_costo: float
    solucion_inicial_referencia: list[list[str]]
    costo_inicial_referencia: float
    gap_porcentaje: float
    mejora_absoluta: float
    mejora_porcentaje: float
    tiempo_segundos: float
    iteraciones_totales: int
    enfriamientos_ejecutados: int
    aceptadas: int
    mejoras: int
    semilla: int | None
    historial_mejor_costo: list[float] = field(default_factory=list)
    historial_temperatura: list[float] = field(default_factory=list)
    ultimo_movimiento_aceptado: MovimientoVecindario | None = None
    archivo_csv: str | None = None


def _copiar_solucion_labels(sol: Sequence[Sequence[Hashable]]) -> list[list[str]]:
    """Copia profunda ligera de una solución en formato por etiquetas."""
    return [[str(x).strip() for x in ruta] for ruta in sol]


def _es_solucion_lista_de_rutas(obj: Any) -> bool:
    """
    Heurística para decidir si ``obj`` parece solución CARP:
    lista/tupla de rutas, donde cada ruta es lista/tupla de tokens hashables.
    """
    # Debe ser secuencia "externa" (lista/tupla), no string/bytes.
    if not isinstance(obj, (list, tuple)):
        return False
    if isinstance(obj, (str, bytes)):
        return False
    # Aceptamos rutas vacías porque una solución puede tener vehículos sin tareas.
    for ruta in obj:
        if not isinstance(ruta, (list, tuple)):
            return False
        if isinstance(ruta, (str, bytes)):
            return False
        for token in ruta:
            if isinstance(token, Mapping):
                return False
    return True


def _extraer_candidatas_desde_objeto(obj: Any, *, max_nodos: int = 20000) -> list[list[list[str]]]:
    """
    Recorre recursivamente un objeto inicial y extrae candidatas con forma de solución.

    Esto permite soportar pickles que guardan:
    - una sola solución,
    - dict con varias soluciones,
    - estructuras anidadas (dict/list de experimentos).
    """
    candidatas: list[list[list[str]]] = []
    visitados = 0

    def _walk(x: Any) -> None:
        nonlocal visitados
        # Protección simple para evitar ciclos/estructuras enormes.
        visitados += 1
        if visitados > max_nodos:
            return

        # Si ya parece solución, la agregamos y no seguimos profundizando ese nodo.
        if _es_solucion_lista_de_rutas(x):
            candidatas.append(_copiar_solucion_labels(x))
            return

        # Si es mapeo, exploramos sus valores.
        if isinstance(x, Mapping):
            for v in x.values():
                _walk(v)
            return

        # Si es contenedor secuencial/genérico, exploramos elementos.
        if isinstance(x, (list, tuple, set)):
            for v in x:
                _walk(v)
            return

    _walk(obj)
    return candidatas


def _evaluar_costo_solucion(
    solucion: Sequence[Sequence[Hashable]],
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    marcador_depot_etiqueta: str | None,
    usar_gpu: bool,
) -> float:
    """Evalúa costo total reutilizando el módulo central de costos."""
    return costo_solucion(
        solucion,
        data,
        G,
        detalle=False,
        marcador_depot_etiqueta=marcador_depot_etiqueta,
        usar_gpu=usar_gpu,
    ).costo_total


def _seleccionar_mejor_inicial(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    marcador_depot_etiqueta: str | None,
    usar_gpu: bool,
) -> tuple[list[list[str]], float]:
    """
    Encuentra automáticamente la mejor solución dentro del objeto inicial.

    Si ninguna candidata es válida para costo, lanza ValueError con contexto.
    """
    candidatas = _extraer_candidatas_desde_objeto(inicial_obj)
    if not candidatas:
        raise ValueError(
            "No se encontraron soluciones candidatas en el objeto inicial. "
            "Se esperaba lista de rutas o estructura (dict/list) que la contenga."
        )

    mejor_sol: list[list[str]] | None = None
    mejor_cost = float("inf")
    errores = 0

    # Evaluamos todas las candidatas y nos quedamos con la de menor costo.
    for cand in candidatas:
        try:
            c = _evaluar_costo_solucion(
                cand,
                data,
                G,
                marcador_depot_etiqueta=marcador_depot_etiqueta,
                usar_gpu=usar_gpu,
            )
        except Exception:  # noqa: BLE001 - intencional para filtrar candidatas inválidas.
            errores += 1
            continue
        if c < mejor_cost:
            mejor_cost = c
            mejor_sol = cand

    if mejor_sol is None:
        raise ValueError(
            "Ninguna solución candidata del objeto inicial pudo evaluarse con costo_solucion. "
            f"Candidatas detectadas: {len(candidatas)} | inválidas: {errores}."
        )

    return mejor_sol, mejor_cost


def recocido_simulado(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    temperatura_inicial: float = 1000.0,
    temperatura_minima: float = 1e-3,
    alpha: float = 0.95,
    iteraciones_por_temperatura: int = 120,
    max_enfriamientos: int = 250,
    semilla: int | None = None,
    operadores: Iterable[str] = OPERADORES_POPULARES,
    marcador_depot_etiqueta: str | None = None,
    usar_gpu: bool = False,
    backend_vecindario: Literal["labels", "ids"] = "labels",
    guardar_historial: bool = True,
    guardar_csv: bool = False,
    ruta_csv: str | None = None,
    nombre_instancia: str = "instancia",
    id_corrida: str | None = None,
    config_id: str | None = None,
    repeticion: int | None = None,
) -> RecocidoSimuladoResult:
    """
    Recocido simulado clásico para minimizar costo de solución CARP.

    Flujo clásico:
    1) Arranca en la mejor solución inicial disponible.
    2) Genera vecino aleatorio.
    3) Acepta mejoras siempre; empeoramientos con prob. ``exp(-delta/T)``.
    4) Enfría ``T = alpha * T`` y repite.
    """
    # Validaciones básicas de parámetros para evitar corridas silenciosamente inválidas.
    if temperatura_inicial <= 0:
        raise ValueError("temperatura_inicial debe ser > 0.")
    if temperatura_minima <= 0:
        raise ValueError("temperatura_minima debe ser > 0.")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha debe estar en (0, 1).")
    if iteraciones_por_temperatura <= 0:
        raise ValueError("iteraciones_por_temperatura debe ser > 0.")
    if max_enfriamientos <= 0:
        raise ValueError("max_enfriamientos debe ser > 0.")

    # Inicializamos RNG reproducible cuando se proporciona semilla.
    rng = random.Random(semilla)

    # Medimos el tiempo total de la metaheurística.
    t0 = time.perf_counter()

    # Detectamos automáticamente la mejor referencia dentro del objeto inicial.
    sol_ref, costo_ref = _seleccionar_mejor_inicial(
        inicial_obj,
        data,
        G,
        marcador_depot_etiqueta=marcador_depot_etiqueta,
        usar_gpu=usar_gpu,
    )

    # El estado "actual" arranca exactamente en esa referencia.
    sol_actual = _copiar_solucion_labels(sol_ref)
    costo_actual = costo_ref

    # El estado "mejor" también arranca en la misma solución.
    sol_mejor = _copiar_solucion_labels(sol_ref)
    costo_mejor = costo_ref

    # Si se usa backend indexado, precompilamos encoding una sola vez.
    encoding = build_search_encoding(data) if backend_vecindario == "ids" else None

    # Inicializamos la temperatura.
    T = float(temperatura_inicial)

    # Contadores de métricas para reporte final.
    iteraciones_totales = 0
    enfriamientos = 0
    aceptadas = 0
    mejoras = 0
    ultimo_mov_aceptado: MovimientoVecindario | None = None

    # Historial opcional para graficar convergencia.
    historial_best: list[float] = []
    historial_temp: list[float] = []

    # Bucle externo de enfriamiento.
    while T > temperatura_minima and enfriamientos < max_enfriamientos:
        # Guardamos estado de la "meseta térmica" si el usuario pidió historial.
        if guardar_historial:
            historial_temp.append(T)
            historial_best.append(costo_mejor)

        # Bucle interno: varias perturbaciones por cada temperatura.
        for _ in range(iteraciones_por_temperatura):
            iteraciones_totales += 1

            # Proponemos vecino con los operadores definidos.
            vecino, mov = generar_vecino(
                sol_actual,
                rng=rng,
                operadores=operadores,
                marcador_depot=marcador_depot_etiqueta or "D",
                devolver_con_deposito=True,
                usar_gpu=usar_gpu,
                backend=backend_vecindario,
                encoding=encoding,
            )

            # Evaluamos costo del vecino.
            costo_vecino = _evaluar_costo_solucion(
                vecino,
                data,
                G,
                marcador_depot_etiqueta=marcador_depot_etiqueta,
                usar_gpu=usar_gpu,
            )

            # Delta clásico de minimización.
            delta = costo_vecino - costo_actual

            # Regla de aceptación clásica de SA.
            if delta <= 0:
                aceptar = True
            else:
                # Probabilidad de aceptar un empeoramiento para escapar de óptimos locales.
                p = math.exp(-delta / T)
                aceptar = rng.random() < p

            # Si aceptamos, actualizamos estado actual.
            if aceptar:
                sol_actual = vecino
                costo_actual = costo_vecino
                aceptadas += 1
                ultimo_mov_aceptado = mov

                # Si además mejoró el mejor global, lo registramos.
                if costo_actual < costo_mejor:
                    costo_mejor = costo_actual
                    sol_mejor = _copiar_solucion_labels(sol_actual)
                    mejoras += 1

        # Enfriamiento geométrico clásico.
        T *= alpha
        enfriamientos += 1

    # Medimos tiempo total al finalizar.
    elapsed = time.perf_counter() - t0

    # Gap solicitado: mejor inicial de referencia vs mejor hallada.
    if costo_ref == 0:
        gap = 0.0 if costo_mejor == 0 else float("inf")
    else:
        gap = ((costo_mejor - costo_ref) / costo_ref) * 100.0

    # Métricas de mejora en términos absolutos/relativos.
    mejora_abs = costo_ref - costo_mejor
    mejora_pct = 0.0 if costo_ref == 0 else (mejora_abs / costo_ref) * 100.0

    archivo_csv: str | None = None
    if guardar_csv:
        ruta = ruta_csv or f"resultados_recocido_simulado_{nombre_instancia}.csv"
        detalle_txt, costo_total_reporte = generar_reporte_detallado(
            sol_mejor,
            data,
            G,
            nombre_instancia=nombre_instancia,
            marcador_depot_etiqueta=marcador_depot_etiqueta,
            usar_gpu=usar_gpu,
        )
        fila = {
            "metaheuristica": "recocido_simulado",
            "instancia": nombre_instancia,
            "id_corrida": id_corrida or "",
            "config_id": config_id or "",
            "repeticion": repeticion if repeticion is not None else "",
            "semilla": semilla,
            "tiempo_segundos": elapsed,
            "iteraciones_totales": iteraciones_totales,
            "enfriamientos_ejecutados": enfriamientos,
            "aceptadas": aceptadas,
            "mejoras": mejoras,
            "costo_inicial_referencia": costo_ref,
            "mejor_costo": costo_mejor,
            "gap_porcentaje": gap,
            "mejora_absoluta": mejora_abs,
            "mejora_porcentaje": mejora_pct,
            "mejor_solucion_tr_legible": solucion_legible_humana(sol_mejor),
            "reporte_detalle_deadheading": detalle_txt,
            "costo_total_desde_reporte": costo_total_reporte,
        }
        archivo_csv = guardar_resultado_csv(fila=fila, ruta_csv=ruta)

    return RecocidoSimuladoResult(
        mejor_solucion=sol_mejor,
        mejor_costo=costo_mejor,
        solucion_inicial_referencia=sol_ref,
        costo_inicial_referencia=costo_ref,
        gap_porcentaje=gap,
        mejora_absoluta=mejora_abs,
        mejora_porcentaje=mejora_pct,
        tiempo_segundos=elapsed,
        iteraciones_totales=iteraciones_totales,
        enfriamientos_ejecutados=enfriamientos,
        aceptadas=aceptadas,
        mejoras=mejoras,
        semilla=semilla,
        historial_mejor_costo=historial_best,
        historial_temperatura=historial_temp,
        ultimo_movimiento_aceptado=ultimo_mov_aceptado,
        archivo_csv=archivo_csv,
    )


def recocido_simulado_desde_instancia(
    nombre_instancia: str,
    *,
    root: str | None = None,
    temperatura_inicial: float = 1000.0,
    temperatura_minima: float = 1e-3,
    alpha: float = 0.95,
    iteraciones_por_temperatura: int = 120,
    max_enfriamientos: int = 250,
    semilla: int | None = None,
    operadores: Iterable[str] = OPERADORES_POPULARES,
    marcador_depot_etiqueta: str | None = None,
    usar_gpu: bool = False,
    backend_vecindario: Literal["labels", "ids"] = "labels",
    guardar_historial: bool = True,
    guardar_csv: bool = False,
    ruta_csv: str | None = None,
    id_corrida: str | None = None,
    config_id: str | None = None,
    repeticion: int | None = None,
) -> RecocidoSimuladoResult:
    """
    Helper que carga instancia + grafo + objeto inicial y ejecuta recocido.
    """
    # Cargamos estructuras base desde el paquete.
    data = load_instances(nombre_instancia, root=root)
    G = cargar_objeto_gexf(nombre_instancia, root=root)
    inicial_obj = cargar_solucion_inicial(nombre_instancia, root=root)

    # Delegamos toda la lógica al núcleo de recocido.
    return recocido_simulado(
        inicial_obj,
        data,
        G,
        temperatura_inicial=temperatura_inicial,
        temperatura_minima=temperatura_minima,
        alpha=alpha,
        iteraciones_por_temperatura=iteraciones_por_temperatura,
        max_enfriamientos=max_enfriamientos,
        semilla=semilla,
        operadores=operadores,
        marcador_depot_etiqueta=marcador_depot_etiqueta,
        usar_gpu=usar_gpu,
        backend_vecindario=backend_vecindario,
        guardar_historial=guardar_historial,
        guardar_csv=guardar_csv,
        ruta_csv=ruta_csv,
        nombre_instancia=nombre_instancia,
        id_corrida=id_corrida,
        config_id=config_id,
        repeticion=repeticion,
    )
