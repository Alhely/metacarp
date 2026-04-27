from __future__ import annotations

import random
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx

from .busqueda_indices import build_search_encoding
from .cargar_grafos import cargar_objeto_gexf
from .cargar_soluciones_iniciales import cargar_solucion_inicial
from .instances import load_instances
from .metaheuristicas_utils import (
    calcular_metricas_gap,
    copiar_solucion_labels,
    evaluar_costo_solucion,
    generar_reporte_detallado,
    guardar_resultado_csv,
    seleccionar_mejor_inicial,
    solucion_legible_humana,
)
from .vecindarios import MovimientoVecindario, OPERADORES_POPULARES, generar_vecino

__all__ = [
    "BusquedaTabuResult",
    "busqueda_tabu",
    "busqueda_tabu_desde_instancia",
]


@dataclass(frozen=True, slots=True)
class BusquedaTabuResult:
    """Resultado de búsqueda tabú con métricas de calidad y tiempo."""

    mejor_solucion: list[list[str]]
    mejor_costo: float
    solucion_inicial_referencia: list[list[str]]
    costo_inicial_referencia: float
    gap_porcentaje: float
    mejora_absoluta: float
    mejora_porcentaje: float
    tiempo_segundos: float
    iteraciones_totales: int
    vecinos_evaluados: int
    movimientos_tabu_bloqueados: int
    mejoras: int
    semilla: int | None
    historial_mejor_costo: list[float] = field(default_factory=list)
    ultimo_movimiento_aceptado: MovimientoVecindario | None = None
    archivo_csv: str | None = None


def _clave_tabu(mov: MovimientoVecindario) -> tuple[Any, ...]:
    """
    Clave hashable del movimiento para memoria tabú.
    Incluye operador, índices y labels movidos cuando estén disponibles.
    """
    return (
        mov.operador,
        mov.ruta_a,
        mov.ruta_b,
        mov.i,
        mov.j,
        mov.k,
        mov.l,
        tuple(mov.labels_movidos),
    )


def busqueda_tabu(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    iteraciones: int = 400,
    tam_vecindario: int = 25,
    tenure_tabu: int = 20,
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
) -> BusquedaTabuResult:
    """
    Búsqueda tabú clásica (short-term memory):
    - Explora un vecindario por iteración.
    - Selecciona el mejor vecino no tabú (o por aspiración).
    - Registra el movimiento en lista tabú por ``tenure_tabu`` iteraciones.
    """
    if iteraciones <= 0:
        raise ValueError("iteraciones debe ser > 0.")
    if tam_vecindario <= 0:
        raise ValueError("tam_vecindario debe ser > 0.")
    if tenure_tabu <= 0:
        raise ValueError("tenure_tabu debe ser > 0.")

    rng = random.Random(semilla)
    t0 = time.perf_counter()

    sol_ref, costo_ref = seleccionar_mejor_inicial(
        inicial_obj,
        data,
        G,
        marcador_depot_etiqueta=marcador_depot_etiqueta,
        usar_gpu=usar_gpu,
    )
    sol_actual = copiar_solucion_labels(sol_ref)
    costo_actual = costo_ref
    sol_mejor = copiar_solucion_labels(sol_ref)
    costo_mejor = costo_ref

    encoding = build_search_encoding(data) if backend_vecindario == "ids" else None

    # Memoria tabú: clave_mov -> iteración de expiración.
    tabu_hasta: dict[tuple[Any, ...], int] = {}

    vecinos_evaluados = 0
    bloqueados = 0
    mejoras = 0
    ultimo_mov_aceptado: MovimientoVecindario | None = None
    historial_best: list[float] = []

    for it in range(iteraciones):
        if guardar_historial:
            historial_best.append(costo_mejor)

        candidatos: list[tuple[float, list[list[str]], MovimientoVecindario, bool]] = []

        # Generamos un lote de vecinos para evaluar.
        for _ in range(tam_vecindario):
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
            c = evaluar_costo_solucion(
                vecino,
                data,
                G,
                marcador_depot_etiqueta=marcador_depot_etiqueta,
                usar_gpu=usar_gpu,
            )
            vecinos_evaluados += 1

            key = _clave_tabu(mov)
            es_tabu = tabu_hasta.get(key, -1) > it
            # Aspiración: permitimos movimiento tabú si mejora el mejor global.
            aspiracion = c < costo_mejor
            candidatos.append((c, vecino, mov, es_tabu and not aspiracion))

        # Seleccionamos el mejor admisible.
        admisibles = [x for x in candidatos if not x[3]]
        if not admisibles:
            # Si todos quedaron bloqueados, tomamos el mejor igual para no estancarnos.
            bloqueados += len(candidatos)
            elegido = min(candidatos, key=lambda t: t[0])
        else:
            bloqueados += sum(1 for x in candidatos if x[3])
            elegido = min(admisibles, key=lambda t: t[0])

        costo_sig, sol_sig, mov_sig, _bloq = elegido
        sol_actual = sol_sig
        costo_actual = costo_sig
        ultimo_mov_aceptado = mov_sig

        # Guardamos el movimiento aceptado en la memoria tabú.
        tabu_hasta[_clave_tabu(mov_sig)] = it + tenure_tabu

        # Limpieza liviana de entradas vencidas.
        if it % 25 == 0 and tabu_hasta:
            expiradas = [k for k, vence in tabu_hasta.items() if vence <= it]
            for k in expiradas:
                del tabu_hasta[k]

        # Actualizamos mejor global.
        if costo_actual < costo_mejor:
            costo_mejor = costo_actual
            sol_mejor = copiar_solucion_labels(sol_actual)
            mejoras += 1

    elapsed = time.perf_counter() - t0
    gap, mejora_abs, mejora_pct = calcular_metricas_gap(costo_ref, costo_mejor)

    archivo_csv: str | None = None
    if guardar_csv:
        ruta = ruta_csv or f"resultados_busqueda_tabu_{nombre_instancia}.csv"
        detalle_txt, costo_total_reporte = generar_reporte_detallado(
            sol_mejor,
            data,
            G,
            nombre_instancia=nombre_instancia,
            marcador_depot_etiqueta=marcador_depot_etiqueta,
            usar_gpu=usar_gpu,
        )
        fila = {
            "metaheuristica": "busqueda_tabu",
            "instancia": nombre_instancia,
            "id_corrida": id_corrida or "",
            "config_id": config_id or "",
            "repeticion": repeticion if repeticion is not None else "",
            "semilla": semilla,
            "tiempo_segundos": elapsed,
            "iteraciones_totales": iteraciones,
            "vecinos_evaluados": vecinos_evaluados,
            "movimientos_tabu_bloqueados": bloqueados,
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

    return BusquedaTabuResult(
        mejor_solucion=sol_mejor,
        mejor_costo=costo_mejor,
        solucion_inicial_referencia=sol_ref,
        costo_inicial_referencia=costo_ref,
        gap_porcentaje=gap,
        mejora_absoluta=mejora_abs,
        mejora_porcentaje=mejora_pct,
        tiempo_segundos=elapsed,
        iteraciones_totales=iteraciones,
        vecinos_evaluados=vecinos_evaluados,
        movimientos_tabu_bloqueados=bloqueados,
        mejoras=mejoras,
        semilla=semilla,
        historial_mejor_costo=historial_best,
        ultimo_movimiento_aceptado=ultimo_mov_aceptado,
        archivo_csv=archivo_csv,
    )


def busqueda_tabu_desde_instancia(
    nombre_instancia: str,
    *,
    root: str | None = None,
    iteraciones: int = 400,
    tam_vecindario: int = 25,
    tenure_tabu: int = 20,
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
) -> BusquedaTabuResult:
    """Helper de ejecución tabú cargando recursos desde la instancia."""
    data = load_instances(nombre_instancia, root=root)
    G = cargar_objeto_gexf(nombre_instancia, root=root)
    inicial_obj = cargar_solucion_inicial(nombre_instancia, root=root)
    return busqueda_tabu(
        inicial_obj,
        data,
        G,
        iteraciones=iteraciones,
        tam_vecindario=tam_vecindario,
        tenure_tabu=tenure_tabu,
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
