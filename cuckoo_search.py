from __future__ import annotations

import math
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
    "CuckooSearchResult",
    "cuckoo_search",
    "cuckoo_search_desde_instancia",
]


@dataclass(frozen=True, slots=True)
class CuckooSearchResult:
    """Resultado de Cuckoo Search con reemplazo por nidos y abandono parcial."""

    mejor_solucion: list[list[str]]
    mejor_costo: float
    solucion_inicial_referencia: list[list[str]]
    costo_inicial_referencia: float
    gap_porcentaje: float
    mejora_absoluta: float
    mejora_porcentaje: float
    tiempo_segundos: float
    iteraciones_totales: int
    nidos: int
    abandonos_totales: int
    reemplazos_exitosos: int
    mejoras: int
    semilla: int | None
    historial_mejor_costo: list[float] = field(default_factory=list)
    ultimo_movimiento_aceptado: MovimientoVecindario | None = None
    archivo_csv: str | None = None


def _vecino_un_paso(
    sol: list[list[str]],
    *,
    rng: random.Random,
    operadores: Iterable[str],
    marcador_depot_etiqueta: str | None,
    usar_gpu: bool,
    backend_vecindario: Literal["labels", "ids"],
    encoding: Any,
) -> tuple[list[list[str]], MovimientoVecindario]:
    return generar_vecino(
        sol,
        rng=rng,
        operadores=operadores,
        marcador_depot=marcador_depot_etiqueta or "D",
        devolver_con_deposito=True,
        usar_gpu=usar_gpu,
        backend=backend_vecindario,
        encoding=encoding,
    )


def _vuelo_levy_discreto(
    base: list[list[str]],
    *,
    rng: random.Random,
    pasos_base: int,
    beta: float,
    operadores: Iterable[str],
    marcador_depot_etiqueta: str | None,
    usar_gpu: bool,
    backend_vecindario: Literal["labels", "ids"],
    encoding: Any,
) -> tuple[list[list[str]], MovimientoVecindario | None]:
    """
    Aproximación discreta de vuelo Levy:
    número de pasos ~ 1 + floor(|N(0,1)|^(1/beta) * pasos_base), acotado.
    """
    if beta <= 0:
        beta = 1.5
    x = abs(rng.gauss(0.0, 1.0))
    n_pasos = 1 + int((x ** (1.0 / beta)) * max(1, pasos_base))
    n_pasos = min(n_pasos, 12)  # tope razonable para no disparar costo por iteración.

    sol = copiar_solucion_labels(base)
    ultimo: MovimientoVecindario | None = None
    for _ in range(n_pasos):
        sol, ultimo = _vecino_un_paso(
            sol,
            rng=rng,
            operadores=operadores,
            marcador_depot_etiqueta=marcador_depot_etiqueta,
            usar_gpu=usar_gpu,
            backend_vecindario=backend_vecindario,
            encoding=encoding,
        )
    return sol, ultimo


def cuckoo_search(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    iteraciones: int = 260,
    num_nidos: int = 20,
    pa_abandono: float = 0.25,
    pasos_levy_base: int = 3,
    beta_levy: float = 1.5,
    semilla: int | None = None,
    operadores: Iterable[str] = OPERADORES_POPULARES,
    marcador_depot_etiqueta: str | None = None,
    usar_gpu: bool = False,
    backend_vecindario: Literal["labels", "ids"] = "labels",
    guardar_historial: bool = True,
    guardar_csv: bool = False,
    ruta_csv: str | None = None,
    nombre_instancia: str = "instancia",
) -> CuckooSearchResult:
    """
    Cuckoo Search clásico adaptado a espacio discreto:
    - Cada cuckoo genera una solución candidata (vuelo tipo Levy).
    - Si mejora un nido aleatorio, lo reemplaza.
    - Una fracción de peores nidos se abandona y reinicializa.
    """
    if iteraciones <= 0:
        raise ValueError("iteraciones debe ser > 0.")
    if num_nidos <= 1:
        raise ValueError("num_nidos debe ser >= 2.")
    if not (0.0 < pa_abandono < 1.0):
        raise ValueError("pa_abandono debe estar en (0, 1).")
    if pasos_levy_base <= 0:
        raise ValueError("pasos_levy_base debe ser > 0.")

    rng = random.Random(semilla)
    t0 = time.perf_counter()

    sol_ref, costo_ref = seleccionar_mejor_inicial(
        inicial_obj,
        data,
        G,
        marcador_depot_etiqueta=marcador_depot_etiqueta,
        usar_gpu=usar_gpu,
    )
    encoding = build_search_encoding(data) if backend_vecindario == "ids" else None

    # Inicialización de nidos: mejor referencia + perturbaciones.
    nidos_sol: list[list[list[str]]] = [copiar_solucion_labels(sol_ref)]
    nidos_cost: list[float] = [costo_ref]
    ultimos_movs: list[MovimientoVecindario | None] = [None]
    while len(nidos_sol) < num_nidos:
        cand, mov = _vecino_un_paso(
            copiar_solucion_labels(sol_ref),
            rng=rng,
            operadores=operadores,
            marcador_depot_etiqueta=marcador_depot_etiqueta,
            usar_gpu=usar_gpu,
            backend_vecindario=backend_vecindario,
            encoding=encoding,
        )
        c = evaluar_costo_solucion(
            cand,
            data,
            G,
            marcador_depot_etiqueta=marcador_depot_etiqueta,
            usar_gpu=usar_gpu,
        )
        nidos_sol.append(cand)
        nidos_cost.append(c)
        ultimos_movs.append(mov)

    idx_best = min(range(num_nidos), key=nidos_cost.__getitem__)
    sol_mejor = copiar_solucion_labels(nidos_sol[idx_best])
    costo_mejor = nidos_cost[idx_best]
    mejoras = 0
    reemplazos = 0
    abandonos = 0
    historial_best: list[float] = []
    ultimo_mov_aceptado: MovimientoVecindario | None = None

    for _it in range(iteraciones):
        if guardar_historial:
            historial_best.append(costo_mejor)

        # Generación de cuckoos y competencia con nidos aleatorios.
        for i in range(num_nidos):
            cuckoo_sol, mov = _vuelo_levy_discreto(
                nidos_sol[i],
                rng=rng,
                pasos_base=pasos_levy_base,
                beta=beta_levy,
                operadores=operadores,
                marcador_depot_etiqueta=marcador_depot_etiqueta,
                usar_gpu=usar_gpu,
                backend_vecindario=backend_vecindario,
                encoding=encoding,
            )
            cuckoo_cost = evaluar_costo_solucion(
                cuckoo_sol,
                data,
                G,
                marcador_depot_etiqueta=marcador_depot_etiqueta,
                usar_gpu=usar_gpu,
            )

            j = rng.randrange(num_nidos)
            if cuckoo_cost < nidos_cost[j]:
                nidos_sol[j] = cuckoo_sol
                nidos_cost[j] = cuckoo_cost
                ultimos_movs[j] = mov
                reemplazos += 1
                ultimo_mov_aceptado = mov

        # Abandono de una fracción de peores nidos.
        n_abandonar = max(1, int(math.floor(pa_abandono * num_nidos)))
        peores = sorted(range(num_nidos), key=nidos_cost.__getitem__, reverse=True)[:n_abandonar]
        idx_best = min(range(num_nidos), key=nidos_cost.__getitem__)
        base_best = nidos_sol[idx_best]
        for idx in peores:
            nuevo, mov = _vuelo_levy_discreto(
                base_best,
                rng=rng,
                pasos_base=pasos_levy_base,
                beta=beta_levy,
                operadores=operadores,
                marcador_depot_etiqueta=marcador_depot_etiqueta,
                usar_gpu=usar_gpu,
                backend_vecindario=backend_vecindario,
                encoding=encoding,
            )
            c = evaluar_costo_solucion(
                nuevo,
                data,
                G,
                marcador_depot_etiqueta=marcador_depot_etiqueta,
                usar_gpu=usar_gpu,
            )
            nidos_sol[idx] = nuevo
            nidos_cost[idx] = c
            ultimos_movs[idx] = mov
            abandonos += 1

        # Mejor global.
        idx_best = min(range(num_nidos), key=nidos_cost.__getitem__)
        if nidos_cost[idx_best] < costo_mejor:
            costo_mejor = nidos_cost[idx_best]
            sol_mejor = copiar_solucion_labels(nidos_sol[idx_best])
            mejoras += 1

    elapsed = time.perf_counter() - t0
    gap, mejora_abs, mejora_pct = calcular_metricas_gap(costo_ref, costo_mejor)

    archivo_csv: str | None = None
    if guardar_csv:
        ruta = ruta_csv or f"resultados_cuckoo_search_{nombre_instancia}.csv"
        detalle_txt, costo_total_reporte = generar_reporte_detallado(
            sol_mejor,
            data,
            G,
            nombre_instancia=nombre_instancia,
            marcador_depot_etiqueta=marcador_depot_etiqueta,
            usar_gpu=usar_gpu,
        )
        fila = {
            "metaheuristica": "cuckoo_search",
            "instancia": nombre_instancia,
            "semilla": semilla,
            "tiempo_segundos": elapsed,
            "iteraciones_totales": iteraciones,
            "nidos": num_nidos,
            "abandonos_totales": abandonos,
            "reemplazos_exitosos": reemplazos,
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

    return CuckooSearchResult(
        mejor_solucion=sol_mejor,
        mejor_costo=costo_mejor,
        solucion_inicial_referencia=sol_ref,
        costo_inicial_referencia=costo_ref,
        gap_porcentaje=gap,
        mejora_absoluta=mejora_abs,
        mejora_porcentaje=mejora_pct,
        tiempo_segundos=elapsed,
        iteraciones_totales=iteraciones,
        nidos=num_nidos,
        abandonos_totales=abandonos,
        reemplazos_exitosos=reemplazos,
        mejoras=mejoras,
        semilla=semilla,
        historial_mejor_costo=historial_best,
        ultimo_movimiento_aceptado=ultimo_mov_aceptado,
        archivo_csv=archivo_csv,
    )


def cuckoo_search_desde_instancia(
    nombre_instancia: str,
    *,
    root: str | None = None,
    iteraciones: int = 260,
    num_nidos: int = 20,
    pa_abandono: float = 0.25,
    pasos_levy_base: int = 3,
    beta_levy: float = 1.5,
    semilla: int | None = None,
    operadores: Iterable[str] = OPERADORES_POPULARES,
    marcador_depot_etiqueta: str | None = None,
    usar_gpu: bool = False,
    backend_vecindario: Literal["labels", "ids"] = "labels",
    guardar_historial: bool = True,
    guardar_csv: bool = False,
    ruta_csv: str | None = None,
) -> CuckooSearchResult:
    """Helper que carga recursos y ejecuta Cuckoo Search."""
    data = load_instances(nombre_instancia, root=root)
    G = cargar_objeto_gexf(nombre_instancia, root=root)
    inicial_obj = cargar_solucion_inicial(nombre_instancia, root=root)
    return cuckoo_search(
        inicial_obj,
        data,
        G,
        iteraciones=iteraciones,
        num_nidos=num_nidos,
        pa_abandono=pa_abandono,
        pasos_levy_base=pasos_levy_base,
        beta_levy=beta_levy,
        semilla=semilla,
        operadores=operadores,
        marcador_depot_etiqueta=marcador_depot_etiqueta,
        usar_gpu=usar_gpu,
        backend_vecindario=backend_vecindario,
        guardar_historial=guardar_historial,
        guardar_csv=guardar_csv,
        ruta_csv=ruta_csv,
        nombre_instancia=nombre_instancia,
    )
