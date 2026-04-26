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
    "AbejasResult",
    "busqueda_abejas",
    "busqueda_abejas_desde_instancia",
]


@dataclass(frozen=True, slots=True)
class AbejasResult:
    """Resultado de la metaheurística Artificial Bee Colony (ABC) simplificada."""

    mejor_solucion: list[list[str]]
    mejor_costo: float
    solucion_inicial_referencia: list[list[str]]
    costo_inicial_referencia: float
    gap_porcentaje: float
    mejora_absoluta: float
    mejora_porcentaje: float
    tiempo_segundos: float
    iteraciones_totales: int
    fuentes_alimento: int
    scouts_reinicios: int
    mejoras: int
    semilla: int | None
    historial_mejor_costo: list[float] = field(default_factory=list)
    ultimo_movimiento_aceptado: MovimientoVecindario | None = None
    archivo_csv: str | None = None


def _fitness_por_costo(c: float) -> float:
    """Fitness positiva para minimización (costos bajos => fitness alta)."""
    return 1.0 / (1.0 + max(c, 0.0))


def _vecino_local(
    sol: list[list[str]],
    *,
    rng: random.Random,
    operadores: Iterable[str],
    marcador_depot_etiqueta: str | None,
    usar_gpu: bool,
    backend_vecindario: Literal["labels", "ids"],
    encoding: Any,
) -> tuple[list[list[str]], MovimientoVecindario]:
    """Perturbación local un paso usando operadores de vecindario."""
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


def busqueda_abejas(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    iteraciones: int = 250,
    num_fuentes: int = 16,
    limite_abandono: int = 35,
    semilla: int | None = None,
    operadores: Iterable[str] = OPERADORES_POPULARES,
    marcador_depot_etiqueta: str | None = None,
    usar_gpu: bool = False,
    backend_vecindario: Literal["labels", "ids"] = "labels",
    guardar_historial: bool = True,
    guardar_csv: bool = False,
    ruta_csv: str | None = None,
    nombre_instancia: str = "instancia",
) -> AbejasResult:
    """
    Artificial Bee Colony simplificada:
    - Empleadas exploran cada fuente.
    - Observadoras exploran fuentes con probabilidad proporcional al fitness.
    - Scouts reinician fuentes estancadas.
    """
    if iteraciones <= 0:
        raise ValueError("iteraciones debe ser > 0.")
    if num_fuentes <= 1:
        raise ValueError("num_fuentes debe ser >= 2.")
    if limite_abandono <= 0:
        raise ValueError("limite_abandono debe ser > 0.")

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

    # Inicialización de fuentes de alimento: base + perturbaciones.
    fuentes_sol: list[list[list[str]]] = [copiar_solucion_labels(sol_ref)]
    fuentes_cost: list[float] = [costo_ref]
    trials: list[int] = [0]
    ultimo_mov: list[MovimientoVecindario | None] = [None]

    while len(fuentes_sol) < num_fuentes:
        base = copiar_solucion_labels(sol_ref)
        vecino, mov = _vecino_local(
            base,
            rng=rng,
            operadores=operadores,
            marcador_depot_etiqueta=marcador_depot_etiqueta,
            usar_gpu=usar_gpu,
            backend_vecindario=backend_vecindario,
            encoding=encoding,
        )
        c = evaluar_costo_solucion(
            vecino,
            data,
            G,
            marcador_depot_etiqueta=marcador_depot_etiqueta,
            usar_gpu=usar_gpu,
        )
        fuentes_sol.append(vecino)
        fuentes_cost.append(c)
        trials.append(0)
        ultimo_mov.append(mov)

    best_idx = min(range(len(fuentes_cost)), key=fuentes_cost.__getitem__)
    sol_mejor = copiar_solucion_labels(fuentes_sol[best_idx])
    costo_mejor = fuentes_cost[best_idx]
    mejoras = 0
    scouts = 0
    historial_best: list[float] = []
    ultimo_mov_aceptado: MovimientoVecindario | None = None

    for _it in range(iteraciones):
        if guardar_historial:
            historial_best.append(costo_mejor)

        # 1) Fase de abejas empleadas.
        for i in range(num_fuentes):
            vecino, mov = _vecino_local(
                fuentes_sol[i],
                rng=rng,
                operadores=operadores,
                marcador_depot_etiqueta=marcador_depot_etiqueta,
                usar_gpu=usar_gpu,
                backend_vecindario=backend_vecindario,
                encoding=encoding,
            )
            c = evaluar_costo_solucion(
                vecino,
                data,
                G,
                marcador_depot_etiqueta=marcador_depot_etiqueta,
                usar_gpu=usar_gpu,
            )
            if c < fuentes_cost[i]:
                fuentes_sol[i] = vecino
                fuentes_cost[i] = c
                trials[i] = 0
                ultimo_mov[i] = mov
                ultimo_mov_aceptado = mov
            else:
                trials[i] += 1

        # 2) Fase de abejas observadoras (selección por fitness).
        fitness = [_fitness_por_costo(c) for c in fuentes_cost]
        total_fit = sum(fitness)
        if total_fit <= 0:
            probs = [1.0 / num_fuentes] * num_fuentes
        else:
            probs = [f / total_fit for f in fitness]

        for _ in range(num_fuentes):
            idx = rng.choices(range(num_fuentes), weights=probs, k=1)[0]
            vecino, mov = _vecino_local(
                fuentes_sol[idx],
                rng=rng,
                operadores=operadores,
                marcador_depot_etiqueta=marcador_depot_etiqueta,
                usar_gpu=usar_gpu,
                backend_vecindario=backend_vecindario,
                encoding=encoding,
            )
            c = evaluar_costo_solucion(
                vecino,
                data,
                G,
                marcador_depot_etiqueta=marcador_depot_etiqueta,
                usar_gpu=usar_gpu,
            )
            if c < fuentes_cost[idx]:
                fuentes_sol[idx] = vecino
                fuentes_cost[idx] = c
                trials[idx] = 0
                ultimo_mov[idx] = mov
                ultimo_mov_aceptado = mov
            else:
                trials[idx] += 1

        # 3) Fase de scouts: reinicio de fuentes estancadas.
        best_idx = min(range(len(fuentes_cost)), key=fuentes_cost.__getitem__)
        base_best = fuentes_sol[best_idx]
        for i in range(num_fuentes):
            if trials[i] < limite_abandono:
                continue
            # Reinicio alrededor del mejor actual para preservar intensificación.
            reinicio, mov = _vecino_local(
                base_best,
                rng=rng,
                operadores=operadores,
                marcador_depot_etiqueta=marcador_depot_etiqueta,
                usar_gpu=usar_gpu,
                backend_vecindario=backend_vecindario,
                encoding=encoding,
            )
            c = evaluar_costo_solucion(
                reinicio,
                data,
                G,
                marcador_depot_etiqueta=marcador_depot_etiqueta,
                usar_gpu=usar_gpu,
            )
            fuentes_sol[i] = reinicio
            fuentes_cost[i] = c
            trials[i] = 0
            ultimo_mov[i] = mov
            scouts += 1

        # Actualización del mejor global.
        best_idx = min(range(len(fuentes_cost)), key=fuentes_cost.__getitem__)
        if fuentes_cost[best_idx] < costo_mejor:
            costo_mejor = fuentes_cost[best_idx]
            sol_mejor = copiar_solucion_labels(fuentes_sol[best_idx])
            mejoras += 1

    elapsed = time.perf_counter() - t0
    gap, mejora_abs, mejora_pct = calcular_metricas_gap(costo_ref, costo_mejor)

    archivo_csv: str | None = None
    if guardar_csv:
        ruta = ruta_csv or f"resultados_busqueda_abejas_{nombre_instancia}.csv"
        detalle_txt, costo_total_reporte = generar_reporte_detallado(
            sol_mejor,
            data,
            G,
            nombre_instancia=nombre_instancia,
            marcador_depot_etiqueta=marcador_depot_etiqueta,
            usar_gpu=usar_gpu,
        )
        fila = {
            "metaheuristica": "busqueda_abejas",
            "instancia": nombre_instancia,
            "semilla": semilla,
            "tiempo_segundos": elapsed,
            "iteraciones_totales": iteraciones,
            "fuentes_alimento": num_fuentes,
            "scouts_reinicios": scouts,
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

    return AbejasResult(
        mejor_solucion=sol_mejor,
        mejor_costo=costo_mejor,
        solucion_inicial_referencia=sol_ref,
        costo_inicial_referencia=costo_ref,
        gap_porcentaje=gap,
        mejora_absoluta=mejora_abs,
        mejora_porcentaje=mejora_pct,
        tiempo_segundos=elapsed,
        iteraciones_totales=iteraciones,
        fuentes_alimento=num_fuentes,
        scouts_reinicios=scouts,
        mejoras=mejoras,
        semilla=semilla,
        historial_mejor_costo=historial_best,
        ultimo_movimiento_aceptado=ultimo_mov_aceptado,
        archivo_csv=archivo_csv,
    )


def busqueda_abejas_desde_instancia(
    nombre_instancia: str,
    *,
    root: str | None = None,
    iteraciones: int = 250,
    num_fuentes: int = 16,
    limite_abandono: int = 35,
    semilla: int | None = None,
    operadores: Iterable[str] = OPERADORES_POPULARES,
    marcador_depot_etiqueta: str | None = None,
    usar_gpu: bool = False,
    backend_vecindario: Literal["labels", "ids"] = "labels",
    guardar_historial: bool = True,
    guardar_csv: bool = False,
    ruta_csv: str | None = None,
) -> AbejasResult:
    """Helper de ABC cargando recursos desde la instancia."""
    data = load_instances(nombre_instancia, root=root)
    G = cargar_objeto_gexf(nombre_instancia, root=root)
    inicial_obj = cargar_solucion_inicial(nombre_instancia, root=root)
    return busqueda_abejas(
        inicial_obj,
        data,
        G,
        iteraciones=iteraciones,
        num_fuentes=num_fuentes,
        limite_abandono=limite_abandono,
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
