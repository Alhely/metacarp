from __future__ import annotations

import csv
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Hashable

import networkx as nx

from .costo_solucion import costo_solucion
from .reporte_solucion import reporte_solucion

__all__ = [
    "copiar_solucion_labels",
    "extraer_candidatas_desde_objeto",
    "evaluar_costo_solucion",
    "seleccionar_mejor_inicial",
    "calcular_metricas_gap",
    "solucion_legible_humana",
    "generar_reporte_detallado",
    "guardar_resultado_csv",
]


def copiar_solucion_labels(sol: Sequence[Sequence[Hashable]]) -> list[list[str]]:
    """Copia ligera a formato de etiquetas string."""
    return [[str(x).strip() for x in ruta] for ruta in sol]


def _es_solucion_lista_de_rutas(obj: Any) -> bool:
    """
    Heurística para detectar estructura de solución:
    lista/tupla de rutas y cada ruta lista/tupla de tokens.
    """
    if not isinstance(obj, (list, tuple)):
        return False
    if isinstance(obj, (str, bytes)):
        return False
    for ruta in obj:
        if not isinstance(ruta, (list, tuple)):
            return False
        if isinstance(ruta, (str, bytes)):
            return False
        for token in ruta:
            if isinstance(token, Mapping):
                return False
    return True


def extraer_candidatas_desde_objeto(obj: Any, *, max_nodos: int = 20000) -> list[list[list[str]]]:
    """
    Busca recursivamente candidatas en un objeto inicial arbitrario.
    Soporta solución única, dict de soluciones y estructuras anidadas.
    """
    candidatas: list[list[list[str]]] = []
    visitados = 0

    def _walk(x: Any) -> None:
        nonlocal visitados
        visitados += 1
        if visitados > max_nodos:
            return

        if _es_solucion_lista_de_rutas(x):
            candidatas.append(copiar_solucion_labels(x))
            return

        if isinstance(x, Mapping):
            for v in x.values():
                _walk(v)
            return

        if isinstance(x, (list, tuple, set)):
            for v in x:
                _walk(v)
            return

    _walk(obj)
    return candidatas


def evaluar_costo_solucion(
    solucion: Sequence[Sequence[Hashable]],
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    marcador_depot_etiqueta: str | None,
    usar_gpu: bool,
) -> float:
    """Evalúa costo total con el evaluador central del paquete."""
    return costo_solucion(
        solucion,
        data,
        G,
        detalle=False,
        marcador_depot_etiqueta=marcador_depot_etiqueta,
        usar_gpu=usar_gpu,
    ).costo_total


def seleccionar_mejor_inicial(
    inicial_obj: Any,
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    marcador_depot_etiqueta: str | None,
    usar_gpu: bool,
) -> tuple[list[list[str]], float]:
    """Devuelve la candidata inicial de menor costo y su valor."""
    candidatas = extraer_candidatas_desde_objeto(inicial_obj)
    if not candidatas:
        raise ValueError(
            "No se encontraron soluciones candidatas en el objeto inicial. "
            "Se esperaba lista de rutas o estructura anidada que la contenga."
        )

    mejor_sol: list[list[str]] | None = None
    mejor_cost = float("inf")
    errores = 0
    for cand in candidatas:
        try:
            c = evaluar_costo_solucion(
                cand,
                data,
                G,
                marcador_depot_etiqueta=marcador_depot_etiqueta,
                usar_gpu=usar_gpu,
            )
        except Exception:  # noqa: BLE001 - candidatas heterogéneas en pickle.
            errores += 1
            continue
        if c < mejor_cost:
            mejor_cost = c
            mejor_sol = cand

    if mejor_sol is None:
        raise ValueError(
            "Ninguna candidata inicial pudo evaluarse con costo_solucion. "
            f"Candidatas detectadas: {len(candidatas)} | inválidas: {errores}."
        )
    return mejor_sol, mejor_cost


def calcular_metricas_gap(costo_inicial: float, costo_mejor: float) -> tuple[float, float, float]:
    """
    Devuelve (gap_pct, mejora_abs, mejora_pct).
    Gap negativo indica mejora contra la referencia inicial.
    """
    if costo_inicial == 0:
        gap = 0.0 if costo_mejor == 0 else float("inf")
        mejora_abs = costo_inicial - costo_mejor
        mejora_pct = 0.0
        return gap, mejora_abs, mejora_pct
    gap = ((costo_mejor - costo_inicial) / costo_inicial) * 100.0
    mejora_abs = costo_inicial - costo_mejor
    mejora_pct = (mejora_abs / costo_inicial) * 100.0
    return gap, mejora_abs, mejora_pct


def solucion_legible_humana(solucion: Sequence[Sequence[Hashable]]) -> str:
    """
    Convierte una solución a texto legible:
    R1: D -> TR1 -> TR5 -> D || R2: D -> TR3 -> D
    """
    rutas_txt: list[str] = []
    for i, ruta in enumerate(solucion, start=1):
        seq = [str(x).strip() for x in ruta]
        rutas_txt.append(f"R{i}: " + " -> ".join(seq))
    return " || ".join(rutas_txt)


def generar_reporte_detallado(
    solucion: Sequence[Sequence[Hashable]],
    data: Mapping[str, Any],
    G: nx.Graph,
    *,
    nombre_instancia: str = "instancia",
    marcador_depot_etiqueta: str | None,
    usar_gpu: bool,
) -> tuple[str, float]:
    """
    Genera detalle con deadheading y costo total reutilizando reporte_solucion.
    """
    rep = reporte_solucion(
        solucion,
        data,
        G,
        nombre_instancia=nombre_instancia,
        marcador_depot_etiqueta=marcador_depot_etiqueta,
        usar_gpu=usar_gpu,
        guardar=False,
    )
    return rep.texto, rep.costo_total


def guardar_resultado_csv(
    *,
    fila: Mapping[str, Any],
    ruta_csv: str | Path,
) -> str:
    """
    Guarda una ejecución en CSV (una fila por corrida, una columna por item).
    Si el archivo no existe, escribe encabezado.
    """
    path = Path(ruta_csv).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Normalizamos valores complejos a texto para mantener formato tabular.
    normalizada: dict[str, Any] = {}
    for k, v in fila.items():
        if isinstance(v, (list, tuple, set, dict)):
            normalizada[k] = str(v)
        else:
            normalizada[k] = v

    existe = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(normalizada.keys()))
        if not existe:
            writer.writeheader()
        writer.writerow(normalizada)
    return str(path)
