"""
Script de experimentación para metaheurísticas CARP.

Objetivo:
- Ejecutar SA / Tabu / Abejas / Cuckoo por instancia.
- Guardar CSV por metaheurística e instancia.
- Permitir corridas reproducibles y configuración sencilla.

Ejemplos:
    python metacarp/scripts/experimentos.py
    python metacarp/scripts/experimentos.py --instancias gdb19 --metaheuristicas sa tabu
    python metacarp/scripts/experimentos.py --seed 7 --salida-dir resultados/experimentos
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from metacarp import (
    busqueda_abejas_desde_instancia,
    busqueda_tabu_desde_instancia,
    cuckoo_search_desde_instancia,
    nombres_soluciones_iniciales_disponibles,
    recocido_simulado_desde_instancia,
)

# Alias canónicos de metaheurísticas disponibles en este script.
METAHEURISTICAS_VALIDAS = ("sa", "tabu", "abejas", "cuckoo")


@dataclass(frozen=True, slots=True)
class MetaRunner:
    nombre: str
    run: Callable[..., object]
    parametros_default: dict[str, object]


def _construir_runners() -> dict[str, MetaRunner]:
    """
    Define configuración default de cada metaheurística.
    Ajusta estos valores para campañas más largas/cortas.
    """
    return {
        "sa": MetaRunner(
            nombre="sa",
            run=recocido_simulado_desde_instancia,
            parametros_default={
                "temperatura_inicial": 250.0,
                "temperatura_minima": 1e-3,
                "alpha": 0.95,
                "iteraciones_por_temperatura": 80,
                "max_enfriamientos": 60,
            },
        ),
        "tabu": MetaRunner(
            nombre="tabu",
            run=busqueda_tabu_desde_instancia,
            parametros_default={
                "iteraciones": 250,
                "tam_vecindario": 24,
                "tenure_tabu": 20,
            },
        ),
        "abejas": MetaRunner(
            nombre="abejas",
            run=busqueda_abejas_desde_instancia,
            parametros_default={
                "iteraciones": 220,
                "num_fuentes": 14,
                "limite_abandono": 28,
            },
        ),
        "cuckoo": MetaRunner(
            nombre="cuckoo",
            run=cuckoo_search_desde_instancia,
            parametros_default={
                "iteraciones": 220,
                "num_nidos": 16,
                "pa_abandono": 0.25,
                "pasos_levy_base": 3,
                "beta_levy": 1.5,
            },
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta campañas de experimentación para metaheurísticas CARP."
    )
    parser.add_argument(
        "--instancias",
        nargs="*",
        default=["all"],
        help="Lista de instancias (por defecto: all = todas con solución inicial).",
    )
    parser.add_argument(
        "--metaheuristicas",
        nargs="*",
        default=["all"],
        help="Subconjunto a ejecutar: sa tabu abejas cuckoo (por defecto: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla base para reproducibilidad.",
    )
    parser.add_argument(
        "--salida-dir",
        type=str,
        default="resultados/experimentos",
        help="Carpeta raíz donde se guardan los CSV.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root de datos opcional (si no, usa configuración por defecto del paquete).",
    )
    return parser.parse_args()


def _resolver_instancias(raw: Sequence[str], *, root: str | None) -> list[str]:
    if not raw or (len(raw) == 1 and raw[0].lower() == "all"):
        return sorted(nombres_soluciones_iniciales_disponibles(root=root))
    return list(raw)


def _resolver_metaheuristicas(raw: Sequence[str]) -> list[str]:
    if not raw or (len(raw) == 1 and raw[0].lower() == "all"):
        return list(METAHEURISTICAS_VALIDAS)

    seleccion = [x.lower() for x in raw]
    invalidas = [x for x in seleccion if x not in METAHEURISTICAS_VALIDAS]
    if invalidas:
        raise ValueError(
            f"Metaheurísticas inválidas: {invalidas}. "
            f"Válidas: {METAHEURISTICAS_VALIDAS}."
        )
    return seleccion


def _ruta_csv(salida_dir: Path, meta: str, instancia: str) -> Path:
    # Un CSV por metaheurística e instancia (se irá acumulando una fila por corrida).
    return salida_dir / meta / f"{instancia}.csv"


def main() -> None:
    args = _parse_args()
    salida_dir = Path(args.salida_dir).expanduser().resolve()
    salida_dir.mkdir(parents=True, exist_ok=True)

    runners = _construir_runners()
    instancias = _resolver_instancias(args.instancias, root=args.root)
    metas = _resolver_metaheuristicas(args.metaheuristicas)

    if not instancias:
        raise RuntimeError("No se encontraron instancias para ejecutar.")

    print("=" * 96)
    print("EXPERIMENTOS METAHEURÍSTICAS")
    print("=" * 96)
    print(f"Instancias       : {instancias}")
    print(f"Metaheurísticas  : {metas}")
    print(f"Seed base        : {args.seed}")
    print(f"Salida CSV       : {salida_dir}")
    print("-" * 96)

    total_ok = 0
    total_fail = 0

    for idx_inst, instancia in enumerate(instancias):
        print(f"\n[INSTANCIA] {instancia}")

        for idx_meta, meta in enumerate(metas):
            runner = runners[meta]
            ruta_csv = _ruta_csv(salida_dir, meta, instancia)
            # Semilla derivada para que cada par (instancia, meta) sea estable y distinto.
            semilla = args.seed + (idx_inst * 1000) + idx_meta

            kwargs = dict(runner.parametros_default)
            kwargs.update(
                {
                    "root": args.root,
                    "semilla": semilla,
                    "guardar_csv": True,
                    "ruta_csv": str(ruta_csv),
                    "guardar_historial": True,
                }
            )

            print(f"  -> {meta:7s} | semilla={semilla} | csv={ruta_csv}")
            try:
                res = runner.run(instancia, **kwargs)
                # Se asume interfaz homogénea ya implementada en los Result dataclasses.
                print(
                    "     OK "
                    f"| mejor_costo={res.mejor_costo:.6f} "
                    f"| gap={res.gap_porcentaje:.4f}% "
                    f"| tiempo={res.tiempo_segundos:.4f}s"
                )
                total_ok += 1
            except Exception as exc:  # noqa: BLE001 - queremos continuar campaña.
                print(f"     FAIL | {type(exc).__name__}: {exc}")
                total_fail += 1

    print("\n" + "-" * 96)
    print(f"Ejecuciones OK   : {total_ok}")
    print(f"Ejecuciones FAIL : {total_fail}")
    print(f"CSV raíz         : {salida_dir}")
    print("-" * 96)


if __name__ == "__main__":
    main()
