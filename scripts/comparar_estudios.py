#!/usr/bin/env python
"""comparar_estudios.py -- comparacion pareada por fold de dos estudios, lista para presentar.

Lee los kfold_summary de checkpoints/ y aplica la regla de decision del repo: un cambio se
adopta si gana por mas de 0.86 pp en al menos 4 de 5 folds (IEMOCAP, LOSO).

    python scripts\\comparar_estudios.py 107 98            # el 107 frente al 98, semilla 42
    python scripts\\comparar_estudios.py 98 74 --semillas 42 123 456   # media de semillas
    python scripts\\comparar_estudios.py 98 71 --metrica wa

Solo lee JSON; no toca la GPU.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAVE = {"ua": "test_unweighted_accuracy", "wa": "test_weighted_accuracy"}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def por_fold(estudio: int, semilla: int, metrica: str) -> dict:
    sufijo = "" if semilla == 42 else f"_seed{semilla}"
    ruta = ROOT / "checkpoints" / f"ablation_{estudio}" / f"kfold_summary_ablation_{estudio}{sufijo}.json"
    if not ruta.is_file() and semilla == 42:
        ruta = ruta.with_name(f"kfold_summary_ablation_{estudio}_seed42.json")
    if not ruta.is_file():
        raise SystemExit(f"No existe el resumen {ruta}")
    d = json.load(open(ruta))
    if not d.get("complete", True):
        print(f"[aviso] {ruta.name} esta INCOMPLETO")
    return {int(f["test_session"]): 100 * float(f[CLAVE[metrica]]) for f in d["folds"]}


def media(estudio: int, semillas, metrica: str) -> dict:
    tablas = [por_fold(estudio, s, metrica) for s in semillas]
    comunes = set.intersection(*(set(t) for t in tablas))
    return {k: sum(t[k] for t in tablas) / len(tablas) for k in sorted(comunes)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=int, help="estudio candidato")
    ap.add_argument("b", type=int, help="estudio de referencia")
    ap.add_argument("--semillas", type=int, nargs="+", default=[42])
    ap.add_argument("--semillas-b", type=int, nargs="+", default=None, dest="semillas_b")
    ap.add_argument("--metrica", choices=list(CLAVE), default="ua")
    ap.add_argument("--banda", type=float, default=0.86)
    args = ap.parse_args()

    a = media(args.a, args.semillas, args.metrica)
    b = media(args.b, args.semillas_b or args.semillas, args.metrica)
    folds = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in folds]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    h = T975.get(n - 1, 1.96) * sd / math.sqrt(n) if n > 1 else float("nan")
    signos = list(itertools.product([1, -1], repeat=n))
    p = sum(abs(sum(s * x for s, x in zip(g, d)) / n) >= abs(m) - 1e-12 for g in signos) / len(signos)
    met = args.metrica.upper()

    print(f"\n  {met} de test (%), estudio {args.a} frente a {args.b}"
          f"   (semillas {args.semillas}{'' if not args.semillas_b else ' / ' + str(args.semillas_b)})\n")
    print(f"  {'fold':<6}{args.b:>10}{args.a:>10}{'delta':>10}")
    for k, x in zip(folds, d):
        print(f"  S{k:<5}{b[k]:>10.2f}{a[k]:>10.2f}{x:>+10.2f}")
    ma = sum(a[k] for k in folds) / n
    mb = sum(b[k] for k in folds) / n
    print(f"  {'media':<6}{mb:>10.2f}{ma:>10.2f}{m:>+10.2f}")
    print(f"\n  delta = {m:+.2f} +- {sd:.2f} pp   IC 95 % [{m - h:+.2f}, {m + h:+.2f}]   "
          f"p (permutacion) = {p:.4f}")
    gana = sum(x > args.banda for x in d)
    pierde = sum(x < -args.banda for x in d)
    print(f"  folds por encima de +{args.banda}: {gana}/{n}    por debajo de -{args.banda}: {pierde}/{n}")
    if gana >= 4:
        veredicto = f"el {args.a} GANA segun la regla (> {args.banda} pp en >= 4/5 folds)"
    elif pierde >= 4:
        veredicto = f"el {args.a} PIERDE segun la regla"
    else:
        veredicto = "EMPATE: la diferencia no sale de la banda de ruido"
    print(f"  veredicto: {veredicto}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
