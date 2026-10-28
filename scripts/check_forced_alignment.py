#!/usr/bin/env python
"""check_forced_alignment.py -- verificar el parser de .syseg ANTES de construir el cache.

El parser de src/data/speechcue.py (read_forced_alignment) se escribio sin acceso a
IEMOCAP. Este script lo contrasta con los ficheros reales:

  1. imprime las primeras lineas de unos .syseg tal cual estan en disco;
  2. lista las etiquetas que el parser trata como silencio/relleno y las que cuenta
     como silaba (si aparece algo raro en la segunda lista, el filtro esta mal);
  3. compara, utterance a utterance, el numero de silabas alineadas con el de la
     heuristica de vocales sobre la transcripcion.

    python scripts\\check_forced_alignment.py
    python scripts\\check_forced_alignment.py --sessions 1 --n 300

Lo que deberia salir: correlacion de conteos alta (r > 0.8) y una razon mediana
alineacion/heuristica cercana a 1. Si no, NO correr los estudios 88 ni 94.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.speechcue import (  # noqa: E402
    count_syllables, forced_alignment_path, is_fa_silence, read_forced_alignment,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--sessions", default="1,2,3,4,5")
    ap.add_argument("--n", type=int, default=0, help="limitar a N utterances (0 = todas)")
    ap.add_argument("--show", type=int, default=3, help="ficheros a imprimir en crudo")
    args = ap.parse_args()

    from scripts.fit_speechcue_stats import enumerate_utterances
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    items = enumerate_utterances(cfg, [int(x) for x in args.sessions.split(",")])
    if args.n:
        items = items[:args.n]

    silencio, silabas = Counter(), Counter()
    pares, faltan, errores = [], [], []
    for k, (uid, path, transcript) in enumerate(items):
        fa = forced_alignment_path(path)
        if not fa.exists():
            faltan.append(uid)
            continue
        if k < args.show:
            print(f"\n----- {fa}")
            for line in fa.read_text(encoding="utf-8", errors="replace").splitlines()[:8]:
                print("   " + line)
        for line in fa.read_text(encoding="utf-8", errors="replace").splitlines():
            tok = line.split()
            if len(tok) >= 4 and tok[0].lstrip("-").isdigit() and tok[1].lstrip("-").isdigit():
                lab = " ".join(tok[3:])
                (silencio if is_fa_silence(lab) else silabas)[lab] += 1
        try:
            n_fa, span = read_forced_alignment(fa)
        except ValueError as e:
            errores.append((uid, str(e)))
            continue
        pares.append((count_syllables(transcript), n_fa, span))

    print(f"\n  utterances: {len(items)}   sin .syseg: {len(faltan)}   errores: {len(errores)}")
    for uid, e in errores[:5]:
        print(f"    {uid}: {e}")
    print(f"\n  etiquetas tratadas como silencio/relleno ({len(silencio)} distintas):")
    print("    " + ", ".join(f"{l} ({c})" for l, c in silencio.most_common(15)))
    print(f"\n  etiquetas contadas como silaba ({len(silabas)} distintas), las 25 mas comunes:")
    print("    " + ", ".join(f"{l} ({c})" for l, c in silabas.most_common(25)))

    if len(pares) > 2:
        import numpy as np
        a = np.array(pares, dtype=float)
        r = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
        ok = a[:, 0] > 0
        ratio = np.median(a[ok, 1] / a[ok, 0]) if ok.any() else float("nan")
        rate = a[a[:, 2] > 0, 1] / a[a[:, 2] > 0, 2]
        print(f"\n  silabas heuristica vs alineacion: r = {r:.3f}   "
              f"razon mediana (alineacion/heuristica) = {ratio:.2f}")
        print(f"  tasa alineada (sil/s): mediana {np.median(rate):.2f}   "
              f"p10 {np.percentile(rate, 10):.2f}   p90 {np.percentile(rate, 90):.2f}")
        veredicto = r > 0.8 and 0.7 < ratio < 1.4
        print("\n  " + ("OK: el parser cuenta silabas de forma coherente."
                        if veredicto else
                        "REVISAR: el conteo no cuadra con la transcripcion; "
                        "no construir el cache FA hasta entender por que."))
        return 0 if veredicto and not errores else 1
    print("\n  sin pares suficientes para comparar")
    return 1


if __name__ == "__main__":
    sys.exit(main())
