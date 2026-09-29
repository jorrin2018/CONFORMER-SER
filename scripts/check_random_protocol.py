#!/usr/bin/env python
"""
check_random_protocol.py -- Verifica el reparto del protocolo 'random' antes de entrenar.

El protocolo aleatorio (Yoon et al. 2018) existe para medir, con el mismo modelo y el
mismo corpus, cuanto vale el reparto por si solo frente a LOSO. Para que esa medicion
signifique algo hay que comprobar tres cosas:

  1. Los cinco folds son una PARTICION: cada utterance es test exactamente una vez.
  2. Dentro de cada fold, train / val / test son disjuntos y suman 5 531.
  3. El reparto es DEPENDIENTE DE HABLANTE -- los mismos actores en train y test.
     Si saliera independiente de hablante, no estariamos midiendo lo que creemos.

    python scripts/check_random_protocol.py
"""
from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ablation_studies import load_base_config
from src.data import build_dataset, corpus_from_config
from src.train import _random_fold_indices

logging.basicConfig(level=logging.ERROR)


def main() -> int:
    cfg = load_base_config()
    cfg['dataset']['all_sessions'] = [1, 2, 3, 4, 5]

    sonda = build_dataset(
        corpus=corpus_from_config(cfg),
        data_path=cfg['dataset']['data_path'],
        sessions=[1, 2, 3, 4, 5],
        emotions=cfg['dataset']['emotions'],
        sample_rate=cfg['dataset']['sample_rate'],
        audio_preprocessor=None, text_preprocessor=None,
        max_audio_length=cfg['audio']['max_length'], training=False,
    )
    total = len(sonda.data)
    hablante = lambda u: f"{u[:5]}_{u.split('_')[-1][0]}"

    log = logging.getLogger("check")
    log.warning = lambda *a, **k: None          # silenciar el aviso por fold
    log.info = lambda *a, **k: None

    vistos = Counter()
    ok = True
    print(f"\n  corpus: {total} utterances\n")
    print(f"  {'fold':<6}{'train':>8}{'val':>7}{'test':>7}{'suma':>8}"
          f"{'hablantes train∩test':>24}{'clases en test':>18}")
    print("  " + "-" * 78)

    for k in range(5):
        cfg['dataset']['random_folds'] = {'n_folds': 5, 'fold_index': k, 'val_fraction': 0.06}
        tr, va, te = _random_fold_indices(cfg, log)
        vistos.update(te)

        h_tr = {hablante(sonda.data[i]['utterance_id']) for i in tr}
        h_te = {hablante(sonda.data[i]['utterance_id']) for i in te}
        comp = len(h_tr & h_te)
        clases = len({sonda.data[i]['emotion'] for i in te})
        suma = len(tr) + len(va) + len(te)
        bien = suma == total and comp == len(h_te) and clases == 4
        ok &= bien
        print(f"  {k:<6}{len(tr):>8}{len(va):>7}{len(te):>7}{suma:>8}"
              f"{f'{comp} de {len(h_te)}':>24}{clases:>18}")

    print("  " + "-" * 78)
    una_vez = sum(1 for v in vistos.values() if v == 1)
    print(f"\n  cada utterance es test exactamente una vez: {una_vez} de {total}"
          f"   {'ok' if una_vez == total else 'FALLO'}")
    ok &= una_vez == total

    if ok:
        print("\n  Reparto correcto, y DEPENDIENTE DE HABLANTE como debe ser.")
        print("  Recordatorio: este numero no es el principal. Se reporta junto al")
        print("  'standard', nunca solo.\n")
    else:
        print("\n  FALLO en el reparto. No lanzar.\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
