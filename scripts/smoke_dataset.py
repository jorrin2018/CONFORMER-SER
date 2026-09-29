#!/usr/bin/env python
"""
smoke_dataset.py -- Carga los tres splits sin tocar la GPU ni el modelo.

Confirma en segundos que el loader parcheado ve el corpus completo, en vez de
esperar dos epocas de entrenamiento para descubrir que no.

    python scripts/smoke_dataset.py
    python scripts/smoke_dataset.py --config config/config.yaml
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import build_dataset, corpus_from_config  # noqa: E402

EXPECTED = {"train": 3259, "val": 1031, "test": 1241}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    ds_cfg = cfg["dataset"]
    print(f"\n  config    : {args.config}")
    print(f"  data_path : {ds_cfg['data_path']}\n")

    splits = {
        "train": ds_cfg["train_sessions"],
        "val": ds_cfg["val_sessions"],
        "test": ds_cfg["test_sessions"],
    }

    ok = True
    first_impro = None
    for name, sessions in splits.items():
        ds = build_dataset(
            corpus=corpus_from_config(cfg),
            data_path=ds_cfg["data_path"],
            sessions=sessions,
            emotions=ds_cfg["emotions"],
            sample_rate=ds_cfg["sample_rate"],
            training=(name == "train"),
        )
        counts = Counter(d["emotion"] for d in ds.data)
        n_impro = sum(1 for d in ds.data if "impro" in d["utterance_id"])
        exp = EXPECTED[name]
        good = len(ds) == exp and n_impro > 0
        ok &= good

        print(f"  {name:<6} sessions={sessions}  n={len(ds):<5} "
              f"(esperado {exp})  impro={n_impro:<5} "
              f"{'OK' if good else '<-- REVISAR'}")
        print("         " + "  ".join(
            f"{k}={counts[k]}" for k in ds_cfg["emotions"]))

        if first_impro is None:
            for d in ds.data:
                if "impro" in d["utterance_id"]:
                    first_impro = d
                    break

    if first_impro:
        p = Path(first_impro["audio_path"])
        print(f"\n  muestra improvisada : {first_impro['utterance_id']}")
        print(f"  ruta                : {p}")
        print(f"  existe              : {p.exists()}")
        print(f"  transcripcion       : {first_impro['transcription'][:70]!r}")
        try:
            import torchaudio
            wav, sr = torchaudio.load(str(p))
            print(f"  audio               : {wav.shape[1] / sr:.2f} s @ {sr} Hz")
        except Exception as e:  # noqa: BLE001
            print(f"  audio               : ERROR al cargar -> {e}")
            ok = False

    print("\n  " + ("Loader correcto. Listo para entrenar.\n"
                    if ok else "Algo no cuadra. NO entrenes todavia.\n"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
