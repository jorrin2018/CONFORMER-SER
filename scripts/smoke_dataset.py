#!/usr/bin/env python
"""
smoke_dataset.py -- Carga los tres splits sin tocar la GPU ni el modelo.

Confirma en segundos que el loader parcheado ve el corpus completo, en vez de
esperar dos epocas de entrenamiento para descubrir que no.

    python scripts/smoke_dataset.py
    python scripts/smoke_dataset.py --config config/config.yaml
    python scripts/smoke_dataset.py --config config/ablation/ablation_79_meld_trimodal.yaml

Sirve para cualquier corpus registrado: el numero esperado por split se busca en
ESPERADO por (corpus, n_clases), y si esa combinacion no esta ahi solo se informa el
conteo sin juzgarlo (el contrato duro lo impone _assert_integrity del propio loader).

La clave lleva el numero de clases porque un mismo corpus tiene contratos DISTINTOS
segun las etiquetas que se usen: MELD a 4 clases son 8244/896/2211 y a 7 clases son
9988/1108/2610. Compararlo solo por nombre de corpus daba una falsa alarma en el par
81/82.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import build_dataset, corpus_from_config  # noqa: E402

#: (corpus, n_clases) -> {split: n esperado}.
#: IEMOCAP: el reparto del config por defecto (sesiones 1-3 / 4 / 5), 4 clases.
#: MELD: el split oficial, ya descontadas las 2 utterances sin medio
#: (ver src/data/meld_dataset.py). Los dos contratos de MELD estan tabulados porque
#: los configs 79/80 corren a 4 clases y los 81/82 a 7.
ESPERADO = {
    ("iemocap", 4): {"train": 3259, "val": 1031, "test": 1241},
    # IEMOCAP con sus siete clases propias (excited separado de happy, frustrated y
    # surprised incluidas): REF_7CLASS_PER_SESSION {1:1390, 2:1365, 3:1561, 4:1531,
    # 5:1640}. Con el reparto 1-3 / 4 / 5 son 4316 / 1531 / 1640 = 7487.
    ("iemocap", 7): {"train": 4316, "val": 1531, "test": 1640},
    ("meld", 4): {"train": 8244, "val": 896, "test": 2211},
    ("meld", 7): {"train": 9988, "val": 1108, "test": 2610},
    # RAVDESS (habla): 28 utterances de 4 clases por actor (8 angry + 8 happy +
    # 8 sad + 4 neutral, porque neutral no tiene intensidad strong). Los grupos de
    # actores salen 6/6/4/4/4, asi que con el reparto 1-3 / 4 / 5 son
    # (6+6+4)x28 = 448 / 4x28 = 112 / 4x28 = 112 = 672. Verificado el 2026-10-20.
    ("ravdess", 4): {"train": 448, "val": 112, "test": 112},
}


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

    corpus = corpus_from_config(cfg).strip().lower()
    #: El numero de clases sale de la lista de emociones del config, que es lo que
    #: el cargador usa de verdad para filtrar. num_classes del bloque model deberia
    #: coincidir; si no, se avisa, porque es un desajuste que rompe el entrenamiento
    #: mucho mas adelante y con un error que no apunta aqui.
    n_clases = len(ds_cfg.get("emotions") or [])
    n_modelo = (cfg.get("model") or {}).get("num_classes")
    if n_modelo is not None and n_modelo != n_clases:
        print(f"  AVISO: dataset.emotions tiene {n_clases} clases pero "
              f"model.num_classes es {n_modelo}. Cuadralos antes de entrenar.\n")
    esperado_corpus = ESPERADO.get((corpus, n_clases))
    if esperado_corpus is None:
        print(f"  ({corpus}, {n_clases} clases) sin cifras de referencia aqui: se "
              f"informan los conteos sin juzgarlos.\n")

    ok = True
    muestra = None
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
        exp = (esperado_corpus or {}).get(name)
        # En IEMOCAP el conteo de utterances improvisadas es el guardarrail que
        # detecto el bug de troceado que perdia 2,943 de 5,531: se conserva, pero
        # solo tiene sentido ahi (ningun otro corpus marca la improvisacion).
        extra, bien_extra = "", True
        if corpus == "iemocap":
            n_impro = sum(1 for d in ds.data if "impro" in d["utterance_id"])
            extra = f"  impro={n_impro:<5}"
            bien_extra = n_impro > 0
        good = bien_extra and (exp is None or len(ds) == exp)
        ok &= good

        print(f"  {name:<6} sessions={sessions}  n={len(ds):<5} "
              f"({'esperado ' + str(exp) if exp is not None else 'sin referencia'})"
              f"{extra}  {'OK' if good else '<-- REVISAR'}")
        print("         " + "  ".join(
            f"{k}={counts[k]}" for k in ds_cfg["emotions"]))

        if muestra is None and len(ds):
            if corpus == "iemocap":
                muestra = next((d for d in ds.data if "impro" in d["utterance_id"]),
                               ds.data[0])
            else:
                muestra = ds.data[0]

    if muestra:
        p = Path(muestra["audio_path"])
        print(f"\n  muestra             : {muestra['utterance_id']}")
        print(f"  ruta                : {p}")
        print(f"  existe              : {p.exists()}")
        print(f"  transcripcion       : {muestra['transcription'][:70]!r}")
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
