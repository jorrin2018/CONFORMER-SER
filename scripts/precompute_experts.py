#!/usr/bin/env python
"""precompute_experts.py -- Etapa 1 de la compuerta con expertos: predicciones por fold.

Idea (pedida el 04/11/2026): los clasificadores que deciden el conflicto no se entrenan
junto con la fusion, sino ANTES y POR SEPARADO, cada uno sobre su modalidad; despues
se congelan y solo aportan sus probabilidades al entrenamiento global.

Los expertos de audio y texto ya existen: son las lineas base unimodales LOSO
(estudio 71, audio; estudio 73, texto), un checkpoint por fold, con el mismo reparto
que el 74 (4 sesiones de train, 12 % estratificado para validacion, semilla 42).
Falta el de imagen: el ViT va congelado en CAMF, asi que su experto es una regresion
logistica sobre el [CLS] del ViT congelado, ajustada en el train de cada fold y con
C elegido por UA de validacion. Segundos en CPU.

Como los expertos estan congelados, sus predicciones se calculan UNA vez por fold y se
guardan: la etapa 2 no necesita cargar tres encoders mas en la GPU.

    python scripts\\precompute_experts.py --device cuda
    python scripts\\precompute_experts.py --device cuda --folds 1 2

Escribe cache\\experts\\fold_S{k}.json con, por utterance: reparto (train/val/test),
etiqueta y las probabilidades de cada experto.

Comprobacion de carga: el UA de test que da cada experto se compara con el
final_results.json de su corrida. Si difiere en mas de 0.5 pp, el script PARA: un
checkpoint mal cargado daria predicciones plausibles y falsas.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_ablation import create_ablation_model  # noqa: E402
from src.data import collate_fn  # noqa: E402
from src.models.forward_utils import multimodal_forward  # noqa: E402
from src.train import create_dataloaders  # noqa: E402

OUT_DEFAULT = ROOT / "cache" / "experts"
TOL_PP = 0.5


def unweighted_accuracy(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    recalls = [float((pred[y == c] == c).mean()) for c in range(n_classes) if (y == c).any()]
    return float(np.mean(recalls))


def fold_run_dir(study: int, fold: int) -> Path:
    """La corrida que entra en el kfold_summary del estudio (no la ultima que haya)."""
    summ = ROOT / "checkpoints" / f"ablation_{study}" / f"kfold_summary_ablation_{study}.json"
    if not summ.exists():
        raise SystemExit(f"No encuentro {summ}")
    for f in json.loads(summ.read_text(encoding="utf-8"))["folds"]:
        if int(f["test_session"]) == fold:
            d = ROOT / Path(f["results_dir"].replace("\\", "/"))
            if not (d / "best_model.pt").exists():
                raise SystemExit(f"Falta best_model.pt en {d}")
            return d
    raise SystemExit(f"El fold S{fold} no esta en {summ}")


def eval_config(run_dir: Path) -> dict:
    """config.json de la corrida, sin aumento de datos."""
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cfg.setdefault("audio", {})["apply_waveform_augment"] = False
    cfg.setdefault("text", {})["apply_text_augment"] = False
    cfg.setdefault("compute", {})["num_workers"] = 0
    return cfg


def split_datasets(cfg: dict, log):
    train_loader, val_loader, test_loader, _ = create_dataloaders(cfg, log)
    out = {}
    for name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        ds = loader.dataset
        ds.training = False            # sin recorte aleatorio de los > 8 s
        out[name] = ds
    return out


def loader(ds, bs: int) -> DataLoader:
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate_fn)


@torch.no_grad()
def predict(model, datasets: dict, device: str, bs: int):
    probs, labels, split = {}, {}, {}
    model.eval()
    for name, ds in datasets.items():
        for b in loader(ds, bs):
            audio = b["audio"].to(device)
            ids = b["text_input_ids"].to(device)
            mask = b["text_attention_mask"].to(device)
            out = multimodal_forward(model, audio, ids, mask, batch=b)
            p = torch.softmax(out["logits"].float(), dim=-1).cpu().numpy()
            for uid, pi, y in zip(b["utterance_ids"], p, b["labels"].tolist()):
                probs[uid], labels[uid], split[uid] = pi.tolist(), int(y), name
    return probs, labels, split


def ua_of(probs: dict, labels: dict, split: dict, which: str, n_classes: int) -> float:
    uids = [u for u in probs if split[u] == which]
    y = np.array([labels[u] for u in uids])
    pred = np.array([int(np.argmax(probs[u])) for u in uids])
    return unweighted_accuracy(y, pred, n_classes)


def run_expert(study: int, fold: int, device: str, bs: int, log):
    run_dir = fold_run_dir(study, fold)
    cfg = eval_config(run_dir)
    datasets = split_datasets(cfg, log)
    model = create_ablation_model(study, cfg, device, log)
    ck = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    probs, labels, split = predict(model, datasets, device, bs)
    n_classes = len(cfg["dataset"]["emotions"])
    ua = ua_of(probs, labels, split, "test", n_classes)
    ref = json.loads((run_dir / "final_results.json").read_text(encoding="utf-8"))
    ref_ua = float(ref["test_metrics"]["unweighted_accuracy"])
    print(f"    estudio {study}: test UA {100 * ua:.2f}  (final_results {100 * ref_ua:.2f})")
    if abs(ua - ref_ua) * 100 > TOL_PP:
        raise SystemExit(
            f"  El UA de test reproducido ({100 * ua:.2f}) no coincide con el de la corrida "
            f"({100 * ref_ua:.2f}) en {run_dir}. Checkpoint o config mal cargados: PARO.")
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return dict(probs=probs, labels=labels, split=split, run_dir=str(run_dir),
                emotions=cfg["dataset"]["emotions"], test_ua=ua, ref_test_ua=ref_ua,
                train_ua=ua_of(probs, labels, split, "train", n_classes),
                val_ua=ua_of(probs, labels, split, "val", n_classes))


# ------------------------------------------------------------------ vision
@torch.no_grad()
def vit_cls_features(vision_study: int, device: str, bs: int, log) -> dict:
    """[CLS] del ViT congelado para TODAS las utterances, con la tuberia log-mel del CAMF."""
    from torchaudio.transforms import MelSpectrogram
    from transformers import ViTModel
    from src.ablation_studies.ablation_11_model import Ablation11Model

    run_dir = fold_run_dir(vision_study, 1)
    cfg = eval_config(run_dir)
    m, a, d = cfg["model"], cfg.get("audio", {}), cfg["dataset"]
    # El mismo codigo que usa el modelo (_waveform_to_vit_image), con un "self" minimo.
    me = types.SimpleNamespace(
        mel_transform=MelSpectrogram(
            sample_rate=d.get("sample_rate", 16000),
            n_fft=int(m.get("mel_n_fft", a.get("n_fft", 400))),
            hop_length=int(m.get("mel_hop_length", a.get("hop_length", 160))),
            win_length=int(m.get("mel_win_length", a.get("win_length", 400))),
            n_mels=int(m.get("mel_n_mels", a.get("n_mels", 80))),
            normalized=True),
        vit_image_size=int(m.get("vision_image_size", 224)),
        _vision_mel_speech_only=False, _vision_mel_spec_augment=False,
        _vit_freq_mask=None, _vit_time_mask=None, training=False)
    name = m.get("vision_encoder", "google/vit-base-patch16-224")
    try:
        vit = ViTModel.from_pretrained(name, use_safetensors=True)
    except Exception:  # noqa: BLE001  -- solo .bin: el error, si lo hay, sera el real
        vit = ViTModel.from_pretrained(name)
    vit = vit.to(device).eval()
    feats = {}
    for ds in split_datasets(cfg, log).values():         # train+val+test = 5 sesiones
        for b in loader(ds, bs):
            img = Ablation11Model._waveform_to_vit_image(me, b["audio"]).to(device)
            cls = vit(pixel_values=img).last_hidden_state[:, 0, :].float().cpu().numpy()
            for uid, f in zip(b["utterance_ids"], cls):
                feats[uid] = f
    print(f"    ViT [CLS]: {len(feats)} utterances ({name}, mel de {run_dir.name})")
    return feats


def fit_vision_expert(feats: dict, labels: dict, split: dict, n_classes: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    tr = [u for u in split if split[u] == "train"]
    va = [u for u in split if split[u] == "val"]
    X = lambda us: np.stack([feats[u] for u in us])  # noqa: E731
    y = lambda us: np.array([labels[u] for u in us])  # noqa: E731
    sc = StandardScaler().fit(X(tr))
    best = None
    for C in (0.001, 0.01, 0.1, 1.0):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
        clf.fit(sc.transform(X(tr)), y(tr))
        ua = unweighted_accuracy(y(va), clf.predict(sc.transform(X(va))), n_classes)
        if best is None or ua > best[0]:
            best = (ua, C, clf)
    _, C, clf = best
    uids = list(split)
    P = clf.predict_proba(sc.transform(X(uids)))
    probs = {u: p.tolist() for u, p in zip(uids, P)}
    return probs, C


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-study", type=int, default=71)
    ap.add_argument("--text-study", type=int, default=73)
    ap.add_argument("--vision-mel-from", type=int, default=74,
                    help="estudio del que se copia la tuberia log-mel del ViT")
    ap.add_argument("--ref-study", type=int, default=None,
                    help="opcional: guarda tambien las probabilidades de un modelo de "
                         "referencia (p. ej. 74) para comparar su calibracion (ECE, Brier)")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--force", action="store_true",
                    help="recalcular los expertos aunque el fold ya este en disco")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    logging.basicConfig(level=logging.WARNING)
    log = logging.getLogger("experts")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Los folds ya escritos no se recalculan: si la corrida se corto en S3, se
    # reanuda en S4. Con --ref-study solo se anade la referencia a lo que ya hay.
    existentes = {k: json.loads((out / f"fold_S{k}.json").read_text(encoding="utf-8"))
                  for k in args.folds
                  if (out / f"fold_S{k}.json").exists() and not args.force}
    pendientes = [k for k in args.folds if k not in existentes]
    if existentes:
        print(f"  ya en disco (no se recalculan): {['S%d' % k for k in sorted(existentes)]}")

    feats = None
    if pendientes:
        print("\n  ViT congelado: extrayendo [CLS] una sola vez")
        feats = vit_cls_features(args.vision_mel_from, args.device, args.batch_size, log)

    for k in args.folds:
        print(f"\n  Fold S{k}")
        path = out / f"fold_S{k}.json"
        if k in existentes:
            payload = existentes[k]
        else:
            audio = run_expert(args.audio_study, k, args.device, args.batch_size, log)
            text = run_expert(args.text_study, k, args.device, args.batch_size, log)
            if audio["split"] != text["split"] or audio["labels"] != text["labels"]:
                raise SystemExit("  Los expertos de audio y texto no ven el mismo reparto: PARO.")
            n_classes = len(audio["emotions"])
            missing = [u for u in audio["split"] if u not in feats]
            if missing:
                raise SystemExit(f"  {len(missing)} utterances sin [CLS] del ViT: {missing[:3]}")
            vprobs, C = fit_vision_expert(feats, audio["labels"], audio["split"], n_classes)
            vua = {s: ua_of(vprobs, audio["labels"], audio["split"], s, n_classes)
                   for s in ("train", "val", "test")}
            print(f"    vision (logistica, C={C}): train {100 * vua['train']:.2f}  "
                  f"val {100 * vua['val']:.2f}  test {100 * vua['test']:.2f}")
            payload = {
                "meta": {
                    "fold": k, "emotions": audio["emotions"],
                    "audio": {"study": args.audio_study, "run_dir": audio["run_dir"],
                              "ua": {"train": audio["train_ua"], "val": audio["val_ua"],
                                     "test": audio["test_ua"]}},
                    "text": {"study": args.text_study, "run_dir": text["run_dir"],
                             "ua": {"train": text["train_ua"], "val": text["val_ua"],
                                    "test": text["test_ua"]}},
                    "vision": {"model": "logistic regression on frozen ViT [CLS]", "C": C,
                               "mel_from_study": args.vision_mel_from, "ua": vua},
                    "counts": {s: sum(1 for v in audio["split"].values() if v == s)
                               for s in ("train", "val", "test")},
                },
                "split": audio["split"], "labels": audio["labels"],
                "probs": {"audio": audio["probs"], "text": text["probs"], "vision": vprobs},
            }
        if args.ref_study is not None and "ref" not in payload:
            ref = run_expert(args.ref_study, k, args.device, args.batch_size, log)
            if ref["labels"] != payload["labels"] or ref["split"] != payload["split"]:
                # El 74 de la semilla 42 tiene el mismo reparto que 71/73; otra
                # semilla daria otra validacion y la comparacion no seria pareada.
                raise SystemExit(f"  El estudio {args.ref_study} no ve el mismo reparto que "
                                 "los expertos en este fold: PARO.")
            payload["ref"] = {"study": args.ref_study, "run_dir": ref["run_dir"],
                              "probs": ref["probs"],
                              "ua": {"train": ref["train_ua"], "val": ref["val_ua"],
                                     "test": ref["test_ua"]}}
        path.write_text(json.dumps(payload), encoding="utf-8")
        print(f"    -> {path}")
    print("\n  Siguiente paso: python scripts\\expert_gate_offline.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
