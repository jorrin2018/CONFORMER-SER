#!/usr/bin/env python
"""demo_emocion.py -- demo en vivo: que emocion predice el modelo, y que aporta cada flujo.

Carga los checkpoints YA ENTRENADOS de un fold (por defecto, el fold S5: modelos que
nunca vieron a los dos hablantes de la sesion 5) y, para cada enunciado, muestra lado a
lado lo que predicen:

    solo audio (estudio 71)  |  solo texto (73)  |  fusion audio + texto (98)

Es la forma mas directa de ensenar en vivo el hallazgo del articulo: en *happy* el audio
suele dudar y el texto decide; en *angry* el audio ya basta.

    # 5 enunciados al azar de la sesion de test (hablantes no vistos), con su etiqueta real
    python scripts\\demo_emocion.py
    python scripts\\demo_emocion.py --n 8 --clase happy
    python scripts\\demo_emocion.py --utt Ses05F_impro03_F012
    # tu propio audio, con o sin transcripcion
    python scripts\\demo_emocion.py --wav mi_audio.wav --texto "I can't believe you did that"
    python scripts\\demo_emocion.py --wav mi_audio.wav --asr          # transcribe con Whisper
    python scripts\\demo_emocion.py --grabar 4 --asr                  # graba 4 s del microfono

Si estas entrenando a la vez en la GPU, usa --device cpu (tarda ~2-4 s por enunciado).
Solo lee checkpoints: no entrena ni escribe nada en checkpoints/.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import random
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NOMBRES = {"angry": "enojo", "happy": "alegria", "sad": "tristeza", "neutral": "neutral"}


# --------------------------------------------------------------------- checkpoints
def run_dir(study: int, fold: int) -> Path:
    """Carpeta del fold `fold` del estudio, segun su kfold_summary (semilla 42)."""
    resumen = ROOT / "checkpoints" / f"ablation_{study}" / f"kfold_summary_ablation_{study}.json"
    if not resumen.is_file():
        raise SystemExit(f"No existe {resumen}: el estudio {study} no se ha corrido.")
    for f in json.load(open(resumen))["folds"]:
        if int(f["test_session"]) == fold:
            rd = Path(f["results_dir"].replace("\\", "/"))
            rd = rd if rd.is_absolute() else ROOT / rd
            if not (rd / "best_model.pt").is_file():
                raise SystemExit(f"Falta {rd / 'best_model.pt'}")
            return rd
    raise SystemExit(f"El resumen del estudio {study} no tiene el fold S{fold}.")


def cargar_modelo(study: int, rd: Path, device: str, tiny: bool):
    from scripts.run_ablation import create_ablation_model
    cfg = json.load(open(rd / "config.json"))
    sc = (cfg.get("text", {}) or {}).get("speechcue", {}) or {}
    if sc.get("enabled"):
        raise SystemExit("La demo no soporta modelos con SpeechCue de 9 tokens.")
    log = logging.getLogger("demo")
    ctx = contextlib.nullcontext()
    if tiny:
        from tests.tiny_encoders import offline_config, tiny_encoders
        cfg = offline_config(cfg)
        ctx = tiny_encoders()
    with ctx, contextlib.redirect_stdout(open(os.devnull, "w")):
        model = create_ablation_model(study, cfg, device, log)
    ck = torch.load(rd / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    return model, cfg


# --------------------------------------------------------------------------- datos
def dataset_para(cfg: dict, registros=None, sesion=None):
    """Dataset con el preprocesado EXACTO de test (sin aumentacion), via BaseSERDataset."""
    from src.data.base_dataset import BaseSERDataset
    from src.data.preprocessing import AudioPreprocessor, TextPreprocessor
    from src.data.registry import build_dataset, corpus_from_config

    sr = cfg["dataset"].get("sample_rate", 16000)
    audio_pre = AudioPreprocessor(sample_rate=sr, use_mel_spectrogram=False,
                                  apply_spec_augment=False, apply_waveform_augment=False)
    audio_pre.eval()
    tcfg = cfg.get("text", {}) or {}
    text_pre = TextPreprocessor(model_name=tcfg["model_name"],
                                max_length=tcfg.get("max_length", 128),
                                apply_text_augment=False)
    text_pre.eval()
    comunes = dict(emotions=cfg["dataset"]["emotions"], sample_rate=sr,
                   audio_preprocessor=audio_pre, text_preprocessor=text_pre,
                   max_audio_length=cfg["audio"]["max_length"], training=False,
                   use_prosody_cues=bool(tcfg.get("use_prosody_cues", False)),
                   pad_after_preprocessing=bool(cfg["audio"].get("pad_after_preprocessing", False)))
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        if registros is None:
            return build_dataset(corpus=corpus_from_config(cfg),
                                 data_path=cfg["dataset"]["data_path"],
                                 sessions=[sesion], **comunes)

        class _Demo(BaseSERDataset):
            CORPUS = "demo"

            def _collect_records(self):
                return registros
        return _Demo(data_path=".", sessions=[0], **comunes)


def predecir(model, cfg, item, device):
    from src.data.dataset import collate_fn
    from src.models.forward_utils import multimodal_forward
    b = collate_fn([item])
    audio = b["audio"].to(device)
    ids, mask = b["text_input_ids"].to(device), b["text_attention_mask"].to(device)
    with torch.no_grad():
        if cfg["model"].get("modality") in ("audio", "text"):
            out = model(audio, ids, mask)
        else:
            out = multimodal_forward(model, audio, ids, mask, batch=b)
    return torch.softmax(out["logits"].float(), -1)[0].cpu()


# --------------------------------------------------------------------------- salida
W = 36  # ancho de cada columna


def barra(p: float, ancho: int = 16) -> str:
    n = int(round(p * ancho))
    return "#" * n + "." * (ancho - n)


def mostrar(titulo, verdad, texto_leido, filas, emociones):
    print("\n" + "=" * (2 + W * len(filas)))
    print(titulo)
    if verdad:
        print(f"  etiqueta real: {NOMBRES.get(verdad, verdad).upper()}")
    if texto_leido:
        print(f"  lo que lee la rama de texto: \"{texto_leido[:140]}\"")
    print("-" * (2 + W * len(filas)))
    print("  " + "".join(nombre.ljust(W) for nombre, _ in filas))
    for k, emo in enumerate(emociones):
        celdas = [f"{NOMBRES.get(emo, emo):<9}{barra(float(p[k]))} {100 * float(p[k]):3.0f}%"
                  for _, p in filas]
        print("  " + "".join(c.ljust(W) for c in celdas))
    pie = []
    for _, p in filas:
        top = emociones[int(p.argmax())]
        marca = "" if not verdad else ("  [acierta]" if top == verdad else "  [falla]")
        pie.append(f"-> {NOMBRES.get(top, top).upper()}{marca}")
    print("  " + "".join(c.ljust(W) for c in pie))


# --------------------------------------------------------------------------- entrada
def grabar(segundos: float, sr: int = 16000) -> str:
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        raise SystemExit("Para --grabar instala:  pip install sounddevice soundfile")
    print(f"Grabando {segundos:.0f} s... habla ahora.")
    x = sd.rec(int(segundos * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    ruta = os.path.join(tempfile.gettempdir(), "demo_emocion.wav")
    sf.write(ruta, x[:, 0], sr)
    print(f"Listo ({ruta}).")
    return ruta


def transcribir(wav: str, device: str) -> str:
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small.en",
                   device=0 if device == "cuda" else -1)
    return asr(wav)["text"].strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Demo en vivo de CAMF (audio, texto, fusion).")
    ap.add_argument("--fold", type=int, default=5, help="sesion de test del fold a cargar")
    ap.add_argument("--fusion", type=int, default=98, help="estudio de fusion (98)")
    ap.add_argument("--audio", type=int, default=71, help="estudio solo audio (71); 0 = no")
    ap.add_argument("--texto-estudio", type=int, default=73, dest="texto_estudio",
                    help="estudio solo texto (73); 0 = no")
    ap.add_argument("--n", type=int, default=5, help="enunciados al azar de la sesion de test")
    ap.add_argument("--clase", choices=list(NOMBRES), help="solo enunciados de esta clase")
    ap.add_argument("--utt", nargs="+", help="ids concretos, p. ej. Ses05F_impro03_F012")
    ap.add_argument("--wav", help="tu propio audio (.wav)")
    ap.add_argument("--texto", help="transcripcion de --wav")
    ap.add_argument("--asr", action="store_true", help="transcribir --wav / --grabar con Whisper")
    ap.add_argument("--grabar", type=float, help="grabar N segundos del microfono")
    ap.add_argument("--semilla", type=int, default=None, help="para repetir la misma muestra")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tiny", action="store_true", help=argparse.SUPPRESS)  # solo tests
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)

    estudios = [("FUSION audio+texto (98)" if args.fusion == 98 else f"FUSION ({args.fusion})", args.fusion)]
    if args.audio:
        estudios.insert(0, (f"SOLO AUDIO ({args.audio})", args.audio))
    if args.texto_estudio:
        estudios.insert(1 if args.audio else 0, (f"SOLO TEXTO ({args.texto_estudio})", args.texto_estudio))

    print(f"Cargando modelos del fold S{args.fold} en {args.device}...")
    modelos = []
    for nombre, st in estudios:
        rd = run_dir(st, args.fold)
        m, cfg = cargar_modelo(st, rd, args.device, args.tiny)
        modelos.append((nombre, m, cfg))
        print(f"  {nombre:<26} {rd.name}")
    emociones = modelos[-1][2]["dataset"]["emotions"]

    # ---- que enunciados
    propios = None
    if args.grabar:
        args.wav = grabar(args.grabar)
    if args.wav:
        texto = args.texto
        if args.asr or not texto:
            if not args.asr:
                raise SystemExit("Con --wav da --texto \"...\" o usa --asr.")
            print("Transcribiendo con Whisper...")
            texto = transcribir(args.wav, args.device)
        propios = [dict(audio_path=args.wav, transcription=texto, emotion=emociones[0],
                        emotion_idx=0, utterance_id="Ses00X_demo_X000")]

    datasets = []
    for _, _, cfg in modelos:
        datasets.append(dataset_para(cfg, registros=propios,
                                     sesion=None if propios else args.fold))
    base = datasets[-1]
    if propios:
        indices, conocida = [0], False
    else:
        conocida = True
        uids = [d["utterance_id"] for d in base.data]
        if args.utt:
            faltan = [u for u in args.utt if u not in uids]
            if faltan:
                raise SystemExit(f"No estan en la sesion {args.fold}: {faltan}")
            indices = [uids.index(u) for u in args.utt]
        else:
            pool = [i for i, d in enumerate(base.data)
                    if args.clase is None or d["emotion"] == args.clase]
            indices = random.Random(args.semilla).sample(pool, min(args.n, len(pool)))

    aciertos = [0] * len(modelos)
    for i in indices:
        filas = []
        for k, ((nombre, m, cfg), ds) in enumerate(zip(modelos, datasets)):
            j = i if propios else [d["utterance_id"] for d in ds.data].index(base.data[i]["utterance_id"])
            p = predecir(m, cfg, ds[j], args.device)
            filas.append((nombre, p))
            if conocida and emociones[int(p.argmax())] == base.data[i]["emotion"]:
                aciertos[k] += 1
        rec = base.data[i]
        tok = base.text_preprocessor.tokenizer
        leido = tok.decode(base[i]["text_input_ids"], skip_special_tokens=True)
        titulo = (f"  {rec['utterance_id']}" if conocida else f"  tu audio: {args.wav}") + \
                 f"   |   transcripcion: \"{rec['transcription'][:90]}\""
        mostrar(titulo, rec["emotion"] if conocida else None, leido, filas, emociones)

    if conocida and len(indices) > 1:
        print("\n  Aciertos en esta muestra: " + "   ".join(
            f"{n.split(' (')[0].lower()}: {a}/{len(indices)}" for (n, _, _), a in zip(modelos, aciertos)))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
