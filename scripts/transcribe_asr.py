#!/usr/bin/env python
"""transcribe_asr.py -- texto de ASR (Whisper) para el estudio 100.

El estudio 100 es el 74 con UN cambio: el texto que entra a DistilBERT sale de un
reconocedor automatico en vez de la transcripcion manual del corpus. Mide cuanto del
aporte de la fusion sobrevive sin transcripciones humanas, que es la pregunta que hara
cualquier revisor ("en produccion no hay transcripcion manual") y la condicion con la
que Wu et al. 2021 reportan 71.90 UA.

Escribe cache/asr/<corpus>_<modelo>.json:
    {"corpus": ..., "model": ..., "complete": true, "wer": 0.17,
     "n": 5531, "transcripts": {utterance_id: texto}}

Reanudable: guarda cada --save-every utterances (escritura atomica) y, si lo
interrumpes, al volver a lanzarlo salta lo ya transcrito. El WER se calcula contra la
transcripcion manual del corpus, con normalizacion simple (minusculas, sin puntuacion
ni etiquetas tipo [LAUGHTER]); va en el JSON y hay que reportarlo en el articulo.

    python scripts\\transcribe_asr.py --config config\\ablation\\ablation_100_asr_transcripts.yaml
    python scripts\\transcribe_asr.py --config ... --model openai/whisper-medium.en --device cuda

Coste aproximado en una RTX 3070 con whisper-small.en: decenas de minutos para IEMOCAP.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def normalizar(texto: str) -> list[str]:
    texto = re.sub(r"\[[^\]]*\]|<[^>]*>", " ", texto.lower())
    texto = re.sub(r"[^a-z0-9' ]+", " ", texto)
    return texto.split()


def distancia(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein por palabras (sustituciones + borrados + inserciones)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(pares) -> float:
    err = tot = 0
    for ref, hyp in pares:
        r, h = normalizar(ref), normalizar(hyp)
        err += distancia(r, h)
        tot += len(r)
    return err / max(tot, 1)


def escribir_atomico(ruta: Path, datos: dict) -> None:
    tmp = ruta.with_suffix(ruta.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=0)
    os.replace(tmp, ruta)


def registros(cfg: dict) -> list[dict]:
    from src.data.registry import build_dataset, corpus_from_config
    d = cfg["dataset"]
    sesiones = d.get("all_sessions") or [1, 2, 3, 4, 5]
    ds = build_dataset(corpus=corpus_from_config(cfg), data_path=d["data_path"],
                       sessions=sesiones, emotions=d["emotions"],
                       sample_rate=d.get("sample_rate", 16000), training=False)
    return ds.data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML del estudio (da corpus y ruta)")
    ap.add_argument("--model", default="openai/whisper-small.en")
    ap.add_argument("--out", default=None, help="por defecto cache/asr/<corpus>_<modelo>.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=200)
    args = ap.parse_args()

    import torch
    import torchaudio
    from transformers import pipeline
    from scripts.run_ablation import load_ablation_config
    from src.ablation_studies import load_base_config
    from src.data.registry import corpus_from_config

    cfg = load_ablation_config(args.config, load_base_config())
    corpus = corpus_from_config(cfg)
    out = Path(args.out or ROOT / "cache" / "asr" /
               f"{corpus}_{args.model.split('/')[-1]}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    datos = {"corpus": corpus, "model": args.model, "complete": False, "transcripts": {}}
    if out.exists():
        previo = json.load(open(out, encoding="utf-8"))
        if previo.get("model") != args.model or previo.get("corpus") != corpus:
            raise SystemExit(f"{out} es de otro modelo/corpus; usa --out distinto")
        datos = previo
    hechos = datos["transcripts"]

    recs = registros(cfg)
    pendientes = [r for r in recs if r["utterance_id"] not in hechos]
    print(f"{corpus}: {len(recs)} utterances, {len(hechos)} ya transcritas, "
          f"{len(pendientes)} pendientes. Modelo {args.model} en {args.device}.")

    if pendientes:
        dev = 0 if args.device == "cuda" else -1
        dtype = torch.float16 if args.device == "cuda" else torch.float32
        asr = pipeline("automatic-speech-recognition", model=args.model, device=dev,
                       torch_dtype=dtype, chunk_length_s=30)
        gen = {} if args.model.endswith(".en") else {"language": "en", "task": "transcribe"}
        t0 = time.time()
        for i in range(0, len(pendientes), args.batch_size):
            lote = pendientes[i:i + args.batch_size]
            audios = []
            for r in lote:
                wav, sr = torchaudio.load(r["audio_path"])
                wav = wav.mean(0)
                if sr != 16000:
                    wav = torchaudio.functional.resample(wav, sr, 16000)
                audios.append({"raw": wav.numpy(), "sampling_rate": 16000})
            salida = asr(audios, batch_size=args.batch_size, generate_kwargs=gen)
            for r, s in zip(lote, salida):
                hechos[r["utterance_id"]] = s["text"].strip()
            hechas = i + len(lote)
            if hechas % args.save_every < args.batch_size or hechas == len(pendientes):
                escribir_atomico(out, datos)
                ritmo = hechas / max(time.time() - t0, 1e-9)
                print(f"  {hechas}/{len(pendientes)}  ({ritmo:.1f} utt/s)")

    faltan = [r["utterance_id"] for r in recs if r["utterance_id"] not in hechos]
    if faltan:
        raise SystemExit(f"Faltan {len(faltan)} utterances; vuelve a lanzarlo.")
    datos["wer"] = round(wer((r["transcription"], hechos[r["utterance_id"]]) for r in recs), 4)
    datos["n"] = len(recs)
    datos["complete"] = True
    escribir_atomico(out, datos)
    vacias = sum(1 for r in recs if not hechos[r["utterance_id"]].strip())
    print(f"Listo: {out}\n  WER contra la transcripcion manual: {100 * datos['wer']:.1f} %"
          f"\n  transcripciones vacias: {vacias}")
    try:
        ruta_cfg = out.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        ruta_cfg = str(out)
    print(f"  Pon en el config: text.asr_cache: \"{ruta_cfg}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
