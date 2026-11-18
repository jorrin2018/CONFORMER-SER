#!/usr/bin/env python
"""extraer_clasicas.py -- caches de los encoders clasicos (estudios 111-119).

Una vez por corpus, antes de entrenar. Escribe en las rutas de model.clasico del config:

  audio_cache   descriptores acusticos por frame (10 ms) de cada utterance del corpus
  glove_cache   vocabulario del corpus con sus vectores GloVe (solo las palabras que
                aparecen en las transcripciones y en el prompt de prosodia)

Uso (PowerShell, desde la raiz del repo, con el venv activado):

    # IEMOCAP (sirve para 111-115)
    python scripts\\extraer_clasicas.py --study 113 --glove D:\\DOCTORADO\\RECURSOS\\glove.840B.300d.zip
    # CREMA-D (116 y 117) y MELD (118 y 119)
    python scripts\\extraer_clasicas.py --study 116 --glove D:\\DOCTORADO\\RECURSOS\\glove.840B.300d.zip
    python scripts\\extraer_clasicas.py --study 118 --glove D:\\DOCTORADO\\RECURSOS\\glove.840B.300d.zip

Fuentes de audio (--fuente, por defecto la del config):

  egemaps   openSMILE eGeMAPSv02, 25 descriptores de bajo nivel (pip install opensmile)
  compare   openSMILE ComParE_2016, 65 descriptores de bajo nivel
  covarep   COVAREP (MATLAB), el de MulT / MISA. Primero en MATLAB, sobre cada carpeta
            de wav, con frames de 10 ms:  COVAREP_feature_extraction('<carpeta>', ...)
            (revisa la firma en tu copia de COVAREP; la de GitHub deja un .mat junto a
            cada .wav con la matriz 'features', frames x 74). Este script solo recoge
            esos .mat. Si estan en otra carpeta, pasala con --covarep-dir; si el paso
            de frame no es 10 ms, dilo con --fps.

Es incremental: si el cache ya existe con la misma fuente, solo extrae las utterances
que falten. GloVe: acepta el .zip o el .txt de https://nlp.stanford.edu/projects/glove/
(glove.840B.300d es el que usa la literatura de MulT; glove.6B.300d tambien sirve).
"""
from __future__ import annotations

import argparse
import glob
import io
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.classic_features import guardar_cache  # noqa: E402
from src.models.classic_encoders import ESPECIALES, tokenizar, variantes  # noqa: E402

PALABRAS_PROSODIA = ("overall energy is low medium high , variability is speaking rate "
                     "slow moderate fast .")


def cargar_config(args) -> dict:
    import yaml
    if args.config:
        ruta = Path(args.config)
    else:
        rutas = glob.glob(str(ROOT / f"config/ablation/ablation_{args.study}_*.yaml"))
        if len(rutas) != 1:
            raise SystemExit(f"config del estudio {args.study}: {rutas}")
        ruta = Path(rutas[0])
    cfg = yaml.safe_load(open(ruta, encoding="utf-8"))
    if not (cfg.get("model", {}) or {}).get("clasico"):
        raise SystemExit(f"{ruta.name} no tiene bloque model.clasico: no es un estudio clasico")
    print(f"config: {ruta.name}")
    return cfg


def registros(cfg: dict):
    from src.data.registry import build_dataset, corpus_from_config
    ds = build_dataset(
        corpus=corpus_from_config(cfg),
        data_path=cfg["dataset"]["data_path"],
        sessions=cfg["dataset"].get("all_sessions") or [1, 2, 3, 4, 5],
        emotions=cfg["dataset"]["emotions"],
        sample_rate=cfg["dataset"]["sample_rate"],
        audio_preprocessor=None, text_preprocessor=None,
        max_audio_length=cfg["audio"]["max_length"], training=False,
    )
    return corpus_from_config(cfg), ds.data


# ---------------------------------------------------------------------------- audio
def extractor_opensmile(fuente: str):
    try:
        import opensmile
    except ImportError:
        raise SystemExit("Falta openSMILE:  pip install opensmile")
    fs = {"egemaps": opensmile.FeatureSet.eGeMAPSv02,
          "compare": opensmile.FeatureSet.ComParE_2016}[fuente]
    smile = opensmile.Smile(feature_set=fs, feature_level=opensmile.FeatureLevel.LowLevelDescriptors)

    def f(ruta_wav: str, sr_obj: int):
        import torchaudio
        w, sr = torchaudio.load(ruta_wav)
        if w.shape[0] > 1:
            w = w.mean(0, keepdim=True)
        if sr != sr_obj:
            w = torchaudio.transforms.Resample(sr, sr_obj)(w)
        df = smile.process_signal(w.squeeze(0).numpy(), sr_obj)
        return torch.tensor(df.values, dtype=torch.float32), list(df.columns)
    return f


def extractor_covarep(covarep_dir):
    from scipy.io import loadmat

    def f(ruta_wav: str, sr_obj: int):
        p = Path(ruta_wav).with_suffix(".mat")
        if covarep_dir is not None:
            # misma ruta relativa que el wav, bajo --covarep-dir; si no, por nombre
            cand = list(Path(covarep_dir).rglob(p.name))
            rel = [c for c in cand if c.parent.name == p.parent.name]
            p = (rel or cand or [p])[0]
        if not p.is_file():
            raise FileNotFoundError(f"no hay .mat de COVAREP para {ruta_wav} (buscado {p})")
        m = loadmat(str(p))
        if "features" not in m:
            claves = [k for k in m if not k.startswith("__")]
            raise KeyError(f"{p} no tiene la matriz 'features' (claves: {claves})")
        x = torch.tensor(m["features"], dtype=torch.float32)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)  # COVAREP deja NaN/inf
        try:
            nombres = [str(n[0]) for n in m["names"].ravel()]
        except Exception:  # noqa: BLE001  (los nombres son solo informativos)
            nombres = []
        return x, nombres
    return f


def extraer_audio(cfg, corpus, datos, fuente, covarep_dir, forzar, fps=100.0):
    ruta = Path(cfg["model"]["clasico"]["audio_cache"])
    feats, meta = {}, None
    if ruta.is_file() and not forzar:
        viejo = torch.load(ruta, map_location="cpu", weights_only=False)
        if viejo["meta"].get("fuente") != fuente:
            raise SystemExit(f"{ruta} es de fuente {viejo['meta'].get('fuente')!r}, no "
                             f"{fuente!r}. Usa otra ruta en model.clasico.audio_cache o --forzar.")
        feats, meta = viejo["feats"], viejo["meta"]
    faltan = [d for d in datos if d["utterance_id"] not in feats]
    print(f"audio ({fuente}): {len(datos)} utterances, {len(datos) - len(faltan)} ya en "
          f"{ruta.name}, {len(faltan)} por extraer")
    if not faltan:
        return
    f = extractor_covarep(covarep_dir) if fuente == "covarep" else extractor_opensmile(fuente)
    sr = int(cfg["dataset"]["sample_rate"])
    nombres, t0 = None, time.time()
    for i, d in enumerate(faltan, 1):
        x, n = f(d["audio_path"], sr)
        nombres = nombres or n
        if meta is not None and x.shape[1] != meta["dim"]:
            raise RuntimeError(f"{d['utterance_id']}: {x.shape[1]} descriptores, el cache tiene {meta['dim']}")
        feats[d["utterance_id"]] = x.to(torch.float16)
        if i % 250 == 0 or i == len(faltan):
            dt = time.time() - t0
            print(f"  {i}/{len(faltan)}  {dt / i * 1000:.0f} ms/utt  quedan "
                  f"{dt / i * (len(faltan) - i) / 60:.1f} min", flush=True)
    dim = next(iter(feats.values())).shape[1]
    if meta is None:
        meta = {"fuente": fuente, "dim": int(dim), "fps": float(fps), "corpus": corpus,
                "nombres": nombres or [], "sample_rate": sr,
                "creado": time.strftime("%Y-%m-%d %H:%M")}
    meta["n"] = len(feats)
    guardar_cache(str(ruta), feats, meta)
    print(f"  -> {ruta} ({len(feats)} utterances, {dim} descriptores/frame)")


# ---------------------------------------------------------------------------- GloVe
def _lineas_glove(ruta: Path):
    if ruta.suffix == ".zip":
        z = zipfile.ZipFile(ruta)
        nombre = [n for n in z.namelist() if n.endswith(".txt")][0]
        with z.open(nombre) as fb:
            for linea in io.TextIOWrapper(fb, encoding="utf-8", errors="replace"):
                yield linea
    else:
        with open(ruta, encoding="utf-8", errors="replace") as fh:
            yield from fh


def construir_glove(cfg, datos, ruta_glove: Path, forzar):
    ruta = Path(cfg["model"]["clasico"]["glove_cache"])
    if ruta.is_file() and not forzar:
        print(f"GloVe: {ruta.name} ya existe (usa --forzar para rehacerlo)")
        return
    conteo = Counter()
    for d in datos:
        conteo.update(tokenizar(d["transcription"]))
    conteo.update(tokenizar(PALABRAS_PROSODIA))
    buscadas = {v for tok in conteo for v in variantes(tok)}
    print(f"GloVe: {len(conteo)} tokens distintos en el corpus; leyendo {ruta_glove.name} ...")
    vec, dim, t0 = {}, None, time.time()
    for k, linea in enumerate(_lineas_glove(ruta_glove)):
        partes = linea.rstrip("\n").split(" ")
        if dim is None:
            dim = len(partes) - 1
        palabra = " ".join(partes[:-dim])
        if palabra in buscadas and palabra not in vec:
            vec[palabra] = torch.tensor([float(x) for x in partes[-dim:]])
        if k and k % 500000 == 0:
            print(f"  {k:,} lineas, {len(vec)} encontradas ({time.time() - t0:.0f} s)", flush=True)
    itos = list(ESPECIALES) + sorted(vec)
    vectores = torch.zeros(len(itos), dim)
    for i, w in enumerate(itos[len(ESPECIALES):], len(ESPECIALES)):
        vectores[i] = vec[w]
    # cobertura: fraccion de tokens del corpus (con repeticion) que tienen vector
    cubiertos = sum(c for tok, c in conteo.items() if any(v in vec for v in variantes(tok)))
    total = sum(conteo.values())
    sin = sorted((t for t in conteo if not any(v in vec for v in variantes(t))),
                 key=lambda t: -conteo[t])
    meta = {"fuente": ruta_glove.name, "dim": dim, "palabras": len(itos) - len(ESPECIALES),
            "cobertura_tokens": f"{100 * cubiertos / max(total, 1):.1f} %",
            "sin_vector_ejemplos": sin[:30], "creado": time.strftime("%Y-%m-%d %H:%M")}
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"itos": itos, "vectors": vectores, "meta": meta}, ruta)
    print(f"  -> {ruta}: {meta['palabras']} palabras, {dim}-d, cobertura {meta['cobertura_tokens']}")
    if sin:
        print(f"     sin vector (las mas frecuentes): {sin[:15]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--study", type=int)
    g.add_argument("--config")
    ap.add_argument("--glove", help="glove.840B.300d.zip / .txt (obligatorio la primera vez)")
    ap.add_argument("--fuente", choices=["egemaps", "compare", "covarep"])
    ap.add_argument("--covarep-dir")
    ap.add_argument("--fps", type=float, default=100.0,
                    help="frames por segundo de los descriptores (openSMILE: 100)")
    ap.add_argument("--solo", choices=["audio", "texto"])
    ap.add_argument("--forzar", action="store_true", help="rehacer aunque el cache exista")
    args = ap.parse_args()

    cfg = cargar_config(args)
    fuente = args.fuente or cfg["model"]["clasico"].get("fuente", "egemaps")
    corpus, datos = registros(cfg)
    print(f"corpus: {corpus}, {len(datos)} utterances")

    if args.solo != "texto":
        extraer_audio(cfg, corpus, datos, fuente, args.covarep_dir, args.forzar, args.fps)
    if args.solo != "audio":
        destino = Path(cfg["model"]["clasico"]["glove_cache"])
        if args.glove:
            construir_glove(cfg, datos, Path(args.glove), args.forzar)
        elif not destino.is_file():
            raise SystemExit(f"Falta {destino}: pasa --glove <glove.840B.300d.zip>")
        else:
            print(f"GloVe: {destino.name} ya existe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
