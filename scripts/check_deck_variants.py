#!/usr/bin/env python
"""check_deck_variants.py -- que cada variante de la serie 87-94 haga lo que dice.

Antes de gastar GPU en un estudio de la serie, este script monta EL MODELO REAL con
EL CONFIG REAL (encoders desde la cache de HuggingFace) y comprueba, una por una,
las banderas que el YAML activa:

  text.speechcue            cache presente, rate_source del cache = el del config
  audio.pad_after_preprocessing + model.audio_masked_pooling
                            el promedio de Wav2Vec2 usa exactamente los frames reales
  model.vision_mel_speech_only
                            la imagen del ViT NO cambia si cambia el relleno
  model.vision_feature: cls la rama visual entrega el [CLS], no el pooler
  vision_unfreeze_last_n_layers
                            bloques del ViT entrenables = los que dice el YAML
  pitch_shift_impl          el aumento cambia de verdad el tono de un seno de 200 Hz

y da un paso de optimizacion por la ruta de entrenamiento (multimodal_forward).

    python scripts\\check_deck_variants.py --study 94
    python scripts\\check_deck_variants.py --study 94 --with-data   # + un batch real
    python scripts\\check_deck_variants.py --study 94 --tiny        # sin HF (solo codigo)

Sale con codigo 0 si todo cuadra; si no, dice que falla y NO hay que lanzar.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import logging
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_ablation import create_ablation_model, load_ablation_config  # noqa: E402
from src.ablation_studies.utils import create_ablation_optimizer  # noqa: E402
from src.data.speechcue import SPEECHCUE_TOKENS, SpeechCueStatsCache  # noqa: E402
from src.models.forward_utils import multimodal_forward  # noqa: E402

SR = 16000
fallos: list[str] = []


def ok(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FALLO'}] {msg}")
    if not cond:
        fallos.append(msg)


def cargar_config(study: int, tiny: bool) -> dict:
    hits = sorted(glob.glob(str(ROOT / f"config/ablation/ablation_{study}_*.yaml")))
    if not hits:
        raise SystemExit(f"No encuentro el config del estudio {study}")
    if tiny:
        from tests.tiny_encoders import offline_config
        return offline_config(load_ablation_config(hits[0], {}))
    from src.ablation_studies import load_base_config
    return load_ablation_config(hits[0], load_base_config())


def check_speechcue(cfg: dict) -> None:
    sc = (cfg.get("text", {}) or {}).get("speechcue", {}) or {}
    if not sc.get("enabled"):
        print("  SpeechCue de tokens: desactivado"
              + (" (prompt en lenguaje natural)" if cfg["text"].get("use_prosody_cues") else ""))
        return
    path = Path(sc.get("stats_cache", "cache/speechcue_stats.json"))
    ok(path.exists(), f"cache SpeechCue presente: {path}")
    if path.exists():
        want = sc.get("rate_source", "transcript")
        have = SpeechCueStatsCache.load_meta(path).get("rate_source", "transcript")
        ok(want == have, f"rate_source config={want} / cache={have}")
        n = len(SpeechCueStatsCache.load(path))
        ok(n == 5531 or cfg["dataset"].get("corpus", "iemocap") != "iemocap",
           f"utterances en el cache: {n}")
    print(f"  normalizacion de terciles: {sc.get('normalize', 'speaker')}")


def check_pitch(cfg: dict) -> None:
    aug = (cfg.get("audio", {}) or {}).get("augmentation", {}) or {}
    if not aug.get("apply_pitch_shift"):
        return
    impl = aug.get("pitch_shift_impl", "legacy_noop")
    from src.data.augmentation import AudioAugmentation
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = AudioAugmentation(apply_time_masking=False, apply_noise=False,
                              apply_speed_perturb=False, apply_pitch_shift=True,
                              pitch_prob=1.0, pitch_range=tuple(aug.get("pitch_range", (-2, 2))),
                              pitch_shift_impl=impl, sample_rate=SR)
    a.train()
    x = torch.sin(2 * math.pi * 200 * torch.arange(2 * SR) / SR)
    random.seed(0)
    y = a(x)
    spec = torch.fft.rfft(y * torch.hann_window(len(y))).abs()
    f = (int(spec[5:].argmax()) + 5) * SR / len(y)
    semis = 12 * math.log2(f / 200.0)
    if impl == "legacy_noop":
        print(f"  pitch shift: legacy_noop -> no se aplica (desplazamiento medido {semis:+.2f} st)")
    else:
        ok(abs(semis) > 0.7, f"pitch shift real: {semis:+.2f} semitonos sobre un seno de 200 Hz")


def check_model(study: int, cfg: dict, device: str, tiny: bool) -> None:
    sc = (cfg.get("text", {}) or {}).get("speechcue", {}) or {}
    if sc.get("enabled") and not cfg["text"].get("_tokenizer_len"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["text"]["model_name"])
        cfg["text"]["_tokenizer_len"] = len(tok) + len(SPEECHCUE_TOKENS)
    ctx = contextlib.nullcontext()
    if tiny:
        from tests.tiny_encoders import tiny_encoders
        ctx = tiny_encoders()
    torch.manual_seed(0)
    with ctx:
        model = create_ablation_model(study, cfg, device, logging.getLogger("check"))
    m = cfg["model"]

    vit = model.vision_encoder
    n_layers = len(vit.encoder.layer)
    trainable = [i for i, l in enumerate(vit.encoder.layer)
                 if any(p.requires_grad for p in l.parameters())]
    want = 0 if m.get("freeze_vision_encoder", True) else int(m.get("vision_unfreeze_last_n_layers", 0))
    ok(len(trainable) == want and (not trainable or trainable[-1] == n_layers - 1),
       f"ViT: bloques entrenables {[i + 1 for i in trainable] or 'ninguno'} (esperados {want})")

    g = torch.Generator().manual_seed(1)
    B, T = 2, int(cfg["audio"].get("max_length", 8)) * SR
    L = torch.tensor([3 * SR, 5 * SR])
    audio = torch.randn(B, T, generator=g)
    for i, n in enumerate(L.tolist()):
        audio[i, n:] = 0.0
    ids = torch.randint(1000, 2000, (B, 24), generator=g)
    mask = torch.ones(B, 24, dtype=torch.long)
    y = torch.tensor([0, 1])
    audio, ids, mask, y, L = (t.to(device) for t in (audio, ids, mask, y, L))
    batch = {"audio_lengths": L}

    print(f"  uses_audio_lengths = {getattr(model, 'uses_audio_lengths', False)}")
    model.eval()
    if m.get("audio_masked_pooling"):
        cap = {}
        orig = model.pool_audio_features

        def spy(features, attention_mask=None):
            cap["f"], cap["m"] = features, attention_mask
            return orig(features, attention_mask)
        model.pool_audio_features = spy
        with torch.no_grad():
            out = multimodal_forward(model, audio, ids, mask, y, batch=batch)
        model.pool_audio_features = orig
        frames = model.audio_encoder._get_feat_extract_output_lengths(L)
        good = cap.get("m") is not None and cap["m"].sum(1).long().tolist() == frames.tolist()
        good = good and all(torch.allclose(out["audio_features"][i], cap["f"][i, :n].mean(0), atol=1e-4)
                            for i, n in enumerate(frames.tolist()))
        ok(good, f"pooling de audio sobre {frames.tolist()} frames reales de {cap['f'].shape[1]}")

    if m.get("vision_mel_speech_only"):
        a2 = audio.clone()
        a2[0, L[0]:] = 0.3 * torch.randn(T - int(L[0]), device=device)
        with torch.no_grad():
            same = torch.allclose(model._waveform_to_vit_image(audio, L),
                                  model._waveform_to_vit_image(a2, L))
        ok(same, "log-mel del ViT invariante al relleno")

    if m.get("vision_feature", "pooler") == "cls":
        with torch.no_grad():
            out = multimodal_forward(model, audio, ids, mask, y, batch=batch)
            img = model._waveform_to_vit_image(audio, L if getattr(model, "_vision_mel_speech_only", False) else None)
            cls = model.vision_encoder(pixel_values=img.to(device)).last_hidden_state[:, 0, :]
        ok(torch.allclose(out["vision_features"], cls, atol=1e-5), "rama visual = [CLS] del ViT")

    model.train()
    opt = create_ablation_optimizer(model, cfg, study)
    out = multimodal_forward(model, audio, ids, mask, y, batch=batch)
    loss = torch.nn.functional.cross_entropy(out["logits"], y)
    before = model.fusion.audio_proj.weight.detach().clone()
    loss.backward()
    opt.step()
    ok(torch.isfinite(loss).item() and not torch.equal(before, model.fusion.audio_proj.weight),
       f"paso de optimizacion por la ruta real (loss {loss.item():.4f})")


def check_data(cfg: dict) -> None:
    from src.train import create_dataloaders
    log = logging.getLogger("check_data")
    train_loader, _, _, _ = create_dataloaders(cfg, log)
    b = next(iter(train_loader))
    L, a = b["audio_lengths"], b["audio"]
    print(f"  batch real: audio {tuple(a.shape)}, audio_lengths {L.tolist()}")
    if cfg["audio"].get("pad_after_preprocessing"):
        ok(all(torch.count_nonzero(a[i, n:]) == 0 for i, n in enumerate(L.tolist())),
           "relleno en ceros exactos tras audio_length")
    sc = (cfg.get("text", {}) or {}).get("speechcue", {}) or {}
    if sc.get("enabled"):
        tok = train_loader.dataset.text_preprocessor.tokenizer
        cue_ids = set(tok.convert_tokens_to_ids(SPEECHCUE_TOKENS))
        pos = [b["text_input_ids"][i, 1:4].tolist() for i in range(len(L))]
        ok(all(set(p) <= cue_ids for p in pos),
           f"tokens SpeechCue tras [CLS]: {[tok.convert_ids_to_tokens(p) for p in pos[:2]]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tiny", action="store_true",
                    help="encoders diminutos aleatorios (sin HuggingFace); prueba solo el codigo")
    ap.add_argument("--with-data", action="store_true",
                    help="ademas, construir el dataloader real y revisar un batch")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cfg = cargar_config(args.study, args.tiny)
    m, au = cfg["model"], cfg["audio"]
    print(f"\n  Estudio {args.study}")
    for k, v in (("audio.pad_after_preprocessing", au.get("pad_after_preprocessing", False)),
                 ("model.audio_masked_pooling", m.get("audio_masked_pooling", False)),
                 ("model.vision_mel_speech_only", m.get("vision_mel_speech_only", False)),
                 ("model.vision_feature", m.get("vision_feature", "pooler")),
                 ("model.freeze_vision_encoder", m.get("freeze_vision_encoder", True)),
                 ("model.vision_unfreeze_last_n_layers", m.get("vision_unfreeze_last_n_layers", 0)),
                 ("audio.augmentation.pitch_shift_impl",
                  (au.get("augmentation") or {}).get("pitch_shift_impl", "legacy_noop"))):
        print(f"    {k:<40} {v}")
    print()
    check_speechcue(cfg)
    check_pitch(cfg)
    check_model(args.study, cfg, args.device, args.tiny)
    if args.with_data:
        check_data(cfg)

    if fallos:
        print(f"\n  {len(fallos)} comprobacion(es) fallaron. NO lanzar el estudio {args.study}.\n")
        return 1
    print(f"\n  Todo cuadra. El estudio {args.study} mide lo que dice medir.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
