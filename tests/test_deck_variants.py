"""Tests de la serie 87-94 (variantes que alinean el codigo con la presentacion).

Corren sin GPU, sin IEMOCAP y sin red: encoders diminutos aleatorios
(tests/tiny_encoders.py), wavs sinteticos y .syseg falsos.

    python -m pytest tests/test_deck_variants.py -q
"""
from __future__ import annotations

import glob
import logging
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.augmentation import AudioAugmentation  # noqa: E402
from src.data.base_dataset import BaseSERDataset  # noqa: E402
from src.data.dataset import collate_fn  # noqa: E402
from src.data.preprocessing import AudioPreprocessor, TextPreprocessor  # noqa: E402
from src.data.speechcue import (  # noqa: E402
    SPEECHCUE_TOKENS, SpeechCueStats, SpeechCueStatsCache, forced_alignment_path,
    rate_from_forced_alignment, read_forced_alignment,
)
from src.models.forward_utils import multimodal_forward  # noqa: E402
from tests.tiny_encoders import LOCAL_TOKENIZER, offline_config, tiny_encoders  # noqa: E402

SR = 16000
LOG = logging.getLogger("test")


# ------------------------------------------------------------------ helpers
def _peak_hz(x: torch.Tensor) -> float:
    spec = torch.fft.rfft(x * torch.hann_window(len(x)))
    k = int(spec.abs()[5:].argmax()) + 5
    return k * SR / len(x)


def _aug(**kw):
    base = dict(apply_time_masking=False, apply_noise=False, apply_speed_perturb=False,
                apply_freq_masking=False, sample_rate=SR)
    base.update(kw)
    a = AudioAugmentation(**base)
    a.train()
    return a


def _config(study: int) -> dict:
    from scripts.run_ablation import load_ablation_config
    path = glob.glob(str(ROOT / f"config/ablation/ablation_{study}_*.yaml"))[0]
    return offline_config(load_ablation_config(path, {}))


def _model(study: int, cfg=None):
    from scripts.run_ablation import create_ablation_model
    cfg = cfg or _config(study)
    if cfg.get("text", {}).get("speechcue", {}).get("enabled"):
        cfg["text"]["_tokenizer_len"] = 30522 + len(SPEECHCUE_TOKENS)
    torch.manual_seed(0)
    with tiny_encoders():
        return create_ablation_model(study, cfg, "cpu", LOG)


def _batch(lengths=(3 * SR, 5 * SR), total=8 * SR, pad_value=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    B = len(lengths)
    audio = torch.randn(B, total, generator=g)
    for i, n in enumerate(lengths):
        audio[i, n:] = pad_value
    ids = torch.randint(1000, 2000, (B, 24), generator=g)
    mask = torch.ones(B, 24, dtype=torch.long)
    return audio, ids, mask, torch.tensor(lengths), torch.arange(B) % 4


# ------------------------------------------------------------- pitch shift
def test_pitch_shift_legacy_is_noop():
    x = torch.sin(2 * math.pi * 200 * torch.arange(SR) / SR)
    with pytest.warns(UserWarning, match="NO se aplica"):
        a = _aug(apply_pitch_shift=True, pitch_prob=1.0)
    assert torch.equal(a(x), x)


def test_pitch_shift_legacy_consumes_same_rng():
    """Con legacy_noop la secuencia aleatoria es la de siempre: corridas viejas reproducibles."""
    x = torch.randn(2 * SR)
    outs = []
    for pitch in (False, True):
        random.seed(7)
        torch.manual_seed(7)
        with pytest.warns(UserWarning) if pitch else _nullcontext():
            a = AudioAugmentation(apply_pitch_shift=pitch, sample_rate=SR)
        a.train()
        y = a(x)
        outs.append((y, random.random()))
    assert torch.equal(outs[0][0], outs[1][0]) and outs[0][1] == outs[1][1]


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_pitch_shift_torchaudio_shifts_by_semitones():
    x = torch.sin(2 * math.pi * 200 * torch.arange(2 * SR) / SR)
    a = _aug(apply_pitch_shift=True, pitch_prob=1.0, pitch_shift_impl="torchaudio")
    random.seed(0)
    seen = set()
    for _ in range(8):
        y = a(x)
        assert y.shape == x.shape and torch.isfinite(y).all()
        ratio = _peak_hz(y) / 200.0
        semis = 12 * math.log2(ratio)
        assert round(semis) in (-2, -1, 1, 2) and abs(semis - round(semis)) < 0.35
        seen.add(round(semis))
    assert len(seen) >= 2


def test_bad_pitch_impl_rejected():
    with pytest.raises(ValueError):
        AudioAugmentation(pitch_shift_impl="librosa")


def test_speed_perturb_natural_length():
    x = torch.randn(SR)
    a = _aug(apply_speed_perturb=True, speed_prob=1.0, speed_range=(0.9, 0.9),
             preserve_length=False)
    assert a(x).shape[0] == int(SR / 0.9)
    b = _aug(apply_speed_perturb=True, speed_prob=1.0, speed_range=(0.9, 0.9))
    assert b(x).shape[0] == SR


# ---------------------------------------------------------------- dataset
class _FakeCorpus(BaseSERDataset):
    CORPUS = "fake"
    WAVS: list = []

    def _collect_records(self):
        return [dict(audio_path=p, transcription="hello there", emotion="angry",
                     emotion_idx=0, utterance_id=f"Ses01F_impro01_F{i:03d}")
                for i, p in enumerate(self.WAVS)]


def _write_wavs(tmp_path, secs=(1.0, 3.0)):
    import soundfile as sf
    paths = []
    rng = np.random.default_rng(0)
    for i, s in enumerate(secs):
        p = tmp_path / f"u{i}.wav"
        sf.write(p, (0.1 * rng.standard_normal(int(s * SR)) + 0.05).astype("float32"), SR)
        paths.append(str(p))
    return paths


def _dataset(paths, pad_after, max_len=2.0):
    _FakeCorpus.WAVS = paths
    pre = AudioPreprocessor(sample_rate=SR, apply_waveform_augment=False)
    return _FakeCorpus(data_path=".", sessions=[1], emotions=["angry"], sample_rate=SR,
                       audio_preprocessor=pre, max_audio_length=max_len,
                       training=False, pad_after_preprocessing=pad_after)


def test_dataset_pad_after_preprocessing(tmp_path):
    paths = _write_wavs(tmp_path)
    new = _dataset(paths, True)
    s0, s1 = new[0], new[1]
    assert s0["audio"].shape[0] == 2 * SR and s0["audio_length"] == SR
    assert s1["audio_length"] == 2 * SR                       # recortado a max_len
    assert torch.count_nonzero(s0["audio"][SR:]) == 0         # relleno = ceros exactos
    real = s0["audio"][:SR]
    assert abs(float(real.mean())) < 1e-4 and abs(float(real.std()) - 1) < 1e-3


def test_dataset_legacy_unchanged_values(tmp_path):
    """Por defecto: normalizacion sobre la senal rellenada (historico) + audio_length."""
    paths = _write_wavs(tmp_path)
    old = _dataset(paths, False)
    s0 = old[0]
    assert s0["audio_length"] == SR
    full = s0["audio"]
    assert abs(float(full.mean())) < 1e-4 and abs(float(full.std()) - 1) < 1e-3
    assert torch.count_nonzero(full[SR:]) > 0                 # el relleno ya no es cero


def test_collate_carries_lengths(tmp_path):
    paths = _write_wavs(tmp_path)
    ds = _dataset(paths, True)
    tp = TextPreprocessor(model_name=LOCAL_TOKENIZER, max_length=16)
    ds.text_preprocessor = tp
    b = collate_fn([ds[0], ds[1]])
    assert b["audio_lengths"].tolist() == [SR, 2 * SR]


# ------------------------------------------------------ forced alignment
# Formato real de IEMOCAP (.syseg): tres columnas, sin silencios.
SYSEG_REAL = """  SFrm  EFrm Syllable
    45    50 IX_K
    51    79 S_K_Y_UW_Z
    80   105 M_IY
"""

# Formato con puntuacion y silencios (.wdseg); el parser acepta los dos.
SYSEG = """ SFrm  EFrm    SegAScr Syllable
    0    20    -100000 <s>
   21    40    -200000 hh_ax
   41    55    -200000 l_ow
   56    80    -100000 <sil>
   81    99    -200000 dh_eh_r
  100   130    -100000 ++BREATHING++
  131   150    -100000 </s>
 Total score:    -1000000
"""


def test_read_forced_alignment(tmp_path):
    p = tmp_path / "x.syseg"
    p.write_text(SYSEG)
    n, span = read_forced_alignment(p)
    assert n == 3 and span == pytest.approx((99 - 21 + 1) * 0.01)
    rate, sp, src = rate_from_forced_alignment(p)
    assert rate == pytest.approx(3 / 0.79) and sp == pytest.approx(0.79) and src == "syseg"


def test_header_only_syseg_falls_back_to_wdseg(tmp_path):
    """Turnos no lexicos ("Hmm."): el .syseg trae solo cabecera, el .wdseg la palabra."""
    sy = tmp_path / "u.syseg"
    sy.write_text("  SFrm  EFrm Syllable\n")
    assert read_forced_alignment(sy) == (0, 0.0)
    (tmp_path / "u.wdseg").write_text(
        "\t SFrm  EFrm    SegAScr Word\n\t    0    33    1970028 <s>\n"
        "\t   34    58     865133 HMM\n\t   59    93    1973904 <sil>\n"
        "\t   94   173    4236421 </s>\n Total score:     9045486\n")
    rate, span, src = rate_from_forced_alignment(sy, fallback_span=1.75)
    # 'hmm' -> 1 silaba; tramo 0.25 s < 0.30 -> se usa la duracion del fichero
    assert src == "wdseg" and span == pytest.approx(1.75) and rate == pytest.approx(1 / 1.75)
    (tmp_path / "u.wdseg").unlink()
    rate, span, src = rate_from_forced_alignment(sy)
    assert src == "empty" and rate == 0.0           # como la ruta transcript con 0 silabas


def test_read_forced_alignment_real_iemocap_layout(tmp_path):
    p = tmp_path / "Ses01F_impro01_F000.syseg"
    p.write_text(SYSEG_REAL)
    n, span = read_forced_alignment(p)
    assert n == 3 and span == pytest.approx((105 - 45 + 1) * 0.01)


def test_read_forced_alignment_rejects_unknown_format(tmp_path):
    p = tmp_path / "x.syseg"
    p.write_text("esto no es un syseg\n")
    with pytest.raises(ValueError):
        read_forced_alignment(p)


def test_forced_alignment_path():
    a = Path("D:/IEMOCAP/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav")
    assert forced_alignment_path(a).as_posix().endswith(
        "Session1/sentences/ForcedAlignment/Ses01F_impro01/Ses01F_impro01_F000.syseg")


def test_cache_meta_roundtrip(tmp_path):
    p = tmp_path / "c.json"
    SpeechCueStatsCache.save({"u": SpeechCueStats(1.0, 2.0, 3.0, 4.0, 5.0)}, p,
                             meta={"rate_source": "forced_alignment"})
    assert SpeechCueStatsCache.load_meta(p)["rate_source"] == "forced_alignment"
    assert SpeechCueStatsCache.load(p).get("u").rate == 3.0


def test_speechcue_tokens_follow_cls():
    tp = TextPreprocessor(model_name=LOCAL_TOKENIZER, max_length=32,
                          additional_tokens=SPEECHCUE_TOKENS)
    ids = tp("[P_LOW] [E_MID] [R_HIGH] hello there")["input_ids"].tolist()
    want = tp.tokenizer.convert_tokens_to_ids(["[P_LOW]", "[E_MID]", "[R_HIGH]"])
    assert ids[0] == tp.tokenizer.cls_token_id and ids[1:4] == want


# ------------------------------------------------------------------ model
def test_default_study_72_ignores_lengths():
    m = _model(72)
    assert not getattr(m, "uses_audio_lengths", False)
    audio, ids, mask, L, y = _batch()
    out = multimodal_forward(m.eval(), audio, ids, mask, y, batch={})   # sin longitudes: vale
    assert out["logits"].shape == (2, 4)


def test_masked_pooling_averages_only_real_frames():
    m = _model(89).eval()
    assert m.uses_audio_lengths
    audio, ids, mask, L, y = _batch()
    capturado = {}
    orig = m.pool_audio_features

    def spy(features, attention_mask=None):
        capturado["f"], capturado["m"] = features, attention_mask
        return orig(features, attention_mask)

    m.pool_audio_features = spy
    with torch.no_grad():
        out = multimodal_forward(m, audio, ids, mask, y, batch={"audio_lengths": L})
    f, msk = capturado["f"], capturado["m"]
    frames = m.audio_encoder._get_feat_extract_output_lengths(L)
    assert msk.sum(1).tolist() == frames.tolist()
    for i, n in enumerate(frames.tolist()):
        assert torch.allclose(out["audio_features"][i], f[i, :n].mean(0), atol=1e-5)


def test_masked_pooling_requires_lengths():
    m = _model(89).eval()
    audio, ids, mask, L, y = _batch()
    with pytest.raises(RuntimeError, match="audio_lengths"):
        multimodal_forward(m, audio, ids, mask, y, batch={})


def test_masked_pooling_requires_pad_after_preprocessing():
    cfg = _config(89)
    cfg["audio"]["pad_after_preprocessing"] = False
    with pytest.raises(ValueError, match="pad_after_preprocessing"):
        _model(89, cfg)


def test_mel_speech_only_ignores_padding():
    m90, m72 = _model(90).eval(), _model(72).eval()
    a1, _, _, L, _ = _batch(pad_value=0.0)
    a2, _, _, _, _ = _batch(pad_value=0.0)
    a2[0, L[0]:] = 0.3 * torch.randn(a2.shape[1] - L[0])     # otro relleno
    i1, i2 = m90._waveform_to_vit_image(a1, L), m90._waveform_to_vit_image(a2, L)
    assert torch.allclose(i1, i2)
    j1, j2 = m72._waveform_to_vit_image(a1), m72._waveform_to_vit_image(a2)
    assert not torch.allclose(j1, j2)                          # historico: el relleno cuenta


def test_vit_cls_feature():
    m = _model(91).eval()
    audio, ids, mask, L, y = _batch()
    with torch.no_grad():
        out = multimodal_forward(m, audio, ids, mask, y)
        img = m._waveform_to_vit_image(audio)
        cls = m.vision_encoder(pixel_values=img).last_hidden_state[:, 0, :]
    assert torch.allclose(out["vision_features"], cls, atol=1e-6)
    assert not any(p.requires_grad for p in m.vision_encoder.pooler.parameters())


def test_vit_partial_finetune_blocks():
    m = _model(92)
    layers = m.vision_encoder.encoder.layer
    assert all(not p.requires_grad for l in layers[:8] for p in l.parameters())
    assert all(p.requires_grad for l in layers[8:] for p in l.parameters())
    assert all(p.requires_grad for p in m.vision_encoder.layernorm.parameters())
    assert not any(p.requires_grad for p in m.vision_encoder.embeddings.parameters())


@pytest.mark.parametrize("study", [72, 87, 88, 89, 90, 91, 92, 93, 94])
def test_one_optimizer_step(study):
    from src.ablation_studies.utils import create_ablation_optimizer
    cfg = _config(study)
    m = _model(study, cfg).train()
    opt = create_ablation_optimizer(m, cfg, study)
    audio, ids, mask, L, y = _batch()
    out = multimodal_forward(m, audio, ids, mask, y, batch={"audio_lengths": L})
    loss = torch.nn.functional.cross_entropy(out["logits"], y)
    assert torch.isfinite(loss)
    antes = m.fusion.audio_proj.weight.detach().clone()
    loss.backward()
    opt.step()
    assert not torch.equal(antes, m.fusion.audio_proj.weight)


def test_build_forced_alignment_cache(tmp_path, monkeypatch):
    import argparse
    import scripts.fit_speechcue_stats as fit
    items = []
    base = {}
    for i in range(3):
        uid = f"Ses01F_impro01_F{i:03d}"
        wav = tmp_path / "Session1/sentences/wav/Ses01F_impro01" / f"{uid}.wav"
        fa = tmp_path / "Session1/sentences/ForcedAlignment/Ses01F_impro01" / f"{uid}.syseg"
        fa.parent.mkdir(parents=True, exist_ok=True)
        fa.write_text(SYSEG_REAL)
        items.append((uid, str(wav), "hello there"))
        base[uid] = SpeechCueStats(150.0 + i, 0.1, 2.0, 2.0, 1.5)
    SpeechCueStatsCache.save(base, tmp_path / "base.json", meta={"version": 4})
    monkeypatch.setattr(fit, "enumerate_utterances", lambda cfg, s: items)
    out = tmp_path / "fa.json"
    args = argparse.Namespace(base_cache=str(tmp_path / "base.json"), out=str(out),
                              max_missing_frac=0.5)
    assert fit.build_forced_alignment_cache(args, {}, [1]) == 0
    c = SpeechCueStatsCache.load(out)
    assert SpeechCueStatsCache.load_meta(out)["rate_source"] == "forced_alignment"
    st = c.get("Ses01F_impro01_F001")
    assert st.pitch == 151.0 and st.rate == pytest.approx(3 / 0.61)
    # un .syseg que falta -> tasa NaN (token MID), registrado en meta
    (tmp_path / "Session1/sentences/ForcedAlignment/Ses01F_impro01/Ses01F_impro01_F002.syseg").unlink()
    out.unlink()
    assert fit.build_forced_alignment_cache(args, {}, [1]) == 0
    assert math.isnan(SpeechCueStatsCache.load(out).get("Ses01F_impro01_F002").rate)
    assert SpeechCueStatsCache.load_meta(out)["fa_unaligned"] == ["Ses01F_impro01_F002"]
    # demasiados sin alinear -> falla y no escribe
    args.max_missing_frac = 0.1
    out.unlink()
    assert fit.build_forced_alignment_cache(args, {}, [1]) == 1 and not out.exists()
    # un formato desconocido -> falla
    args.max_missing_frac = 0.5
    (tmp_path / "Session1/sentences/ForcedAlignment/Ses01F_impro01/Ses01F_impro01_F001.syseg").write_text("basura\n" * 9)
    assert fit.build_forced_alignment_cache(args, {}, [1]) == 1
