"""Tests de la familia clasica (estudios 111-119): eGeMAPS / COVAREP + GloVe + BiLSTM.

Sin GPU, sin corpus y sin red: caches de descriptores y de GloVe falsos en tmp_path,
encoders HF diminutos para el ViT del 115 (tests/tiny_encoders.py).

    python -m pytest tests/test_classic_studies.py -q
"""
from __future__ import annotations

import glob
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_ablation import create_ablation_model, load_ablation_config  # noqa: E402
from src.ablation_studies.utils import create_ablation_optimizer  # noqa: E402
from src.data.base_dataset import BaseSERDataset  # noqa: E402
from src.data.classic_features import ClassicAudioStore, guardar_cache  # noqa: E402
from src.data.dataset import collate_fn  # noqa: E402
from src.data.preprocessing import AudioPreprocessor, TextPreprocessor  # noqa: E402
from src.models.classic_encoders import (  # noqa: E402
    CLS, ESPECIALES, PAD, UNK, ClassicAudioEncoder, ClassicTextEncoder, GloveTokenizer,
    tokenizar,
)
from src.models.forward_utils import multimodal_forward  # noqa: E402
from tests.tiny_encoders import offline_config, tiny_encoders  # noqa: E402

LOG = logging.getLogger("t")
SR = 16000
F = 25
UTTS = [f"Ses01F_impro01_F{i:03d}" for i in range(6)]
PALABRAS = ["hello", "there", "do", "overall", "energy", "is", "high", "low", "medium",
            ",", ".", "variability", "speaking", "rate", "slow", "moderate", "fast", "Yeah", "n't", "'s", "it"]

CLASICOS = (111, 112, 113, 114, 115, 116, 117, 118, 119)
# estudio -> (base, claves que cambian aparte de model.clasico.*)
ESPERADO = {
    111: (71, {"model.audio_encoder": "clasico:lld", "training.learning_rate_audio": 1.0e-3}),
    112: (73, {"model.text_encoder": "clasico:glove", "text.model_name": "clasico:glove",
               "training.learning_rate_text": 1.0e-3}),
    113: (98, {"model.audio_encoder": "clasico:lld", "model.text_encoder": "clasico:glove",
               "text.model_name": "clasico:glove", "training.learning_rate_audio": 1.0e-3,
               "training.learning_rate_text": 1.0e-3}),
    114: (113, {"model.conflict_aware.enabled": True, "model.conflict_aware.aux_loss_weight": 0.3}),
    115: (74, {"model.audio_encoder": "clasico:lld", "model.text_encoder": "clasico:glove",
               "text.model_name": "clasico:glove", "training.learning_rate_audio": 1.0e-3,
               "training.learning_rate_text": 1.0e-3}),
    116: (105, {"model.audio_encoder": "clasico:lld", "model.text_encoder": "clasico:glove",
                "text.model_name": "clasico:glove", "training.learning_rate_audio": 1.0e-3,
                "training.learning_rate_text": 1.0e-3}),
    117: (78, {"model.audio_encoder": "clasico:lld", "training.learning_rate_audio": 1.0e-3}),
    118: (106, {"model.audio_encoder": "clasico:lld", "model.text_encoder": "clasico:glove",
                "text.model_name": "clasico:glove", "training.learning_rate_audio": 1.0e-3,
                "training.learning_rate_text": 1.0e-3}),
    119: (80, {"model.audio_encoder": "clasico:lld", "training.learning_rate_audio": 1.0e-3}),
}


def _cfg(study: int) -> dict:
    path = glob.glob(str(ROOT / f"config/ablation/ablation_{study}_*.yaml"))
    assert len(path) == 1, path
    return load_ablation_config(path[0], {})


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


@pytest.fixture
def caches(tmp_path):
    g = torch.Generator().manual_seed(0)
    feats = {u: (torch.randn(120 + 40 * i, F, generator=g) * 3 + 5).to(torch.float16)
             for i, u in enumerate(UTTS)}
    audio = tmp_path / "fake_egemaps.pt"
    guardar_cache(str(audio), feats, {"fuente": "egemaps", "dim": F, "fps": 100.0})
    itos = list(ESPECIALES) + sorted(PALABRAS)
    vec = torch.zeros(len(itos), 8)
    vec[len(ESPECIALES):] = torch.randn(len(itos) - len(ESPECIALES), 8, generator=g)
    glove = tmp_path / "glove_fake.pt"
    torch.save({"itos": itos, "vectors": vec, "meta": {"fuente": "fake"}}, glove)
    return {"audio_cache": str(audio), "glove_cache": str(glove), "feats": feats}


def _config(study, caches):
    cfg = _cfg(study)
    if not str(cfg["text"].get("model_name", "")).startswith("clasico:"):
        cfg = offline_config(cfg)
    cfg["model"]["clasico"]["audio_cache"] = caches["audio_cache"]
    cfg["model"]["clasico"]["glove_cache"] = caches["glove_cache"]
    return cfg


def _model(study, caches):
    cfg = _config(study, caches)
    torch.manual_seed(0)
    with tiny_encoders():
        return create_ablation_model(study, cfg, "cpu", LOG), cfg


def _batch(caches, B=2, texto_glove=True):
    store = ClassicAudioStore(caches["audio_cache"])
    store.ajustar_normalizacion(UTTS[:4])
    pares = [store.segmento(u, 0.0, 2.0) for u in UTTS[:B]]
    tok = GloveTokenizer(caches["glove_cache"])
    if texto_glove:
        enc = [tok("[SPEECH] hello there [PROSODY] overall energy is high.", max_length=24),
               tok("Yeah, do you?", max_length=24)][:B]
        ids = torch.cat([e["input_ids"] for e in enc])
        mask = torch.cat([e["attention_mask"] for e in enc])
    else:
        ids = torch.randint(1000, 2000, (B, 24))
        mask = torch.ones(B, 24, dtype=torch.long)
    batch = {"audio_feats": torch.stack([p[0] for p in pares]),
             "audio_feats_lengths": torch.tensor([p[1] for p in pares]),
             "audio_lengths": torch.tensor([2 * SR] * B)}
    return torch.randn(B, 2 * SR), ids, mask, torch.arange(B) % 4, batch


# ------------------------------------------------------------------ configs
@pytest.mark.parametrize("study", sorted(ESPERADO))
def test_config_is_base_plus_encoder_change(study):
    base_n, esperado = ESPERADO[study]
    base, cfg = _flat(_cfg(base_n)), _flat(_cfg(study))
    cambios = {k: cfg.get(k) for k in set(base) | set(cfg)
               if not k.startswith(("logging.", "model.clasico.")) and base.get(k) != cfg.get(k)}
    assert cambios == esperado


@pytest.mark.parametrize("study", CLASICOS)
def test_config_clasico_block(study):
    c = _cfg(study)["model"]["clasico"]
    corpus = _cfg(study)["dataset"].get("corpus", "iemocap")
    assert c["fuente"] == "egemaps"
    assert c["audio_cache"] == f"cache/clasicas/{corpus}_egemaps.pt"
    assert c["glove_cache"] == f"cache/clasicas/glove_{corpus}.pt"
    assert _cfg(study)["training"]["learning_rate_fusion"] == 5.0e-6


def test_studies_registered():
    src = (ROOT / "scripts/run_ablation.py").read_text(encoding="utf-8")
    for n in CLASICOS:
        assert f"({n}, \"config/ablation/ablation_{n}_" in src


# ------------------------------------------------------------------ texto
def test_tokenizer(caches):
    tok = GloveTokenizer(caches["glove_cache"])
    assert tokenizar("[SPEECH] Don't go. It's [PROSODY] x") == [
        "[SPEECH]", "Do", "n't", "go", ".", "It", "'s", "[PROSODY]", "x"]
    out = tok("[SPEECH] Hello zzqx, don't [PROSODY]", max_length=10)
    ids, mask = out["input_ids"][0].tolist(), out["attention_mask"][0].tolist()
    itos = tok.itos
    assert ids[0] == CLS
    assert itos[ids[1]] == "[SPEECH]" and itos[ids[2]] == "hello"  # minusculas si no hay forma exacta
    assert ids[3] == UNK                                            # palabra sin vector
    assert [itos[i] for i in ids[5:8]] == ["do", "n't", "[PROSODY]"]  # la negacion se queda
    assert mask == [1] * 8 + [0] * 2 and ids[8:] == [PAD] * 2


def test_text_encoder_glove_frozen_specials_learn(caches):
    d = torch.load(caches["glove_cache"], weights_only=False)
    enc = ClassicTextEncoder(d["vectors"], d_model=16, lstm_hidden=8, lstm_layers=1)
    assert not enc.palabras.weight.requires_grad and enc.especiales.weight.requires_grad
    tok = GloveTokenizer(caches["glove_cache"])
    e = tok("[SPEECH] hello there", max_length=12)
    out = enc(input_ids=e["input_ids"], attention_mask=e["attention_mask"]).last_hidden_state
    assert out.shape == (1, 1, enc.config.hidden_size) == (1, 1, 32)
    out.sum().backward()
    assert enc.especiales.weight.grad.abs().sum() > 0


def test_text_encoder_ignores_padding(caches):
    d = torch.load(caches["glove_cache"], weights_only=False)
    enc = ClassicTextEncoder(d["vectors"], d_model=16, lstm_hidden=8, lstm_layers=2).eval()
    tok = GloveTokenizer(caches["glove_cache"])
    a, b = tok("hello there .", max_length=6), tok("hello there .", max_length=30)
    ya = enc(input_ids=a["input_ids"], attention_mask=a["attention_mask"]).last_hidden_state
    yb = enc(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
    assert torch.allclose(ya, yb, atol=1e-6)


# ------------------------------------------------------------------ audio
def test_audio_encoder_ignores_padding():
    enc = ClassicAudioEncoder(F, d_model=16, lstm_hidden=8, lstm_layers=2).eval()
    x = torch.randn(1, 50, F)
    corto = enc(x, torch.tensor([50])).last_hidden_state
    largo = enc(torch.cat([x, torch.randn(1, 70, F)], 1), torch.tensor([50])).last_hidden_state
    assert torch.allclose(corto, largo, atol=1e-6)
    with pytest.raises(RuntimeError, match="forma de onda"):
        enc(torch.randn(1, 16000), torch.tensor([1]))


def test_store_normalizes_with_train_only_and_crops(caches):
    st = ClassicAudioStore(caches["audio_cache"])
    with pytest.raises(RuntimeError):
        st.segmento(UTTS[0], 0.0, 1.0)
    media, desv = st.ajustar_normalizacion(UTTS[:2])
    x = torch.cat([caches["feats"][u].float() for u in UTTS[:2]])
    assert torch.allclose(media, x.mean(0), atol=1e-3)
    assert torch.allclose(desv, x.std(0, unbiased=False), atol=1e-2)
    seg, n = st.segmento(UTTS[0], 0.0, 2.0)          # 120 frames reales, 200 de largo
    assert seg.shape == (200, F) and n == 120
    assert torch.count_nonzero(seg[120:]) == 0
    seg2, n2 = st.segmento(UTTS[5], 0.5, 1.0)        # recorte desde el frame 50
    esperado = (caches["feats"][UTTS[5]].float()[50:150] - media) / desv
    assert n2 == 100 and torch.allclose(seg2, esperado, atol=1e-5)
    assert st.faltantes(UTTS + ["no_existe"]) == ["no_existe"]


# ------------------------------------------------------------------ dataset
class _FakeCorpus(BaseSERDataset):
    CORPUS = "fake"
    WAVS: list = []

    def _collect_records(self):
        return [dict(audio_path=p, transcription="hello there", emotion="angry",
                     emotion_idx=0, utterance_id=UTTS[i]) for i, p in enumerate(self.WAVS)]


def test_dataset_and_collate_carry_features(tmp_path, caches):
    import soundfile as sf
    paths = []
    for i, s in enumerate((1.2, 3.0)):
        p = tmp_path / f"u{i}.wav"
        sf.write(p, (0.1 * np.random.default_rng(i).standard_normal(int(s * SR))).astype("float32"), SR)
        paths.append(str(p))
    _FakeCorpus.WAVS = paths
    ds = _FakeCorpus(data_path=".", sessions=[1], emotions=["angry"], sample_rate=SR,
                     audio_preprocessor=AudioPreprocessor(sample_rate=SR),
                     text_preprocessor=TextPreprocessor(model_name="clasico:glove", max_length=16,
                                                        glove_cache=caches["glove_cache"]),
                     max_audio_length=2.0, training=False)
    st = ClassicAudioStore(caches["audio_cache"])
    st.ajustar_normalizacion(UTTS[:2])
    ds.classic_audio = st
    b = collate_fn([ds[0], ds[1]])
    assert b["audio_feats"].shape == (2, 200, F)
    assert b["audio_feats_lengths"].tolist() == [120, 160]
    assert b["text_input_ids"][0, 0].item() == CLS


def test_text_preprocessor_classic_requires_cache():
    with pytest.raises(ValueError, match="glove_cache"):
        TextPreprocessor(model_name="clasico:glove")


# ------------------------------------------------------------------ modelos
@pytest.mark.parametrize("study", CLASICOS)
def test_one_optimizer_step(study, caches):
    m, cfg = _model(study, caches)
    m.train()
    texto_glove = cfg["model"]["text_encoder"].startswith("clasico:")
    audio, ids, mask, y, batch = _batch(caches, texto_glove=texto_glove)
    opt = create_ablation_optimizer(m, cfg, study)
    lr_de = {id(p): g["lr"] for g in opt.param_groups for p in g["params"]}
    if cfg["model"].get("modality") != "text" and cfg["model"]["audio_encoder"].startswith("clasico:"):
        assert m.uses_classic_audio
        assert all(lr_de[id(p)] == 1.0e-3 for p in m.audio_encoder.parameters())
    if texto_glove and cfg["model"].get("modality") != "audio":
        assert all(lr_de[id(p)] == 1.0e-3 for p in m.text_encoder.parameters() if p.requires_grad)
        assert not m.text_encoder.palabras.weight.requires_grad
    out = multimodal_forward(m, audio, ids, mask, y, batch=batch)
    loss = torch.nn.functional.cross_entropy(out["logits"], y)
    assert torch.isfinite(loss)
    enc = m.audio_encoder if getattr(m, "uses_classic_audio", False) else m.text_encoder
    antes = [p.detach().clone() for p in enc.parameters() if p.requires_grad]
    loss.backward()
    opt.step()
    despues = [p for p in enc.parameters() if p.requires_grad]
    assert any(not torch.equal(a, b) for a, b in zip(antes, despues))


def test_forward_without_features_fails(caches):
    m, _ = _model(113, caches)
    audio, ids, mask, y, _ = _batch(caches)
    with pytest.raises(RuntimeError, match="audio_feats"):
        multimodal_forward(m, audio, ids, mask, y, batch={})


def test_113_vision_off_and_115_vision_on(caches):
    m113, _ = _model(113, caches)
    m115, _ = _model(115, caches)
    assert m113._vision_stream == "off" and m115._vision_stream == "on"
    m113.eval()
    m113.vision_encoder.forward = lambda *a, **k: (_ for _ in ()).throw(AssertionError("ViT"))
    audio, ids, mask, y, batch = _batch(caches)
    multimodal_forward(m113, audio, ids, mask, y, batch=batch)


def test_114_has_gate_and_113_not(caches):
    m113, _ = _model(113, caches)
    m114, _ = _model(114, caches)
    assert getattr(m113.fusion, "conflict_aware_enabled", None) is False
    assert getattr(m114.fusion, "conflict_aware_enabled", None) is True
