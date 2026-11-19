"""Tests de los estudios 123-126 (revision de noviembre): sin rama mel, MulT, detector
con cross-fitting.

Sin GPU, sin IEMOCAP, sin red: encoders diminutos (tests/tiny_encoders.py) y un cache de
expertos falso en tmp_path.

    python -m pytest tests/test_review_studies.py -q
"""
from __future__ import annotations

import glob
import json
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
from src.data.expert_gate import RASGOS, load_expert_features  # noqa: E402
from src.models.bimodal_fusion import FusionBimodal  # noqa: E402
from src.models.forward_utils import multimodal_forward  # noqa: E402
from src.models.mult import MulTTransmodal  # noqa: E402
from tests.tiny_encoders import offline_config, tiny_encoders  # noqa: E402

LOG = logging.getLogger("t")

MULT = {"model.crossmodal_attention.enabled": True, "model.crossmodal_attention.tipo": "mult",
        "model.crossmodal_attention.d_model": 256, "model.crossmodal_attention.num_heads": 8,
        "model.crossmodal_attention.num_layers": 4, "model.crossmodal_attention.dropout": 0.1,
        "model.crossmodal_attention.attn_dropout": 0.1,
        "model.crossmodal_attention.embed_dropout": 0.1,
        "model.crossmodal_attention.ffn_mult": 4, "model.crossmodal_attention.kernel": 1,
        "training.learning_rate_crossmodal": 1.0e-4}
DET = {"model.detector_conflicto.tipo": "crossfit",
       "model.detector_conflicto.expert_cache_dir": "cache/experts",
       "model.detector_conflicto.oculto": 32, "training.learning_rate_detector": 5.0e-4}
ESPERADO = {
    123: (98, {"model.vision_stream": "remove"}),
    124: (123, MULT),
    125: (123, DET),
    126: (124, DET),
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


def _model(study, cfg=None):
    cfg = cfg or offline_config(_cfg(study))
    torch.manual_seed(0)
    with tiny_encoders():
        return create_ablation_model(study, cfg, "cpu", LOG), cfg


def _batch(B=3):
    torch.manual_seed(0)
    mask = torch.ones(B, 12, dtype=torch.long)
    mask[1, 7:] = 0
    return (torch.randn(B, 16000 * 2), torch.randint(1000, 2000, (B, 12)), mask,
            torch.arange(B) % 4, {"expert_feats": torch.randn(B, len(RASGOS))})


@pytest.mark.parametrize("study", sorted(ESPERADO))
def test_config_is_base_plus_one_change(study):
    base_n, esperado = ESPERADO[study]
    base, cfg = _flat(_cfg(base_n)), _flat(_cfg(study))
    cambios = {k: cfg.get(k) for k in set(base) | set(cfg)
               if not k.startswith("logging.") and base.get(k) != cfg.get(k)}
    assert cambios == esperado


def test_studies_registered():
    src = (ROOT / "scripts/run_ablation.py").read_text(encoding="utf-8")
    for n in ESPERADO:
        assert f"({n}, \"config/ablation/ablation_{n}_" in src


# ------------------------------------------------------------------ sin rama mel
def test_123_has_no_vit_and_two_tokens():
    m, _ = _model(123)
    assert isinstance(m.fusion, FusionBimodal) and m.fusion.detector is None
    assert sum(p.numel() for p in m.vision_encoder.parameters()) == 0
    assert not hasattr(m.fusion, "vision_proj") and not hasattr(m.fusion, "audio_classifier")
    m.eval()
    audio, ids, mask, y, batch = _batch()
    out = multimodal_forward(m, audio, ids, mask, y, batch={})
    assert out["logits"].shape == (3, 4) and "vision_features" not in out
    assert torch.allclose(m.fusion.last_weights, torch.full((3, 2), 0.5))
    with pytest.raises(RuntimeError, match="rama mel"):
        m.vision_encoder(torch.zeros(1))


def test_123_fewer_params_than_98():
    m98, _ = _model(98)
    m123, _ = _model(123)
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(m123) < n(m98)


def test_remove_rejects_simple_detector():
    cfg = offline_config(_cfg(123))
    cfg["model"]["conflict_aware"]["enabled"] = True
    with pytest.raises(ValueError, match="detector simple"):
        _model(123, cfg)


# ------------------------------------------------------------------ MulT
def test_mult_source_is_low_level_and_masks_text():
    torch.manual_seed(0)
    mt = MulTTransmodal(32, 24, d_model=16, num_heads=4, num_layers=3).eval()
    a, t = torch.randn(2, 20, 32), torch.randn(2, 9, 24)
    mask = torch.ones(2, 9, dtype=torch.long); mask[0, 5:] = 0
    za, zt = mt(a, t, mask)
    assert za.shape == (2, 20, 16) and zt.shape == (2, 9, 16)
    # El relleno del texto no cambia lo que ve el audio.
    t2 = t.clone(); t2[0, 5:] = 100.0
    za2, _ = mt(a, t2, mask)
    assert torch.allclose(za[0], za2[0], atol=1e-5)


def test_124_uses_mult_dims_and_trains():
    m, cfg = _model(124)
    assert isinstance(m.crossmodal, MulTTransmodal)
    assert m.fusion.audio_proj.in_features == 256 == m.fusion.text_proj.in_features
    opt = create_ablation_optimizer(m, cfg, 124)
    lr_de = {id(p): g["lr"] for g in opt.param_groups for p in g["params"]}
    assert all(lr_de[id(p)] == 1.0e-4 for p in m.crossmodal.parameters())
    m.train()
    audio, ids, mask, y, _ = _batch()
    out = multimodal_forward(m, audio, ids, mask, y, batch={})
    antes = m.crossmodal.texto_a_audio[0].attn.in_proj_weight.detach().clone()
    torch.nn.functional.cross_entropy(out["logits"], y).backward()
    opt.step()
    assert not torch.equal(antes, m.crossmodal.texto_a_audio[0].attn.in_proj_weight)


# ------------------------------------------------------------------ detector
def test_detector_starts_at_fixed_weights_then_moves():
    f = FusionBimodal(16, 16, 32, 4, num_heads=4, num_layers=1, detector="crossfit")
    x = torch.randn(3, len(RASGOS))
    f(torch.randn(3, 16), torch.randn(3, 16), expert_feats=x)
    assert torch.allclose(f.last_weights, torch.full((3, 2), 0.5))
    opt = torch.optim.Adam(f.parameters(), lr=1e-2)
    for _ in range(5):
        loss = torch.nn.functional.cross_entropy(
            f(torch.randn(3, 16), torch.randn(3, 16), expert_feats=x), torch.tensor([0, 1, 2]))
        opt.zero_grad(); loss.backward(); opt.step()
    f(torch.randn(3, 16), torch.randn(3, 16), expert_feats=x)
    assert not torch.allclose(f.last_weights, torch.full((3, 2), 0.5))
    with pytest.raises(RuntimeError, match="expert_feats"):
        f(torch.randn(3, 16), torch.randn(3, 16))


@pytest.mark.parametrize("study", [125, 126])
def test_detector_studies_need_and_use_expert_feats(study):
    m, cfg = _model(study)
    assert m.uses_expert_feats and m.fusion.detector is not None
    opt = create_ablation_optimizer(m, cfg, study)
    lr_de = {id(p): g["lr"] for g in opt.param_groups for p in g["params"]}
    assert all(lr_de[id(p)] == 5.0e-4 for p in m.fusion.detector.parameters())
    audio, ids, mask, y, batch = _batch()
    with pytest.raises(RuntimeError, match="expert_feats"):
        multimodal_forward(m, audio, ids, mask, y, batch={})
    m.train()
    out = multimodal_forward(m, audio, ids, mask, y, batch=batch)
    torch.nn.functional.cross_entropy(out["logits"], y).backward()
    assert m.fusion.detector[-1].weight.grad.abs().sum() > 0


def _cache(tmp_path):
    rng = np.random.default_rng(0)
    uids = [f"Ses0{s}F_impro01_F{i:03d}" for s in range(1, 6) for i in range(8)]
    split = {u: ("test" if u.startswith("Ses05") else ("val" if u.endswith("7") else "train")) for u in uids}
    def p():
        x = rng.random(4) + 0.1
        return (x / x.sum()).tolist()
    probs = {e: {u: p() for u in uids} for e in ("audio", "text", "vision")}
    oof = {e: {u: p() for u in uids if split[u] == "train"} for e in ("audio", "text", "vision")}
    d = {"meta": {"fold": 5}, "split": split, "labels": {u: 0 for u in uids}, "probs": probs, "probs_oof": oof}
    (tmp_path / "fold_S5.json").write_text(json.dumps(d))
    return uids, split, probs, oof


def test_load_expert_features_uses_oof_for_train(tmp_path):
    uids, split, probs, oof = _cache(tmp_path)
    textos = {u: "[LAUGHTER] I don't know." for u in uids}
    r, meta = load_expert_features(tmp_path, 5, textos)
    assert len(r) == len(uids) and len(next(iter(r.values()))) == len(RASGOS) == 14
    tr = [u for u in uids if split[u] == "train"]
    mu, sd = np.array(meta["media"]), np.array(meta["desv"])
    u = tr[0]
    crudo = np.array(r[u]) * sd + mu
    assert np.allclose(crudo[:4], np.log(np.clip(oof["audio"][u], 1e-6, 1)), atol=1e-4)
    t = [u for u in uids if split[u] == "test"][0]
    assert np.allclose((np.array(r[t]) * sd + mu)[4:8], np.log(np.clip(probs["text"][t], 1e-6, 1)), atol=1e-4)
    assert np.isclose(crudo[13], np.log1p(3))  # "i don't know" -> 3 palabras, sin la etiqueta
    with pytest.raises(FileNotFoundError):
        load_expert_features(tmp_path, 4, textos)


def test_iemocap_path_override_only_touches_iemocap(monkeypatch):
    # Sin config/local_paths.yaml de esta maquina: el test no depende de donde corre.
    from src.utils import local_paths
    monkeypatch.setattr(local_paths, "_cache", {})
    monkeypatch.setenv("CAMF_IEMOCAP_PATH", r"E:\LAB\IEMOCAP_full_release")
    assert _cfg(123)["dataset"]["data_path"] == r"E:\LAB\IEMOCAP_full_release"
    assert _cfg(105)["dataset"]["data_path"] == "D:\\DOCTORADO\\DATASETS\\CREMA-D"
    monkeypatch.delenv("CAMF_IEMOCAP_PATH")
    assert _cfg(123)["dataset"]["data_path"].startswith("D:\\DOCTORADO")


def test_local_paths_precedence(monkeypatch, tmp_path):
    """local_paths.yaml sustituye la ruta del YAML; --iemocap-path / CAMF_IEMOCAP_PATH gana."""
    from src.utils import local_paths
    monkeypatch.setattr(local_paths, "_cache", {"iemocap": str(tmp_path)})
    monkeypatch.delenv("CAMF_IEMOCAP_PATH", raising=False)
    assert _cfg(123)["dataset"]["data_path"] == str(tmp_path)
    assert _cfg(105)["dataset"]["data_path"] == r"D:\DOCTORADO\DATASETS\CREMA-D"
    monkeypatch.setenv("CAMF_IEMOCAP_PATH", r"E:\LAB\IEMOCAP_full_release")
    assert _cfg(123)["dataset"]["data_path"] == r"E:\LAB\IEMOCAP_full_release"


def test_local_paths_missing_dir_fails(monkeypatch, tmp_path):
    """Una ruta local que no existe para la carga en vez de volver a la del YAML."""
    from src.utils import local_paths
    monkeypatch.setattr(local_paths, "_cache", {"iemocap": str(tmp_path / "no_existe")})
    monkeypatch.delenv("CAMF_IEMOCAP_PATH", raising=False)
    with pytest.raises(FileNotFoundError):
        _cfg(123)
