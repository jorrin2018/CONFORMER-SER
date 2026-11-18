"""Tests de la alineacion contrastiva audio-texto (estudios 120-122).

Sin GPU, sin IEMOCAP, sin red: encoders diminutos (tests/tiny_encoders.py).

    python -m pytest tests/test_alignment_studies.py -q
"""
from __future__ import annotations

import glob
import logging
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_ablation import create_ablation_model, load_ablation_config  # noqa: E402
from src.ablation_studies.utils import create_ablation_optimizer  # noqa: E402
from src.models.alignment import AlineacionContrastiva  # noqa: E402
from src.models.forward_utils import multimodal_forward  # noqa: E402
from tests.tiny_encoders import offline_config, tiny_encoders  # noqa: E402

LOG = logging.getLogger("t")

ALINEACION = {
    "model.alineacion.enabled": True, "model.alineacion.dim": 128,
    "model.alineacion.temperatura": 0.07, "training.loss.contrastive_weight": 0.1,
    "training.learning_rate_alignment": 1.0e-4,
}
ESPERADO = {
    120: (98, {**ALINEACION, "model.alineacion.tipo": "memoria", "model.alineacion.memoria": 512}),
    121: (98, {**ALINEACION, "model.alineacion.tipo": "prototipos",
               "model.alineacion.momento_prototipos": 0.9}),
    122: (120, {"model.crossmodal_attention.enabled": True,
                "model.crossmodal_attention.num_heads": 8,
                "model.crossmodal_attention.num_layers": 1,
                "model.crossmodal_attention.dropout": 0.1,
                "model.crossmodal_attention.ffn_mult": 2,
                "training.learning_rate_crossmodal": 1.0e-4}),
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


def _batch(B=4):
    torch.manual_seed(0)
    mask = torch.ones(B, 12, dtype=torch.long)
    mask[1, 8:] = 0
    return torch.randn(B, 16000 * 2), torch.randint(1000, 2000, (B, 12)), mask, torch.arange(B) % 4


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


# ------------------------------------------------------------------ el modulo
def test_memoria_queue_fills_and_wraps():
    m = AlineacionContrastiva(16, 16, tipo="memoria", dim=8, memoria=6)
    for _ in range(4):
        m(torch.randn(4, 16), torch.randn(4, 16))
    assert int(m.llenos) == 6 and int(m.puntero) == 16 % 6
    assert torch.allclose(m.cola_audio.norm(dim=1), torch.ones(6), atol=1e-5)


def test_memoria_adds_negatives():
    """Con cola llena, la perdida inicial ~ log(B + K): la cola SI entra como negativos."""
    torch.manual_seed(0)
    m = AlineacionContrastiva(16, 16, tipo="memoria", dim=8, memoria=60, temperatura=100.0)
    sin_cola = float(m(torch.randn(4, 16), torch.randn(4, 16)))
    for _ in range(20):
        m(torch.randn(4, 16), torch.randn(4, 16))
    con_cola = float(m(torch.randn(4, 16), torch.randn(4, 16)))
    assert abs(sin_cola - math.log(4)) < 0.05
    assert abs(con_cola - math.log(64)) < 0.05


def test_memoria_matched_pairs_get_lower_loss():
    torch.manual_seed(0)
    m = AlineacionContrastiva(16, 16, tipo="memoria", dim=8, memoria=32)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(150):  # pares nuevos en cada paso: audio y texto con la misma senal
        x = torch.randn(4, 16)
        loss = m(x, x + 0.05 * torch.randn(4, 16))
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < 0.5 * math.log(4 + 32)


def test_prototipos_need_labels_and_track_classes():
    m = AlineacionContrastiva(16, 16, tipo="prototipos", dim=8, num_classes=4)
    with pytest.raises(RuntimeError, match="etiquetas"):
        m(torch.randn(4, 16), torch.randn(4, 16))
    y = torch.tensor([0, 0, 1, 1])
    loss = m(torch.randn(4, 16), torch.randn(4, 16), y)
    assert torch.isfinite(loss)
    assert m.proto_listo.tolist() == [True, True, False, False]
    loss.backward()
    assert m.proy_audio[0].weight.grad.abs().sum() > 0
    m(torch.randn(4, 16), torch.randn(4, 16), torch.tensor([2, 3, 3, 0]))
    assert bool(m.proto_listo.all())


def test_bad_tipo_rejected():
    with pytest.raises(ValueError):
        AlineacionContrastiva(16, 16, tipo="otra")


# ------------------------------------------------------------------ en el modelo
def test_98_has_no_alignment():
    m, _ = _model(98)
    assert m.alineacion is None
    m.train()
    out = multimodal_forward(m, *_batch()[:3], _batch()[3], batch={})
    assert "loss_contrastive" not in out


@pytest.mark.parametrize("study", sorted(ESPERADO))
def test_one_step_trains_alignment_and_encoders(study):
    m, cfg = _model(study)
    assert m.alineacion is not None and m.alineacion.tipo == cfg["model"]["alineacion"]["tipo"]
    opt = create_ablation_optimizer(m, cfg, study)
    lr_de = {id(p): g["lr"] for g in opt.param_groups for p in g["params"]}
    assert all(lr_de[id(p)] == 1.0e-4 for p in m.alineacion.parameters())
    m.train()
    audio, ids, mask, y = _batch()
    out = multimodal_forward(m, audio, ids, mask, y, batch={})
    assert torch.isfinite(out["loss_contrastive"])
    loss = torch.nn.functional.cross_entropy(out["logits"], y) + 0.1 * out["loss_contrastive"]
    proj = m.alineacion.proy_audio[0].weight.detach().clone()
    loss.backward()
    top = [p for p in m.text_encoder.parameters() if p.requires_grad]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in top)
    opt.step()
    assert not torch.equal(proj, m.alineacion.proy_audio[0].weight)
    if study == 122:
        assert m.crossmodal is not None


def test_eval_has_no_contrastive_loss_and_same_logits():
    m, _ = _model(120)
    m.eval()
    audio, ids, mask, y = _batch()
    with torch.no_grad():
        a = multimodal_forward(m, audio, ids, mask, y, batch={})
        b = multimodal_forward(m, audio, ids, mask, None, batch={})
    assert "loss_contrastive" not in a
    assert torch.equal(a["logits"], b["logits"])
    assert int(m.alineacion.llenos) == 0  # en evaluacion la cola no se toca


def test_alignment_without_weight_fails():
    cfg = offline_config(_cfg(120))
    cfg["training"]["loss"]["contrastive_weight"] = 0.0
    with pytest.raises(ValueError, match="contrastive_weight"):
        _model(120, cfg)


def test_alignment_lr_missing_warns():
    m, cfg = _model(120)
    del cfg["training"]["learning_rate_alignment"]
    with pytest.warns(UserWarning, match="learning_rate_alignment"):
        create_ablation_optimizer(m, cfg, 120)
