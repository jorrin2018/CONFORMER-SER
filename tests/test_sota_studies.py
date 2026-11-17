"""Tests de los estudios 98-110 (filas pendientes del articulo y mejoras candidatas).

Sin GPU, sin IEMOCAP, sin red: encoders diminutos y aleatorios (tests/tiny_encoders.py).

    python -m pytest tests/test_sota_studies.py -q
"""
from __future__ import annotations

import glob
import json
import logging
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_ablation import create_ablation_model, load_ablation_config  # noqa: E402
from src.ablation_studies.utils import create_ablation_optimizer  # noqa: E402
from src.models.crossmodal import CrossModalCoAttention  # noqa: E402
from tests.tiny_encoders import offline_config, tiny_encoders  # noqa: E402

LOG = logging.getLogger("t")


def _cfg(study: int) -> dict:
    path = glob.glob(str(ROOT / f"config/ablation/ablation_{study}_*.yaml"))
    assert len(path) == 1, f"config del estudio {study}: {path}"
    return load_ablation_config(path[0], {})


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


# estudio -> (base, claves que cambian frente a la base)
ESPERADO = {
    98: (74, {"model.vision_stream": "off"}),
    99: (98, {"training.model_selection": "last", "training.num_epochs": 14}),
    100: (98, {"text.transcript_source": "asr",
               "text.asr_cache": "cache/asr/iemocap_whisper-small.en.json"}),
    101: (74, {"model.crossmodal_attention.enabled": True,
               "model.crossmodal_attention.num_heads": 8,
               "model.crossmodal_attention.num_layers": 1,
               "model.crossmodal_attention.dropout": 0.1,
               "model.crossmodal_attention.ffn_mult": 2,
               "training.learning_rate_crossmodal": 1.0e-4}),
    102: (74, {"model.audio_encoder": "microsoft/wavlm-base-plus"}),
    103: (98, {"model.text_encoder": "roberta-base", "text.model_name": "roberta-base"}),
    104: (98, {"audio.pad_after_preprocessing": True, "model.audio_masked_pooling": True}),
    105: (77, {"model.vision_stream": "off"}),
    106: (79, {"model.vision_stream": "off"}),
    107: (98, {"text.use_prosody_cues": False}),
    108: (98, {"audio.apply_waveform_augment": False}),
    109: (98, {"model.fusion_num_layers": 0}),
    110: (81, {"model.vision_stream": "off"}),
}


@pytest.mark.parametrize("study", sorted(ESPERADO))
def test_config_is_base_plus_one_change(study):
    """Cada estudio es su base con exactamente las claves declaradas cambiadas."""
    base_n, esperado = ESPERADO[study]
    base = _flat(_cfg(base_n))
    cfg = _flat(_cfg(study))
    ignorar = ("logging.",)
    cambios = {k: cfg.get(k) for k in set(base) | set(cfg)
               if not k.startswith(ignorar) and base.get(k) != cfg.get(k)}
    assert cambios == esperado


def test_studies_registered():
    src = (ROOT / "scripts/run_ablation.py").read_text(encoding="utf-8")
    for n in ESPERADO:
        assert f"({n}, \"config/ablation/ablation_{n}_" in src


def _batch(B=2, S_t=12):
    torch.manual_seed(0)
    mask = torch.ones(B, S_t, dtype=torch.long)
    mask[1, 8:] = 0
    return (torch.randn(B, 16000 * 2), torch.randint(1000, 2000, (B, S_t)), mask,
            torch.tensor([0, 1]))


def _model(study):
    cfg = offline_config(_cfg(study))
    torch.manual_seed(0)
    with tiny_encoders():
        m = create_ablation_model(study, cfg, "cpu", LOG)
    return m, cfg


# ----------------------------------------------------------------- estudio 98
def test_98_does_not_run_vit_and_feeds_zeros():
    m, _ = _model(98)
    m.eval()

    def boom(*a, **k):
        raise AssertionError("el ViT no debe ejecutarse en el estudio 98")
    m.vision_encoder.forward = boom
    audio, ids, mask, y = _batch()
    with torch.no_grad():
        out = m(audio, ids, mask, y)
    assert out["logits"].shape == (2, 4)
    assert torch.count_nonzero(out["vision_features"]) == 0


def test_98_same_parameters_as_74():
    m74, _ = _model(74)
    m98, _ = _model(98)
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert n(m74) == n(m98)


def test_74_unchanged_runs_vit_and_has_no_crossmodal():
    m, _ = _model(74)
    assert m.crossmodal is None and m._vision_stream == "on"
    llamado = []
    orig = m.vision_encoder.forward

    def espia(*a, **k):
        llamado.append(1)
        return orig(*a, **k)
    m.vision_encoder.forward = espia
    m.eval()
    with torch.no_grad():
        m(*_batch())
    assert llamado


# ---------------------------------------------------------------- estudio 101
def test_101_has_crossmodal_and_its_own_lr_group():
    m, cfg = _model(101)
    assert isinstance(m.crossmodal, CrossModalCoAttention)
    opt = create_ablation_optimizer(m, cfg, 101)
    cm = {id(p) for n, p in m.named_parameters() if n.startswith("crossmodal.")}
    grupos = [g for g in opt.param_groups if {id(p) for p in g["params"]} == cm]
    assert len(grupos) == 1 and grupos[0]["lr"] == pytest.approx(1e-4)


def test_101_trains_crossmodal():
    m, cfg = _model(101)
    m.train()
    opt = create_ablation_optimizer(m, cfg, 101)
    antes = {n: p.detach().clone() for n, p in m.named_parameters() if n.startswith("crossmodal.")}
    out = m(*_batch())
    torch.nn.functional.cross_entropy(out["logits"], _batch()[3]).backward()
    sin_grad = [n for n, p in m.named_parameters()
                if n.startswith("crossmodal.") and (p.grad is None or not p.grad.abs().sum())]
    assert not sin_grad, sin_grad
    opt.step()
    assert any(not torch.equal(antes[n], p) for n, p in m.named_parameters() if n in antes)


def test_crossmodal_ignores_padded_text_tokens():
    torch.manual_seed(0)
    cm = CrossModalCoAttention(audio_dim=16, text_dim=24, num_heads=4).eval()
    a, t = torch.randn(2, 30, 16), torch.randn(2, 10, 24)
    mask = torch.ones(2, 10, dtype=torch.long)
    mask[:, 6:] = 0
    t2 = t.clone()
    t2[:, 6:] = 99.0  # basura en el relleno
    with torch.no_grad():
        a1, t1 = cm(a, t, mask)
        a2, t2o = cm(a, t2, mask)
    assert a1.shape == a.shape and t1.shape == t.shape
    assert torch.allclose(a1, a2, atol=1e-6)            # el audio no ve el relleno
    assert torch.allclose(t1[:, :6], t2o[:, :6], atol=1e-6)


def test_crossmodal_without_lr_warns():
    m, cfg = _model(101)
    cfg["training"].pop("learning_rate_crossmodal")
    with pytest.warns(UserWarning, match="learning_rate_crossmodal"):
        create_ablation_optimizer(m, cfg, 101)


# ---------------------------------------------------------------- estudio 104
def test_104_pools_only_real_frames():
    """Con audio_masked_pooling, cambiar el relleno no cambia la prediccion... del pooling.

    wav2vec2-base no recibe mascara de atencion, asi que el relleno SI llega al encoder;
    lo que el 104 garantiza es que el promedio solo usa frames reales. Se comprueba con
    el modelo que el vector de audio no depende de frames fuera de la longitud real.
    """
    m, _ = _model(104)
    assert m.uses_audio_lengths and m._audio_masked_pooling and m._vision_stream == "off"
    m.eval()
    feats = torch.randn(2, 100, 32)
    L = torch.tensor([16000 * 2, 16000])          # 2 s y 1 s de habla real
    mask = m._audio_frame_mask(L, feats.size(1))
    a = m.pool_audio_features(feats, mask)
    feats2 = feats.clone()
    feats2[1, int(mask[1].sum()):] = 1e3            # basura fuera del enunciado real
    b = m.pool_audio_features(feats2, mask)
    assert torch.allclose(a, b)
    audio, ids, msk, y = _batch()
    with torch.no_grad():
        out = m(audio, ids, msk, y, audio_lengths=torch.tensor([32000, 16000]))
    assert out["logits"].shape == (2, 4)


# ---------------------------------------------------------------- estudio 99
def test_99_selection_code_paths_exist():
    src = (ROOT / "scripts/run_ablation.py").read_text(encoding="utf-8")
    assert "seleccion = config['training'].get('model_selection', 'best_val')" in src
    assert "if seleccion == 'last':" in src
    assert "'model_selection': seleccion" in src


# ---------------------------------------------------------------- estudio 100
def _fake_dataset(override):
    from src.data.base_dataset import BaseSERDataset

    class Fake(BaseSERDataset):
        CORPUS = "fake"

        def _collect_records(self):
            return [dict(audio_path="x.wav", transcription=f"manual {i}", emotion="angry",
                         emotion_idx=0, utterance_id=f"Ses01F_impro01_F{i:03d}")
                    for i in range(3)]
    return Fake(data_path=".", sessions=[1], emotions=["angry"], transcript_override=override)


def test_transcript_override_replaces_text():
    ov = {f"Ses01F_impro01_F{i:03d}": f"asr {i}" for i in range(3)}
    ds = _fake_dataset(ov)
    assert [d["transcription"] for d in ds.data] == ["asr 0", "asr 1", "asr 2"]


def test_transcript_override_refuses_partial_map():
    with pytest.raises(KeyError, match="sin transcripcion"):
        _fake_dataset({"Ses01F_impro01_F000": "asr 0"})


def test_load_transcript_override(tmp_path):
    from src.train import load_transcript_override
    p = tmp_path / "asr.json"
    cfg = {"text": {"transcript_source": "asr", "asr_cache": str(p)}, "dataset": {}}
    p.write_text(json.dumps({"corpus": "iemocap", "complete": False, "transcripts": {}}))
    with pytest.raises(ValueError, match="incompleto"):
        load_transcript_override(cfg, LOG)
    p.write_text(json.dumps({"corpus": "meld", "complete": True, "transcripts": {}}))
    with pytest.raises(ValueError, match="corpus"):
        load_transcript_override(cfg, LOG)
    p.write_text(json.dumps({"corpus": "iemocap", "complete": True, "model": "w",
                             "wer": 0.1, "transcripts": {"a": "b"}}))
    assert load_transcript_override(cfg, LOG) == {"a": "b"}
    assert load_transcript_override({"text": {}}, LOG) is None


def test_wer():
    from scripts.transcribe_asr import wer
    assert wer([("Hello, there! [LAUGHTER]", "hello there")]) == 0
    assert wer([("a b c d", "a x c")]) == pytest.approx(0.5)


# ------------------------------------------- borrado de last_model.pt al terminar
def _logger():
    return logging.getLogger("t")


def test_last_model_is_deleted_only_when_fold_finished(tmp_path):
    import scripts.run_ablation as ra
    d = tmp_path / "run"
    d.mkdir()
    (d / "last_model.pt").write_bytes(b"x" * 10)
    (d / "best_model.pt").write_bytes(b"y" * 10)
    ra._borrar_estado_de_reanudacion(str(d), {}, "best_val", _logger())
    assert (d / "last_model.pt").exists()          # sin final_results: fold a medias
    (d / "final_results.json").write_text("{}")
    ra._borrar_estado_de_reanudacion(str(d), {}, "last", _logger())
    assert (d / "last_model.pt").exists()          # model_selection last: es el evaluado
    ra._borrar_estado_de_reanudacion(str(d), {"training": {"keep_last_model": True}},
                                     "best_val", _logger())
    assert (d / "last_model.pt").exists()          # pedido explicito de conservarlo
    ra._borrar_estado_de_reanudacion(str(d), {}, "best_val", _logger())
    assert not (d / "last_model.pt").exists()
    assert (d / "best_model.pt").exists()          # el mejor modelo nunca se toca


def test_resume_of_finished_fold_does_not_retrain(tmp_path):
    import scripts.run_ablation as ra
    model = torch.nn.Linear(3, 2)
    exp = tmp_path / "exp_fold"
    exp.mkdir()
    torch.save({"epoch": 12, "model_state_dict": model.state_dict()}, exp / "best_model.pt")
    (exp / "final_results.json").write_text(json.dumps(
        {"final_epoch": 20, "best_val_ua": 0.8, "resume_points": []}))
    (exp / "metrics.json").write_text(json.dumps(
        {str(e): {"val_unweighted_accuracy": 0.5} for e in range(1, 21)}))
    otro = torch.nn.Linear(3, 2)
    opt = torch.optim.SGD(otro.parameters(), lr=0.1)
    cfg = {"training": {"early_stopping_patience": 8}}
    start, mejor, paciencia, puntos, _ = ra._load_resume_state(
        str(tmp_path), "exp_fold", otro, opt, None, "cpu", "val_ua", "max", _logger(),
        config_actual=cfg)
    assert start == 21 and paciencia >= 8 and mejor == 0.8 and puntos == []
    assert torch.equal(otro.weight, model.weight)  # se cargaron los mejores pesos


def test_109_fusion_has_no_attention_layers():
    m, _ = _model(109)
    assert len(m.fusion.cross_attn_layers) == 0 and len(m.fusion.layer_norms) == 0
    m.eval()
    with torch.no_grad():
        out = m(*_batch())
    assert out["logits"].shape == (2, 4)


def test_random_protocol_summary_does_not_overwrite_standard():
    src = (ROOT / "scripts/run_ablation.py").read_text(encoding="utf-8")
    assert "sufijo_protocolo = '_random' if protocol == 'random' else ''" in src
