"""Tests de la compuerta con expertos congelados (precompute_experts / expert_gate_offline).

Sin GPU, sin IEMOCAP, sin red.

    python -m pytest tests/test_expert_gate.py -q
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.expert_gate_offline as off  # noqa: E402
import scripts.precompute_experts as pre  # noqa: E402
from tests.test_deck_variants import _FakeCorpus, _write_wavs  # noqa: E402
from tests.tiny_encoders import LOCAL_TOKENIZER, tiny_encoders  # noqa: E402


# ------------------------------------------------------------- reglas offline
def _P(*rows):
    """rows: por experto, lista de distribuciones (N, C)."""
    return np.array(rows, dtype=float)


def test_rules_consensus_is_average():
    P = _P([[.7, .1, .1, .1]], [[.6, .2, .1, .1]], [[.4, .3, .2, .1]])
    _, preds, g = off.apply_rules(P, np.array([0]))
    assert not g[0] and preds["compuerta"][0] == preds["promedio"][0] == 0


def test_rules_conflict_weights_by_confidence():
    # audio seguro de 1 (0.7); texto y vision tibios en 0 (0.5) -> el promedio dice 0
    # (1/3 vs 0.30), la compuerta (0.29 vs 0.35) y la seleccion dicen 1
    P = _P([[0., .7, .15, .15]], [[.5, .1, .2, .2]], [[.5, .1, .2, .2]])
    _, preds, g = off.apply_rules(P, np.array([1]))
    assert g[0]
    assert preds["promedio"][0] == 0
    assert preds["seleccion"][0] == 1 and preds["compuerta"][0] == 1
    assert preds["voto"][0] == 0 and preds["oraculo"][0] == 1


def test_vote_tie_falls_back_to_average():
    P = _P([[.9, .1, 0, 0]], [[.1, .8, .1, 0]], [[.1, .1, .7, .1]])
    _, preds, _ = off.apply_rules(P, np.array([0]))
    assert preds["voto"][0] == preds["promedio"][0]


def test_paired_stats():
    m, s, t, n = off.paired([1.0, 2.0, 3.0])
    assert m == 2.0 and s == pytest.approx(1.0) and n == 3 and t == pytest.approx(2 * 3 ** .5)


def test_offline_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    for k in range(1, 6):
        uids = [f"Ses0{k}F_x_{i:03d}" for i in range(120)]
        y = rng.integers(0, 4, len(uids))
        split = ["train"] * 80 + ["val"] * 10 + ["test"] * 30
        probs = {}
        for e in off.EXPERTS:
            noisy = np.eye(4)[y] * 2 + rng.random((len(uids), 4))
            p = noisy / noisy.sum(1, keepdims=True)
            probs[e] = {u: pi.tolist() for u, pi in zip(uids, p)}
        (tmp_path / f"fold_S{k}.json").write_text(json.dumps({
            "meta": {"fold": k}, "split": dict(zip(uids, split)),
            "labels": {u: int(v) for u, v in zip(uids, y)}, "probs": probs}))
    sys_argv = sys.argv
    sys.argv = ["x", "--dir", str(tmp_path)]
    try:
        assert off.main() == 0
    finally:
        sys.argv = sys_argv
    summ = json.loads((tmp_path / "offline_summary.json").read_text())
    assert len(summ["folds"]) == 5 and "compuerta - promedio" in summ["comparaciones"]
    assert "brier - promedio" in summ["comparaciones"] and isinstance(summ["ganan"], list)
    cal = summ["folds"][0]["calibracion_test"]
    assert {"promedio", "brier", "fiabilidad", "audio (crudo)"} <= set(cal)


# ------------------------------------------------------------- calibracion
from src.utils.calibration import (  # noqa: E402
    ReliabilityMap, apply_temperature, brier, ece, fit_temperature)


def _calibrated_sample(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, 4)) * 2
    P = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(4, p=p) for p in P])     # y ~ P: calibrado por construccion
    return P, y


def test_ece_and_brier_basic():
    P, y = _calibrated_sample()
    assert ece(P, y) < 0.02
    onehot = np.eye(4)[[0, 1]]
    assert brier(onehot, np.array([0, 1])) == 0.0
    assert brier(np.full((1, 4), .25), np.array([0])) == pytest.approx(.75)


def test_temperature_undoes_overconfidence():
    P, y = _calibrated_sample()
    sharp = apply_temperature(P, 1 / 3)                # sobreconfiado
    assert ece(sharp, y) > 0.1
    T = fit_temperature(sharp, y)
    assert T == pytest.approx(3.0, rel=0.15)
    assert ece(apply_temperature(sharp, T), y) < 0.03


def test_reliability_map_tracks_accuracy():
    P, y = _calibrated_sample()
    sharp = apply_temperature(P, 1 / 3)
    rel = ReliabilityMap().fit(sharp, y)
    est = rel(sharp)
    hit = (sharp.argmax(1) == y)
    assert abs(est.mean() - hit.mean()) < 0.02 and est.max() <= 1.0


# -------------------------------------------------------- experto de imagen
def test_fit_vision_expert_learns_separable_features():
    rng = np.random.default_rng(1)
    uids = [f"u{i}" for i in range(400)]
    y = rng.integers(0, 4, 400)
    feats = {u: np.eye(4)[c].repeat(8) * 3 + rng.normal(size=32) for u, c in zip(uids, y)}
    split = {u: ("train" if i < 300 else "val" if i < 350 else "test") for i, u in enumerate(uids)}
    labels = dict(zip(uids, y.tolist()))
    probs, C = pre.fit_vision_expert(feats, labels, split, 4)
    assert set(probs) == set(uids)
    assert pre.ua_of(probs, labels, split, "test", 4) > 0.9
    assert np.allclose([sum(p) for p in probs.values()], 1.0)


# ------------------------------------------------ experto de audio/texto real
def _fake_run(tmp_path, study, modality):
    """Una corrida falsa: config.json + best_model.pt de un modelo diminuto + final_results."""
    from scripts.run_ablation import create_ablation_model
    cfg = {
        "dataset": {"emotions": ["angry", "happy", "sad", "neutral"], "sample_rate": 16000},
        "audio": {"max_length": 2.0},
        "text": {"model_name": LOCAL_TOKENIZER, "max_length": 16},
        "model": {"modality": modality, "num_classes": 4, "hidden_dim": 64,
                  "classification_head_type": "simple",
                  "audio_unfreeze_last_n_layers": 6, "text_unfreeze_last_n_layers": 4},
        "training": {}, "compute": {}, "seed": 42,
    }
    d = tmp_path / f"run_{study}"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    torch.manual_seed(study)
    with tiny_encoders():
        m = create_ablation_model(study, cfg, "cpu", logging.getLogger("t"))
    torch.save({"model_state_dict": m.state_dict()}, d / "best_model.pt")
    return d, cfg


def test_run_expert_reproduces_and_checks_test_ua(tmp_path, monkeypatch):
    from src.data.preprocessing import AudioPreprocessor, TextPreprocessor
    paths = _write_wavs(tmp_path, secs=(1.0, 1.5, 0.8, 2.0, 1.2, 0.9))
    _FakeCorpus.WAVS = paths

    def fake_split(cfg, log):
        out = {}
        for name, sl in (("train", slice(0, 2)), ("val", slice(2, 4)), ("test", slice(4, 6))):
            ds = _FakeCorpus(data_path=".", sessions=[1], emotions=["angry"], sample_rate=16000,
                             audio_preprocessor=AudioPreprocessor(sample_rate=16000),
                             text_preprocessor=TextPreprocessor(model_name=LOCAL_TOKENIZER,
                                                                max_length=16),
                             max_audio_length=2.0, training=False)
            ds.data = ds.data[sl]
            out[name] = ds
        return out

    run_dir, cfg = _fake_run(tmp_path, 71, "audio")
    monkeypatch.setattr(pre, "fold_run_dir", lambda study, fold, outer=None: run_dir)
    monkeypatch.setattr(pre, "split_datasets", fake_split)

    # 1) calcular el UA real de este modelo y escribirlo como final_results
    (run_dir / "final_results.json").write_text(json.dumps(
        {"test_metrics": {"unweighted_accuracy": 0.5}}))
    with tiny_encoders():
        with pytest.raises(SystemExit, match="no coincide"):
            pre.run_expert(71, 1, "cpu", 2, logging.getLogger("t"))   # UA de una clase: 0 o 1, nunca 0.5
    # 2) con el UA correcto en final_results, pasa y devuelve 6 utterances con reparto
    with tiny_encoders():
        import scripts.run_ablation as ra
        m = ra.create_ablation_model(71, json.loads((run_dir / "config.json").read_text()),
                                     "cpu", logging.getLogger("t"))
        m.load_state_dict(torch.load(run_dir / "best_model.pt")["model_state_dict"])
        probs, labels, split = pre.predict(m, fake_split(cfg, None), "cpu", 2)
        model_ua = pre.ua_of(probs, labels, split, "test", 4)
    (run_dir / "final_results.json").write_text(json.dumps(
        {"test_metrics": {"unweighted_accuracy": model_ua}}))
    with tiny_encoders():
        r = pre.run_expert(71, 1, "cpu", 2, logging.getLogger("t"))
    assert len(r["probs"]) == 6 and sorted(set(r["split"].values())) == ["test", "train", "val"]
    assert all(abs(sum(p) - 1) < 1e-5 for p in r["probs"].values())


def test_vit_cls_features_uses_model_mel_pipeline(tmp_path, monkeypatch):
    """Mismo log-mel que el modelo: misma imagen que _waveform_to_vit_image del CAMF."""
    import transformers
    from src.data.preprocessing import AudioPreprocessor
    from tests.tiny_encoders import tiny_vit
    paths = _write_wavs(tmp_path, secs=(1.0, 1.5))
    _FakeCorpus.WAVS = paths
    run = tmp_path / "run74"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({
        "dataset": {"sample_rate": 16000}, "audio": {}, "text": {},
        "model": {"mel_n_mels": 80, "mel_n_fft": 400, "mel_hop_length": 160,
                  "mel_win_length": 400, "vision_image_size": 224}}))

    def fake_split(cfg, log):
        ds = _FakeCorpus(data_path=".", sessions=[1], emotions=["angry"], sample_rate=16000,
                         audio_preprocessor=AudioPreprocessor(sample_rate=16000),
                         max_audio_length=2.0, training=False)
        return {"train": ds}

    torch.manual_seed(0)
    vit = tiny_vit()
    monkeypatch.setattr(pre, "fold_run_dir", lambda s, f, outer=None: run)
    monkeypatch.setattr(pre, "split_datasets", fake_split)
    monkeypatch.setattr(transformers.ViTModel, "from_pretrained", classmethod(lambda cls, *a, **k: vit))
    feats = pre.vit_cls_features(74, "cpu", 2, logging.getLogger("t"))
    assert len(feats) == 2 and next(iter(feats.values())).shape == (32,)


# ------------------------------------------------------------- cross-fitting
def test_outer_test_excludes_session_everywhere(tmp_path, monkeypatch):
    import scripts.run_ablation as ra
    vistos = []

    def fake_run(study_number, config, device, experiment_name, seed_override, resume):
        vistos.append(json.loads(json.dumps(config)))
        return {"best_val_ua": 0.7, "test_unweighted_accuracy": 0.6,
                "test_weighted_accuracy": 0.6, "final_epoch": 1}

    monkeypatch.setattr(ra, "_run_ablation_with_config", fake_run)
    monkeypatch.setattr(ra, "load_base_config", lambda: {})
    monkeypatch.chdir(tmp_path)
    (tmp_path / "checkpoints/ablation_95/outer_S1").mkdir(parents=True)   # lo crean los folds reales
    cfg = str(ROOT / "config/ablation/ablation_95_expert_audio_crossfit.yaml")
    ra.run_ablation_study_kfold(95, cfg, device="cpu", protocol="standard", outer_test=1)
    assert len(vistos) == 4
    assert sorted(c["dataset"]["test_sessions"][0] for c in vistos) == [2, 3, 4, 5]
    for c in vistos:
        d = c["dataset"]
        assert 1 not in d["train_sessions"] + d["val_sessions"] + d["test_sessions"]
        assert len(d["train_sessions"]) == 3 and d["val_fraction"] == 0.12
        assert "outer_S1" in c["logging"]["checkpoint_dir"]
    summ = tmp_path / "checkpoints/ablation_95/outer_S1/kfold_summary_ablation_95.json"
    assert summ.exists()
    assert all(f["outer_test"] == 1 for f in json.loads(summ.read_text())["folds"])


def test_crossfit_fold_fills_every_train_utterance(monkeypatch):
    import argparse
    rng = np.random.default_rng(0)
    uids, split, labels = [], {}, {}
    for s in (2, 3, 4, 5):
        for i in range(30):
            u = f"Ses0{s}F_x_F{i:03d}"
            uids.append(u)
            split[u] = "val" if i < 4 else "train"
            labels[u] = int(rng.integers(0, 4))
    for i in range(20):
        u = f"Ses01F_x_F{i:03d}"
        uids.append(u); split[u] = "test"; labels[u] = int(rng.integers(0, 4))
    probs = {e: {u: [0.25] * 4 for u in uids} for e in ("audio", "text", "vision")}
    payload = {"meta": {"emotions": ["a", "h", "s", "n"], "vision": {"C": 0.1}},
               "split": split, "labels": labels, "probs": probs}
    llamadas = []

    def fake_run_expert(study, fold, device, bs, log, outer=None, only=None):
        llamadas.append((study, fold, outer, only))
        mine = [u for u in uids if int(u[3:5]) == fold]
        return {"probs": {u: [0.7, 0.1, 0.1, 0.1] for u in mine},
                "labels": {u: labels[u] for u in mine}, "run_dir": f"r{study}_{fold}"}

    monkeypatch.setattr(pre, "run_expert", fake_run_expert)
    feats = {u: rng.normal(size=8) for u in uids}
    args = argparse.Namespace(audio_crossfit_study=95, text_crossfit_study=96,
                              device="cpu", batch_size=2)
    out = pre.crossfit_fold(payload, 1, args, feats, None)
    train = [u for u in uids if split[u] == "train"]
    for e in ("audio", "text", "vision"):
        assert set(out["probs_oof"][e]) == set(train)
    assert sorted({(c[0], c[1]) for c in llamadas}) == [(95, s) for s in (2, 3, 4, 5)] + [(96, s) for s in (2, 3, 4, 5)]
    assert all(c[2] == 1 and c[3] == ("test",) for c in llamadas)


def test_offline_uses_oof_rows_for_train(tmp_path):
    uids = [f"Ses02F_x_{i:03d}" for i in range(8)]
    split = {u: ("train" if i < 4 else "val" if i < 6 else "test") for i, u in enumerate(uids)}
    labels = {u: 0 for u in uids}
    full = {e: {u: [0.9, 0.05, 0.03, 0.02] for u in uids} for e in off.EXPERTS}
    oof = {e: {u: [0.1, 0.7, 0.1, 0.1] for u in uids if split[u] == "train"} for e in off.EXPERTS}
    f = tmp_path / "fold_S1.json"
    f.write_text(json.dumps({"meta": {"fold": 1}, "split": split, "labels": labels,
                             "probs": full, "probs_oof": oof}))
    meta, sp, y, P, _ = off.load_fold(f)
    tr = sp == "train"
    assert (P[:, tr].argmax(-1) == 1).all() and (P[:, ~tr].argmax(-1) == 0).all()
    assert meta["_oof"]
