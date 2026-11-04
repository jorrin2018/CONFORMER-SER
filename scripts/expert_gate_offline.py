#!/usr/bin/env python
"""expert_gate_offline.py -- Prueba SIN ENTRENAR de fusiones con expertos congelados.

Con las predicciones de precompute_experts.py responde, antes de gastar GPU:

  1. Compuerta (04/11/2026): con expertos entrenados por separado, ¿ponderar por su
     confianza cuando discrepan le gana a promediarlos?
  2. Fusion por incertidumbre (propuesta de la asesora: ECE y Brier): ¿pesar a cada
     experto por lo bien CALIBRADO que esta le gana a promediarlos?

Todos los calibradores y pesos se ajustan en VALIDACION de cada fold y se evaluan en
TEST. Los expertos nunca vieron el test; la validacion solo la usaron para el early
stopping.

Reglas (probabilidades de los tres expertos: audio, texto, imagen)
  sin calibrar
    promedio        media de las tres distribuciones                      <- referencia
    compuerta       si los tres argmax coinciden, promedio; si no, w_e = c_e / sum(c),
                    c_e = softmax maxima del experto (la regla de la presentacion)
    seleccion       decide el experto mas seguro
    voto            mayoria; empate -> promedio
  calibradas (escalado de temperatura por experto, T ajustada en val por NLL)
    promedio_cal    media de las distribuciones calibradas
    compuerta_cal   la compuerta con confianzas calibradas
    brier           pesos fijos w_e proporcionales a 1 / Brier_e(val)
    fiabilidad      pesos por muestra w_e(x) = exactitud esperada del experto a su
                    confianza, leida de su curva de fiabilidad en val (la curva que
                    resume la ECE)
    producto        producto de expertos calibrados / prior^(E-1) (fusion bayesiana
                    con independencia condicional)
  oraculo           acierta si alguno acierta (techo, NO es un metodo)

REGLA DE DECISION, fijada antes de ver los numeros: una fusion merece una version
entrenada solo si le gana al promedio en UA de test por mas de 0.86 pp (banda de ruido
de re-ejecucion, REWRITE_NOTES sec. 7) de media pareada Y en al menos 4 de 5 folds.
Se prueban varias reglas a la vez: una que pase por poco hay que confirmarla con otra
semilla antes de escribirla.

Ademas mide la calibracion (ECE de 15 bins, Brier, NLL) de cada experto, de cada fusion
y, si precompute_experts.py se corrio con --ref-study 74, del modelo trimodal.

    python scripts\\expert_gate_offline.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_calibration():
    """src/utils/calibration.py cargado por ruta: `import src.utils` arrastra torch y
    sklearn (metrics.py), y este analisis solo necesita numpy."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "camf_calibration", ROOT / "src" / "utils" / "calibration.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cal = _load_calibration()
ReliabilityMap, apply_temperature, brier, ece, fit_temperature, nll = (
    _cal.ReliabilityMap, _cal.apply_temperature, _cal.brier, _cal.ece,
    _cal.fit_temperature, _cal.nll)

EXPERTS = ("audio", "text", "vision")
RULES = ("promedio", "compuerta", "seleccion", "voto",
         "promedio_cal", "compuerta_cal", "brier", "fiabilidad", "producto", "oraculo")
CANDIDATAS = ("compuerta", "compuerta_cal", "brier", "fiabilidad", "producto", "promedio_cal")
NOISE_PP = 0.86


def ua(y, pred, n_classes):
    return float(np.mean([(pred[y == c] == c).mean() for c in range(n_classes) if (y == c).any()]))


def wa(y, pred):
    return float((y == pred).mean())


def _gated(P):
    arg = P.argmax(-1)
    g = ~(arg == arg[0]).all(0)
    conf = P.max(-1)
    w = conf / conf.sum(0, keepdims=True)
    return np.where(g[:, None], (w[..., None] * P).sum(0), P.mean(0)), g


def _weighted(P, w):
    """P (E, N, C); w (E,) o (E, N)."""
    w = np.asarray(w, dtype=float)
    if w.ndim == 1:
        w = np.repeat(w[:, None], P.shape[1], 1)
    w = w / w.sum(0, keepdims=True)
    return (w[..., None] * P).sum(0)


def apply_rules(P: np.ndarray, y: np.ndarray, calib: dict | None = None):
    """P: (E, N, C) sin calibrar. calib (opcional, de validacion): temperaturas, pesos
    de Brier, mapas de fiabilidad y prior. Devuelve ({regla: (N, C) o None}, {regla:
    pred}, g)."""
    E, N, C = P.shape
    arg = P.argmax(-1)
    mean = P.mean(0)
    gated, g = _gated(P)
    conf = P.max(-1)
    sel_e = conf.argmax(0)
    sel = P[sel_e, np.arange(N)]
    votes = np.stack([(arg == c).sum(0) for c in range(C)], 1)
    unique_top = (votes == votes.max(1, keepdims=True)).sum(1) == 1
    vote = np.where(unique_top, votes.argmax(1), mean.argmax(1))
    oracle = np.where((arg == y[None]).any(0), y, mean.argmax(1))
    dist = {"promedio": mean, "compuerta": gated, "seleccion": sel, "voto": None,
            "oraculo": None}
    pred = {"voto": vote, "oraculo": oracle}
    if calib is not None:
        Pc = np.stack([apply_temperature(P[e], calib["T"][e]) for e in range(E)])
        dist["promedio_cal"] = Pc.mean(0)
        dist["compuerta_cal"] = _gated(Pc)[0]
        dist["brier"] = _weighted(Pc, 1.0 / np.asarray(calib["brier_val"]))
        rel = np.stack([calib["rel"][e](Pc[e]) for e in range(E)])
        dist["fiabilidad"] = _weighted(Pc, np.clip(rel, 1e-3, None))
        logp = np.log(np.clip(Pc, 1e-12, 1)).sum(0) - (E - 1) * np.log(calib["prior"])
        logp -= logp.max(1, keepdims=True)
        prod = np.exp(logp)
        dist["producto"] = prod / prod.sum(1, keepdims=True)
    for r, d in dist.items():
        if d is not None:
            pred[r] = d.argmax(1)
    return dist, pred, g


def fit_calibration(P_val: np.ndarray, y_val: np.ndarray, y_train: np.ndarray, C: int):
    T = [fit_temperature(P_val[e], y_val) for e in range(P_val.shape[0])]
    Pc = [apply_temperature(P_val[e], T[e]) for e in range(P_val.shape[0])]
    prior = np.bincount(y_train, minlength=C) / len(y_train)
    return {"T": T, "brier_val": [brier(p, y_val) for p in Pc],
            "ece_val": [ece(p, y_val) for p in Pc],
            "rel": [ReliabilityMap().fit(p, y_val) for p in Pc], "prior": prior}


def load_fold(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    uids = sorted(d["split"])
    split = np.array([d["split"][u] for u in uids])
    y = np.array([d["labels"][u] for u in uids])
    P = np.stack([np.array([d["probs"][e][u] for u in uids]) for e in EXPERTS])
    ref = None
    if "ref" in d:
        ref = (d["ref"]["study"], np.array([d["ref"]["probs"][u] for u in uids]))
    return d["meta"], split, y, P, ref


def ref_74(fold: int):
    f = ROOT / "checkpoints" / "ablation_74" / "kfold_summary_ablation_74.json"
    if not f.exists():
        return None
    for r in json.loads(f.read_text(encoding="utf-8")).get("folds", []):
        if int(r.get("test_session", -1)) == fold:
            return float(r["test_unweighted_accuracy"])
    return None


def paired(diffs):
    d = np.array(diffs, dtype=float)
    n = len(d)
    m = float(d.mean())
    s = float(d.std(ddof=1)) if n > 1 else float("nan")
    t = m / (s / math.sqrt(n)) if n > 1 and s > 0 else float("nan")
    return m, s, t, int((d > 0).sum())


def analyse_fold(path: Path):
    meta, split, y, P, ref = load_fold(path)
    k, C = int(meta["fold"]), P.shape[2]
    va, te, tr = split == "val", split == "test", split == "train"
    calib = fit_calibration(P[:, va], y[va], y[tr], C)
    fila = {"fold": k, "T": calib["T"], "brier_val": calib["brier_val"],
            "ece_val": calib["ece_val"], "ref_74": ref_74(k)}
    # Temperatura sobre la distribucion YA FUSIONADA (ajustada en val): no cambia el
    # argmax ni el UA, solo la calibracion. Calibrar cada experto por separado y luego
    # promediar deja la mezcla infraconfiada cuando discrepan; esto lo corrige.
    dist_val, _, _ = apply_rules(P[:, va], y[va], calib)
    T_fus = {r: fit_temperature(dist_val[r], y[va]) for r in ("promedio", "compuerta")}
    fila["T_fusion"] = T_fus
    for s, m in (("train", tr), ("val", va), ("test", te)):
        dist, pred, g = apply_rules(P[:, m], y[m], calib)
        fila[s] = {r: ua(y[m], pred[r], C) for r in RULES}
        fila[s]["g_rate"] = float(g.mean())
        for e, Pe in zip(EXPERTS, P[:, m]):
            fila[s][f"experto_{e}"] = ua(y[m], Pe.argmax(-1), C)
        if s == "test":
            fila["conflicto_test"] = {r: float((pred[r][g] == y[m][g]).mean()) if g.any()
                                      else float("nan") for r in RULES}
            cal = {}
            for e, Pe in zip(EXPERTS, P[:, m]):
                i = EXPERTS.index(e)
                Pce = apply_temperature(Pe, calib["T"][i])
                cal[f"{e} (crudo)"] = (ece(Pe, y[m]), brier(Pe, y[m]), nll(Pe, y[m]))
                cal[f"{e} (T={calib['T'][i]:.2f})"] = (ece(Pce, y[m]), brier(Pce, y[m]), nll(Pce, y[m]))
            for r, d in dist.items():
                if d is not None and r != "oraculo":
                    cal[r] = (ece(d, y[m]), brier(d, y[m]), nll(d, y[m]))
            for r, T in T_fus.items():
                d = apply_temperature(dist[r], T)
                cal[f"{r} (+T fusion)"] = (ece(d, y[m]), brier(d, y[m]), nll(d, y[m]))
            if ref is not None:
                Pr = ref[1][m]
                fila["ref_test_ua"] = ua(y[m], Pr.argmax(1), C)
                cal[f"estudio {ref[0]}"] = (ece(Pr, y[m]), brier(Pr, y[m]), nll(Pr, y[m]))
            fila["calibracion_test"] = cal
    return fila


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "cache" / "experts"))
    args = ap.parse_args()
    files = sorted(Path(args.dir).glob("fold_S*.json"))
    if not files:
        print(f"  No hay predicciones en {args.dir}; corre antes scripts\\precompute_experts.py")
        return 2
    filas = [analyse_fold(f) for f in files]
    n = len(filas)

    pct = lambda v: "  n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:5.2f}"  # noqa: E731
    row = lambda cells, w=9: "  " + "".join(f"{c:>{w}}" for c in cells)  # noqa: E731

    print("\n  UA de TEST por fold (%): expertos y fusiones sin calibrar")
    cab = ["fold", "audio", "texto", "imagen", "promedio", "compuerta", "seleccion", "voto",
           "oraculo", "74"]
    print(row(cab, 10))
    for r in filas:
        t = r["test"]
        print(row([f"S{r['fold']}"] + [pct(t[x]) for x in
                  ("experto_audio", "experto_text", "experto_vision", "promedio", "compuerta",
                   "seleccion", "voto", "oraculo")] + [pct(r["ref_74"])], 10))

    print("\n  UA de TEST por fold (%): fusiones guiadas por calibracion (ajustadas en val)")
    cab = ["fold", "promedio", "prom_cal", "comp_cal", "brier", "fiabilid", "producto"]
    print(row(cab, 10))
    for r in filas:
        t = r["test"]
        print(row([f"S{r['fold']}"] + [pct(t[x]) for x in
                  ("promedio", "promedio_cal", "compuerta_cal", "brier", "fiabilidad",
                   "producto")], 10))

    print("\n  Calibracion de los expertos en validacion (la que se usa para pesar)")
    print(row(["fold", "T audio", "T texto", "T imagen", "Br aud", "Br tex", "Br img",
               "ECE aud", "ECE tex", "ECE img"], 9))
    for r in filas:
        print(row([f"S{r['fold']}"] + [f"{v:.2f}" for v in r["T"]]
                  + [f"{v:.3f}" for v in r["brier_val"]] + [f"{v:.3f}" for v in r["ece_val"]], 9))

    print("\n  Calibracion en TEST, media de los folds (menor es mejor)")
    claves = list(filas[0]["calibracion_test"])
    print(f"    {'':<26}{'ECE':>8}{'Brier':>8}{'NLL':>8}")
    for c in claves:
        vals = np.array([r["calibracion_test"][c] for r in filas if c in r["calibracion_test"]])
        if len(vals):
            e, b, l_ = vals.mean(0)
            print(f"    {c:<26}{e:8.3f}{b:8.3f}{l_:8.3f}")

    print("    Ojo: con el protocolo standard la validacion sale de las MISMAS sesiones (y")
    print("    hablantes) que el train, asi que la temperatura ajustada en val subestima la")
    print("    sobreconfianza ante hablantes nuevos: comparar la ECE de val con la de test.")

    print("\n  Exactitud en las muestras de test CON conflicto (g = 1)")
    print(row(["fold", "promedio", "compuerta", "comp_cal", "brier", "fiabilid", "oraculo"], 10))
    for r in filas:
        c = r["conflicto_test"]
        print(row([f"S{r['fold']}"] + [pct(c[x]) for x in
                  ("promedio", "compuerta", "compuerta_cal", "brier", "fiabilidad", "oraculo")], 10))

    print("\n  Sobreconfianza en train (los expertos se entrenaron con esas utterances)")
    print(row(["fold", "g train", "g val", "g test", "audio tr-te", "texto tr-te"], 14))
    gaps = []
    for r in filas:
        gaps.append(r["test"]["g_rate"] - r["train"]["g_rate"])
        print(row([f"S{r['fold']}", pct(r["train"]["g_rate"]), pct(r["val"]["g_rate"]),
                   pct(r["test"]["g_rate"]),
                   f"{100 * (r['train']['experto_audio'] - r['test']['experto_audio']):+.2f}",
                   f"{100 * (r['train']['experto_text'] - r['test']['experto_text']):+.2f}"], 14))

    print("\n  Pareado por fold contra el PROMEDIO sin calibrar, UA de test (pp)")
    comps = {}
    for a in CANDIDATAS + ("seleccion", "voto"):
        m, s, t, n_pos = paired([100 * (r["test"][a] - r["test"]["promedio"]) for r in filas])
        comps[f"{a} - promedio"] = dict(mean=m, sd=s, t=t, folds_pos=n_pos, n=n)
        print(f"    {a + ' - promedio':<26} {m:+6.2f} ± {s:4.2f}   t({n - 1}) = {t:+6.2f}   "
              f"{n_pos}/{n} folds a favor")
    con_74 = [r for r in filas if r["ref_74"] is not None]
    if con_74:
        m, s, t, n_pos = paired([100 * (r["test"]["promedio"] - r["ref_74"]) for r in con_74])
        comps["promedio expertos - 74"] = dict(mean=m, sd=s, t=t, folds_pos=n_pos, n=len(con_74))
        print(f"    {'promedio expertos - 74':<26} {m:+6.2f} ± {s:4.2f}   t({len(con_74) - 1}) = "
              f"{t:+6.2f}   {n_pos}/{len(con_74)} folds a favor")

    print("\n  VEREDICTO (regla fijada antes de ver los numeros: > 0.86 pp y >= 4/5 folds)")
    ganan = [a for a in CANDIDATAS
             if comps[f"{a} - promedio"]["mean"] > NOISE_PP
             and comps[f"{a} - promedio"]["folds_pos"] >= min(4, n)]
    if ganan:
        print(f"    Merecen version entrenada: {', '.join(ganan)}")
    else:
        print("    Ninguna fusion le gana al promedio de los expertos: no se entrena ninguna.")
    print(f"    Tasa de g: train {100 * np.mean([r['train']['g_rate'] for r in filas]):.1f} % "
          f"vs test {100 * np.mean([r['test']['g_rate'] for r in filas]):.1f} % "
          f"(diferencia {100 * float(np.mean(gaps)):+.1f} pp).")
    if float(np.mean(gaps)) > 0.10:
        print("    AVISO: en train la compuerta dispara mucho menos que en test. Una version "
              "entrenada necesitaria predicciones de train por cross-fitting.")
    if n < 5:
        print(f"    Solo {n} fold(s): el veredicto es provisional.")

    out = Path(args.dir) / "offline_summary.json"
    for r in filas:
        r["calibracion_test"] = {k: dict(zip(("ece", "brier", "nll"), v))
                                 for k, v in r["calibracion_test"].items()}
    out.write_text(json.dumps({"folds": filas, "comparaciones": comps, "ganan": ganan,
                               "g_rate_gap_test_minus_train": float(np.mean(gaps))},
                              indent=2), encoding="utf-8")
    print(f"\n  Resumen en {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
