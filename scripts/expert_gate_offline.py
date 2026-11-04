#!/usr/bin/env python
"""expert_gate_offline.py -- Prueba SIN ENTRENAR de la compuerta con expertos congelados.

Antes de gastar ~30 h de GPU en la etapa 2, responde con las predicciones de
precompute_experts.py la unica pregunta que decide si merece la pena:

    con expertos entrenados por separado, ¿ponderar por su confianza cuando
    discrepan le gana a promediarlos?

En el 75/76 (clasificadores lineales entrenados junto con la fusion) la respuesta fue
no: elegir por confianza 69.0 % frente a promediar 70.5 % en las muestras de conflicto.
Aqui los expertos son modelos unimodales completos y no co-adaptan con la fusion.

Reglas que se comparan, por fold, sobre test (y val):
  promedio        media de las tres distribuciones
  compuerta       la regla de la presentacion con confianzas de experto: si los tres
                  argmax coinciden (g = 0) promedio; si no (g = 1) w_e = c_e / sum(c),
                  c_e = max softmax del experto
  ponderado       w_e = c_e / sum(c) siempre, sin compuerta
  seleccion       el experto mas seguro decide
  voto            mayoria; empate -> promedio
  oraculo         acierta si alguno acierta (techo, NO es un metodo)

REGLA DE DECISION, fijada antes de ver los numeros: se corre la etapa 2 solo si la
compuerta le gana al promedio en UA de test por mas de 0.86 pp (banda de ruido de
re-ejecucion, REWRITE_NOTES sec. 7) de media pareada Y en al menos 4 de 5 folds.

Ademas mide la SOBRECONFIANZA EN TRAIN: los expertos se entrenaron con esas mismas
utterances, asi que en train discrepan menos que en test. Si la tasa de g en train y en
test difiere mucho, la etapa 2 entrenaria con una compuerta que casi no dispara y se
evaluaria con otra que dispara a menudo; entonces habria que hacer cross-fitting.

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
EXPERTS = ("audio", "text", "vision")
RULES = ("promedio", "compuerta", "ponderado", "seleccion", "voto", "oraculo")
NOISE_PP = 0.86


def ua(y, pred, n_classes):
    return float(np.mean([(pred[y == c] == c).mean() for c in range(n_classes) if (y == c).any()]))


def wa(y, pred):
    return float((y == pred).mean())


def apply_rules(P: np.ndarray, y: np.ndarray):
    """P: (E, N, C). Devuelve {regla: pred (N,)} y g (N,)."""
    arg = P.argmax(-1)                                   # (E, N)
    g = ~(arg == arg[0]).all(0)                          # True si no coinciden los tres
    conf = P.max(-1)                                     # (E, N)
    w = conf / conf.sum(0, keepdims=True)
    mean = P.mean(0)
    weighted = (w[..., None] * P).sum(0)
    gated = np.where(g[:, None], weighted, mean)
    sel = arg[conf.argmax(0), np.arange(P.shape[1])]
    votes = np.stack([(arg == c).sum(0) for c in range(P.shape[2])], 1)   # (N, C)
    top = votes.max(1)
    unique_top = (votes == top[:, None]).sum(1) == 1
    vote = np.where(unique_top, votes.argmax(1), mean.argmax(1))
    any_ok = (arg == y[None]).any(0)
    oracle = np.where(any_ok, y, mean.argmax(1))
    return {"promedio": mean.argmax(1), "compuerta": gated.argmax(1),
            "ponderado": weighted.argmax(1), "seleccion": sel, "voto": vote,
            "oraculo": oracle}, g


def load_fold(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    uids = sorted(d["split"])
    split = np.array([d["split"][u] for u in uids])
    y = np.array([d["labels"][u] for u in uids])
    P = np.stack([np.array([d["probs"][e][u] for u in uids]) for e in EXPERTS])
    return d["meta"], split, y, P


def ref_74(fold: int):
    f = ROOT / "checkpoints" / "ablation_74" / "kfold_summary_ablation_74.json"
    if not f.exists():
        return None
    for r in json.loads(f.read_text(encoding="utf-8")).get("folds", []):
        if int(r.get("test_session", -1)) == fold:
            return float(r["test_unweighted_accuracy"])
    return None


def paired(diffs):
    d = np.array(diffs)
    n = len(d)
    m, s = float(d.mean()), float(d.std(ddof=1)) if n > 1 else float("nan")
    t = m / (s / math.sqrt(n)) if n > 1 and s > 0 else float("nan")
    return m, s, t, int((d > 0).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "cache" / "experts"))
    args = ap.parse_args()
    files = sorted(Path(args.dir).glob("fold_S*.json"))
    if not files:
        print(f"  No hay predicciones en {args.dir}; corre antes scripts\\precompute_experts.py")
        return 2

    filas, resumen = [], {"folds": []}
    for f in files:
        meta, split, y, P = load_fold(f)
        k, C = int(meta["fold"]), P.shape[2]
        fila = {"fold": k}
        for s in ("train", "val", "test"):
            m = split == s
            preds, g = apply_rules(P[:, m], y[m])
            fila[s] = {r: ua(y[m], preds[r], C) for r in RULES}
            fila[s]["wa_promedio"] = wa(y[m], preds["promedio"])
            fila[s]["g_rate"] = float(g.mean())
            for e, Pe in zip(EXPERTS, P[:, m]):
                fila[s][f"experto_{e}"] = ua(y[m], Pe.argmax(-1), C)
            if s == "test":
                cm = g
                fila["conflicto_test"] = {
                    r: float((preds[r][cm] == y[m][cm]).mean()) if cm.any() else float("nan")
                    for r in RULES}
        fila["ref_74"] = ref_74(k)
        filas.append(fila)

    pct = lambda v: "  n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:5.2f}"  # noqa: E731
    print("\n  UA de TEST por fold (%)")
    cab = ["fold", "audio", "texto", "vision"] + list(RULES) + ["74"]
    print("  " + "".join(f"{c:>10}" for c in cab))
    for r in filas:
        t = r["test"]
        vals = [t["experto_audio"], t["experto_text"], t["experto_vision"]] + [t[x] for x in RULES]
        print("  " + f"{'S' + str(r['fold']):>10}" + "".join(f"{pct(v):>10}" for v in vals)
              + f"{pct(r['ref_74']):>10}")

    print("\n  Exactitud en las muestras de test CON conflicto (g = 1)")
    print("  " + "".join(f"{c:>10}" for c in ["fold"] + list(RULES)))
    for r in filas:
        print("  " + f"{'S' + str(r['fold']):>10}"
              + "".join(f"{pct(r['conflicto_test'][x]):>10}" for x in RULES))

    print("\n  Sobreconfianza en train (los expertos se entrenaron con esas utterances)")
    print("  " + "".join(f"{c:>14}" for c in ("fold", "g train", "g val", "g test",
                                             "audio tr-te", "texto tr-te")))
    gaps = []
    for r in filas:
        gap = r["test"]["g_rate"] - r["train"]["g_rate"]
        gaps.append(gap)
        print("  " + f"{'S' + str(r['fold']):>14}"
              + "".join(f"{pct(v):>14}" for v in (r["train"]["g_rate"], r["val"]["g_rate"],
                                                 r["test"]["g_rate"]))
              + f"{100 * (r['train']['experto_audio'] - r['test']['experto_audio']):>+14.2f}"
              + f"{100 * (r['train']['experto_text'] - r['test']['experto_text']):>+14.2f}")

    print("\n  Pareado por fold, UA de test (pp)")
    comps = {}
    for nombre, a, b in (("compuerta - promedio", "compuerta", "promedio"),
                         ("ponderado - promedio", "ponderado", "promedio"),
                         ("seleccion - promedio", "seleccion", "promedio"),
                         ("voto - promedio", "voto", "promedio")):
        m, s, t, n_pos = paired([100 * (r["test"][a] - r["test"][b]) for r in filas])
        comps[nombre] = dict(mean=m, sd=s, t=t, folds_pos=n_pos, n=len(filas))
        print(f"    {nombre:<24} {m:+6.2f} ± {s:4.2f}   t({len(filas) - 1}) = {t:+5.2f}   "
              f"{n_pos}/{len(filas)} folds a favor")
    con_74 = [r for r in filas if r["ref_74"] is not None]
    if con_74:
        m, s, t, n_pos = paired([100 * (r["test"]["promedio"] - r["ref_74"]) for r in con_74])
        comps["promedio expertos - 74"] = dict(mean=m, sd=s, t=t, folds_pos=n_pos, n=len(con_74))
        print(f"    {'promedio expertos - 74':<24} {m:+6.2f} ± {s:4.2f}   t({len(con_74) - 1}) = {t:+5.2f}   "
              f"{n_pos}/{len(con_74)} folds a favor")

    c = comps["compuerta - promedio"]
    go = c["mean"] > NOISE_PP and c["folds_pos"] >= min(4, len(filas))
    gap_g = float(np.mean(gaps))
    print("\n  VEREDICTO (regla fijada antes de ver los numeros)")
    if go:
        print(f"    SE CORRE LA ETAPA 2: la compuerta gana al promedio por {c['mean']:+.2f} pp "
              f"(> {NOISE_PP}) en {c['folds_pos']}/{c['n']} folds.")
    else:
        print(f"    NO se corre la etapa 2: compuerta - promedio = {c['mean']:+.2f} pp en "
              f"{c['folds_pos']}/{c['n']} folds (hace falta > {NOISE_PP} y >= 4/5).")
    print(f"    Tasa de g: train {100 * np.mean([r['train']['g_rate'] for r in filas]):.1f} % "
          f"vs test {100 * np.mean([r['test']['g_rate'] for r in filas]):.1f} % "
          f"(diferencia {100 * gap_g:+.1f} pp).")
    if gap_g > 0.10:
        print("    AVISO: en train la compuerta dispara mucho menos que en test. Si se corre la "
              "etapa 2, hacerlo con predicciones de train por cross-fitting.")
    if len(filas) < 5:
        print(f"    Solo {len(filas)} fold(s): el veredicto es provisional.")

    resumen["folds"] = filas
    resumen["comparaciones"] = comps
    resumen["go_stage2"] = bool(go)
    resumen["g_rate_gap_test_minus_train"] = gap_g
    out = Path(args.dir) / "offline_summary.json"
    out.write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    print(f"\n  Resumen en {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
