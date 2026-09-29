#!/usr/bin/env python
"""
rebuild_kfold_summary.py -- Reconstruye el resumen k-fold desde los final_results.json
de cada fold.

Entrenar cinco folds cuesta dias; si la agregacion posterior falla, los resultados
siguen en disco y no hay que repetir nada. Cada fold guarda su propio
final_results.json y su config.json, que es de donde salen el protocolo y el reparto
de sesiones.

    python scripts/rebuild_kfold_summary.py --study 66 --experiment std
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _val_label(f: dict) -> str:
    """Como se eligio la validacion de ese fold, para la tabla."""
    if f.get("val_speaker"):
        return str(f["val_speaker"])
    if f.get("val_fraction"):
        return "{:.0f}% estrat.".format(100 * f["val_fraction"])
    return "ses {}".format(f.get("val_session"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", type=int, required=True)
    ap.add_argument("--experiment", default=None,
                    help="prefijo del nombre de experimento, p.ej. 'std'")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = ROOT / "checkpoints" / f"ablation_{args.study}"
    if not root.is_dir():
        print(f"No existe {root}")
        return 1

    folds = []
    for fr in sorted(root.glob("fold_S*_test")):
        cands = [d for d in sorted(fr.iterdir()) if (d / "final_results.json").exists()]
        if args.experiment:
            cands = [d for d in cands if d.name.startswith(args.experiment)]
        if not cands:
            continue
        run = max(cands, key=lambda d: (d / "final_results.json").stat().st_mtime)
        res = json.loads((run / "final_results.json").read_text(encoding="utf-8"))
        tm = res.get("test_metrics", {})
        cfg = {}
        if (run / "config.json").exists():
            cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        ds = cfg.get("dataset", {})
        folds.append({
            "best_val_ua": res.get("best_val_ua"),
            "test_unweighted_accuracy": tm.get("unweighted_accuracy"),
            "test_weighted_accuracy": tm.get("weighted_accuracy"),
            "final_epoch": res.get("final_epoch"),
            "results_dir": str(run),
            "test_session": (ds.get("test_sessions") or [None])[0],
            "val_speaker": ds.get("val_speaker"),
            "val_fraction": ds.get("val_fraction"),
            "val_session": (None if (ds.get("val_speaker") or ds.get("val_fraction"))
                            else (ds.get("val_sessions") or [None])[0]),
            # El protocolo se deduce del reparto que quedo escrito en el config del
            # fold: 'standard' usa una fraccion estratificada, 'speaker' un hablante,
            # 'legacy' una sesion entera de validacion. Deducirlo solo de val_speaker
            # etiquetaba como 'legacy' todas las corridas estandar.
            "protocol": ("speaker" if ds.get("val_speaker")
                         else "standard" if ds.get("val_fraction")
                         else "legacy"),
            "train_sessions": ds.get("train_sessions"),
        })

    if not folds:
        print("No se encontro ningun fold con final_results.json")
        return 1

    def agg(key):
        v = [f[key] for f in folds if f[key] is not None]
        return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0) if v else (None, None)

    protos = {f["protocol"] for f in folds}
    summary = {
        "folds": folds,
        "protocol": protos.pop() if len(protos) == 1 else sorted(protos),
        "num_folds": len(folds),
        "rebuilt_from_fold_results": True,
    }
    for key, name in (("best_val_ua", "best_val_ua"),
                      ("test_unweighted_accuracy", "test_unweighted_accuracy"),
                      ("test_weighted_accuracy", "test_weighted_accuracy")):
        m, sd = agg(key)
        summary[f"mean_{name}"] = m
        summary[f"std_{name}"] = sd

    out = Path(args.out) if args.out else root / f"kfold_summary_ablation_{args.study:02d}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n  estudio {args.study}   protocolo: {summary['protocol']}   "
          f"folds: {len(folds)}\n")
    print(f"  {'test':<6}{'train ses':<16}{'validacion':<12}{'val UA':>8}{'test UA':>9}{'test WA':>9}{'ep':>5}")
    print("  " + "-" * 66)
    for f in folds:
        print(f"  S{f['test_session']:<5}{str(f['train_sessions']):<16}"
              f"{_val_label(f):<12}"
              f"{100*f['best_val_ua']:>8.2f}{100*f['test_unweighted_accuracy']:>9.2f}"
              f"{100*f['test_weighted_accuracy']:>9.2f}{f['final_epoch']:>5}")
    print("  " + "-" * 66)
    print(f"  {'media':<34}{100*summary['mean_best_val_ua']:>8.2f}"
          f"{100*summary['mean_test_unweighted_accuracy']:>9.2f}"
          f"{100*summary['mean_test_weighted_accuracy']:>9.2f}")
    print(f"  {'sigma':<34}{100*summary['std_best_val_ua']:>8.2f}"
          f"{100*summary['std_test_unweighted_accuracy']:>9.2f}"
          f"{100*summary['std_test_weighted_accuracy']:>9.2f}")
    print(f"\n  guardado en {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
