"""
Agrega final_results.json bajo checkpoints/ para analisis de ranking (WA, UA, gap val-test en UA).
Ejecutar desde la raiz del proyecto: python scripts/aggregate_ablation_results.py
Genera ademas docs/RANKING_Y_RESULTADOS.md si se pasa --write-docs
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone


def study_from_path(rel: str) -> int | None:
    s = rel.replace("\\", "/")
    m = re.search(r"post_fix_conflict_chain_study_(\d+)", s)
    if m:
        return int(m.group(1))
    m = re.search(r"post_fix_baseline_study_(\d+)", s)
    if m:
        return int(m.group(1))
    m = re.search(r"post_fix_glu_study_(\d+)", s)
    if m:
        return int(m.group(1))
    m = re.search(r"ablation_(\d+)_kfold_20", s)
    if m:
        return int(m.group(1))
    if re.search(r"ablation_20_kfold_20", s):
        return 20
    if re.search(r"ablation_12_loso_", s):
        return 12
    m = re.search(r"ablation_(\d+)_\d{8}_", s)
    if m:
        return int(m.group(1))
    m = re.search(r"ablation_(\d+)/", s)
    if m:
        return int(m.group(1))
    m = re.search(r"ablation_(\d+)\\", s)
    if m:
        return int(m.group(1))
    return None


def build_rows(root: Path):
    rows: list[
        tuple[int, str, float | None, float | None, float | None, float | None, float | None]
    ] = []
    for p in root.rglob("final_results.json"):
        rel = p.as_posix()
        st = study_from_path(rel)
        if st is None:
            m = re.search(r"ablation_(\d+)", rel)
            st = int(m.group(1)) if m else None
        if st is None:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        tm = data.get("test_metrics", {})
        wa = tm.get("weighted_accuracy")
        ua = tm.get("unweighted_accuracy")
        f1m = tm.get("f1_macro")
        bvu = data.get("best_val_ua")
        gap = (bvu - ua) if bvu is not None and ua is not None else None
        rows.append((st, rel, wa, ua, f1m, bvu, gap))
    return rows


def aggregate(out: list[dict], rows: list) -> None:
    by_st: dict[int, list] = defaultdict(list)
    for r in rows:
        by_st[r[0]].append(r)

    for st in sorted(by_st):
        runs = by_st[st]
        is_kfold = len(runs) > 1 and any("fold_" in r[1] for r in runs)
        if is_kfold:
            was = [x[2] for x in runs if x[2] is not None]
            uas = [x[3] for x in runs if x[3] is not None]
            f1s = [x[4] for x in runs if x[4] is not None]
            gaps = [x[6] for x in runs if x[6] is not None]
            out.append(
                {
                    "study": st,
                    "n_runs": len(runs),
                    "wa": sum(was) / len(was) if was else None,
                    "ua": sum(uas) / len(uas) if uas else None,
                    "f1_macro": sum(f1s) / len(f1s) if f1s else None,
                    "best_val_ua": None,
                    "gap_ua": sum(gaps) / len(gaps) if gaps else None,
                    "note": "media_kfold",
                    "example_path": runs[0][1],
                }
            )
        else:
            best = max(
                runs,
                key=lambda x: (x[2] is not None, x[2] if x[2] is not None else 0.0),
            )
            out.append(
                {
                    "study": st,
                    "n_runs": len(runs),
                    "wa": best[2],
                    "ua": best[3],
                    "f1_macro": best[4],
                    "best_val_ua": best[5],
                    "gap_ua": best[6],
                    "note": "mejor_WA" if len(runs) > 1 else "unico",
                    "example_path": best[1],
                }
            )


def print_report(out: list[dict]) -> None:
    valid = [o for o in out if o["wa"] is not None]
    by_wa = sorted(valid, key=lambda o: o["wa"], reverse=True)  # type: ignore
    by_ua = sorted(valid, key=lambda o: o["ua"] or 0, reverse=True)  # type: ignore
    with_gap = [o for o in out if o["gap_ua"] is not None and o["ua"] and o["ua"] > 0.70]
    by_gen = sorted(with_gap, key=lambda o: (o["gap_ua"], -o["ua"]))  # type: ignore

    print("--- TOP 15 por WA (test) ---")
    for i, o in enumerate(by_wa[:15], 1):
        g = o["gap_ua"]
        gs = f"{g:.4f}" if g is not None else "n/a"
        print(
            f"{i:2d}. study {o['study']:2d} WA={o['wa']:.4f} UA={o['ua']:.4f} "
            f"gap_ua={gs} {o['note']}"
        )
    print("\n--- TOP 15 por UA (test) ---")
    for i, o in enumerate(by_ua[:15], 1):
        g = o["gap_ua"]
        gs = f"{g:.4f}" if g is not None else "n/a"
        print(
            f"{i:2d}. study {o['study']:2d} UA={o['ua']:.4f} WA={o['wa']:.4f} "
            f"gap_ua={gs} {o['note']}"
        )
    print("\n--- Generalizacion (menor gap_ua, UA test > 0.70) top 10 ---")
    for i, o in enumerate(by_gen[:10], 1):
        print(
            f"{i:2d}. study {o['study']:2d} gap_ua={o['gap_ua']:.4f} UA={o['ua']:.4f} WA={o['wa']:.4f}"
        )
    print(f"\nEstudios con al menos un resultado: {len(out)}")


def write_ranking_doc(out: list[dict], path: Path) -> None:
    valid = [o for o in out if o["wa"] is not None and o["ua"] is not None]
    by_wa = sorted(valid, key=lambda o: o["wa"], reverse=True)  # type: ignore
    by_ua = sorted(valid, key=lambda o: o["ua"], reverse=True)  # type: ignore
    with_gap = [o for o in out if o["gap_ua"] is not None and o["ua"] and o["ua"] > 0.70]
    by_gen = sorted(with_gap, key=lambda o: (o["gap_ua"], -o["ua"]))  # type: ignore

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Ranking y analisis de resultados (checkpoints locales)",
        "",
        f"**Generado**: {ts} ejecutando `python scripts/aggregate_ablation_results.py --write-docs`.",
        "",
        "## Criterios",
        "",
        "- **WA (weighted accuracy)**: accuracy global ponderada por soporte de clase; suele alinearse con lo que muchos papers citan como accuracy.",
        "- **UA (unweighted accuracy)**: media de accuracy por clase; mas justa con clases minoritarias.",
        "- **gap_ua**: `best_val_ua - UA_test`. Indica cuanto cae el modelo del mejor punto de validacion UA al test. **Valores mas bajos** suelen indicar mejor transferencia (menos sobreajuste a val), siempre comparando runs comparables.",
        "- **Agregacion por estudio**: si hay varios `final_results.json` en carpetas `fold_*`, se usa **media** en test (K-Fold). Si hay varios runs en un solo split, se elige el run con **mayor WA** (`mejor_WA`).",
        "",
        "## Aviso sobre el estudio 34 historico",
        "",
        "En `config/ablation/ABLATION_SUMMARY.md` se cita un run canonico de Ablation 34 con ~77.98% UA por su bajo gap. Eso **no sustituye** al ranking de esta tabla: aqui solo entran runs que existen hoy bajo `checkpoints/` con `final_results.json`. Un run concreto puede superar al 34 en WA/UA segun configuracion y semilla.",
        "",
        "## Top 3 por WA (test)",
        "",
        "| Pos | Estudio | WA | UA | gap_ua | Nota |",
        "|-----|---------|----|----|--------|------|",
    ]
    for i, o in enumerate(by_wa[:3], 1):
        g = o["gap_ua"]
        gs = f"{g:.4f}" if g is not None else "n/a"
        lines.append(
            f"| {i} | {o['study']} | {o['wa']:.4f} | {o['ua']:.4f} | {gs} | {o['note']} |"
        )
    lines.extend(
        [
            "",
            "## Top 3 por UA (test)",
            "",
            "| Pos | Estudio | UA | WA | gap_ua | Nota |",
            "|-----|---------|----|----|--------|------|",
        ]
    )
    for i, o in enumerate(by_ua[:3], 1):
        g = o["gap_ua"]
        gs = f"{g:.4f}" if g is not None else "n/a"
        lines.append(
            f"| {i} | {o['study']} | {o['ua']:.4f} | {o['wa']:.4f} | {gs} | {o['note']} |"
        )
    lines.extend(
        [
            "",
            "## Mejor generalizacion (menor gap_ua con UA test > 0.70)",
            "",
            "Orden: gap creciente (mejor primero), desempate por UA mayor.",
            "",
            "| Pos | Estudio | gap_ua | UA | WA |",
            "|-----|---------|--------|----|----|",
        ]
    )
    for i, o in enumerate(by_gen[:5], 1):
        lines.append(
            f"| {i} | {o['study']} | {o['gap_ua']:.4f} | {o['ua']:.4f} | {o['wa']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Tabla completa (todos los estudios agregados)",
            "",
            "| Estudio | WA | UA | F1 macro | gap_ua | n | Nota |",
            "|---------|-----|-----|----------|--------|---|------|",
        ]
    )
    for o in sorted(out, key=lambda x: x["study"]):
        wa = o["wa"]
        ua = o["ua"]
        f1 = o["f1_macro"]
        g = o["gap_ua"]
        wa_s = f"{wa:.4f}" if wa is not None else ""
        ua_s = f"{ua:.4f}" if ua is not None else ""
        f1_s = f"{f1:.4f}" if f1 is not None else ""
        g_s = f"{g:.4f}" if g is not None else ""
        lines.append(
            f"| {o['study']} | {wa_s} | {ua_s} | {f1_s} | {g_s} | {o['n_runs']} | {o['note']} |"
        )
    lines.extend(
        [
            "",
            "## Como regenerar",
            "",
            "```powershell",
            "cd d:\\DOCTORADO\\PROYECTOS\\CONFORMER-SER",
            "python scripts\\aggregate_ablation_results.py --write-docs",
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--write-docs",
        action="store_true",
        help="Escribe docs/RANKING_Y_RESULTADOS.md",
    )
    args = ap.parse_args()
    root = Path("checkpoints")
    rows = build_rows(root)
    out: list[dict] = []
    aggregate(out, rows)
    print_report(out)
    if args.write_docs:
        write_ranking_doc(out, Path("docs/RANKING_Y_RESULTADOS.md"))


if __name__ == "__main__":
    main()
