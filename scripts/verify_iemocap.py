#!/usr/bin/env python
"""
verify_iemocap.py — Chequeo de integridad de una copia de IEMOCAP.

Ejecutar SIEMPRE tras descargar un espejo (Kaggle, HF, etc.) y ANTES de entrenar.
Fue exactamente la ausencia de este chequeo lo que produjo el corpus scripted-only
(2 588 utterances en lugar de 5 531) sobre el que se entrenó CAMF.

Uso:
    python scripts/verify_iemocap.py "D:/DOCTORADO/DATASETS/IEMOCAP/IEMOCAP_full_release"
    python scripts/verify_iemocap.py <ruta> --leak-check

Salida: código 0 si el corpus está completo, 1 si no.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ----------------------------------------------------------------------------
# Contrato del dataset. Protocolo de 4 clases: ang / hap+exc / sad / neu.
# ----------------------------------------------------------------------------
EXPECTED_4CLASS_PER_SESSION = {1: 1085, 2: 1023, 3: 1151, 4: 1031, 5: 1241}
EXPECTED_4CLASS_TOTAL = 5531          # suma de la línea anterior
EXPECTED_ALL_UTTERANCES = 10039       # corpus completo, todas las etiquetas
TOLERANCE = 0.01                      # 1 % de holgura por variaciones de espejo

EMOTION_MAP = {
    "ang": "angry",
    "hap": "happy",
    "exc": "happy",   # excited se fusiona en happy (protocolo estándar)
    "sad": "sad",
    "neu": "neutral",
}

LINE_RE = re.compile(r"\[[\d.]+\s*-\s*[\d.]+\]\s+(\S+)\s+(\w+)")

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def _txt_files(folder: Path):
    """Ficheros .txt reales, ignorando los '._x' que dejan los zips de macOS.

    Los espejos de Kaggle suelen traerlos y un glob ingenuo los parsea como si
    fueran anotaciones, produciendo ids fantasma.
    """
    for f in sorted(folder.glob("*.txt")):
        if f.name.startswith("._") or f.name.startswith("."):
            continue
        yield f


def parse_session(root: Path, session: int):
    """Devuelve (labels, transcripts, wav_ok, wav_missing) de una sesión."""
    spath = root / f"Session{session}"
    labels: dict[str, str] = {}
    transcripts: dict[str, str] = {}
    wav_ok, wav_missing = 0, 0

    eval_dir = spath / "dialog" / "EmoEvaluation"
    if eval_dir.is_dir():
        for f in _txt_files(eval_dir):
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line[0] in "%" or line[:2] in ("C-", "A-"):
                    continue
                m = LINE_RE.match(line)
                if m:
                    labels[m.group(1)] = m.group(2).lower()

    trans_dir = spath / "dialog" / "transcriptions"
    if trans_dir.is_dir():
        for f in _txt_files(trans_dir):
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                if ":" not in line:
                    continue
                head, text = line.split(":", 1)
                parts = head.split()
                if parts:
                    transcripts[parts[0]] = text.strip()

    wav_root = spath / "sentences" / "wav"
    for uid, code in labels.items():
        if EMOTION_MAP.get(code) is None:
            continue
        dialog_id = "_".join(uid.split("_")[:-1])
        if (wav_root / dialog_id / f"{uid}.wav").exists():
            wav_ok += 1
        else:
            wav_missing += 1

    return labels, transcripts, wav_ok, wav_missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="ruta a IEMOCAP_full_release")
    ap.add_argument("--leak-check", action="store_true",
                    help="mide la fuga léxica entre sesiones 1-3 y sesión 5")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"{RED}La ruta no existe:{RESET} {root}")
        return 1

    print(f"\n  Corpus: {root}\n")
    header = f"  {'Ses':<5}{'total':>8}{'4-clases':>10}{'esperado':>10}{'impro':>8}{'script':>8}{'wav?':>8}{'texto?':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    ok = True
    grand_4c = grand_all = grand_impro = grand_script = 0
    per_class = Counter()
    corpus: dict[int, tuple] = {}

    for s in (1, 2, 3, 4, 5):
        labels, transcripts, wav_ok, wav_missing = parse_session(root, s)
        corpus[s] = (labels, transcripts)

        four = {u: EMOTION_MAP[c] for u, c in labels.items() if c in EMOTION_MAP}
        n_impro = sum(1 for u in four if "impro" in u)
        n_script = len(four) - n_impro
        n_text = sum(1 for u in four if transcripts.get(u))
        exp = EXPECTED_4CLASS_PER_SESSION[s]

        per_class.update(four.values())
        grand_4c += len(four)
        grand_all += len(labels)
        grand_impro += n_impro
        grand_script += n_script

        bad = (abs(len(four) - exp) > TOLERANCE * exp) or n_impro == 0 or wav_missing > 0
        ok &= not bad
        colour = RED if bad else GREEN

        print(f"  {colour}S{s}{RESET}   {len(labels):>8}{colour}{len(four):>10}{RESET}{exp:>10}"
              f"{n_impro:>8}{n_script:>8}"
              f"{(str(wav_missing) + ' falt.') if wav_missing else 'ok':>8}"
              f"{n_text:>8}")

    print("  " + "-" * (len(header) - 2))
    print(f"  {'TOT':<5}{grand_all:>8}{grand_4c:>10}{EXPECTED_4CLASS_TOTAL:>10}"
          f"{grand_impro:>8}{grand_script:>8}\n")

    print(f"  Distribución 4 clases: " + "  ".join(
        f"{k}={per_class[k]}" for k in ("angry", "happy", "sad", "neutral")))
    print(f"  Utterances anotadas (todas las etiquetas): {grand_all} "
          f"(referencia {EXPECTED_ALL_UTTERANCES})\n")

    if grand_impro == 0:
        print(f"{RED}  FALLO: no hay ni un diálogo improvisado. La copia está incompleta")
        print(f"  y el texto de train y test comparte guion — fuga léxica garantizada.{RESET}\n")
        ok = False
    elif abs(grand_4c - EXPECTED_4CLASS_TOTAL) > TOLERANCE * EXPECTED_4CLASS_TOTAL:
        print(f"{RED}  FALLO: {grand_4c} utterances de 4 clases, se esperaban "
              f"~{EXPECTED_4CLASS_TOTAL}.{RESET}\n")
        ok = False
    else:
        print(f"{GREEN}  Corpus completo. Se puede entrenar.{RESET}\n")

    if args.leak_check:
        print("  Fuga léxica train(S1-3) -> test(S5)")
        print("  " + "-" * (len(header) - 2))
        norm = lambda t: re.sub(r"[^a-z ]", "", t.lower()).strip()
        train_txt = set()
        for s in (1, 2, 3):
            labels, transcripts = corpus[s]
            train_txt.update(norm(transcripts[u]) for u in labels
                             if labels[u] in EMOTION_MAP and transcripts.get(u))
        train_txt.discard("")

        labels5, trans5 = corpus[5]
        for kind in ("impro", "script", "todo"):
            uids = [u for u in labels5 if labels5[u] in EMOTION_MAP and trans5.get(u)
                    and (kind == "todo" or kind in u)]
            if not uids:
                continue
            hits = sum(1 for u in uids if norm(trans5[u]) in train_txt)
            pct = 100 * hits / len(uids)
            colour = RED if pct > 20 else (YELLOW if pct > 5 else GREEN)
            print(f"  {kind:<8}{len(uids):>6} utt.   {colour}{hits:>5} idénticas  "
                  f"({pct:5.1f} %){RESET}")
        print(f"\n{DIM}  Un porcentaje alto en 'script' es normal y esperado: las tres obras\n"
              f"  las representan las cinco parejas de actores. Por eso el corpus\n"
              f"  scripted-only inflaba la rama textual. Repórtalo en el artículo.{RESET}\n")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
