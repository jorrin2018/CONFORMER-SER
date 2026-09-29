#!/usr/bin/env python
"""
fit_speechcue_stats.py -- Precompute the three SpeechCue statistics for the
whole corpus, once, and cache them by utterance_id.

pYIN costs ~0.5-2 s per utterance, so it cannot run inside __getitem__. This
script does the expensive pass once; training then only looks the values up.

    python scripts/fit_speechcue_stats.py                       # all 5 sessions
    python scripts/fit_speechcue_stats.py --workers 8
    python scripts/fit_speechcue_stats.py --tracker yin         # ~10x faster

The run is resumable: utterances already in the cache are skipped, and the file
is flushed every --flush-every items, so Ctrl-C loses at most that many.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.speechcue import (  # noqa: E402
    SpeechCueConfig, SpeechCueNormalizer, SpeechCueStats, SpeechCueStatsCache,
    SpeechCueTertiles, compute_stats, speaker_id,
)

# Bump when the definition of any statistic changes, so stale caches are not reused.
CACHE_VERSION = 4

_CFG: SpeechCueConfig | None = None


def _init(cfg_kwargs: dict) -> None:
    global _CFG
    _CFG = SpeechCueConfig(**cfg_kwargs)


def _work(item):
    """(uid, audio_path, transcript) -> (uid, stats_tuple | None, error)."""
    uid, path, transcript = item
    try:
        import librosa
        y, _ = librosa.load(path, sr=_CFG.sample_rate, mono=True)
        return uid, compute_stats(y, transcript, _CFG).as_tuple(), None
    except Exception as e:  # noqa: BLE001
        return uid, None, f"{type(e).__name__}: {e}"


def enumerate_utterances(cfg: dict, sessions):
    """Same utterance set the training pipeline sees."""
    from src.data import build_dataset, corpus_from_config
    ds_cfg = cfg["dataset"]
    items = []
    for s in sessions:
        ds = build_dataset(
            corpus=corpus_from_config(cfg),
            data_path=ds_cfg["data_path"], sessions=[s],
            emotions=ds_cfg["emotions"], sample_rate=ds_cfg["sample_rate"],
            training=False,
        )
        items += [(d["utterance_id"], d["audio_path"], d["transcription"])
                  for d in ds.data]
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--out", default="cache/speechcue_stats.json")
    ap.add_argument("--sessions", default="1,2,3,4,5")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tracker", choices=["pyin", "yin"], default="pyin")
    ap.add_argument("--flush-every", type=int, default=250)
    ap.add_argument("--force", action="store_true", help="recompute everything")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    sessions = [int(x) for x in args.sessions.split(",")]
    out = Path(args.out)

    table: dict[str, SpeechCueStats] = {}
    if out.exists() and not args.force:
        import json
        meta = (json.loads(out.read_text(encoding="utf-8")) or {}).get("meta", {})
        if meta.get("version") != CACHE_VERSION:
            print(f"  cache v{meta.get('version')} obsoleto (actual v{CACHE_VERSION}): "
                  f"se recalcula todo")
        else:
            table = SpeechCueStatsCache.load(out).table
            print(f"  cache existente: {len(table)} utterances")

    items = enumerate_utterances(cfg, sessions)
    todo = [it for it in items if args.force or it[0] not in table]
    print(f"\n  utterances totales : {len(items)}")
    print(f"  por calcular       : {len(todo)}")
    print(f"  tracker            : {args.tracker}   workers: {args.workers}\n")

    cue_kwargs = dict(sample_rate=cfg["dataset"]["sample_rate"], tracker=args.tracker)
    errors, done, t0 = [], 0, time.time()

    if todo:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init, initargs=(cue_kwargs,)) as ex:
            futures = [ex.submit(_work, it) for it in todo]
            for fut in as_completed(futures):
                uid, stats, err = fut.result()
                if err:
                    errors.append((uid, err))
                else:
                    table[uid] = SpeechCueStats(*stats)
                done += 1
                if done % args.flush_every == 0 or done == len(todo):
                    SpeechCueStatsCache.save(
                        table, out,
                        meta={"tracker": args.tracker, "sessions": sessions,
                              "version": CACHE_VERSION})
                    el = time.time() - t0
                    eta = el / done * (len(todo) - done)
                    print(f"  {done}/{len(todo)}  ({el/60:.1f} min transcurridos, "
                          f"ETA {eta/60:.1f} min)", flush=True)

    SpeechCueStatsCache.save(table, out,
                             meta={"tracker": args.tracker, "sessions": sessions,
                                   "version": CACHE_VERSION})
    print(f"\n  cache guardado en {out}  ({len(table)} utterances)")
    if errors:
        print(f"  {len(errors)} errores; primeros 5:")
        for uid, e in errors[:5]:
            print(f"    {uid}: {e}")

    # ---------------------------------------------------------------- resumen
    import math
    from collections import defaultdict

    pit = defaultdict(list)
    for uid, st in table.items():
        if st.pitch and math.isfinite(st.pitch):
            pit[speaker_id(uid)[-1]].append(st.pitch)
    if pit:
        print("\n  F0 medio por genero del hablante (control del salto de octava):")
        for g in sorted(pit):
            v = sorted(pit[g])
            print(f"    {g}: n={len(v):<5} mediana={v[len(v)//2]:6.1f} Hz   "
                  f"p10={v[len(v)//10]:6.1f}  p90={v[9*len(v)//10]:6.1f}")

    n_nan = sum(1 for st in table.values()
                if st.pitch is None or not math.isfinite(st.pitch))
    print(f"    sin F0 medible: {n_nan} ({100*n_nan/len(table):.1f}%) -> caen en [P_MID]")
    per_ses = defaultdict(lambda: [0, 0])
    for uid, st in table.items():
        c = per_ses[uid[:5]]
        c[1] += 1
        if st.pitch is None or not math.isfinite(st.pitch):
            c[0] += 1
    print("      por sesion: " + "   ".join(
        f"{k}={100*v[0]/v[1]:.1f}%" for k, v in sorted(per_ses.items())))

    rates = sorted(st.rate for st in table.values() if st.rate and math.isfinite(st.rate))
    if rates:
        print(f"\n  tasa silabica (sobre el tramo hablado): mediana="
              f"{rates[len(rates)//2]:.2f} sil/s   p10={rates[len(rates)//10]:.2f}   "
              f"p90={rates[9*len(rates)//10]:.2f}   max={rates[-1]:.1f}")
        print(f"    utterances con tasa > 12 sil/s (implausible): "
              f"{sum(1 for r in rates if r > 12)}")
    spans = [(st.duration, st.speech_span) for st in table.values()
             if math.isfinite(st.duration) and math.isfinite(st.speech_span)]
    if spans:
        d = sum(a for a,_ in spans)/len(spans); sp = sum(b for _,b in spans)/len(spans)
        print(f"    duracion media del fichero {d:.2f} s -> tramo hablado {sp:.2f} s "
              f"({100*sp/d:.0f}%)")

    train_sessions = cfg["dataset"]["train_sessions"]
    train_uids = [u for u, _, _ in enumerate_utterances(cfg, train_sessions)]
    uids = [u for u in train_uids if u in table]
    stats = [table[u] for u in uids]
    if len(stats) > 100:
        norm = SpeechCueNormalizer.fit(table)
        for label, nz in (("globales (articulo actual)", None),
                          ("normalizados por hablante", norm)):
            tert = SpeechCueTertiles.fit(stats, train_sessions,
                                         normalizer=nz, utterance_ids=uids)
            print(f"\n  terciles {label}, sesiones {train_sessions}:")
            for name, (lo, hi) in tert.bounds.items():
                unit = "" if nz and name in nz.stat_names else \
                    {"pitch": "Hz", "energy": "", "rate": "sil/s"}[name]
                print(f"    {name:<7} LOW < {lo:.4g} {unit} <= MID <= {hi:.4g} {unit} < HIGH")
        print("\n  (el entrenamiento recalcula los terciles por fold; esto es informativo)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
