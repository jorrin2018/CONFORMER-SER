"""
SpeechCue: prosody prompting for a text encoder, as described in the paper.

Three utterance-level statistics are computed over voiced frames -- mean
fundamental frequency (pYIN), root-mean-square energy, and syllable rate from
the transcript. Each is discretised into LOW / MID / HIGH using tertile
boundaries estimated ONLY on the training partition of the current fold. Each
level has a dedicated learnable token; the three active tokens are inserted
after [CLS] and before the lexical word pieces.

The nine-token vocabulary adds exactly nine embeddings to the text encoder and
leaves the existing tokenisation untouched.

Cost note: pYIN is far too slow to run inside __getitem__. Statistics are
computed once for the whole corpus by scripts/fit_speechcue_stats.py and cached
by utterance_id; training only does a dict lookup plus two comparisons.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Nine-token vocabulary
# ---------------------------------------------------------------------------
PITCH_TOKENS = ("[P_LOW]", "[P_MID]", "[P_HIGH]")
ENERGY_TOKENS = ("[E_LOW]", "[E_MID]", "[E_HIGH]")
RATE_TOKENS = ("[R_LOW]", "[R_MID]", "[R_HIGH]")
SPEECHCUE_TOKENS: List[str] = list(PITCH_TOKENS + ENERGY_TOKENS + RATE_TOKENS)
assert len(SPEECHCUE_TOKENS) == 9

STAT_NAMES = ("pitch", "energy", "rate")


@dataclass
class SpeechCueConfig:
    """Analysis parameters for the three statistics."""

    sample_rate: int = 16000
    # pYIN search range for speech (covers male ~85 Hz to female ~255 Hz with margin)
    fmin: float = 65.0
    # 300 Hz cubre voz femenina con margen. 400 dejaba pasar saltos de octava:
    # con fmax=400 la media de F0 de los hablantes masculinos salia en 165-188 Hz,
    # muy por encima del rango tipico (100-130 Hz).
    fmax: float = 300.0
    frame_length: int = 1024        # 64 ms at 16 kHz
    hop_length: int = 160           # 10 ms
    # A frame counts as voiced when pYIN's voiced probability exceeds this.
    voiced_prob_threshold: float = 0.5
    tracker: str = "pyin"           # "pyin" (paper) or "yin" (faster fallback)
    # Octave-jump filter: keep only frames within [median/1.5, median*1.5] of the
    # utterance median before averaging. Standard practice; the reported value is
    # still the mean F0 over voiced frames, just without the doubled frames.
    octave_filter: bool = True
    octave_tolerance: float = 1.5
    # Silence trimming for the syllable-rate denominator.
    trim_top_db: float = 35.0
    min_speech_span: float = 0.30   # below this, fall back to the file duration


@dataclass
class SpeechCueStats:
    """Utterance-level statistics. NaN means "not measurable".

    `duration` is the whole file; `speech_span` is what is left after trimming
    leading and trailing silence, and it is the denominator of the syllable rate.
    IEMOCAP sentence wavs carry generous silence padding, which would otherwise
    drag the rate down for reasons that have nothing to do with prosody.
    """

    pitch: float
    energy: float
    rate: float
    duration: float = float("nan")        # full file
    speech_span: float = float("nan")     # after trimming leading/trailing silence

    def as_tuple(self) -> Tuple[float, ...]:
        return (self.pitch, self.energy, self.rate, self.duration, self.speech_span)


# ---------------------------------------------------------------------------
# Syllable rate
# ---------------------------------------------------------------------------
# IEMOCAP transcripts carry non-lexical markers such as [LAUGHTER], [BREATHING],
# [GARBAGE]; they are not spoken syllables and are removed before counting.
_MARKER_RE = re.compile(r"\[[^\]]*\]")
_WORD_RE = re.compile(r"[a-z']+")
_VOWELS = set("aeiouy")


def count_syllables_word(word: str) -> int:
    """Vowel-group heuristic for English syllable counting."""
    word = word.strip("'")
    if not word:
        return 0
    n, prev_vowel = 0, False
    for ch in word:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev_vowel:
            n += 1
        prev_vowel = is_vowel
    # silent terminal 'e' ("time" = 1, but "little"/"bye" keep theirs)
    if word.endswith("e") and n > 1 and not word.endswith(("le", "ye", "ee")):
        n -= 1
    return max(n, 1)


def speaker_id(utterance_id: str) -> str:
    """Ses05F_impro05_M034 -> 'Ses05_M'.

    Each IEMOCAP session is one male/female pair, and the turn prefix says which
    of the two is talking, so session + turn gender identifies all ten actors.
    """
    return f"{utterance_id[:5]}_{utterance_id.split('_')[-1][0]}"


def count_syllables(transcript: str) -> int:
    text = _MARKER_RE.sub(" ", (transcript or "").lower())
    return sum(count_syllables_word(w) for w in _WORD_RE.findall(text))


# ---------------------------------------------------------------------------
# Statistic extraction
# ---------------------------------------------------------------------------
def compute_stats(
    waveform: np.ndarray,
    transcript: str,
    config: Optional[SpeechCueConfig] = None,
) -> SpeechCueStats:
    """Compute (mean F0, RMS energy, syllable rate) for one utterance.

    Pitch and energy are averaged over voiced frames only, as the paper states.
    Rate is syllables per second over the utterance duration.
    """
    import librosa

    cfg = config or SpeechCueConfig()
    y = np.asarray(waveform, dtype=np.float32).squeeze()
    if y.ndim > 1:
        y = y.mean(axis=0)

    duration = len(y) / cfg.sample_rate
    if duration <= 0 or not np.any(np.isfinite(y)) or np.allclose(y, 0.0):
        return SpeechCueStats(np.nan, np.nan, np.nan)

    # --- F0 and voicing ----------------------------------------------------
    # pYIN's voicing decision degrades on quiet recordings: session 3 of IEMOCAP
    # is roughly 5x quieter than session 1 and produced 34.3% unvoiced utterances
    # against 13.7% in session 5. F0 does not depend on amplitude but the
    # estimator's thresholds do, so the tracker gets a peak-normalised copy.
    # Energy is measured on the original signal.
    peak = float(np.max(np.abs(y)))
    y_pitch = y / peak if peak > 1e-6 else y

    if cfg.tracker == "pyin":
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y_pitch,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            sr=cfg.sample_rate,
            frame_length=cfg.frame_length,
            hop_length=cfg.hop_length,
        )
        voiced = np.asarray(voiced_flag, dtype=bool)
        if voiced_prob is not None:
            voiced &= np.asarray(voiced_prob) >= cfg.voiced_prob_threshold
        # pYIN's own decision plus a 0.5 probability floor left ~21% of utterances
        # with no voiced frame at all -- mostly short backchannels ("Yeah.", "Mm-hm").
        # Those would silently fall back to [P_MID]. Retry on the flag alone before
        # giving up, so the token means something for as many utterances as possible.
        if not voiced.any():
            voiced = np.asarray(voiced_flag, dtype=bool)
        # Still nothing: fall back to plain YIN over the loud half of the
        # utterance. YIN always returns an estimate, so restricting it to frames
        # with above-median energy keeps it off the silence.
        if not voiced.any():
            f0 = librosa.yin(y_pitch, fmin=cfg.fmin, fmax=cfg.fmax,
                             sr=cfg.sample_rate, frame_length=cfg.frame_length,
                             hop_length=cfg.hop_length)
            f0 = np.asarray(f0, dtype=np.float64)
            e = librosa.feature.rms(y=y, frame_length=cfg.frame_length,
                                    hop_length=cfg.hop_length, center=True)[0]
            m = min(len(f0), len(e))
            f0, e = f0[:m], e[:m]
            voiced = (np.isfinite(f0) & (f0 > cfg.fmin) & (f0 < cfg.fmax)
                      & (e >= np.median(e)))
    else:  # "yin": no voicing decision, keep frames with an in-range estimate
        f0 = librosa.yin(
            y_pitch,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            sr=cfg.sample_rate,
            frame_length=cfg.frame_length,
            hop_length=cfg.hop_length,
        )
        voiced = np.isfinite(f0) & (f0 > cfg.fmin) & (f0 < cfg.fmax)

    f0 = np.asarray(f0, dtype=np.float64)
    voiced &= np.isfinite(f0)

    voiced_full = voiced.copy()          # before the octave filter narrows it

    # Drop octave jumps before averaging (see SpeechCueConfig.octave_filter).
    if cfg.octave_filter and voiced.any():
        med = float(np.median(f0[voiced]))
        if np.isfinite(med) and med > 0:
            lo, hi = med / cfg.octave_tolerance, med * cfg.octave_tolerance
            keep = voiced & (f0 >= lo) & (f0 <= hi)
            if keep.sum() >= 3:
                voiced = keep

    # The octave filter narrows the mask; keep the wider one for energy so the
    # average is not taken over a handful of frames.
    voiced_pitch = voiced
    pitch = float(np.mean(f0[voiced_pitch])) if voiced_pitch.any() else np.nan

    # --- RMS energy over voiced frames -------------------------------------
    rms = librosa.feature.rms(
        y=y, frame_length=cfg.frame_length, hop_length=cfg.hop_length, center=True
    )[0]
    n = min(len(rms), len(voiced_full))
    rms_c, voiced_e = rms[:n], voiced_full[:n]
    if voiced_e.any():
        energy = float(np.mean(rms_c[voiced_e]))
    elif n:
        energy = float(np.mean(rms_c))
    else:
        energy = np.nan

    # --- syllable rate over the SPEECH duration ----------------------------
    # Dividing by the whole file would penalise utterances with more silence
    # padding, which is a recording artefact. But voiced frames are NOT speech
    # frames -- unvoiced consonants and short pauses are speech too, and using
    # them as the denominator produced rates up to 57 syllables/s. Trim leading
    # and trailing silence by energy instead: that is the spoken span.
    try:
        _, (b, e) = librosa.effects.trim(y, top_db=cfg.trim_top_db)
        span = float(max(e - b, 0) / cfg.sample_rate)
    except Exception:  # noqa: BLE001
        span = duration
    if span < cfg.min_speech_span:
        span = duration
    n_syl = count_syllables(transcript)
    rate = float(n_syl / span) if n_syl > 0 and span > 0 else 0.0

    return SpeechCueStats(pitch, energy, rate, duration, span)


# ---------------------------------------------------------------------------
# Per-speaker normalisation
# ---------------------------------------------------------------------------
class SpeechCueNormalizer:
    """Z-score of log-pitch and log-energy within each speaker.

    Measured on the full corpus, the speaker explains 31.3% of the variance of
    mean F0 and 20.6% of RMS energy, but only 4.6% of syllable rate. Global
    tertiles therefore turn the pitch token into a speaker-gender detector
    (male medians 165-188 Hz, female 218-259 Hz, global tertile boundaries at
    192 and 245 Hz land exactly between the two distributions), and the energy
    token into a per-session recording-gain detector (speaker means span
    0.010 to 0.051). Normalising per speaker makes the token mean "high FOR THIS
    SPEAKER", which is the prosodic quantity the paper is after. Rate is left
    global, since speaker identity barely explains it.

    Statistics are estimated from each speaker's own audio only -- no labels are
    involved -- but they do use the test speakers' recordings, so this is
    unsupervised speaker adaptation (transductive) and must be stated as such.
    """

    NORMALIZED = ("pitch", "energy")
    LOG_SCALE = ("pitch", "energy")

    def __init__(self, params: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
                 stat_names: Sequence[str] = NORMALIZED):
        self.params = params or {}
        self.stat_names = tuple(stat_names)

    @classmethod
    def fit(cls, table: Dict[str, "SpeechCueStats"],
            stat_names: Sequence[str] = NORMALIZED) -> "SpeechCueNormalizer":
        buckets: Dict[str, Dict[str, list]] = {}
        for uid, st in table.items():
            spk = speaker_id(uid)
            b = buckets.setdefault(spk, {n: [] for n in stat_names})
            for name in stat_names:
                v = getattr(st, name)
                if v is None or not np.isfinite(v) or (name in cls.LOG_SCALE and v <= 0):
                    continue
                b[name].append(np.log(v) if name in cls.LOG_SCALE else v)

        params: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for spk, b in buckets.items():
            params[spk] = {}
            for name, vals in b.items():
                if len(vals) < 10:
                    params[spk][name] = (0.0, 1.0)   # identity: too little evidence
                else:
                    arr = np.asarray(vals, dtype=np.float64)
                    sd = float(arr.std())
                    params[spk][name] = (float(arr.mean()), sd if sd > 1e-8 else 1.0)
        return cls(params, stat_names)

    def transform(self, utterance_id: str, stats: "SpeechCueStats") -> "SpeechCueStats":
        spk = speaker_id(utterance_id)
        p = self.params.get(spk)
        if p is None:
            return stats
        out = dict(pitch=stats.pitch, energy=stats.energy, rate=stats.rate,
                   duration=stats.duration, speech_span=stats.speech_span)
        for name in self.stat_names:
            v = out[name]
            if v is None or not np.isfinite(v):
                continue
            if name in self.LOG_SCALE:
                if v <= 0:
                    continue
                v = np.log(v)
            mu, sd = p[name]
            out[name] = float((v - mu) / sd)
        return SpeechCueStats(**out)

    def to_dict(self) -> dict:
        return {"stat_names": list(self.stat_names),
                "params": {s: {n: list(v) for n, v in d.items()}
                           for s, d in self.params.items()}}

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "SpeechCueNormalizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({s: {n: tuple(v) for n, v in x.items()} for s, x in d["params"].items()},
                   d.get("stat_names", cls.NORMALIZED))

    def __repr__(self) -> str:
        return (f"SpeechCueNormalizer(speakers={len(self.params)}, "
                f"stats={list(self.stat_names)})")


# ---------------------------------------------------------------------------
# Tertile boundaries -- fitted on the training partition only
# ---------------------------------------------------------------------------
class SpeechCueTertiles:
    """LOW / MID / HIGH boundaries per statistic.

    Fit on the training sessions of the current fold and reuse unchanged for
    validation and test, so no distributional information leaks into the prompt.
    """

    def __init__(self, bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 fitted_on: Optional[Sequence[int]] = None,
                 n_utterances: int = 0,
                 normalizer: Optional["SpeechCueNormalizer"] = None):
        self.bounds: Dict[str, Tuple[float, float]] = bounds or {}
        self.fitted_on = list(fitted_on or [])
        self.n_utterances = n_utterances
        self.normalizer = normalizer

    # -- fitting ------------------------------------------------------------
    @classmethod
    def fit(cls, stats: Iterable[SpeechCueStats],
            sessions: Optional[Sequence[int]] = None,
            normalizer: Optional["SpeechCueNormalizer"] = None,
            utterance_ids: Optional[Sequence[str]] = None) -> "SpeechCueTertiles":
        stats = list(stats)
        if normalizer is not None:
            if utterance_ids is None or len(utterance_ids) != len(stats):
                raise ValueError("fit(normalizer=...) requires matching utterance_ids")
            stats = [normalizer.transform(u, st) for u, st in zip(utterance_ids, stats)]
        bounds = {}
        for i, name in enumerate(STAT_NAMES):
            vals = np.array([s.as_tuple()[i] for s in stats], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size < 3:
                raise ValueError(f"SpeechCue: not enough finite values for '{name}'")
            q33, q67 = np.percentile(vals, [100 / 3.0, 200 / 3.0])
            bounds[name] = (float(q33), float(q67))
        return cls(bounds, sessions, len(stats), normalizer)

    # -- application --------------------------------------------------------
    def level(self, name: str, value: float) -> int:
        """0 = LOW, 1 = MID, 2 = HIGH. Unmeasurable values fall back to MID."""
        if value is None or not np.isfinite(value):
            return 1
        low, high = self.bounds[name]
        if value < low:
            return 0
        if value > high:
            return 2
        return 1

    def tokens_for(self, stats: SpeechCueStats,
                   utterance_id: Optional[str] = None) -> List[str]:
        if self.normalizer is not None and utterance_id is not None:
            stats = self.normalizer.transform(utterance_id, stats)
        return [
            PITCH_TOKENS[self.level("pitch", stats.pitch)],
            ENERGY_TOKENS[self.level("energy", stats.energy)],
            RATE_TOKENS[self.level("rate", stats.rate)],
        ]

    def prompt_for(self, stats: SpeechCueStats,
                   utterance_id: Optional[str] = None) -> str:
        return " ".join(self.tokens_for(stats, utterance_id))

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        return {"bounds": {k: list(v) for k, v in self.bounds.items()},
                "fitted_on_sessions": self.fitted_on,
                "n_utterances": self.n_utterances}

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "SpeechCueTertiles":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({k: tuple(v) for k, v in d["bounds"].items()},
                   d.get("fitted_on_sessions"), d.get("n_utterances", 0))

    def __repr__(self) -> str:
        b = "  ".join(f"{k}:[{v[0]:.4g},{v[1]:.4g}]" for k, v in self.bounds.items())
        norm = "speaker-normalised" if self.normalizer is not None else "global"
        return (f"SpeechCueTertiles({norm}, sessions={self.fitted_on}, "
                f"n={self.n_utterances}, {b})")


# ---------------------------------------------------------------------------
# Cached corpus statistics
# ---------------------------------------------------------------------------
class SpeechCueStatsCache:
    """utterance_id -> SpeechCueStats, produced by scripts/fit_speechcue_stats.py."""

    def __init__(self, table: Dict[str, SpeechCueStats]):
        self.table = table

    def __len__(self) -> int:
        return len(self.table)

    def __contains__(self, uid: str) -> bool:
        return uid in self.table

    def get(self, uid: str) -> Optional[SpeechCueStats]:
        return self.table.get(uid)

    def subset(self, uids: Iterable[str]) -> List[SpeechCueStats]:
        return [self.table[u] for u in uids if u in self.table]

    @classmethod
    def load(cls, path) -> "SpeechCueStatsCache":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = raw["stats"] if isinstance(raw, dict) and "stats" in raw else raw
        # Older caches stored only (pitch, energy, rate).
        return cls({u: SpeechCueStats(*v) for u, v in rows.items()})

    @classmethod
    def save(cls, table: Dict[str, SpeechCueStats], path, meta: Optional[dict] = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {"meta": meta or {},
                   "stats": {u: list(s.as_tuple()) for u, s in table.items()}}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")
