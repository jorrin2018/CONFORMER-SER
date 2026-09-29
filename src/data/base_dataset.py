"""
Base comun para los corpus de reconocimiento de emociones.

POR QUE EXISTE. Hasta el 22/09/2026 el repo solo sabia leer IEMOCAP: `IEMOCAPDataset`
mezclaba en una clase lo que es propio del corpus (donde estan los wav, como se parsean
las etiquetas y las transcripciones, cuantas utterances debe haber por sesion) con lo
que no lo es (cargar el audio, recortarlo, tokenizar el texto, calcular pesos de clase,
filtrar por hablante, tomar un subconjunto por indices). El articulo necesita un segundo
corpus, asi que esa segunda mitad sube aqui y cada corpus implementa solo la primera.

LO QUE UNA SUBCLASE TIENE QUE IMPLEMENTAR:

    _collect_records()   Devuelve la lista de registros del corpus, en orden
                         determinista. Cada registro es un dict con las claves
                         audio_path, transcription, emotion, emotion_idx,
                         utterance_id y session. El orden importa: `subset_indices`
                         se apoya en que dos construcciones con las mismas sesiones
                         produzcan la misma lista.

    _assert_integrity()  Opcional pero MUY recomendable. Es el guardarrail que en
                         IEMOCAP detecto que se estaban perdiendo silenciosamente
                         2,943 de 5,531 utterances por un bug de troceado de cadenas.
                         Un corpus sin contrato verificable es un corpus que puede
                         mentir durante meses. La base no lo impone porque un corpus
                         nuevo puede no tener todavia las cifras de referencia.

LO QUE NO CAMBIA: `__getitem__`, el recorte de audio, la prosodia, los SpeechCue, los
pesos de clase, el filtro de hablante y el subconjunto por indices son identicos a lo
que habia. El refactor se verifica con `scripts/check_dataset_refactor.py`, que compara
registro a registro contra una copia congelada del archivo anterior.
"""

import os
import re
import torch
import torchaudio
import random
import gc
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional

from src.data.prosody_cues import ProsodyCueExtractor, ProsodyCueConfig
from src.data.speechcue import SpeechCueStatsCache, SpeechCueTertiles, speaker_id


class BaseSERDataset(Dataset):
    """Parte independiente del corpus: audio, texto, pesos, filtros, subconjuntos."""

    #: Etiqueta corta del corpus. Solo se usa en los mensajes de consola.
    CORPUS = 'base'

    def __init__(
        self,
        data_path: str,
        sessions: List[int],
        emotions: List[str],
        sample_rate: int = 16000,
        audio_preprocessor=None,
        text_preprocessor=None,
        max_audio_length: float = 10.0,
        training: bool = True,
        use_prosody_cues: bool = False,
        prosody_config: Optional[Dict] = None,
        strict_integrity: bool = True,
        speechcue_tertiles: Optional['SpeechCueTertiles'] = None,
        speechcue_stats: Optional['SpeechCueStatsCache'] = None,
        only_speakers: Optional[List[str]] = None,
        exclude_speakers: Optional[List[str]] = None,
        subset_indices: Optional[List[int]] = None
    ):
        """
        Args:
            data_path: Path to IEMOCAP_full_release directory
            sessions: List of session numbers to include (e.g., [1, 2, 3])
            emotions: List of emotion labels to include
            sample_rate: Target sample rate for audio
            audio_preprocessor: Audio preprocessing function
            text_preprocessor: Text preprocessing function
            max_audio_length: Maximum audio length in seconds
            training: If True, use random segment sampling for augmentation. If False, use deterministic start.
            speechcue_tertiles: LOW/MID/HIGH boundaries fitted on this fold's TRAIN
                sessions only. When given (together with speechcue_stats), the three
                SpeechCue tokens are prepended to the transcript, so the tokenizer
                places them right after [CLS] and before the lexical word pieces.
                This is the module described in the paper; `use_prosody_cues` selects
                the older natural-language prompt and is kept only for reproducing
                the earlier ablations.
            speechcue_stats: Corpus-wide statistics cache produced by
                scripts/fit_speechcue_stats.py (pYIN is too slow for __getitem__).
            only_speakers / exclude_speakers: Speaker ids ("Ses02_M") to keep or drop.
                Used to carve a speaker-disjoint validation split out of the training
                sessions, so the standard LOSO protocol can train on four sessions
                without validation and training ever sharing a speaker. Applied AFTER
                the integrity check, which always sees the full session.
            strict_integrity: If True, raise if the loaded corpus does not match the
                IEMOCAP 4-class contract (see _assert_integrity). Disable only for
                tests or debugging on a subset - never for a run that goes in a paper.
        """
        self.data_path = Path(data_path)
        self.sessions = sessions
        self.emotions = emotions
        self.sample_rate = sample_rate
        self.audio_preprocessor = audio_preprocessor
        self.text_preprocessor = text_preprocessor
        self.max_audio_length = max_audio_length
        self.training = training
        self.strict_integrity = strict_integrity
        self.subset_indices = subset_indices
        self.only_speakers = set(only_speakers) if only_speakers else None
        self.exclude_speakers = set(exclude_speakers) if exclude_speakers else None
        self.speechcue_tertiles = speechcue_tertiles
        self.speechcue_stats = speechcue_stats
        self._speechcue_missing = 0
        
        # Prosody cue extractor (optional, for SpeechCue-style prompts)
        if use_prosody_cues:
            cfg_dict = prosody_config or {}
            cfg = ProsodyCueConfig(
                sample_rate=sample_rate,
                **{k: v for k, v in cfg_dict.items() if hasattr(ProsodyCueConfig, k)}
            )
            self.prosody_extractor = ProsodyCueExtractor(cfg)
        else:
            self.prosody_extractor = None

        # Emotion to index mapping
        self.emotion2idx = {emotion: idx for idx, emotion in enumerate(emotions)}
        self.idx2emotion = {idx: emotion for emotion, idx in self.emotion2idx.items()}
        
        # Load dataset
        self.data = self._load_data()
        print(f"Loaded {len(self.data)} samples from sessions {sessions}")
        
        # Print class distribution
        self._print_class_distribution()

    def _load_data(self) -> List[Dict]:
        """Enumera el corpus, verifica el contrato y aplica filtros.

        La enumeracion la pone la subclase; todo lo que sigue es comun.
        """
        data = self._collect_records()

        self._assert_integrity(data)

        # Speaker filter last: the contract check above must always see the full
        # session, otherwise carving out a validation speaker would look like a
        # corrupt corpus.
        if self.only_speakers or self.exclude_speakers:
            before = len(data)
            if self.only_speakers:
                data = [d for d in data
                        if self.speaker_of(d) in self.only_speakers]
            if self.exclude_speakers:
                data = [d for d in data
                        if self.speaker_of(d) not in self.exclude_speakers]
            kept = sorted({self.speaker_of(d) for d in data})
            print(f"[{self.CORPUS}] filtro de hablante: {before} -> {len(data)} "
                  f"utterances; hablantes = {kept}")

        # Subconjunto por indices: permite partir las mismas sesiones en train/val
        # manteniendo preprocesadores distintos para cada mitad. El orden de
        # self.data es determinista, asi que los indices son estables entre dos
        # construcciones con las mismas sesiones.
        if self.subset_indices is not None:
            data = [data[i] for i in self.subset_indices]
            print(f"[{self.CORPUS}] subconjunto por indices: {len(data)} utterances")

        return data

    def _collect_records(self) -> List[Dict]:
        """Enumeracion propia del corpus. La subclase la implementa."""
        raise NotImplementedError(
            f'{type(self).__name__} tiene que implementar _collect_records().')

    def speaker_of(self, record: Dict) -> str:
        """Identificador de hablante de un registro.

        Por defecto usa `speechcue.speaker_id`, que lee el formato de IEMOCAP
        ('Ses02F_impro03_M012' -> 'Ses02_M'). UN CORPUS NUEVO CASI SEGURO TIENE QUE
        SOBRESCRIBIRLO: si no, el filtro de hablante -- del que depende la validacion
        disjunta por hablante del protocolo `speaker` -- agruparia mal sin avisar, y
        train y validacion compartirian locutor. Ese fallo no lanza ninguna excepcion;
        solo infla el numero.
        """
        return speaker_id(record['utterance_id'])

    def _assert_integrity(self, data: List[Dict]) -> None:
        """Contrato del corpus. Por defecto no comprueba nada; sobrescribir."""
        return

    def _print_class_distribution(self):
        """Print distribution of emotion classes."""
        emotion_counts = {}
        for sample in self.data:
            emotion = sample['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print("\nClass distribution:")
        for emotion in self.emotions:
            count = emotion_counts.get(emotion, 0)
            percentage = (count / len(self.data)) * 100 if len(self.data) > 0 else 0
            print(f"  {emotion:12s}: {count:4d} ({percentage:5.2f}%)")
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced dataset."""
        emotion_counts = {}
        for sample in self.data:
            emotion = sample['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Calculate weights: inverse of frequency
        total_samples = len(self.data)
        weights = []
        for emotion in self.emotions:
            count = emotion_counts.get(emotion, 1)  # Avoid division by zero
            weight = total_samples / (len(self.emotions) * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        sample = self.data[idx]
        audio_path = sample['audio_path']
        max_samples = int(self.max_audio_length * self.sample_rate)

        # Load only the needed segment when possible to reduce host-memory pressure
        # on Windows and avoid full-file allocations for long utterances.
        try:
            metadata = torchaudio.info(audio_path)
            original_sr = int(metadata.sample_rate)
            total_frames = int(metadata.num_frames)

            max_frames_in_original_sr = max_samples
            if original_sr > 0 and original_sr != self.sample_rate:
                max_frames_in_original_sr = int(self.max_audio_length * original_sr)

            if total_frames > max_frames_in_original_sr:
                if self.training:
                    start_frame = random.randint(0, total_frames - max_frames_in_original_sr)
                else:
                    start_frame = 0
                waveform, sr = torchaudio.load(
                    audio_path,
                    frame_offset=start_frame,
                    num_frames=max_frames_in_original_sr,
                )
            else:
                waveform, sr = torchaudio.load(audio_path)
        except (RuntimeError, OSError, MemoryError):
            # Best-effort recovery from transient allocation pressure.
            gc.collect()
            waveform, sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Ensure fixed length after possible segment loading/resampling
        waveform_length = waveform.shape[1]
        if waveform_length > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform_length < max_samples:
            padding = max_samples - waveform_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Apply audio preprocessing
        if self.audio_preprocessor is not None:
            audio_features = self.audio_preprocessor(waveform.squeeze(0))
        else:
            audio_features = waveform.squeeze(0)
        
        # Build (optionally enriched) transcription
        transcription = sample['transcription']
        if self.speechcue_tertiles is not None and self.speechcue_stats is not None:
            # SpeechCue: three discrete tokens before the word pieces.
            # Tokenizers prepend [CLS] themselves, so putting the tokens at the
            # front of the string yields [CLS] [P_x] [E_x] [R_x] <transcript>.
            stats = self.speechcue_stats.get(sample['utterance_id'])
            if stats is None:
                self._speechcue_missing += 1
                enriched_text = transcription
            else:
                enriched_text = f"{self.speechcue_tertiles.prompt_for(stats, sample['utterance_id'])} {transcription}".strip()
        elif self.prosody_extractor is not None:
            prosody_desc = self.prosody_extractor(waveform)
            # Experiment 1: texto original primero, prosodia después (menos disruptivo)
            # Ejemplo: "[SPEECH] ... [PROSODY] overall energy is high, variability is medium, speaking rate is fast."
            enriched_text = f"[SPEECH] {transcription} [PROSODY] {prosody_desc}."
        else:
            enriched_text = transcription

        # Apply text preprocessing
        if self.text_preprocessor is not None:
            text_features = self.text_preprocessor(enriched_text)
        else:
            text_features = {'input_ids': torch.tensor([0]), 'attention_mask': torch.tensor([1])}
        
        return {
            'audio': audio_features,
            'text_input_ids': text_features['input_ids'],
            'text_attention_mask': text_features['attention_mask'],
            'label': sample['emotion_idx'],
            'emotion': sample['emotion'],
            'utterance_id': sample['utterance_id']
        }
