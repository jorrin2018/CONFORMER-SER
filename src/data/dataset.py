"""
IEMOCAP Dataset implementation for multimodal emotion recognition.

Lo generico (audio, texto, pesos de clase, filtro de hablante, subconjunto por indices)
vive en src/data/base_dataset.py desde el refactor del 22/09/2026. Aqui queda solo lo
que es propio de IEMOCAP.
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
from src.data.base_dataset import BaseSERDataset


class IEMOCAPDataset(BaseSERDataset):
    """
    IEMOCAP Dataset for multimodal (audio + text) emotion recognition.

    The dataset structure:
    IEMOCAP_full_release/
        Session1/
            sentences/
                wav/
                    Ses01F_impro01/
                        Ses01F_impro01_F000.wav
                    ...
            dialog/
                EmoEvaluation/
                    Ses01F_impro01.txt
                transcriptions/
                    Ses01F_impro01.txt
    """

    CORPUS = 'IEMOCAP'

    EMOTION_MAP = {
        'ang': 'angry',
        'hap': 'happy',
        # 'exc' NO se resuelve aqui: lo hace _emotion_from_code(), porque depende de
        # si el config pide `excited` como clase propia. Ver ese metodo. Se deja la
        # entrada con el valor historico para que cualquier lector de EMOTION_MAP
        # suelto vea el comportamiento por defecto (4 clases, exc fusionado en happy).
        'exc': 'happy',
        'sad': 'sad',
        'fru': 'frustrated',
        'fea': 'fearful',
        'sur': 'surprised',
        'neu': 'neutral',
        'dis': 'disgusted',
        'oth': 'other',
        'xxx': 'other',  # Unknown mapped to other
    }
    

    def _collect_records(self) -> List[Dict]:
        """Enumera las utterances de las sesiones pedidas, en orden determinista."""
        data = []
        
        for session in self.sessions:
            session_name = f"Session{session}"
            session_path = self.data_path / session_name
            
            if not session_path.exists():
                print(f"Warning: {session_path} does not exist, skipping...")
                continue
            
            # Parse emotion labels
            emotion_labels = self._parse_emotion_labels(session_path)
            
            # Parse transcriptions
            transcriptions = self._parse_transcriptions(session_path)
            
            # Match audio files with labels and transcriptions
            wav_path = session_path / "sentences" / "wav"
            
            for utterance_id, emotion in emotion_labels.items():
                # Map to our emotion categories
                if emotion not in self.emotions:
                    continue
                
                # Find audio file
                audio_file = self._find_audio_file(wav_path, utterance_id)
                if audio_file is None:
                    continue
                
                # Get transcription
                transcription = transcriptions.get(utterance_id, "")
                
                data.append({
                    'audio_path': str(audio_file),
                    'transcription': transcription,
                    'emotion': emotion,
                    'emotion_idx': self.emotion2idx[emotion],
                    'utterance_id': utterance_id,
                    'session': session
                })

        return data

    # IEMOCAP, protocolo de 4 clases (ang / hap+exc / sad / neu).
    # Verificado el 2026-09-10 con scripts/verify_iemocap.py.
    REF_4CLASS_PER_SESSION = {1: 1085, 2: 1023, 3: 1151, 4: 1031, 5: 1241}

    #: IEMOCAP con sus SIETE clases propias, `excited` SEPARADO de `happy`:
    #: frustrated, neutral, angry, sad, excited, happy, surprised. Total 7,487.
    #: Contado el 2026-10-20 directo de los ficheros de EmoEvaluation de las cinco
    #: sesiones. Cuadra con REF_4CLASS_PER_SESSION en el subconjunto de 4 clases
    #: (1103 ang + 595 hap + 1041 exc + 1084 sad + 1708 neu = 5,531), que es la
    #: confirmacion de que el conteo es el mismo que ve el cargador.
    #: NO son las 7 clases de MELD: `disgusted` tiene 2 utterances en todo IEMOCAP y
    #: `fearful` 40, con tres sesiones a cero en disgusted. Un macro-F1 sobre eso no
    #: significa nada, asi que la condicion de 7 clases de IEMOCAP usa SUS clases.
    REF_7CLASS_PER_SESSION = {1: 1390, 2: 1365, 3: 1561, 4: 1531, 5: 1640}

    #: Los dos conjuntos de clases con contrato duro.
    SET_4CLASS = frozenset({'angry', 'happy', 'sad', 'neutral'})
    SET_7CLASS = frozenset({'frustrated', 'neutral', 'angry', 'sad',
                            'excited', 'happy', 'surprised'})

    def _emotion_from_code(self, emotion_code: str) -> str:
        """Codigo de EmoEvaluation -> nombre de emocion del proyecto.

        El unico codigo con resolucion condicional es `exc`. El protocolo de 4 clases
        de IEMOCAP fusiona excitement en happy, y esa fusion es la que produce el
        contrato de 5,531 utterances; pero es tambien una decision que un revisor
        cuestiona con razon, porque infla `happy` de 595 a 1,636.

        Regla: si el config pide `excited` en su lista `emotions`, se devuelve
        `excited` y las dos clases quedan separadas. Si no lo pide -- y ningun config
        anterior al 2026-10-20 lo hace --, se devuelve `happy`, que es el
        comportamiento historico sin un solo cambio.
        """
        if emotion_code == 'exc':
            return 'excited' if 'excited' in self.emotions else 'happy'
        return self.EMOTION_MAP.get(emotion_code, 'other')

    def _assert_integrity(self, data: List[Dict]) -> None:
        """Fail loudly when the corpus is not what the protocol expects.

        Historical reason this exists: _find_audio_file() used to derive the
        dialog folder with utterance_id.split('_')[:3], which is correct for
        scripted ids (4 fields) but returns the whole id for improvised ones
        (3 fields). Every improvised utterance was therefore dropped silently,
        leaving 2,588 of 5,531 utterances - scripted only, with heavy lexical
        overlap between train and test. Nothing warned. This check does.
        """
        n_impro = sum(1 for d in data if 'impro' in d['utterance_id'])
        n_script = len(data) - n_impro
        n_notext = sum(1 for d in data if not d['transcription'].strip())
        print(f"[IEMOCAP] sessions={self.sessions} total={len(data)} "
              f"impro={n_impro} script={n_script} no_transcript={n_notext}")

        if not self.strict_integrity:
            return

        # El contrato de total depende del conjunto de clases pedido. La comprobacion
        # de improvisados de mas abajo NO: vale para cualquier conjunto y es la que
        # atrapa el bug que una vez descarto 2,943 de 5,531 utterances.
        pedidas = frozenset(self.emotions)
        if pedidas == self.SET_4CLASS:
            ref, etiqueta = self.REF_4CLASS_PER_SESSION, '4-class'
        elif pedidas == self.SET_7CLASS:
            ref, etiqueta = self.REF_7CLASS_PER_SESSION, '7-class'
        else:
            ref, etiqueta = {}, 'custom'
        expected = sum(ref[s] for s in self.sessions if s in ref)
        if n_impro == 0:
            raise RuntimeError(
                f"IEMOCAP integrity: 0 improvised dialogs loaded for sessions "
                f"{self.sessions}. Either the corpus copy is scripted-only or "
                f"_find_audio_file() is dropping them. Train and test would "
                f"share scripted text. data_path={self.data_path}"
            )
        if expected and abs(len(data) - expected) > 0.02 * expected:
            raise RuntimeError(
                f"IEMOCAP integrity: loaded {len(data)} {etiqueta} utterances, "
                f"expected ~{expected} for sessions {self.sessions} "
                f"(impro={n_impro}, script={n_script}). "
                f"Run scripts/verify_iemocap.py. data_path={self.data_path}"
            )
        if not expected:
            print(f"[IEMOCAP] conjunto de clases '{etiqueta}' sin contrato de total "
                  f"tabulado: se comprueban los improvisados pero NO el conteo. "
                  f"clases={sorted(pedidas)}")
    
    def _parse_emotion_labels(self, session_path: Path) -> Dict[str, str]:
        """Parse emotion labels from evaluation files."""
        emotion_labels = {}
        eval_path = session_path / "dialog" / "EmoEvaluation"
        
        if not eval_path.exists():
            return emotion_labels
        
        for eval_file in sorted(eval_path.glob("*.txt")):
            # ._foo.txt: resource forks de zips hechos en macOS
            if eval_file.name.startswith("."):
                continue
            with open(eval_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Skip header lines (%), annotator lines (C-, A-), and other non-emotion lines
                    if line.startswith('%') or line.startswith('C-') or line.startswith('A-'):
                        continue
                    
                    # Parse line format: [start - end]\tUtteranceID\tEmotion\t[...]
                    # Example: [6.2901 - 8.2357]  Ses01F_impro01_F000     neu     [2.5000, 2.5000, 2.5000]
                    match = re.match(r'\[[\d.]+\s*-\s*[\d.]+\]\s+(\S+)\s+(\w+)', line)
                    if match:
                        utterance_id = match.group(1)
                        emotion_code = match.group(2).lower()
                        
                        # Map emotion code to emotion name
                        emotion = self._emotion_from_code(emotion_code)
                        emotion_labels[utterance_id] = emotion
        
        return emotion_labels
    
    def _parse_transcriptions(self, session_path: Path) -> Dict[str, str]:
        """Parse transcriptions from transcription files."""
        transcriptions = {}
        trans_path = session_path / "dialog" / "transcriptions"
        
        if not trans_path.exists():
            return transcriptions
        
        for trans_file in sorted(trans_path.glob("*.txt")):
            # ._foo.txt: resource forks de zips hechos en macOS
            if trans_file.name.startswith("."):
                continue
            with open(trans_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse line format: UtteranceID [start-end]: Transcription
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        utterance_id = parts[0].split()[0]
                        transcription = parts[1].strip()
                        transcriptions[utterance_id] = transcription
        
        return transcriptions
    
    def _find_audio_file(self, wav_path: Path, utterance_id: str) -> Optional[Path]:
        """Find audio file for given utterance ID."""
        # Extract dialog ID from utterance ID (e.g., Ses01F_impro01_F000 -> Ses01F_impro01)
        # Ses01F_impro01_F000      -> Ses01F_impro01
        # Ses05F_script01_1_F000   -> Ses05F_script01_1
        # [:3] rompia los impro (3 campos): devolvia el id completo
        # y la carpeta no existia, descartando las 2 943 improvisadas.
        dialog_id = '_'.join(utterance_id.split('_')[:-1])
        
        dialog_path = wav_path / dialog_id
        if not dialog_path.exists():
            return None
        
        audio_file = dialog_path / f"{utterance_id}.wav"
        if audio_file.exists():
            return audio_file
        
        return None

def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching."""
    # Stack tensors
    audio = torch.stack([item['audio'] for item in batch])
    text_input_ids = torch.stack([item['text_input_ids'] for item in batch])
    text_attention_mask = torch.stack([item['text_attention_mask'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    # Keep metadata
    emotions = [item['emotion'] for item in batch]
    utterance_ids = [item['utterance_id'] for item in batch]
    
    return {
        'audio': audio,
        'text_input_ids': text_input_ids,
        'text_attention_mask': text_attention_mask,
        'labels': labels,
        'emotions': emotions,
        'utterance_ids': utterance_ids
    }


def collate_fn_audio_only(batch: List[Dict]) -> Dict:
    """Custom collate function for audio-only batching."""
    # Stack tensors
    audio = torch.stack([item['audio'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    # Keep metadata
    emotions = [item['emotion'] for item in batch]
    utterance_ids = [item['utterance_id'] for item in batch]
    
    return {
        'audio': audio,
        'labels': labels,
        'emotions': emotions,
        'utterance_ids': utterance_ids
    }

