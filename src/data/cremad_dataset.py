"""
CREMA-D: cargador para el segundo corpus.

POR QUE ESTE CORPUS, y no es relleno. CREMA-D tiene **12 frases fijas**: los 91 actores
dicen las mismas doce oraciones en las seis emociones. El texto es por tanto
DEMOSTRABLEMENTE NO INFORMATIVO sobre la emocion -- no "probablemente poco util", sino
ortogonal por diseno. Eso convierte la caida de trimodal a audio+vision en una medida
directa de **cuanto depende la fusion del contenido lexico**, que es un experimento
controlado que ningun corpus espontaneo (IEMOCAP, MSP-IMPROV) permite montar.

Por eso la transcripcion NO sale de un ASR ni de un fichero: sale de la tabla FRASES,
que es la verdad por construccion.

DECISIONES DECLARADAS. Todas verificadas el 23/09/2026 contra el repositorio
(https://github.com/CheyneyComputerScience/CREMA-D) y el articulo de Cao et al. 2014
(IEEE TAFFC 5(4), 377-390):

1. ETIQUETA = LA INTENCIONADA (campo 3 del nombre de fichero), no la percibida.
   CREMA-D trae dos verdades posibles y NO son intercambiables: la etiqueta percibida
   por oyentes solo-audio (`VoiceVote` en processedResults/summaryTable.csv) coincide
   con la intencionada en el **41.6 %** de los clips. Y para 4 clases `VoiceVote` es
   inservible como objetivo: el **70 %** de los clips se perciben como neutral
   (NEU 3,897 vs HAP 353). Usar la intencionada es ademas lo que hace el cargador de
   TensorFlow Datasets, que es por donde pasa la mayoria de la literatura. HAY QUE
   DECIRLO EN EL ARTICULO, con el 41.6 % como justificacion de que la eleccion no es
   trivial.

2. 4 CLASES = ANG / HAP / SAD / NEU -> angry / happy / sad / neutral.
   **4,900 clips** (1,271 x 3 + 1,087). No hay fusion excited->happy: CREMA-D no tiene
   excited. Neutral es mas pequeno porque IEO-neutral tiene un clip donde las otras
   cinco emociones tienen tres (LO/MD/HI). Desbalance leve (NEU 22.2 % frente a 25.9 %);
   reportar UA y macro-F1, no solo exactitud.

3. LOS FOLDS SON GRUPOS DE ACTORES, no sesiones. CREMA-D no tiene sesiones diadicas.
   Los 91 actores se reparten en 5 grupos disjuntos **estratificados por sexo**, con
   semilla fija e independiente de config['seed'], para que los cinco folds sean una
   particion de verdad. Asi el protocolo sigue siendo independiente de hablante y toda
   la maquinaria de LOSO de run_ablation.py funciona sin tocarla: `sessions=[k]` es
   "el grupo de actores k".
   NO existe una particion canonica de CREMA-D en la literatura -- cada trabajo se
   inventa la suya -- asi que esta hay que declararla, no presentarla como estandar.

4. NO HAY LINEA BASE COMPARABLE DE 4 CLASES EN EL DOMINIO. Lo unico publicado que se
   pudo verificar para 4 clases es SELM (Interspeech 2024), y es **fuera de dominio**
   (42.79 UA, CREMA-D nunca en entrenamiento): no es comparable. Los numeros de 6
   clases que circulan (SepTr 70.47 %, AuxFormer 71.3 F1, VAVL 0.779 F1) NO son
   comparables con uno de 4 clases, y ademas difieren entre si en que etiqueta usan y
   en si la particion es independiente de hablante -- SepTr ni siquiera lo dice.
   **Aqui se esta fijando una linea base propia, y el articulo tiene que decirlo asi.**

5. SE DESCARTA `1076_MTI_SAD_XX`: son 2.57 s de silencio digital exacto (pico 0, RMS 0),
   con tamano de fichero normal, asi que ninguna comprobacion de tamano lo detecta. Es
   1 clip de 4,900. Declararlo.

6. EL DISENO NO ESTA COMPLETO: faltan 20 clips por problemas tecnicos de extraccion
   (los actores 1008/WSI, 1009/MTI y 1019/ITH enteros, mas 1002/TIE_NEU y 1076/ITS_NEU).
   Por eso no se puede afirmar 82 clips por actor en ningun sitio.

LICENCIA: Open Database License 1.0. Uso experimental y publicar resultados, sin
problema, citando a Cao et al. 2014. Ojo con redistribuir una base derivada (un
subconjunto curado, una tabla de caracteristicas): la ODbL es compartir-igual para
bases de datos.

Estructura esperada en disco:

    CREMA-D/
        AudioWAV/                 7,442 .wav a 16 kHz mono 16 bits
        VideoDemographics.csv     ActorID, Age, Sex, Race, Ethnicity
        processedResults/         (no se usa: aqui se etiqueta por nombre de fichero)
"""

import csv
import os
import random as _rnd
import re
import torch
import torchaudio
import gc
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional

from src.data.base_dataset import BaseSERDataset

#: Las doce frases, verbatim del README del repositorio (apostrofos ASCII).
#: Esta tabla ES la transcripcion: no hay ASR ni fichero de texto en CREMA-D.
FRASES = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}

#: Codigo del nombre de fichero -> nombre de emocion del proyecto.
MAPA_EMOCION = {
    "ANG": "angry",
    "HAP": "happy",
    "SAD": "sad",
    "NEU": "neutral",
    "DIS": "disgusted",
    "FEA": "fearful",
}

#: Silencio digital verificado. Ver decision 5 en el docstring.
DESCARTADOS = {"1076_MTI_SAD_XX"}

PATRON = re.compile(r"^(\d{4})_([A-Z]{3})_([A-Z]{3})_(LO|MD|HI|XX)$")

N_GRUPOS = 5
#: Semilla del reparto de actores en grupos. FIJA y ajena a config['seed'], para que
#: los cinco folds sean una particion y no cinco muestreos distintos.
SEMILLA_GRUPOS = 20260923


def grupos_de_actores(data_path, n_grupos: int = N_GRUPOS) -> Dict[str, int]:
    """Reparte los 91 actores en `n_grupos` grupos disjuntos, estratificados por sexo.

    Devuelve {actor_id: numero_de_grupo}, con los grupos numerados desde 1 para que
    encajen con el parametro `sessions` que usa el resto del repo.

    Estratifica por sexo a proposito: con 91 actores y 5 grupos, un reparto al azar sin
    estratificar puede dejar un fold con 13 hombres y 5 mujeres, y esa varianza acaba
    en la sigma entre folds sin que nadie sepa de donde sale.
    """
    demo = Path(data_path) / "VideoDemographics.csv"
    if not demo.is_file():
        raise FileNotFoundError(
            f"Falta {demo}. Hace falta para repartir los actores en folds "
            f"estratificando por sexo. Viene en la raiz del repositorio de CREMA-D."
        )
    por_sexo: Dict[str, List[str]] = {}
    with open(demo, newline="", encoding="utf-8-sig") as f:
        for fila in csv.DictReader(f):
            actor = str(fila["ActorID"]).strip()
            por_sexo.setdefault(fila["Sex"].strip(), []).append(actor)

    asignacion: Dict[str, int] = {}
    rng = _rnd.Random(SEMILLA_GRUPOS)
    for sexo in sorted(por_sexo):
        lista = sorted(por_sexo[sexo])
        rng.shuffle(lista)
        for i, actor in enumerate(lista):
            asignacion[actor] = i % n_grupos + 1
    return asignacion


class CremaDDataset(BaseSERDataset):
    """CREMA-D con etiqueta intencionada, folds por grupo de actores y texto fijo."""

    CORPUS = "CREMA-D"

    #: Contrato de 4 clases, medido sobre el repositorio el 23/09/2026.
    #: 4,900 clips menos el silencioso = 4,899.
    REF_4CLASS_TOTAL = 4899
    REF_4CLASS_POR_EMOCION = {"angry": 1271, "happy": 1271, "sad": 1270, "neutral": 1087}
    REF_N_ACTORES = 91

    def _collect_records(self) -> List[Dict]:
        """Enumera los .wav de los grupos de actores pedidos, en orden determinista."""
        raiz = Path(self.data_path)
        audio_dir = raiz / "AudioWAV"
        if not audio_dir.is_dir():
            raise FileNotFoundError(
                f"No encuentro {audio_dir}. `dataset.data_path` tiene que apuntar a la "
                f"raiz del repositorio de CREMA-D, la carpeta que contiene AudioWAV/ y "
                f"VideoDemographics.csv."
            )

        self.grupo_de_actor = grupos_de_actores(raiz, N_GRUPOS)
        pedidos = {int(s) for s in self.sessions}

        data: List[Dict] = []
        self._descartados_silencio = 0
        self._descartados_patron = 0
        for wav in sorted(audio_dir.glob("*.wav")):
            uid = wav.stem
            m = PATRON.match(uid)
            if not m:
                self._descartados_patron += 1
                continue
            actor, cod_frase, cod_emo, _intensidad = m.groups()

            if uid in DESCARTADOS:
                self._descartados_silencio += 1
                continue

            grupo = self.grupo_de_actor.get(actor)
            if grupo is None or grupo not in pedidos:
                continue

            emocion = MAPA_EMOCION.get(cod_emo)
            if emocion is None or emocion not in self.emotions:
                continue

            frase = FRASES.get(cod_frase)
            if frase is None:
                raise RuntimeError(
                    f"Codigo de frase desconocido {cod_frase!r} en {uid}. La tabla "
                    f"FRASES tiene que cubrir las doce; revisar antes de seguir."
                )

            data.append({
                "audio_path": str(wav),
                "transcription": frase,
                "emotion": emocion,
                "emotion_idx": self.emotion2idx[emocion],
                "utterance_id": uid,
                "session": grupo,
            })
        return data

    def speaker_of(self, record: Dict) -> str:
        """El hablante es el actor. Sin esto se usaria el parser de IEMOCAP."""
        return record["utterance_id"].split("_", 1)[0]

    def _assert_integrity(self, data: List[Dict]) -> None:
        """Contrato de CREMA-D. Sin esto, el corpus puede mentir durante meses."""
        actores = sorted({self.speaker_of(d) for d in data})
        por_emocion: Dict[str, int] = {}
        for d in data:
            por_emocion[d["emotion"]] = por_emocion.get(d["emotion"], 0) + 1
        frases = sorted({d["transcription"] for d in data})
        print(f"[{self.CORPUS}] grupos={sorted(self.sessions)} total={len(data)} "
              f"actores={len(actores)} frases={len(frases)} "
              f"descartados(silencio={self._descartados_silencio}, "
              f"patron={self._descartados_patron})")
        print(f"[{self.CORPUS}] por emocion: "
              f"{ {k: por_emocion[k] for k in sorted(por_emocion)} }")

        if not self.strict_integrity:
            return

        if not data:
            raise RuntimeError(
                f"CREMA-D: 0 clips para los grupos {self.sessions}. O data_path no "
                f"apunta al repositorio, o los .wav no se han descargado (git lfs)."
            )
        if self._descartados_patron:
            raise RuntimeError(
                f"CREMA-D: {self._descartados_patron} ficheros no casan con el patron "
                f"ACTOR_FRASE_EMOCION_INTENSIDAD. Los 7,442 del repositorio lo casan "
                f"sin excepcion, asi que la carpeta tiene algo que no es CREMA-D."
            )
        if len(frases) != len(FRASES):
            raise RuntimeError(
                f"CREMA-D: se cargaron {len(frases)} frases distintas y deberian ser "
                f"{len(FRASES)}. El experimento se apoya en que las doce frases sean "
                f"comunes a todas las emociones: si falta alguna, el texto deja de ser "
                f"ortogonal a la emocion y el resultado no mide lo que dice medir."
            )

        # El contrato completo solo se puede comprobar con los cinco grupos a la vez.
        completo = set(int(s) for s in self.sessions) == set(range(1, N_GRUPOS + 1))
        if completo:
            if len(actores) != self.REF_N_ACTORES:
                raise RuntimeError(
                    f"CREMA-D: {len(actores)} actores, esperados "
                    f"{self.REF_N_ACTORES} (1001-1091).")
            esperado = {k: v for k, v in self.REF_4CLASS_POR_EMOCION.items()
                        if k in self.emotions}
            if set(self.emotions) == set(self.REF_4CLASS_POR_EMOCION):
                if len(data) != self.REF_4CLASS_TOTAL:
                    raise RuntimeError(
                        f"CREMA-D: {len(data)} clips de 4 clases, esperados "
                        f"{self.REF_4CLASS_TOTAL} (4,900 menos el clip silencioso). "
                        f"por emocion={por_emocion}")
                if por_emocion != esperado:
                    raise RuntimeError(
                        f"CREMA-D: distribucion por emocion {por_emocion}, esperada "
                        f"{esperado}.")
