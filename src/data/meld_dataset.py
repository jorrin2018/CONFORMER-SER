"""
MELD: tercer corpus. Multimodal EmotionLines Dataset (Poria et al., ACL 2019).

POR QUE ESTE CORPUS. La seccion 11 de docs/REWRITE_NOTES.md dejo un confound
declarado: en CREMA-D la fusion no aporta, y con IEMOCAP + CREMA-D NO se puede
separar "el texto no es informativo" de "CREMA-D es actuado y facil". MELD es el
extremo opuesto medido por la literatura: el texto DOMINA y el audio tiene techo
duro. Cinco fuentes independientes dan la brecha audio-vs-texto en weighted F1:
bcLSTM 39.08/56.44, DialogueRNN 41.79/57.03, EMART 38.05/64.27, wav2vec2-XLS-R-2B
vs DeBERTa 49.3/60.9. El dato mas contundente es `joy` (2,308 ejemplos): audio-only
F1 15.74 frente a 54.31 del texto. Con los tres corpus el eje queda cerrado:

    CREMA-D   texto nulo por construccion, audio suficiente   -> la fusion NO aporta
    IEMOCAP   texto informativo, audio insuficiente           -> la fusion aporta +6.49
    MELD      texto dominante, audio con techo                -> la fusion aporta poco

DECISIONES DECLARADAS. Todas verificadas sobre la descarga real el 10/10/2026 con
scripts/check_meld.py; el detalle completo esta en REWRITE_NOTES seccion 12.

1. EL SPLIT OFICIAL NO ES INDEPENDIENTE DE HABLANTE, y hay que decirlo. Los seis
   protagonistas de Friends estan en los tres splits y son el 82-86 % de las
   utterances de cada uno; 69 de los 100 hablantes de test tambien estan en train.
   El paper nunca usa el termino "speaker-independent": el split lo heredo de
   EmotionLines, que dividio por DIALOGOS sin restriccion de hablante, y como todos
   los dialogos salen de Friends, un split por dialogos no PUEDE ser independiente
   de hablante.
   Se usa el split oficial igualmente, porque nuestra comparacion es PAREADA
   (trimodal vs audio-only sobre el mismo split): la dependencia de hablante es un
   confound COMPARTIDO por las dos condiciones y se cancela en la diferencia. Lo
   que queda invalido es afirmar que el numero absoluto generaliza a hablantes
   nuevos. Si un revisor pregunta: no, y es una propiedad del benchmark.

2. "SESSION" AQUI SIGNIFICA SPLIT, no sesion ni grupo de actores: 1=train, 2=dev,
   3=test. MELD trae un reparto fijo, asi que NO hay LOSO: se corre con
   `--protocol fixed` (un solo fold) y la variabilidad se mide con varias semillas,
   usando la banda de +-0.86 pp de la seccion 7 como criterio de decision.

3. EL AUDIO SE PRE-EXTRAE, no se lee del mp4. MELD no distribuye audio. El
   extractor es scripts/extract_meld_audio.py y escribe
   <raiz>/audio_16k/{train,dev,test}/diaX_uttY.wav a 16 kHz mono. Hace falta
   correrlo ANTES; este cargador falla con un mensaje explicito si no esta.

4. EN TEST, PARA 131 UTTERANCES EL FICHERO CORRECTO NO ES EL QUE DICE EL NOMBRE.
   Hallazgo propio, no publicado en ninguna parte. El directorio de test trae 132
   ficheros `final_videos_testdiaX_uttY.mp4` ADEMAS del `diaX_uttY.mp4`; arbitrando
   con la duracion del CSV, 131 de 131 coinciden al milisegundo con el prefijado y
   0 con el normal, y el normal esta COMPARTIDO entre utterances vecinas. Como el
   read_meld.py oficial construye el nombre desde el CSV, todo pipeline publicado
   lee el corte equivocado en el 5.0 % del test. El extractor lee el prefijado.

5. SE DESCARTAN DOS UTTERANCES de 13,708: `train/dia125_utt3` (mp4 corrupto, issue
   #39 del repo oficial, nunca corregido) y `dev/dia110_utt7` (la fila existe en el
   CSV y el clip NO esta en el release; varias fuentes derivadas reportan "un clip
   de train corrupto y uno de dev ausente" pero NINGUNA lo identifica -- es este).
   Las dos son `neutral`. Quedan 13,706 utterances, que es exactamente lo que
   contiene el dataset derivado `ajyy/MELD_audio` (9,988/1,108/2,610): confirmacion
   independiente de la auditoria.

6. EL `utterance_id` LLEVA EL SPLIT DELANTE. `Dialogue_ID` reinicia en 0 en cada
   split, asi que `dia125_utt3` existe en train Y en test siendo utterances
   distintas (Ross/neutral vs Rachel/neutral). Sin el prefijo, cualquier estructura
   indexada por utterance_id -- el cache de SpeechCue, entre otras -- colisionaria
   en silencio. El id es `train_dia125_utt3`.

7. LA TRANSCRIPCION SE DEJA VERBATIM. Son subtitulos humanos, no ASR. Nueve filas
   de 13,706 contienen acotaciones entre parentesis (p. ej. dia309_utt0 incluye
   "She doesn't hear him and keeps running..."). No se limpian: es lo que usa la
   literatura y son nueve. Se REPORTA el conteo, no se esconde.
   OJO PARA EL ARTICULO: disponer de transcripcion de oro en test es una ventaja
   que no existiria en despliegue. En MELD, DeBERTa saca 60.9 WF1 con transcripcion
   manual y 51.5 con Whisper (arXiv 2509.05634): 9.4 puntos de regalo.

8. 169 FILAS TIENEN HABLANTE NO INDIVIDUAL (`All`, `Both`, `Woman`, `Man`,
   `Director`/`The Director` sin normalizar, y conjunciones tipo `Ross and Rachel`).
   Esos clips contienen mas de una voz. NO se descartan por defecto, porque
   descartarlos rompe la comparabilidad con la tabla publicada; se reporta el
   conteo y existe `descartar_hablante_grupal` para la variante. Declararlo.

9. METRICA. `neutral` es el 48 % del test: la exactitud es casi inutil (una linea
   base de clase mayoritaria da ~48 %). La literatura reporta **weighted F1** y el
   paper original no reporta exactitud. Hay que dar weighted F1, **macro-F1 y F1
   por clase**. Con 4 clases el numero NO es comparable con la literatura de MELD,
   que es de 7 clases: decirlo en el pie de tabla.

10. LOS CSV TRAEN MOJIBAKE DE CP1252 Y HAY QUE ARREGLARLO, NO ES COSMETICO.
    `Utterance` contiene U+0092 donde deberia haber un apostrofo tipografico, mas
    U+0085 (...), U+0097 (--), U+0091, U+0093, U+0094, U+0096. Son 5,386
    ocurrencias en 3,741 utterances: el **27-28 % del corpus**, en los tres splits
    por igual (train 26.94 %, dev 27.77 %, test 28.43 %). El patron es el clasico:
    el fichero original estaba en cp1252, alguien lo decodifico como latin-1 y lo
    re-codifico como UTF-8, asi que el UTF-8 es valido y contiene caracteres de
    control C1.
    POR QUE IMPORTA, y mucho: el `BasicTokenizer` de BERT **borra** los caracteres
    de categoria C antes de segmentar. Sin arreglarlo, "company's" llega al modelo
    como "companys", "must've" como "mustve" y "don't" como "dont". Es decir: en
    una cuarta parte del corpus, TODAS las contracciones quedan deformadas. Y la
    rama de texto es precisamente lo que este articulo mide, asi que el sesgo va en
    la direccion de SUBESTIMAR el aporte de la fusion en MELD -- justo hacia la
    conclusion equivocada sobre la disociacion de la seccion 11.
    Se arregla en el cargador con `normalizar_texto()`, remapeando U+0080-U+009F a
    traves de cp1252. Se REPORTA el conteo en cada construccion.
    VERIFICADO QUE ESTO NO AFECTA A IEMOCAP: sus 151 ficheros de transcripcion son
    UTF-8 valido, usan el apostrofo ASCII (8,160 en una muestra de 200 ficheros) y
    tienen UN solo caracter de control en 10,239 lineas (un tabulador). CREMA-D
    tampoco, porque su texto sale de la tabla FRASES. El defecto es solo de MELD.

LICENCIA: el repositorio oficial (declare-lab/MELD) es GPL-3.0 y el material de
origen son episodios de Friends. Uso de investigacion y publicar resultados, sin
problema citando a Poria et al. 2019; NO redistribuir los medios.

Estructura esperada en disco:

    MELD/
        MELD-RAW/MELD.Raw/
            train/train_sent_emo.csv   train/train_splits/
            dev_sent_emo.csv           dev/dev_splits_complete/
            test_sent_emo.csv          test/output_repeated_splits_test/
        audio_16k/{train,dev,test}/diaX_uttY.wav   <- lo crea el extractor
"""

import csv
from pathlib import Path
from typing import Dict, List

from src.data.base_dataset import BaseSERDataset

#: numero de "session" -> nombre del split oficial. Ver decision 2.
SPLIT_DE_SESSION = {1: 'train', 2: 'dev', 3: 'test'}

#: split -> (csv relativo a MELD.Raw, directorio de medios relativo a MELD.Raw)
RUTAS_SPLIT = {
    'train': ('train/train_sent_emo.csv', 'train/train_splits'),
    'dev':   ('dev_sent_emo.csv',         'dev/dev_splits_complete'),
    'test':  ('test_sent_emo.csv',        'test/output_repeated_splits_test'),
}

#: Etiqueta del CSV (en MINUSCULAS; el README las capitaliza, el CSV no) -> nombre
#: de emocion del proyecto. Cubre las siete; el config elige cuales se usan.
MAPA_EMOCION = {
    'anger':    'angry',
    'joy':      'happy',
    'sadness':  'sad',
    'neutral':  'neutral',
    'disgust':  'disgusted',
    'fear':     'fearful',
    'surprise': 'surprised',
}

#: Utterances sin medio utilizable. Ver decision 5.
DESCARTADOS = {
    ('train', 'dia125_utt3'),   # mp4 corrupto, issue #39
    ('dev',   'dia110_utt7'),   # fila en el CSV, clip ausente del release
}

#: Etiquetas de hablante que NO son un individuo. Ver decision 8.
HABLANTE_NO_INDIVIDUAL = {'All', 'Both', 'Woman', 'Man', 'Guy', 'Girl',
                          'Director', 'The Director', 'Others'}


def normalizar_texto(txt: str) -> tuple:
    """Deshace el mojibake de cp1252 de las transcripciones. Ver decision 10.

    Devuelve (texto_arreglado, n_caracteres_remapeados). Solo toca el rango
    U+0080-U+009F, que son caracteres de control C1 y nunca aparecen de verdad en
    texto; cualquier otro caracter se deja intacto, asi que no puede romper
    puntuacion legitima ni texto no latino. Los cinco bytes que cp1252 no define
    (0x81, 0x8D, 0x8F, 0x90, 0x9D) se dejan como estan.
    """
    if not any('\u0080' <= c <= '\u009f' for c in txt):
        return txt, 0
    salida, n = [], 0
    for c in txt:
        if '\u0080' <= c <= '\u009f':
            try:
                salida.append(bytes([ord(c)]).decode('cp1252'))
                n += 1
                continue
            except UnicodeDecodeError:
                pass  # byte no definido en cp1252: se deja tal cual
        salida.append(c)
    return ''.join(salida), n


def es_hablante_individual(nombre: str) -> bool:
    """False para `All`, `Woman`, `Ross and Rachel`... Ver decision 8."""
    return not (nombre in HABLANTE_NO_INDIVIDUAL or ' and ' in nombre)


def raiz_meld(data_path) -> Path:
    """Acepta .../MELD, .../MELD/MELD-RAW o .../MELD-RAW/MELD.Raw."""
    base = Path(data_path)
    for candidata in (base, base / 'MELD.Raw', base / 'MELD-RAW' / 'MELD.Raw'):
        if (candidata / 'test_sent_emo.csv').is_file():
            return candidata
    raise FileNotFoundError(
        f'No encuentro test_sent_emo.csv bajo {base}. `dataset.data_path` tiene que '
        f'apuntar a la carpeta MELD que contiene MELD-RAW/MELD.Raw.'
    )


class MELDDataset(BaseSERDataset):
    """MELD con split fijo, audio pre-extraido y el fichero correcto en test."""

    CORPUS = 'MELD'

    #: Contrato de 7 clases: filas del CSV menos las dos utterances sin medio.
    REF_7CLASS_POR_SPLIT = {'train': 9988, 'dev': 1108, 'test': 2610}
    #: Contrato de 4 clases (anger/joy/sadness/neutral), ya descontadas las dos.
    #: Las dos descartadas son `neutral`, por eso train y dev pierden una cada uno.
    REF_4CLASS_POR_SPLIT = {'train': 8244, 'dev': 896, 'test': 2211}
    REF_4CLASS_POR_EMOCION = {
        'train':   {'angry': 1109, 'happy': 1743, 'sad': 683, 'neutral': 4709},
        'dev':     {'angry':  153, 'happy':  163, 'sad': 111, 'neutral':  469},
        'test':    {'angry':  345, 'happy':  402, 'sad': 208, 'neutral': 1256},
    }
    #: Filas con hablante no individual, sobre las 13,706 utilizables.
    REF_HABLANTE_GRUPAL = 169

    #: Si True, se descartan las filas de hablante no individual. Ver decision 8.
    descartar_hablante_grupal = False

    def _collect_records(self) -> List[Dict]:
        """Lee los CSV de los splits pedidos, en orden determinista.

        Itera SIEMPRE sobre las filas del CSV, nunca sobre el listado de ficheros ni
        sobre range(n_utterances): `Utterance_ID` tiene huecos (72 dialogos en train,
        2 en dev, 2 en test) y hay 9 clips cortados cuya anotacion se descarto, asi
        que ambos atajos producirian un corpus distinto del anotado.
        """
        raiz = raiz_meld(self.data_path)
        dir_audio = raiz.parent.parent / 'audio_16k'
        if not dir_audio.is_dir():
            raise FileNotFoundError(
                f'No encuentro {dir_audio}. MELD no distribuye audio: hay que '
                f'extraerlo de los .mp4 antes de entrenar.\n'
                f'  python scripts/extract_meld_audio.py --data-path "{Path(self.data_path)}"'
            )

        splits = []
        for s in self.sessions:
            nombre = SPLIT_DE_SESSION.get(int(s))
            if nombre is None:
                raise ValueError(
                    f'session {s!r} no existe en MELD. El reparto es fijo: '
                    f'{SPLIT_DE_SESSION}. MELD no tiene LOSO; usar --protocol fixed.')
            splits.append((int(s), nombre))

        data: List[Dict] = []
        self._descartados_sin_medio = 0
        self._descartados_sin_wav: List[str] = []
        self._descartados_grupal = 0
        self._con_acotacion = 0
        self._hablantes_grupales: set = set()
        self._mojibake_utterances = 0
        self._mojibake_caracteres = 0

        for num_sesion, split in sorted(splits):
            rel_csv, _ = RUTAS_SPLIT[split]
            with open(raiz / rel_csv, encoding='utf-8') as fh:
                filas = list(csv.DictReader(fh))

            for fila in filas:
                uid_corto = f"dia{fila['Dialogue_ID']}_utt{fila['Utterance_ID']}"
                if (split, uid_corto) in DESCARTADOS:
                    self._descartados_sin_medio += 1
                    continue

                hablante = (fila['Speaker'] or '').strip()
                if not es_hablante_individual(hablante):
                    self._hablantes_grupales.add(hablante)
                    if self.descartar_hablante_grupal:
                        self._descartados_grupal += 1
                        continue

                etiqueta = (fila['Emotion'] or '').strip().lower()
                emocion = MAPA_EMOCION.get(etiqueta)
                if emocion is None:
                    raise RuntimeError(
                        f'Emocion desconocida {etiqueta!r} en {split}/{uid_corto}. '
                        f'MELD tiene siete y MAPA_EMOCION las cubre; si aparece otra, '
                        f'el CSV no es el oficial.')
                if emocion not in self.emotions:
                    continue

                wav = dir_audio / split / f'{uid_corto}.wav'
                if not wav.is_file():
                    if len(self._descartados_sin_wav) < 10:
                        self._descartados_sin_wav.append(f'{split}/{uid_corto}')
                    continue

                texto, n_remapeados = normalizar_texto(fila['Utterance'])
                if n_remapeados:
                    self._mojibake_utterances += 1
                    self._mojibake_caracteres += n_remapeados
                if '(' in texto:
                    self._con_acotacion += 1

                data.append({
                    'audio_path': str(wav),
                    'transcription': texto,
                    'emotion': emocion,
                    'emotion_idx': self.emotion2idx[emocion],
                    # Prefijo de split obligatorio: Dialogue_ID reinicia en cada
                    # split y dia125_utt3 existe en train Y en test. Ver decision 6.
                    'utterance_id': f'{split}_{uid_corto}',
                    'session': num_sesion,
                    'speaker': hablante,
                })
        return data

    def speaker_of(self, record: Dict) -> str:
        """El hablante es el personaje. Sin esto se usaria el parser de IEMOCAP.

        No es un ID limpio (ver decision 8) y los personajes se repiten entre
        splits a proposito: eso ES la dependencia de hablante del benchmark.
        """
        return record['speaker']

    def _assert_integrity(self, data: List[Dict]) -> None:
        """Contrato de MELD. Sin esto, el corpus puede mentir durante meses."""
        por_sesion: Dict[int, int] = {}
        por_sesion_emocion: Dict[int, Dict[str, int]] = {}
        for d in data:
            s = int(d['session'])
            por_sesion[s] = por_sesion.get(s, 0) + 1
            por_sesion_emocion.setdefault(s, {})
            por_sesion_emocion[s][d['emotion']] = \
                por_sesion_emocion[s].get(d['emotion'], 0) + 1

        nombres = {s: SPLIT_DE_SESSION[s] for s in sorted(por_sesion)}
        hablantes = sorted({self.speaker_of(d) for d in data})
        print(f'[{self.CORPUS}] splits={nombres} total={len(data)} '
              f'hablantes={len(hablantes)} '
              f'descartados(sin_medio={self._descartados_sin_medio}, '
              f'hablante_grupal={self._descartados_grupal})')
        for s in sorted(por_sesion_emocion):
            print(f'[{self.CORPUS}]   {SPLIT_DE_SESSION[s]:5} {por_sesion[s]:5}  '
                  + '  '.join(f'{k}={v}' for k, v in
                              sorted(por_sesion_emocion[s].items())))
        if self._mojibake_utterances:
            print(f'[{self.CORPUS}] mojibake cp1252 corregido: '
                  f'{self._mojibake_caracteres} caracteres en '
                  f'{self._mojibake_utterances} utterances '
                  f'({100 * self._mojibake_utterances / max(1, len(data)):.1f} %). '
                  f'Sin esto, BERT borraria el caracter y "company\u2019s" llegaria '
                  f'como "companys". Ver decision 10.')
        if self._con_acotacion:
            print(f'[{self.CORPUS}] filas con acotacion entre parentesis en la '
                  f'transcripcion, dejadas verbatim: {self._con_acotacion}')
        if self._hablantes_grupales:
            print(f'[{self.CORPUS}] etiquetas de hablante NO individual presentes '
                  f'({len(self._hablantes_grupales)} distintas, '
                  f'{"descartadas" if self.descartar_hablante_grupal else "CONSERVADAS"}): '
                  f'{sorted(self._hablantes_grupales)[:8]}')

        if not self.strict_integrity:
            return

        if not data:
            raise RuntimeError(
                f'MELD: 0 utterances para las sessions {self.sessions}. O data_path '
                f'no apunta a la carpeta MELD, o falta correr '
                f'scripts/extract_meld_audio.py.')
        if self._descartados_sin_wav:
            raise RuntimeError(
                f'MELD: hay filas anotadas sin su .wav extraido. La extraccion esta '
                f'incompleta: vuelve a correr scripts/extract_meld_audio.py hasta que '
                f'diga "Nada que hacer".\n'
                f'  Faltan (primeros 10): {self._descartados_sin_wav}')

        # El contrato se comprueba por split, porque cada uno tiene su cifra y el
        # runner los pide por separado (train=[1], val=[2], test=[3]).
        cuatro_clases = set(self.emotions) == {'angry', 'happy', 'sad', 'neutral'}
        siete_clases = set(self.emotions) == set(MAPA_EMOCION.values())
        if self.descartar_hablante_grupal or not (cuatro_clases or siete_clases):
            return  # contrato no definido para subconjuntos arbitrarios

        for s, n in sorted(por_sesion.items()):
            split = SPLIT_DE_SESSION[s]
            esperado = (self.REF_4CLASS_POR_SPLIT if cuatro_clases
                        else self.REF_7CLASS_POR_SPLIT)[split]
            if n != esperado:
                raise RuntimeError(
                    f'MELD: split {split} cargo {n} utterances y el contrato dice '
                    f'{esperado} ({"4" if cuatro_clases else "7"} clases, ya '
                    f'descontadas las 2 sin medio). por emocion='
                    f'{por_sesion_emocion[s]}')
            if cuatro_clases:
                ref = self.REF_4CLASS_POR_EMOCION[split]
                if por_sesion_emocion[s] != ref:
                    raise RuntimeError(
                        f'MELD: distribucion de {split} {por_sesion_emocion[s]}, '
                        f'esperada {ref}.')
