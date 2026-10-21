"""
RAVDESS: cargador para el cuarto corpus, el extremo del eje de texto nulo.

POR QUE ESTE CORPUS. El eje del articulo mide cuanto aporta la fusion segun cuanto
informa el canal lexico, y el extremo "texto nulo" descansa hoy en UN solo corpus
(CREMA-D, -1.19 +- 1.97, 1/5 folds, IC que incluye el cero). Un segundo corpus de
frases fijas convierte ese extremo en una replica en vez de un punto suelto: si
RAVDESS tambien da cero, el +6.49 de IEMOCAP no se explica por una peculiaridad de
CREMA-D.

Y RAVDESS es la version EXTREMA de lo mismo. CREMA-D tiene doce frases; RAVDESS tiene
DOS: "Kids are talking by the door" y "Dogs are sitting by the door". Con dos frases,
la identidad de la frase no puede llevar informacion emocional ni por accidente: no es
que el texto sea poco informativo, es que no hay variacion lexica que correlacionar
con la emocion. Es el control mas limpio que se puede montar.

Por eso la transcripcion NO sale de un ASR ni de un fichero: sale de la tabla FRASES,
que es la verdad por construccion, igual que en CREMA-D.

ESTADO DEL CONTRATO: VERIFICADO el 2026-10-20.
Las cifras de REF_* de abajo se derivaron primero del diseno publicado (Livingstone &
Russo, PLoS ONE 13(5): e0196391, 2018) y despues se CONFIRMARON contra la descarga real
con `scripts/check_ravdess.py --fase todo`. Cuadraron todas sin una sola excepcion:

    1,440 .wav de habla                          (esperado 1,440)
    672 en el subconjunto de 4 clases            (192/192/192/96, exacto)
    diseno factorial completo, 60 ensayos/actor  (cero huecos en las 192 celdas)
    2 frases distintas                           (esperado 2)
    0 nombres fuera del patron de siete campos
    0 clips de neutral con intensidad strong
    48 kHz en los 1,440, sin excepciones
    duracion: min 2.94 s, mediana 3.67, p90 4.14, max 5.27 -- ningun fichero vacio

HALLAZGO PROPIO, no documentado en ninguna fuente: CINCO de los 1,440 ficheros vienen
en DOS CANALES y los otros 1,435 en uno. Es estereo decorativo -- los dos canales son
bit a bit identicos (max|L-R| = 0 exacto en los cinco, comprobado muestra a muestra),
o sea dual-mono. El downmix de BaseSERDataset promedia canales, y la media de x con x
es x, asi que NO SE PIERDE NADA y no hay nada que arreglar. Se registra porque un
corpus con un parametro de audio no uniforme es justo el tipo de cosa que mas tarde
explica una diferencia que nadie sabe de donde sale.

    Actor_01/03-01-02-01-01-02-01.wav   calm       (fuera de las 4 clases)
    Actor_01/03-01-08-01-02-02-01.wav   surprised  (fuera)
    Actor_05/03-01-02-01-02-02-05.wav   calm       (fuera)
    Actor_20/03-01-03-01-02-01-20.wav   happy      <-- el unico DENTRO de las 4 clases
    Actor_20/03-01-06-01-01-02-20.wav   fearful    (fuera)

O sea: 1 de las 672 utterances del eje. `check_ravdess.py --fase audio` los nombra y
comprueba que los canales sigan siendo identicos, para que deje de ser un hallazgo
suelto y sea un guardarrail.

CORROBORACION COLATERAL de la decision 1 (solo habla): en la descarga de Zenodo falta
`Video_Song_Actor_18.zip` -- los ficheros van del 17 al 19 --, que es exactamente lo
esperado si al actor 18 le falta el bloque de canto entero. Una confirmacion
independiente de que el canto no es un conjunto completo.

DECISIONES DECLARADAS. Todas se apoyan en el diseno publicado:

1. SOLO HABLA, NO CANTO. RAVDESS trae dos canales vocales: habla (campo 2 = 01) y canto
   (02). Aqui solo se usa HABLA, por tres motivos que hay que declarar: el canto es un
   modo de produccion distinto (prosodia cantada, duracion impuesta por la melodia),
   solo cubre SEIS emociones (sin disgust ni surprised), y al actor 18 le falta el
   bloque de canto entero. Mezclarlos meteria un confusor de modo de produccion dentro
   del propio corpus.

2. ETIQUETA = LA INTENCIONADA (campo 3 del nombre de fichero). RAVDESS publica tambien
   validacion perceptual (247 jueces), pero por agregado y no como etiqueta por clip
   reutilizable, asi que no hay alternativa practica. A diferencia de CREMA-D, aqui no
   hay un `VoiceVote` por clip que contradiga la intencion: no se puede repetir el
   analisis del 41.6 % de acuerdo. DECLARARLO como limitacion, no presentarlo como que
   la etiqueta sea mas fiable.

3. 4 CLASES = angry / happy / sad / neutral, que son las del eje. Se EXCLUYE `calm`
   (codigo 02), que es propia de RAVDESS y no tiene equivalente en IEMOCAP, CREMA-D ni
   MELD; incluirla rompe la comparabilidad del eje. Se excluyen tambien fearful,
   disgust y surprised por la misma razon que en CREMA-D: el eje se mide con las mismas
   cuatro clases en los cuatro corpus.

4. SE CONSERVAN LAS DOS INTENSIDADES (normal y strong). Consecuencia que hay que
   declarar: `neutral` NO TIENE intensidad strong por diseno, asi que sale a la MITAD
   que las otras tres (96 frente a 192 cada una). Es el mismo tipo de desbalance que en
   CREMA-D pero mas marcado: 14.3 % frente a 28.6 %. Reportar UA y macro-F1, nunca
   exactitud sola.

5. LOS FOLDS SON GRUPOS DE ACTORES, no sesiones. Los 24 actores se reparten en 5 grupos
   disjuntos estratificados por sexo, con semilla fija e independiente de
   config['seed'], igual que en CREMA-D. El sexo se deduce del propio identificador --
   impar = hombre, par = mujer, por diseno del corpus --, asi que no hace falta ningun
   CSV de demografia. 24 no es divisible por 5: los grupos salen de 6/6/4/4/4 actores.
   NO existe particion canonica de RAVDESS en la literatura; esta hay que declararla,
   no presentarla como estandar.

6. EL CORPUS ES PEQUENO, Y ESTO LIMITA LO QUE SE PUEDE AFIRMAR. 672 utterances de 4
   clases, frente a 4,899 de CREMA-D y 5,531 de IEMOCAP: una octava parte. Con cinco
   folds son ~134 utterances de test por fold. La sigma entre folds sera mayor que en
   los otros corpus por puro tamano de muestra, asi que RAVDESS sirve para CONFIRMAR un
   cero, no para medir un efecto pequeno. Decirlo asi en el articulo.

7. AUDIO A 48 kHz. Los otros corpus del repo estan a 16 kHz. `BaseSERDataset` ya
   remuestrea solo cuando `sr != sample_rate`, asi que no hay nada que hacer, pero
   conviene saberlo: cada muestra paga un remuestreo 48k->16k en CPU. Con 672 ficheros
   es irrelevante.

LICENCIA: CC BY-NC-SA 4.0. Uso academico y publicar resultados, sin problema, citando a
Livingstone & Russo 2018. OJO: es NO COMERCIAL y compartir-igual, que es MAS restrictiva
que la ODbL de CREMA-D. No redistribuir derivados sin la misma licencia.

DESCARGA (no esta en disco a fecha de 2026-10-20):
    https://zenodo.org/records/1188976  ->  Audio_Speech_Actors_01-24.zip  (~215 MB)
    Descomprimir en D:/DOCTORADO/DATASETS/RAVDESS/ de forma que quede
    RAVDESS/Audio_Speech_Actors_01-24/Actor_01/... o RAVDESS/Actor_01/...
    (el cargador acepta las dos disposiciones).
    NO hace falta el zip de canto (Audio_Song_...): no se usa. Ver decision 1.

Estructura esperada en disco:

    RAVDESS/
        Actor_01/  03-01-01-01-01-01-01.wav ...      (o bien)
        Audio_Speech_Actors_01-24/Actor_01/ ...
"""

import random as _rnd
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.data.base_dataset import BaseSERDataset

#: Las DOS frases, verbatim del articulo. Esta tabla ES la transcripcion: RAVDESS no
#: trae ficheros de texto. Con solo dos frases, la identidad lexica no puede
#: correlacionar con la emocion: ese es el punto del corpus.
FRASES = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}

#: Campo 3 del nombre de fichero -> nombre de emocion del proyecto.
#: `calm` (02) se mapea pero queda fuera del eje por la decision 3: solo entra si un
#: config la pide explicitamente en `emotions`.
MAPA_EMOCION = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgusted",
    "08": "surprised",
}

#: Campo 2: canal vocal. Solo se acepta habla. Ver decision 1.
CANAL_HABLA = "01"
CANAL_CANTO = "02"

#: Campo 1: modalidad. El zip de habla solo-audio trae 03 (audio-only).
MODALIDAD_SOLO_AUDIO = "03"

#: Nombre de fichero: siete campos de dos digitos separados por guion.
#: Ej. 03-01-05-02-01-02-14 = audio-only, habla, angry, strong, frase 1, rep 2, actor 14.
PATRON = re.compile(
    r"^(?P<modalidad>\d{2})-(?P<canal>\d{2})-(?P<emocion>\d{2})-"
    r"(?P<intensidad>\d{2})-(?P<frase>\d{2})-(?P<repeticion>\d{2})-"
    r"(?P<actor>\d{2})$"
)

#: `neutral` no tiene intensidad strong por diseno. Se comprueba, porque si apareciera
#: significaria que la copia en disco no es RAVDESS tal como se publico.
INTENSIDAD_NORMAL = "01"
INTENSIDAD_STRONG = "02"

#: Los cinco ficheros dual-mono (dos canales bit a bit identicos). Verificado el
#: 2026-10-20; ver el docstring. No se descartan ni se tratan distinto: el downmix de
#: BaseSERDataset ya hace lo correcto. Estan aqui para poder nombrarlos sin volver a
#: recorrer el corpus.
DUAL_MONO = {
    "03-01-02-01-01-02-01",   # Actor_01, calm
    "03-01-08-01-02-02-01",   # Actor_01, surprised
    "03-01-02-01-02-02-05",   # Actor_05, calm
    "03-01-03-01-02-01-20",   # Actor_20, happy -- el unico dentro de las 4 clases
    "03-01-06-01-01-02-20",   # Actor_20, fearful
}

N_GRUPOS = 5
#: Semilla del reparto de actores en grupos. FIJA y ajena a config['seed'], para que los
#: cinco folds sean una particion y no cinco muestreos distintos.
SEMILLA_GRUPOS = 20261020

N_ACTORES = 24


def sexo_de_actor(actor: str) -> str:
    """Sexo del actor a partir de su identificador. Impar = hombre, par = mujer.

    Es parte del diseno publicado de RAVDESS, no una heuristica: los 24 actores se
    numeraron alternando sexo. Por eso aqui no hace falta un CSV de demografia, al
    contrario que en CREMA-D.
    """
    return "M" if int(actor) % 2 == 1 else "F"


def grupos_de_actores(n_grupos: int = N_GRUPOS) -> Dict[str, int]:
    """Reparte los 24 actores en `n_grupos` grupos disjuntos, estratificados por sexo.

    Devuelve {actor_id: numero_de_grupo}, numerados desde 1 para que encajen con el
    parametro `sessions` que usa el resto del repo.

    Estratifica por sexo a proposito, por el mismo motivo que en CREMA-D: un reparto al
    azar con 24 actores puede dejar un fold con 4 hombres y 1 mujer, y esa varianza
    acaba en la sigma entre folds sin que nadie sepa de donde sale.

    24 no es divisible por 5, asi que los grupos salen desiguales (6/6/4/4/4 actores).
    Es inevitable y hay que declararlo: los folds no pesan lo mismo.
    """
    por_sexo: Dict[str, List[str]] = {"M": [], "F": []}
    for i in range(1, N_ACTORES + 1):
        actor = f"{i:02d}"
        por_sexo[sexo_de_actor(actor)].append(actor)

    asignacion: Dict[str, int] = {}
    rng = _rnd.Random(SEMILLA_GRUPOS)
    for sexo in sorted(por_sexo):
        lista = sorted(por_sexo[sexo])
        rng.shuffle(lista)
        for i, actor in enumerate(lista):
            asignacion[actor] = i % n_grupos + 1
    return asignacion


class RavdessDataset(BaseSERDataset):
    """RAVDESS (habla) con etiqueta intencionada, folds por grupo de actores y texto fijo."""

    CORPUS = "RAVDESS"

    # -- Contrato VERIFICADO el 2026-10-20 contra la descarga real. Ver el docstring. --
    #: Habla, solo-audio: 24 actores x 60 ensayos.
    #:   neutral  : 1 intensidad x 2 frases x 2 repeticiones =  4 por actor
    #:   las otras: 2 intensidades x 2 frases x 2 repeticiones = 8 por actor
    #:   4 + 7 x 8 = 60 ; 60 x 24 = 1,440 ficheros en el zip de habla.
    REF_TOTAL_HABLA = 1440
    #: Subconjunto de 4 clases del eje: 192 x 3 + 96 = 672.
    REF_4CLASS_TOTAL = 672
    REF_4CLASS_POR_EMOCION = {"angry": 192, "happy": 192, "sad": 192, "neutral": 96}
    REF_N_ACTORES = N_ACTORES
    #: Las cuatro clases del eje, para saber cuando aplicar el contrato de arriba.
    SET_4CLASS = frozenset({"angry", "happy", "sad", "neutral"})

    def _collect_records(self) -> List[Dict]:
        """Enumera los .wav de habla de los grupos de actores pedidos, en orden determinista."""
        raiz = Path(self.data_path)
        base = self._localizar_raiz_audio(raiz)

        self.grupo_de_actor = grupos_de_actores(N_GRUPOS)
        pedidos = {int(s) for s in self.sessions}

        data: List[Dict] = []
        self._descartados_canto = 0
        self._descartados_patron = 0
        self._descartados_modalidad = 0
        #: Nombres que no casan el patron. Un guardarrail que dice "hay 1 fichero mal"
        #: sin decir CUAL cuesta una ida y vuelta entera; que los nombre.
        self._nombres_raros: List[str] = []
        #: neutral con intensidad strong no existe en RAVDESS. Si aparece, la copia no
        #: es la publicada y hay que saberlo antes de entrenar.
        self._neutral_strong: List[str] = []

        for wav in sorted(base.rglob("*.wav")):
            uid = wav.stem
            m = PATRON.match(uid)
            if not m:
                self._descartados_patron += 1
                if len(self._nombres_raros) < 10:
                    self._nombres_raros.append(wav.name)
                continue

            campos = m.groupdict()

            # Decision 1: canto fuera.
            if campos["canal"] == CANAL_CANTO:
                self._descartados_canto += 1
                continue
            if campos["canal"] != CANAL_HABLA:
                self._descartados_patron += 1
                if len(self._nombres_raros) < 10:
                    self._nombres_raros.append(wav.name)
                continue

            # El zip de habla solo-audio trae modalidad 03. Otra modalidad significa que
            # se ha mezclado el paquete audiovisual, cuyos .wav no son los mismos.
            if campos["modalidad"] != MODALIDAD_SOLO_AUDIO:
                self._descartados_modalidad += 1
                continue

            actor = campos["actor"]
            grupo = self.grupo_de_actor.get(actor)
            if grupo is None or grupo not in pedidos:
                continue

            emocion = MAPA_EMOCION.get(campos["emocion"])
            if emocion is None or emocion not in self.emotions:
                continue

            if emocion == "neutral" and campos["intensidad"] == INTENSIDAD_STRONG:
                self._neutral_strong.append(wav.name)

            frase = FRASES.get(campos["frase"])
            if frase is None:
                raise RuntimeError(
                    f"Codigo de frase desconocido {campos['frase']!r} en {uid}. RAVDESS "
                    f"tiene DOS frases (01 y 02) y la tabla FRASES las cubre; si aparece "
                    f"otra, la carpeta tiene algo que no es RAVDESS."
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

    @staticmethod
    def _localizar_raiz_audio(raiz: Path) -> Path:
        """Acepta las dos disposiciones habituales de la descarga.

        El zip de Zenodo se descomprime unas veces como RAVDESS/Actor_01/... y otras
        como RAVDESS/Audio_Speech_Actors_01-24/Actor_01/.... Aceptar las dos evita el
        fallo mas tonto posible, que es un contrato en rojo por un nivel de carpeta.
        """
        if not raiz.is_dir():
            raise FileNotFoundError(
                f"No encuentro {raiz}. `dataset.data_path` tiene que apuntar a la raiz "
                f"de RAVDESS. Descarga: https://zenodo.org/records/1188976 "
                f"(Audio_Speech_Actors_01-24.zip)."
            )
        candidatos = [raiz, raiz / "Audio_Speech_Actors_01-24"]
        for cand in candidatos:
            if cand.is_dir() and any(cand.glob("Actor_*")):
                return cand
        raise FileNotFoundError(
            f"No encuentro carpetas Actor_* bajo {raiz} ni bajo "
            f"{raiz / 'Audio_Speech_Actors_01-24'}. Descomprime "
            f"Audio_Speech_Actors_01-24.zip ahi. Descarga: "
            f"https://zenodo.org/records/1188976"
        )

    def speaker_of(self, record: Dict) -> str:
        """El hablante es el actor: campo 7 del nombre. Sin esto se usaria el de IEMOCAP."""
        return record["utterance_id"].rsplit("-", 1)[1]

    def _assert_integrity(self, data: List[Dict]) -> None:
        """Contrato de RAVDESS. Sin esto, el corpus puede mentir durante meses."""
        actores = sorted({self.speaker_of(d) for d in data})
        por_emocion: Dict[str, int] = {}
        for d in data:
            por_emocion[d["emotion"]] = por_emocion.get(d["emotion"], 0) + 1
        frases = sorted({d["transcription"] for d in data})
        sexos: Dict[str, int] = {}
        for a in actores:
            s = sexo_de_actor(a)
            sexos[s] = sexos.get(s, 0) + 1

        print(f"[{self.CORPUS}] grupos={sorted(self.sessions)} total={len(data)} "
              f"actores={len(actores)} (M={sexos.get('M', 0)}, F={sexos.get('F', 0)}) "
              f"frases={len(frases)} descartados(canto={self._descartados_canto}, "
              f"modalidad={self._descartados_modalidad}, "
              f"patron={self._descartados_patron})")
        print(f"[{self.CORPUS}] por emocion: "
              f"{ {k: por_emocion[k] for k in sorted(por_emocion)} }")
        if self._neutral_strong:
            print(f"[{self.CORPUS}] AVISO: neutral con intensidad strong, que RAVDESS no "
                  f"tiene por diseno: {self._neutral_strong[:5]}")

        if not self.strict_integrity:
            return

        if not data:
            raise RuntimeError(
                f"RAVDESS: 0 clips para los grupos {self.sessions}. O data_path no apunta "
                f"a la raiz del corpus, o no se ha descomprimido "
                f"Audio_Speech_Actors_01-24.zip. "
                f"Descarga: https://zenodo.org/records/1188976"
            )
        if self._descartados_patron:
            raise RuntimeError(
                f"RAVDESS: {self._descartados_patron} ficheros no casan el patron de "
                f"siete campos de dos digitos. Los 1,440 del zip de habla lo casan sin "
                f"excepciones. Si aparece otro, la carpeta tiene algo que no es RAVDESS.\n"
                f"  Nombres: {self._nombres_raros}\n"
                f"  Si es una copia con sufijo (' (1).wav', '.wav.bak') o un fichero "
                f"ajeno, sacalo. Si es un nombre legitimo que el patron no contempla, "
                f"hay que corregir el patron, no el corpus."
            )
        if self._neutral_strong:
            raise RuntimeError(
                f"RAVDESS: {len(self._neutral_strong)} clips de neutral con intensidad "
                f"strong (campo 4 = 02). RAVDESS NO tiene neutral-strong por diseno: "
                f"neutral solo se grabo en intensidad normal. Si existen, la copia en "
                f"disco no es la publicada.\n  Nombres: {self._neutral_strong[:10]}"
            )
        if len(frases) != len(FRASES):
            raise RuntimeError(
                f"RAVDESS: se cargaron {len(frases)} frases distintas y deberian ser "
                f"{len(FRASES)}. El experimento se apoya en que las DOS frases sean "
                f"comunes a todas las emociones: si falta alguna, el texto deja de ser "
                f"ortogonal a la emocion y el resultado no mide lo que dice medir."
            )

        # El contrato completo solo se puede comprobar con los cinco grupos a la vez.
        completo = {int(s) for s in self.sessions} == set(range(1, N_GRUPOS + 1))
        if not completo:
            return

        if len(actores) != self.REF_N_ACTORES:
            raise RuntimeError(
                f"RAVDESS: {len(actores)} actores, esperados {self.REF_N_ACTORES} "
                f"(01-24). Faltan: "
                f"{sorted(set(f'{i:02d}' for i in range(1, N_ACTORES + 1)) - set(actores))}"
            )
        if set(self.emotions) == self.SET_4CLASS:
            esperado = dict(self.REF_4CLASS_POR_EMOCION)
            if len(data) != self.REF_4CLASS_TOTAL:
                raise RuntimeError(
                    f"RAVDESS: {len(data)} clips de 4 clases, esperados "
                    f"{self.REF_4CLASS_TOTAL} (192 x 3 + 96 de neutral, que no tiene "
                    f"intensidad strong). por emocion={por_emocion}. "
                    f"Este contrato se VERIFICO el 2026-10-20 contra la descarga de "
                    f"Zenodo y cuadraba exacto, asi que una desviacion ahora apunta a la "
                    f"copia en disco: pasa scripts/check_ravdess.py --fase todo."
                )
            if por_emocion != esperado:
                raise RuntimeError(
                    f"RAVDESS: distribucion por emocion {por_emocion}, esperada "
                    f"{esperado}."
                )
