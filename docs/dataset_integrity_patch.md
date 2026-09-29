# Parche para `src/data/dataset.py`

Dos cambios. El primero impide que vuelva a pasar; el segundo arregla un bug latente
que muerde específicamente a los espejos de Kaggle.

---

## 1. Aserción de integridad en `IEMOCAPDataset`

### `__init__` — añadir el parámetro

```python
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
    strict_integrity: bool = True,          # <-- NUEVO
):
    ...
    self.strict_integrity = strict_integrity
```

### Al final de `_load_data()`, justo antes del `return data`

```python
        # --- contrato del dataset -------------------------------------------
        n_impro = sum(1 for d in data if "impro" in d["utterance_id"])
        n_script = len(data) - n_impro
        n_notext = sum(1 for d in data if not d["transcription"].strip())
        print(f"[IEMOCAP] sesiones={self.sessions}  total={len(data)}  "
              f"impro={n_impro}  script={n_script}  sin_transcripcion={n_notext}")

        if self.strict_integrity:
            REF = {1: 1085, 2: 1023, 3: 1151, 4: 1031, 5: 1241}
            expected = sum(REF[s] for s in self.sessions)

            if n_impro == 0:
                raise RuntimeError(
                    f"IEMOCAP incompleto: 0 diálogos improvisados en las sesiones "
                    f"{self.sessions}. La copia del corpus solo tiene la parte "
                    f"'scripted', lo que introduce fuga léxica entre particiones."
                )
            if abs(len(data) - expected) > 0.02 * expected:
                raise RuntimeError(
                    f"IEMOCAP incompleto: {len(data)} utterances de 4 clases cargadas, "
                    f"se esperaban ~{expected} para las sesiones {self.sessions} "
                    f"(impro={n_impro}, script={n_script}). Revisa data_path: "
                    f"{self.data_path}"
                )
```

> `strict_integrity=False` solo para tests o para depurar con un subconjunto.
> Nunca para una corrida cuyo resultado vaya a un artículo.

---

## 2. Bug latente: los ficheros `._*` de macOS

`_parse_emotion_labels()` y `_parse_transcriptions()` hacen `eval_path.glob("*.txt")`.
Los zips subidos desde macOS —muy común en espejos de Kaggle— traen un
`._Ses01F_impro01.txt` junto a cada `Ses01F_impro01.txt`. Son forks de recurso
binarios: el `re.match` casi siempre falla, pero cuando acierta produce ids
fantasma que luego se descartan silenciosamente en `_find_audio_file()`.
Es exactamente el mismo modo de fallo que causó el problema original.

En **ambos** métodos, sustituir:

```python
for eval_file in eval_path.glob("*.txt"):
```

por:

```python
for eval_file in sorted(eval_path.glob("*.txt")):
    if eval_file.name.startswith("."):      # ._foo.txt de macOS
        continue
```

Y lo mismo en `_parse_transcriptions()` con `trans_file`.

---

## 3. Determinismo — `src/train.py`, junto a las líneas 847-851

Hoy solo se fija `manual_seed`, así que «semilla 42» no garantiza reproducibilidad.
Los revisores van a pedir varianza; conviene que la varianza que reportes sea la
real entre folds y semillas, no ruido de cuDNN.

```python
import os, random

seed = config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # requerido por cuBLAS
torch.use_deterministic_algorithms(True, warn_only=True)

# y en cada DataLoader:
g = torch.Generator(); g.manual_seed(seed)
DataLoader(..., generator=g, worker_init_fn=lambda wid: np.random.seed(seed + wid))
```

`warn_only=True` evita que reviente si alguna op de cuDNN no tiene versión
determinista; avisará en el log en vez de abortar.

---

## Orden de ejecución

```powershell
# 1. descargar
kaggle datasets download -d dejolilandry/iemocapfullrelease -p D:\DOCTORADO\DATASETS\IEMOCAP\
Expand-Archive D:\DOCTORADO\DATASETS\IEMOCAP\iemocapfullrelease.zip -DestinationPath D:\DOCTORADO\DATASETS\IEMOCAP\

# 2. verificar ANTES de tocar nada  <-- el paso que faltó la vez pasada
python scripts\verify_iemocap.py "D:\DOCTORADO\DATASETS\IEMOCAP\IEMOCAP_full_release" --leak-check

# 3. solo si el paso 2 sale en verde: apuntar el config a la ruta nueva
#    config/config.yaml -> dataset.data_path

# 4. una corrida de humo (1-2 épocas) para confirmar 5 531 en el log
```

El paso 2 tarda segundos y es lo único que separa esta versión del artículo
de la anterior.
