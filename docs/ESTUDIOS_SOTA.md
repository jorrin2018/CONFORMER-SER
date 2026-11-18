# Estudios 98-110: filas pendientes del artículo y candidatos a superar el estado del arte

Rama `feature/sota-studies` (sobre `feature/expert-gate`). Noviembre 2026.

Todos son **el estudio 74 con un solo cambio**, activado por una clave opt-in. Sin esas
claves el código hace exactamente lo mismo que antes: los logits en modo evaluación de los
estudios 72, 74 y 89 salen idénticos bit a bit antes y después de esta rama.

## 0. Actualización (14 nov 2026): la base pasa a ser el 98

Resultados con la semilla 42 (detalle en REWRITE_NOTES 16.1 y 16.2):

| Estudio | UA | Δ frente al 74 | Decisión |
|---|---|---|---|
| 102 WavLM Base+ | 75.96 | +0.47 ± 0.86 (2/5 folds > 0.86) | no se adopta |
| 101 co-atención | 75.44 | −0.04 ± 1.54 (2/5) | no se adopta |
| **98 sin ViT** | **76.43** | **+0.95 ± 1.02** (2/5; el 74 no gana en ningún fold por > 0.86) | **nueva base** |

El ViT **no aporta**; que quitarlo *mejore* no está demostrado (IC 95 % [−0.32, +2.22]).
La base pasa al 98 por parsimonia: igual o mejor, más simple, ~30 % más barato. Los
estudios 99, 100 y 103, que no se habían corrido, ahora parten del 98. El 101 y el 102
se quedan como estaban (ya se corrieron sobre el 74).

**Nuevos estudios:**

| Estudio | Base | Cambio | Para qué |
|---|---|---|---|
| **104** | 98 | el modelo solo ve el enunciado real: relleno al final, aumento y normalización sin relleno, promedio solo de frames reales | el grupo débil son los enunciados cortos (0–2 s: UA 63.5), que son los que más relleno llevan |
| **105** | 77 | CREMA-D sin ViT | el eje léxico con el modelo final y la ablación del ViT en un segundo corpus |
| **106** | 79 | MELD sin ViT (`--protocol fixed`, 2 semillas) | el eje léxico con el modelo final y la ablación del ViT con audio degradado |

**El ViT no desaparece del artículo, pasa a ser una ablación medida.** Con los pares
74/98 (IEMOCAP), 77/105 (CREMA-D) y 79/106 (MELD) queda la ablación del flujo de imagen
en los tres corpus. Si el modelo final termina siendo otro (por ejemplo, el 104 si se
adopta), se corre su versión con ViT para que la ablación sea sobre el modelo final.

**Orden y comandos:**

```powershell
python scripts\check_sota_studies.py --study 104 --fold 1 --device cuda
python scripts\run_ablation.py --study 104 --loso --device cuda
python scripts\check_sota_studies.py --study 105 --fold 1 --device cuda
python scripts\run_ablation.py --study 105 --loso --device cuda
python scripts\check_sota_studies.py --study 106 --device cuda
python scripts\run_ablation.py --study 106 --protocol fixed --seed 42 --device cuda
python scripts\run_ablation.py --study 106 --protocol fixed --seed 123 --device cuda
# Despues, 5 semillas del modelo final (98, o 104 si se adopta):
foreach ($semilla in 123, 456, 789, 1011) {
    python scripts\run_ablation.py --study 98 --loso --seed $semilla --device cuda
}
# Y al final 103, 100 y 99, ya sobre el 98.
```

## 1. Qué hay que superar

Solo cuentan los trabajos con nuestro protocolo: IEMOCAP, 4 clases, una sesión fuera para
test y nivel de enunciado (tabla 9 de `paper/manuscripts/CAMF_results_section_2026-11-09.docx`).

| Trabajo | Modalidades | UA | Nota |
|---|---|---|---|
| Zhao et al. 2022 | audio + texto | **76.31** | media de 25 corridas; co-atención wav2vec 2.0 ↔ BERT |
| Morais et al. 2022 | audio | **77.76** | wav2vec 2.0 + HuBERT afinados, promedio de checkpoints |
| **CAMF (74)** | audio + texto + log-mel | **75.49** | 2 semillas |

Nuestro solo-audio da 68.99 y el de Morais 77.76: **la rama de audio es la diferencia más
grande**. La fusión ya aporta +6.49 pp; el margen está sobre todo en el encoder.

## 2. ¿Alinear audio y texto o atención cruzada?

**En el 74 no hay interacción entre audio y texto a nivel de secuencia.** La "cross-attention"
de `ConflictAwareCrossAttentionFusion` opera sobre tres vectores ya agrupados (promedio de
wav2vec2, [CLS] de DistilBERT y [CLS] del ViT). Es self-attention entre tres tokens. Ningún
frame de audio ve ninguna palabra.

- **Atención cruzada a nivel de token (estudio 101): sí.** Es un factor nuevo de verdad.
  Es lo que hace Zhao et al., el sistema comparable más fuerte. No necesita marcas de tiempo,
  así que funciona igual en MELD, en CREMA-D y con texto de ASR. Además, el análisis de la
  compuerta (REWRITE_NOTES, sección 15) dejó 25.6 pp de oráculo sin explotar: la información
  complementaria existe, pero pesar vectores por muestra no la aprovecha.
- **Alinear "antes de extraer características": no tal cual.** wav2vec2 y DistilBERT
  procesan cada modalidad por separado; alinear la entrada no cambia lo que extraen. La
  alineación sirve *después* de extraer, para promediar los frames de cada palabra y
  fusionar palabra por palabra (la rama síncrona de Wu et al. 2021). Tiene tres costes:
  1. exige tiempos por palabra (IEMOCAP trae `.wdseg`; MELD y CREMA-D no);
  2. la speed perturbation estira esos tiempos y el prompt de prosodia no tiene tiempos;
  3. no sirve con texto de ASR.

  **Siguiente paso, solo si el 101 gana:** usar la alineación como sesgo sobre la
  atención del 101 (penalizar los pares frame-palabra fuera del intervalo de la palabra),
  como un factor más encima del 101.

## 3. Los seis estudios

| Estudio | Cambio frente al 74 | Para qué | Coste aprox. (RTX 3070, 5 folds) |
|---|---|---|---|
| **98** | el ViT no se ejecuta y su vector se sustituye por ceros (mismos parámetros) | la ablación audio + texto: ¿aporta algo la tercera rama? | ~10 h |
| **99** | 17 épocas fijas; en test se evalúa la última época, sin early stopping | comparar con justicia con Chen & Rudnicky (no seleccionan checkpoint) | ~10 h |
| **100** | texto de Whisper en lugar de la transcripción manual | ¿sobrevive el aporte de la fusión sin transcripción humana? | ~13 h + ~1 h de ASR |
| **101** | co-atención frames de audio ↔ tokens de texto antes del pooling | la fusión que hace el SOTA bimodal | ~14 h |
| **102** | WavLM Base+ en lugar de wav2vec 2.0 base (mismo tamaño) | la palanca más grande: +5.2 pp en SUPERB | ~13 h |
| **103** | RoBERTa-base en lugar de DistilBERT (con su tokenizer) | capacidad de la rama de texto | ~14 h |

Las horas son estimaciones a partir del 74; el log de cada fold da la cifra real.

Detalles de diseño que hay que poder defender:

- **98: por qué con ceros y no quitando el token.** Quitar el token cambiaría también las
  formas de la fusión (dos tokens en vez de tres, otra capa de entrada): serían dos factores.
  Con ceros, `vision_proj(0)` es su sesgo, un token constante aprendido. Desaparece solo la
  información del ViT.
- **99: de dónde sale 17.** Es la mediana de la mejor época de validación de las diez
  corridas del 74 (semillas 42 y 123: 12, 14, 21, 20, 8 y 17, 16, 21, 10, 32). Sale de la
  validación, nunca del test. La validación sigue alimentando al scheduler de plateau.
  `final_results.json` registra `model_selection`.
- **101: su tasa de aprendizaje.** Son capas nuevas, desde cero (~9.5 M parámetros con
  768 dimensiones). Con el lr de la fusión (5e-6) apenas se moverían y leeríamos un falso
  nulo, que es lo que anuló los estudios 68 y 69. Llevan su propio grupo,
  `learning_rate_crossmodal: 1e-4`, que forma parte de la definición del módulo; no es un
  segundo factor. Si alguien activa el módulo sin esa clave, el optimizador avisa.
- **102: torch ≥ 2.6.** `wavlm-base-plus` solo publica `.bin`, y transformers se niega a
  leerlo con torch < 2.6. El cargador falla con un mensaje claro en vez de seguir con pesos
  aleatorios.
- **101 y el relleno.** En el régimen del 74 el audio llega rellenado a 8 s. Por defecto la
  co-atención no enmascara esos frames (igual que el pooling del 74), para no cambiar dos
  cosas. El texto sí se enmascara (un test lo comprueba).

## 4. Antes de gastar GPU

```powershell
python -m pytest tests\test_sota_studies.py -q
# Con encoders reales, un batch real y el pico de memoria en tu GPU:
python scripts\check_sota_studies.py --study 102 --fold 1 --device cuda
python scripts\check_sota_studies.py --study 101 --fold 1 --device cuda
python scripts\check_sota_studies.py --study 103 --fold 1 --device cuda
python scripts\check_sota_studies.py --study 98 --fold 1 --device cuda
python scripts\check_sota_studies.py --study 99 --fold 1 --device cuda
# Solo para el 100: primero el ASR (reanudable; reporta el WER, que va al artículo)
python scripts\transcribe_asr.py --config config\ablation\ablation_100_asr_transcripts.yaml --device cuda
python scripts\check_sota_studies.py --study 100 --fold 1 --device cuda
```

`check_sota_studies.py` sale con 1 si algo no cuadra: no lanzar ese estudio.

## 5. Orden y comandos

**Fase A, una semilla (42), un estudio a la vez:**

```powershell
python scripts\run_ablation.py --study 102 --loso --device cuda
python scripts\run_ablation.py --study 101 --loso --device cuda
python scripts\run_ablation.py --study 98  --loso --device cuda
python scripts\run_ablation.py --study 103 --loso --device cuda
python scripts\run_ablation.py --study 100 --loso --device cuda
python scripts\run_ablation.py --study 99  --loso --device cuda
```

El orden pone primero las dos palancas con más fundamento (102 y 101), después la fila que
pedirá el revisor (98) y al final las de protocolo (100 y 99). Si un fold falla:
`--resume --folds N`, como siempre.

**Fase B, combinación.** Solo se combinan los cambios adoptados con la regla de la sección
6. El config del estudio 104 se crea *después* de ver la fase A, nunca antes.

**Fase C, número del artículo: cinco semillas del modelo final y de su solo-audio.**

```powershell
# Ojo: en PowerShell $S y $s son la misma variable; aqui se usa $semilla.
foreach ($semilla in 123, 456, 789, 1011) {
    python scripts\run_ablation.py --study 74 --loso --seed $semilla --device cuda
}
```

Sustituye `74` por el estudio final. Haz lo mismo con el solo-audio que le corresponda
(el 71, o un 71 con WavLM si el 102 entra).

**Fase D, comparación de protocolo** (tabla 11 del documento de resultados):
`--protocol random` sobre el modelo final y el 71; el estudio 70 (réplica de Pepino); los
estudios 81 y 82 (MELD de 7 clases, 2 semillas).

## 6. Regla de decisión, escrita antes de correr

1. **Un cambio se adopta** si gana al 74 por más de **0.86 pp en al menos 4 de 5 folds**
   (semilla 42, pareado por fold). Es la misma regla que se aplicó a la compuerta.
2. **La combinación (104)** solo junta cambios adoptados y se compara pareada contra el 74 y
   contra cada cambio por separado.
3. **El número del artículo es la media de cinco semillas** del modelo final. La selección
   solo vio la semilla 42; se reporta además la media de las cuatro semillas nuevas, que no
   participaron en ninguna decisión. Así se controla la "maldición del ganador": elegir la
   mejor de seis variantes por su resultado en test sube el número por azar.
4. **Todas las filas van al artículo**, también las nulas, en la tabla de ablación.
5. **Contra Zhao (76.31):** solo se escribe "igual o mejor" si la media de cinco semillas lo
   supera. Aun así, se dice que la diferencia es menor que la dispersión entre folds (~3 pp)
   y que su número promedia 25 corridas.

## 7. Qué no hacer

- No probar variantes que no están aquí con la intención de quedarse con la mejor en test.
  Si aparece una idea nueva, va como estudio nuevo, con su lectura escrita antes.
- No usar WavLM *Large* ni RoBERTa *Large* sin pasar antes `check_sota_studies.py` con el
  pico de memoria. Con 8 GB lo más probable es que no quepan con este régimen.
- No afirmar que el ViT "aporta" ni que "no aporta" antes del 98.

## 8. Archivos

| Archivo | Qué hace |
|---|---|
| `src/models/crossmodal.py` | `CrossModalCoAttention` (estudio 101) |
| `src/ablation_studies/ablation_11_model.py` | `model.vision_stream` y `model.crossmodal_attention` |
| `src/ablation_studies/utils.py` | grupo `learning_rate_crossmodal` en el optimizador |
| `scripts/run_ablation.py` | estudios 98-103; `training.model_selection` |
| `src/data/base_dataset.py`, `src/train.py` | `text.transcript_source: asr` y `transcript_override` |
| `scripts/transcribe_asr.py` | Whisper → `cache/asr/<corpus>_<modelo>.json`, con WER |
| `scripts/check_sota_studies.py` | comprobación previa con modelo, datos y GPU reales |
| `tests/test_sota_studies.py` | 19 tests sin GPU ni datos |
| `config/ablation/ablation_98…103_*.yaml` | un config por estudio, con su lectura en la cabecera |

## 9. Fase final: protocolos comparables y ablaciones del modelo final (18 nov 2026)

El modelo del artículo es el **98** (audio + texto). Lo que falta se divide en dos grupos:
filas que permiten compararse con cada trabajo relacionado **bajo su propio protocolo**, y
ablaciones que sostienen cada pieza del 98. Todo está en una cola ordenada y reanudable:

```powershell
.\scripts\cola_articulo.ps1 -SoloListar      # que falta, sin correr nada
.\scripts\cola_articulo.ps1                  # bloques A-D, en orden
.\scripts\cola_articulo.ps1 -Bloques A,B     # solo esos bloques
```

La cola se salta lo que ya tiene un resumen completo y se detiene si encuentra uno
parcial. Antes de cada estudio nuevo pasa `check_sota_studies.py`, y antes del 100
transcribe con Whisper si falta el cache.

### 9.1 Comparación con los trabajos relacionados

| Trabajo | Su protocolo | Nuestra fila | Cómo se obtiene | Estado |
|---|---|---|---|---|
| Zhao et al. 2022 (76.31 UA) | LOSO, media de 25 corridas (5 × 5) | 98 con 5 semillas | bloque A: semillas 789 y 1011 | 3/5 hechas |
| Pepino et al. 2021 (67.2 UA) | LOSO, wav2vec2-base congelado | estudio 70 (réplica) | bloque D | pendiente |
| Chen & Rudnicky 2023 (69.9 / 74.3) | LOSO, épocas fijas, sin selección | estudio 99 | bloque D | pendiente |
| Yoon et al. 2018 (0.718 WAP) | 5 folds aleatorios, dependiente de hablante | 98 y 71 con `--protocol random` | bloque D | pendiente |
| Wu et al. 2021 (71.90 UA con ASR) | LOSO, ASR, ±3 enunciados de contexto | estudio 100 (ASR, sin contexto) | bloque D | pendiente; sigue sin ser comparable por el contexto |
| Literatura de MELD (weighted F1, 7 clases) | split oficial | 110 (y su par 82) con 2 semillas | bloque D | pendiente |
| Literatura de CREMA-D | mayoría con repartos aleatorios | 105 y 78 con `--protocol random` | bloque D | pendiente |
| emotion2vec (72.69 UA) | LOSO, encoder congelado + capa lineal | no se replica | — | se cita como comparable, sin fila propia |
| Morais et al. 2022 (77.76 UA) | LOSO, dos encoders afinados | no se replica | — | ídem |
| Makiuchi et al. 2021 (73.0 UA) | LOSO | — | — | comparable tal cual |

**Regla de reporte:** las filas `random` van **siempre** al lado de las LOSO, nunca solas.
Su propósito es medir cuánto vale el reparto, no dar un número mejor.

**Corrección de código necesaria:** antes de esta entrega, `--protocol random` escribía
`kfold_summary_ablation_NN.json` y **sobrescribía** el resumen LOSO del mismo estudio.
Ahora escribe `kfold_summary_ablation_NN_random.json`.

### 9.2 Ablaciones del modelo final

| Pieza del 98 | Ablación | Estado |
|---|---|---|
| fusión audio + texto | 98 frente a 71 (solo audio) y 73 (solo texto) | hecho con 1 semilla de 71/73; bloque C añade 123 y 456 |
| flujo de imagen (ViT) | 74/98, 77/105, 79/106 | **hecho: no aporta en ningún corpus** |
| compuerta de conflicto | 72, 75, 76, 97 frente al 74 | hecho: no aporta (análisis sobre el 74) |
| prompt de prosodia en el texto | **107**: el 98 sin el prompt | pendiente (bloque B), el más importante |
| atención entre ramas | **109**: el 98 con 0 capas (concatenación + MLP) | pendiente (bloque B) |
| aumentación de audio | **108**: el 98 sin aumentación | pendiente (bloque B) |
| encoder de audio | 102 (WavLM Base+) | hecho: no aporta |
| interacción frame ↔ palabra | 101 (co-atención) | hecho: no aporta |
| relleno del audio | 104 | hecho: no aporta |
| encoder de texto | 103 (RoBERTa-base) | opcional (bloque E) |

**Por qué el 107 es el más importante:** en el 98 el texto lleva una descripción de
prosodia calculada del audio. El argumento del artículo es que el aporte de la fusión es
léxico. Si el 107 empata con el 98, el argumento queda limpio. Si no, el eje se describe
como "transcripción + descripción prosódica".

### 9.3 Coste estimado (RTX 3070)

| Bloque | Corridas | Horas aprox. |
|---|---|---|
| A | 2 × 98 | ~20 |
| B | 107, 109, 108 | ~30 |
| C | 2 × 73, 2 × 71 | ~23 |
| D | random ×4, 70, 99, 100 (+ASR), 110 ×2, 82 ×2 | ~75 |
| **total A-D** | | **~150 h (unos 6-7 días seguidos)** |

Si hay que recortar: A y B son imprescindibles; de D, Yoon (random) y MELD 7 clases son
los que más pide un revisor.

## 10. Demo en vivo y comparación rápida (18 nov 2026)

Dos scripts que solo leen checkpoints y resúmenes; no entrenan nada.

```powershell
# Demo: el mismo enunciado por solo audio (71), solo texto (73) y la fusión (98).
# Modelos del fold S5: nunca vieron a los hablantes de la sesión 5.
python scripts\demo_emocion.py                       # 5 enunciados al azar de S5
python scripts\demo_emocion.py --clase happy --n 4   # donde más se nota la fusión
python scripts\demo_emocion.py --utt Ses05F_impro03_F012
python scripts\demo_emocion.py --wav mi_audio.wav --texto "I can't believe you did that"
python scripts\demo_emocion.py --wav mi_audio.wav --asr     # transcribe con Whisper
python scripts\demo_emocion.py --grabar 4 --asr             # micrófono (pip install sounddevice soundfile)
# Si la GPU está entrenando: --device cpu

# Comparación pareada por fold con la regla de decisión del repo
python scripts\comparar_estudios.py 107 98
python scripts\comparar_estudios.py 98 74 --semillas 42 123 456
```

## 11. Familia clásica: eGeMAPS / COVAREP + GloVe con la misma fusión (estudios 111-119, 18 nov 2026)

### Por qué

El aporte del artículo es el **eje léxico**: la ganancia de la fusión crece con lo que
informa el texto (CREMA-D −0.99, IEMOCAP +6.76, MELD +12.22 con wav2vec2 + DistilBERT). Un
revisor puede preguntar si eso es una propiedad de los **datos** o de los encoders
preentrenados. Repetir el eje con una familia de encoders completamente distinta
(descriptores acústicos hechos a mano y vectores de palabra estáticos, la de MulT / MCTN /
MISA) separa las dos cosas. Además, con ramas más débiles y más distintas, la compuerta de
conflicto (H3) y el ViT sobre log-mel (H1) tienen otra oportunidad.

**No es para superar el estado del arte.** El UA absoluto quedará bastante por debajo del
98; lo que se compara es el signo y el orden de la ganancia de la fusión en los tres corpus.

### Qué cambia (un solo factor: el encoder de cada rama)

| Rama | Base (98) | Familia clásica |
|---|---|---|
| Audio | wav2vec2-base, 6 bloques afinados | openSMILE eGeMAPSv02: 25 descriptores por frame (10 ms) → BiLSTM 2×256 → media y máximo sobre frames reales |
| Texto | DistilBERT, 4 bloques afinados, [CLS] | GloVe 840B 300-d **congelado** → BiLSTM 2×256 → media y máximo sobre tokens reales |
| Tasa del encoder | 3e-5 / 2e-5 (afinar) | 1e-3 (BiLSTM desde cero) |

Todo lo demás es idéntico a la base: fusión de 3 capas sobre los vectores de rama, cabeza
MultiScaleTemporal, pérdida, tasa de la fusión (5e-6), folds, validación y selección de
modelo. El prompt de prosodia sigue (sus palabras tienen vector GloVe). Las contracciones se
parten como en GloVe (`don't` → `do n't`), así que la negación no se pierde.

Lo que **no** se aplica a los descriptores: el aumento sobre la forma de onda (ruido,
velocidad, enmascarado); se calcularon una vez sobre el audio limpio. Sí se recortan del
mismo segmento que la forma de onda y se normalizan con media y desviación del **train de
cada fold**.

| Estudio | Base | Qué es | Comparar con |
|---|---|---|---|
| 111 | 71 | IEMOCAP solo audio (eGeMAPS) | — |
| 112 | 73 | IEMOCAP solo texto (GloVe) | — |
| 113 | 98 | IEMOCAP audio + texto | **113 − 111** (ganancia de la fusión) |
| 114 | 113 | + compuerta de conflicto entrenada (la del 75) | 114 − 113 (H3) |
| 115 | 74 | audio + texto + ViT sobre log-mel | 115 − 113 (H1) |
| 116 | 105 | CREMA-D audio + texto | **116 − 117** |
| 117 | 78 | CREMA-D solo audio | — |
| 118 | 106 | MELD audio + texto (`--protocol fixed`, 2 semillas) | **118 − 119** |
| 119 | 80 | MELD solo audio (`--protocol fixed`, 2 semillas) | — |

### Preparación (una vez)

```powershell
pip install opensmile
# GloVe 840B (unos 2 GB): https://nlp.stanford.edu/data/glove.840B.300d.zip
# No hace falta descomprimirlo. glove.6B.300d.zip (822 MB) también sirve.

# Un cache por corpus (descriptores + vocabulario GloVe del corpus). Es incremental:
# si el cache existe no se rehace. Del orden de 80 ms por utterance en CPU.
python scripts\extraer_clasicas.py --study 113 --glove D:\DOCTORADO\RECURSOS\glove.840B.300d.zip
python scripts\extraer_clasicas.py --study 116 --glove D:\DOCTORADO\RECURSOS\glove.840B.300d.zip
python scripts\extraer_clasicas.py --study 118 --glove D:\DOCTORADO\RECURSOS\glove.840B.300d.zip
```

Los caches quedan en `cache\clasicas\` (`<corpus>_egemaps.pt` y `glove_<corpus>.pt`). El
script imprime la cobertura de GloVe y las palabras más frecuentes sin vector.

**COVAREP (opcional, necesita MATLAB).** Es la fuente de MulT / MISA. Se corre
`COVAREP_feature_extraction` en MATLAB sobre las carpetas de wav (frames de 10 ms) y luego
`python scripts\extraer_clasicas.py --study 113 --fuente covarep` recoge los `.mat`. Para no
mezclarlo con eGeMAPS, cambia en el config `model.clasico.fuente: covarep` y
`audio_cache: cache/clasicas/iemocap_covarep.pt`, y guarda esa variante con otro número de
estudio. La dimensión del encoder se lee del cache, no hay que tocar nada más.

### Corridas

```powershell
.\scripts\cola_articulo.ps1 -Bloques F -SoloListar
.\scripts\cola_articulo.ps1 -Bloques F -Glove D:\DOCTORADO\RECURSOS\glove.840B.300d.zip
# o a mano:
python scripts\run_ablation.py --study 113 --loso --device cuda
python scripts\run_ablation.py --study 118 --protocol fixed --seed 42 --device cuda
python scripts\comparar_estudios.py 113 111
```

Son mucho más baratas que el 98: no hay 95 M de parámetros que afinar; la lectura de los
wav y el prompt de prosodia pasan a ser lo que más tarda.

### Lecturas, escritas antes de correr

- **El eje es del corpus**: 116 − 117 dentro de la banda, 113 − 111 > 0.86 en ≥ 4/5 folds y
  118 − 119 la mayor de las tres. Se reporta como una segunda familia de encoders con el
  mismo orden: el resultado deja de depender de wav2vec2 / DistilBERT.
- **El eje cambia de orden**: se reporta tal cual; el artículo pasa a decir que la ganancia
  depende del corpus *y* del encoder, con las dos tablas.
- **114 − 113 > 0.86 en ≥ 4/5 folds**: la compuerta sirve cuando las ramas son débiles; se
  reporta como condición de uso (no como mejora general) junto al empate con wav2vec2.
- **115 − 113 > 0.86 en ≥ 4/5 folds**: el log-mel + ViT complementa a los descriptores de
  bajo nivel aunque no a wav2vec2; matiza H1.

Archivos: `src/models/classic_encoders.py`, `src/data/classic_features.py`,
`scripts/extraer_clasicas.py`, `config/ablation/ablation_111..119_*.yaml`,
`tests/test_classic_studies.py`.
