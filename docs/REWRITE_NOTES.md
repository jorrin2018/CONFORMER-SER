# Notas para la reescritura de CAMF

Bitácora de todo lo que la auditoría posterior al rechazo de CIARP 2026 obliga a
cambiar en el texto. Cada afirmación lleva su procedencia para poder rehacerla.

**Estado**: código y datos corregidos y verificados. Pendiente: re-entrenar (Fases 2-5)
y escribir. No escribir ninguna cifra de resultados hasta tener las corridas LOSO.

Postmortem completo con la evidencia: artefacto «Postmortem CAMF».

---

## 0. Lo que ya está arreglado y verificado

| Qué | Estado | Verificación |
|---|---|---|
| Corpus completo (5 531 utterances, 4 clases) | corregido | `scripts/verify_iemocap.py` en verde |
| Bug `_find_audio_file` (`[:3]` → `[:-1]`) | corregido | descartaba las 2 943 improvisadas |
| Aserción de integridad en `_load_data` | añadida | revienta si faltan impro o no cuadra el conteo |
| SpeechCue como lo describe el artículo | reimplementado | `src/data/speechcue.py` |
| Normalización de F0/energía por hablante | añadida | `speechcue.normalize: speaker` |
| Régimen de congelado parcial | corregido | `apply_finetuning_policy()` |
| Determinismo (cuDNN, semillas, DataLoader) | añadido | `src/train.py` |
| LOSO para cualquier estudio | añadido | `--loso` en `run_ablation.py` |
| Encoder de audio cargado con pesos reales | corregido | `use_safetensors`; el fallback a pesos aleatorios **eliminado** (sección 13) |


---

## 0.bis RESULTADO LOSO DEL ESTUDIO 45 (11 sep 2026) — el número honesto

**70.73 ± 3.01 UA · 69.65 ± 1.98 WA**, 5 folds, semilla 42, corpus completo.
Fuente: `checkpoints/ablation_45/kfold_summary_ablation_45.json`.

| Fold (test) | val UA | test UA | test WA | época final |
|---|---|---|---|---|
| S1 | 79.45 | 73.19 | 69.49 | 32 |
| S2 | 74.47 | **75.34** | 73.51 | 30 |
| S3 | 70.25 | 68.11 | 68.20 | 54 |
| S4 | 67.15 | **67.72** | 68.48 | 47 |
| S5 | 71.48 | 69.30 | 68.57 | 50 |
| **media** | **72.56 ± 4.17** | **70.73 ± 3.01** | **69.65 ± 1.98** | |

Lecturas:

- El 79.23 del manuscrito era **la fuga léxica**, no suerte de fold: la sesión 5 da ahora
  69.30, el tercer mejor fold, ligeramente por debajo de la media.
- La brecha val→test baja de 2.91 pp (reclamada) a **1.83 pp** real. La generalización es
  ahora honesta.
- σ = 3.01 pp confirma que ningún margen menor que ~3 pp es interpretable.
- Sin colapso de clases en S5: angry 67.65, happy 61.54, sad 75.10, neutral 72.92.

**ADVERTENCIA DE PROTOCOLO — leer antes de comparar con nadie.**
`run_ablation_study_kfold()` usa **3 sesiones de entrenamiento**, 1 de validación y 1 de
test (`val_session = remaining[0]`, el resto entrena). El protocolo LOSO estándar de
IEMOCAP entrena con **4** sesiones. Es decir: estos números salen con un 25 % menos de
datos de entrenamiento que las líneas base publicadas, y la sesión 1 solo entra en
entrenamiento en 1 de los 5 folds. Es exactamente la objeción de R3 («seguir el mismo
protocolo que la mayoría del trabajo sobre IEMOCAP»). **Pendiente de decidir si se
re-corre con 4 sesiones de entrenamiento y validación extraída de dentro.**

Contexto para comparar (UA, 4 clases): Pepino 72.50 (solo audio), Morais 73.80 (solo
audio), MulT 76.10, Kim 77.70, CTNet 78.30, UniMSE 79.14. CAMF queda por debajo de todas,
**incluidas las unimodales de audio**, lo que obliga a medir las líneas base propias bajo
el mismo protocolo antes de sacar conclusiones sobre la fusión.



### RESULTADO DE LAS LÍNEAS BASE (12 sep 2026) — la fusión SÍ aporta

Protocolo actual (3 sesiones de entrenamiento), 5 folds, semilla 42, corpus completo.

| Fold (test) | 45 trimodal | 66 solo audio | 67 solo texto | 45−66 | 45−67 |
|---|---|---|---|---|---|
| S1 | 73.19 | 68.43 | 66.89 | +4.76 | +6.31 |
| S2 | 75.34 | 69.87 | 71.72 | +5.48 | +3.62 |
| S3 | 68.11 | 59.45 | 66.04 | +8.66 | +2.07 |
| S4 | 67.72 | 59.57 | 65.37 | +8.15 | +2.34 |
| S5 | 69.30 | 62.79 | 64.42 | +6.51 | +4.88 |
| **media ± σ** | **70.73 ± 3.37** | **64.02 ± 4.89** | **66.89 ± 2.85** | **+6.71 ± 1.68** | **+3.85 ± 1.77** |

- CAMF gana en **5 de 5 folds** contra ambas líneas base.
- Pareado por fold: **+6.71 ± 1.68 pp** sobre solo-audio (t(4)=8.95, p≈0.0009) y
  **+3.85 ± 1.77 pp** sobre solo-texto (t(4)=4.85, p≈0.008).
- Test de permutación de signos exacto: p = 0.0625 en ambos — que es **el mínimo
  alcanzable con 5 folds** (2/2⁵). Reportar las dos cosas y decir explícitamente que 5/5
  es el resultado más extremo posible a este tamaño muestral. Con 3 semillas (15
  observaciones pareadas) el test podría bajar mucho más.
- La σ pareada (1.7 pp) es la mitad de la σ absoluta entre folds (3.4 pp): **parear por
  fold elimina la mayor parte de la varianza**. Esta es la forma correcta de comparar
  ablaciones en IEMOCAP y merece una frase en el artículo.

**La evidencia de que el protocolo está deprimiendo los números absolutos.** Nuestra línea
base de solo-audio da 64.02 UA con Wav2Vec2-base. Pepino et al. reportan **72.50** con el
mismo encoder. 8.5 pp de diferencia con la misma arquitectura apunta al protocolo: 3
sesiones de entrenamiento en vez de 4 (−25 % de datos), más posiblemente el congelado
parcial y unos hiperparámetros heredados del montaje con fuga.

Conclusión operativa: **las comparaciones internas son válidas y la fusión funciona; los
números absolutos no son comparables con la literatura hasta arreglar el protocolo.**

### Siguiente paso decidido: líneas base unimodales (estudios 66 y 67)

Antes de re-correr nada con el protocolo estándar, medir qué da un solo stream bajo el
protocolo actual, para responder la pregunta que decide todo: **¿la fusión aporta algo?**

- **66** — solo audio (Wav2Vec2-base)
- **67** — solo texto (DistilBERT sobre transcripción de referencia, sin prosodia)

Mismo congelado parcial, misma proyección a 256, misma cabeza MultiScaleTemporal, mismo
optimizador, schedule, label smoothing y folds que el 45. `UnimodalBaselineModel` en
`src/ablation_studies/unimodal_baseline.py`. No cargan ViT, así que son bastante más
rápidos que el trimodal.

Lectura:

- si **66 ≈ 45 (≈70.7)** → las ramas de texto e imagen no suman nada y el artículo tiene
  que decirlo; la contribución pasa a ser el análisis, no la arquitectura.
- si **66 ≈ 65 y 45 ≈ 70.7** → la fusión aporta ~5 pp y el número absoluto bajo es
  cuestión de protocolo (3 sesiones de entrenamiento en vez de 4).
- si **67 sale alto** → buena parte del rendimiento vive en la transcripción humana, lo
  que hace obligatoria la condición ASR antes de reenviar.

**Trampa encontrada al construirlas**: `create_optimizer()` en `src/train.py` asigna las
tasas por **subcadena del nombre del parámetro** — `audio_encoder` → `lr_audio`,
`text_encoder` → `lr_text`, el resto → `lr_fusion`. Un modelo que llame a su encoder de
otra forma entrenaría el encoder a 5e-6 sin ningún aviso.

> **CORRECCIÓN (17/09/2026).** Esa nota apuntaba a la función equivocada. La ruta de
> ablación no usa `create_optimizer()` de `src/train.py` sino
> `create_ablation_optimizer()` de `src/ablation_studies/utils.py`, y allí los estudios
> 44, 45 y 66-69 ni siquiera llegan a agruparse: ver «El régimen de entrenamiento descrito
> nunca se ejecutó» más abajo.


### Los tres protocolos (12-15 sep 2026) — `--protocol {standard,speaker,legacy}`

`run_ablation_study_kfold()` acepta ahora tres repartos, y el elegido queda escrito en
`kfold_summary_*.json` y en cada fold, para que ningún número vuelva a quedar huérfano de
su protocolo.

- **`standard`** (por defecto): entrena con las **cuatro** sesiones que no son test; la
  validación es una **muestra estratificada por clase del 12 %** sacada de dentro. Todos
  los hablantes de entrenamiento están representados en validación, así que la señal para
  elegir el checkpoint es estable. Test sigue siendo independiente de sesión y de
  hablante, que es lo que exige la comparación con la literatura.
- **`speaker`**: igual, pero la validación es **un solo hablante** de las cuatro sesiones
  de entrenamiento, rotando por fold. Independiente de hablante en los tres conjuntos.
  Medido: demasiado ruidoso (σ de val UA 8.00 pp) — se conserva como variante, no por
  defecto.
- **`legacy`**: 3 + 1 + 1, lo corrido hasta el 12/09/2026. Se conserva solo para
  reproducir esos números; usa un 25 % menos de datos y no es comparable con nada.

Reparto de `speaker` (verificado sobre ids reales: los tres conjuntos son disjuntos por
hablante en los cinco folds y suman 5 531 exactos):

| Fold (test) | sesiones de entrenamiento | hablante de validación | train | val | test |
|---|---|---|---|---|---|
| S1 | 2, 3, 4, 5 | Ses02_F | 3 965 | 481 | 1 085 |
| S2 | 1, 3, 4, 5 | Ses03_M | 3 879 | 629 | 1 023 |
| S3 | 1, 2, 4, 5 | Ses04_F | 3 852 | 528 | 1 151 |
| S4 | 1, 2, 3, 5 | Ses05_M | 3 849 | 651 | 1 031 |
| S5 | 1, 2, 3, 4 | Ses01_F | 3 762 | 528 | 1 241 |

La primera regla que escribí (`speakers[fold_idx % 8]`) repetía Ses02_F en dos folds y no
usaba nunca a los de las sesiones 4 y 5; verificado y corregido antes de gastar GPU.

Con `standard` y `speaker` el entrenamiento medio sube a **~3 860 utterances frente a
3 259 del legacy (+18 %)**.

Frase para el artículo: *«Five-fold leave-one-session-out. Each fold trains on the four
remaining sessions, holding out a class-stratified 12 % of them for validation and
checkpoint selection; the test session is unseen in both speaker and session identity.»*

---

### REFUTACIÓN DE LA HIPÓTESIS DEL PROTOCOLO (15 sep 2026)

La hipótesis era que los 8.5 pp de diferencia contra Pepino et al. (72.50 con el mismo
encoder) venían del protocolo: menos datos de entrenamiento y una validación ruidosa.
**Está refutada.** El mismo estudio 66 (solo audio, Wav2Vec2-base, semilla 42) corrido
bajo los tres repartos:

| Protocolo | train medio | S1 | S2 | S3 | S4 | S5 | **Test UA** | Test WA | Val UA |
|---|---|---|---|---|---|---|---|---|---|
| `legacy` 3+1+1 | 3 259 | 68.43 | 69.87 | 59.45 | 59.57 | 62.79 | **64.02 ± 4.89** | 62.29 ± 3.74 | 68.46 ± 1.69 |
| `speaker` (1 hablante val) | 3 861 | 68.16 | 69.77 | 58.46 | 60.21 | 61.57 | **63.63 ± 5.02** | 61.11 ± 4.49 | 65.07 ± 8.00 |
| `standard` (12 % estratif.) | 3 861 | 66.74 | 65.93 | 60.88 | 61.31 | 64.79 | **63.93 ± 2.68** | 61.96 ± 0.90 | 70.30 ± 2.37 |

Tres repartos, una sola respuesta: **~64 UA**. El rango entre el mejor y el peor es de
0.39 pp, muy por debajo de la σ entre folds de cualquiera de ellos. Ni el +18 % de datos
de entrenamiento ni la calidad de la validación mueven el número de test.

Lo que sí hizo el reparto estratificado —y es un hallazgo metodológico aprovechable:

- **σ de val UA: 8.00 → 2.37 pp**, y la media de val UA sube de 65.07 a 70.30. La
  validación de un solo hablante es una señal tan ruidosa que elegía el checkpoint casi al
  azar.
- **σ de test UA: 4.89 → 2.68 pp; σ de test WA: 3.74 → 0.90 pp.** Mirando fold a fold, el
  reparto estratificado sube los folds malos (S3 59.45→60.88, S4 59.57→61.31, S5
  62.79→64.79) y baja los buenos (S1 68.43→66.74, S2 69.87→65.93). No mejora el modelo:
  deja de premiar los folds afortunados. **Buena parte de la σ de 4–5 pp que la literatura
  de SER reporta entre folds LOSO es ruido en la selección del checkpoint, no varianza del
  hablante.** Vale una subsección; es transferible a cualquier trabajo con LOSO en IEMOCAP.

Por tanto, para el artículo: **`standard` es el protocolo por defecto**, no porque suba el
número, sino porque es el comparable con la literatura y porque su σ es honesta. Todas las
corridas del legacy quedan como histórico.

**Hipótesis viva tras esta refutación.** Si no son los datos ni la validación, quedan dos:

1. **Selección de capas.** CAMF usa `last_hidden_state` de wav2vec2-base. Pepino et al.
   usan una suma ponderada aprendida de las 13 capas ocultas, y su contribución central es
   precisamente esa: las capas intermedias del SSL cargan la información paralingüística,
   la última está especializada hacia fonética. Es el estudio **68** (mismo 66 con
   `audio_weighted_hidden_states: true`, un solo factor cambiado). **En curso.**
2. **Hiperparámetros.** lr, schedule, épocas y label smoothing siguen siendo los heredados
   del montaje con fuga léxica, ajustados sobre un corpus que ya no existe.

Si el 68 sube de ~64 a ~70, el cuello de botella era la selección de capas y la corrección
se aplica igual a las tres ramas del trimodal. Si se queda en 64, el siguiente paso es un
barrido de hiperparámetros sobre el corpus limpio, y hay que escribirlo así en el artículo:
el número absoluto está limitado por un montaje que no se re-optimizó, no por la
arquitectura.

---

### REFUTACIÓN DE LA HIPÓTESIS DE LAS CAPAS — estudio 68, 5 folds (16 sep 2026)

Un solo factor cambiado respecto del 66: en vez de `last_hidden_state`, una combinación
softmax aprendida de las 13 capas ocultas de wav2vec2-base (Pepino et al. 2021). Mismo
protocolo `standard`, misma semilla, mismos folds.

| Fold (test) | 66 `last_hidden_state` | 68 suma ponderada | Δ test | val 66 | val 68 |
|---|---|---|---|---|---|
| S1 | 66.74 | 67.35 | +0.60 | 69.30 | 70.91 |
| S2 | 65.93 | 69.46 | +3.53 | 69.09 | 76.84 |
| S3 | 60.88 | 58.04 | −2.84 | 74.53 | 74.18 |
| S4 | 61.31 | 61.38 | +0.06 | 69.43 | 69.34 |
| S5 | 64.79 | 64.06 | −0.73 | 69.16 | 69.30 |
| **media ± σ** | **63.93 ± 2.68** | **64.06 ± 4.56** | **+0.13 ± 2.31** | 70.30 ± 2.37 | 72.11 ± 3.31 |

**Pareado: +0.13 ± 2.31 pp, t(4) = 0.12, 3 de 5 folds.** Un cero. No es que el efecto sea
pequeño y no lo detectemos con 5 folds: la media del efecto *es* 0.13 pp.

Lo que sí mueve la suma ponderada:

- **Validación +1.81 pp** (70.30 → 72.11) sin nada en test.
- **El hueco validación−test se abre de 6.37 a 8.06 pp.**
- **σ de test casi se dobla: 2.68 → 4.56 pp** (y la de WA, 0.90 → 3.57).

Es decir: las capas extra le dan al modelo más formas de ajustar validación, y ninguna
transfiere. Esto **no contradice a Pepino et al.** —su medición es sobre un montaje
distinto— pero sí dice que en este montaje no es el cuello de botella.

*(El fold S4 está marcado `resumed: true`: se reanudó desde la época 20 tras un reinicio
de la máquina. Su Δ es +0.06 pp, así que no cambia la conclusión, pero si el 68 acabara
sosteniendo alguna afirmación del artículo habría que rehacerlo limpio. Dado que es un
nulo, no hace falta.)*

### Dónde queda el diagnóstico del número absoluto

Dos hipótesis puestas a prueba, las dos refutadas con un solo factor cambiado:

| Hipótesis | Prueba | Resultado |
|---|---|---|
| El protocolo deprime el número (menos datos, validación ruidosa) | 66 bajo `legacy` / `speaker` / `standard` | 64.02 / 63.63 / 63.93 — **rango de 0.39 pp** |
| La selección de capas (`last_hidden_state` vs suma ponderada) | 68 vs 66 | **+0.13 ± 2.31 pp** |

Queda la tercera: **los hiperparámetros**, que siguen siendo los heredados del montaje con
fuga léxica y nunca se re-optimizaron sobre el corpus limpio. La evidencia a favor es
consistente en las dos corridas: con `train UA ≈ 78 %` y `val UA ≈ 67 %` dentro de un
mismo fold, y un hueco validación−test de 6-8 pp, esto es **sobreajuste**, no falta de
capacidad. Sospechosos concretos, por orden de tamaño:

1. **La cabeza MultiScaleTemporal**: BiLSTM de 2 capas × 512 + self-attention + tres
   pooling, sobre 3 860 utterances. Pepino et al. usan pooling temporal y una densa. Para
   una rama de audio sola es una cantidad de parámetros difícil de justificar.
2. **`warmup_ratio: 0.1` nunca se ejecuta** (ya documentado en §2): se declara y no se usa.
3. **lr_audio 3e-5 sobre 6 bloques descongelados**, sin re-ajustar desde el cambio de corpus.
4. `dropout 0.5` + `label_smoothing 0.1` elegidos sobre el corpus con fuga.

### CORRECCIÓN MAYOR: «Pepino 72.50» no existe (17 sep 2026)

Los estudios 66 (protocolo), 68 (capas) y 69 (cabeza) se lanzaron persiguiendo un hueco de
8.5 pp contra «Pepino 72.50 (solo audio)», cifra que estaba en la tabla de estado del arte
del manuscrito de CAMF y que se arrastró a estas notas como contexto **sin verificarla**.
Verificada contra la fuente, está mal. Pepino, Riera y Ferrer (Interspeech 2021) reportan:

| Lo que reportan | Valor |
|---|---|
| Mejor resultado, wav2vec2-base, 4 clases, 5-fold LOSO | **67.2 ± 0.7** average recall (= UA) |
| Su configuración en la tabla comparativa | **65.8 ± 0.3** |
| Régimen del encoder | **congelado** — *«the weights of the wav2vec 2.0 model remain unaltered, so it serves as a feature extractor»* |
| Cabeza | 2 densas de 128 por frame → media temporal → softmax |
| Su σ | sobre **5 semillas**, no sobre folds — no comparable con la nuestra |

Fuente: `isca-archive.org/interspeech_2021/pepino21_interspeech.pdf`.

**Nuestra línea base (66) da 63.93 ± 2.68. El hueco real es de 3.3 pp contra 67.2, y de
1.9 pp contra 65.8. No de 8.5.**

Consecuencia que hay que asumir antes de reescribir: **toda la tabla de comparación con el
estado del arte del manuscrito hay que re-verificarla contra las fuentes primarias**,
incluidos CTNet 78.30 y UniMSE 79.14, que son los números con los que se decidió el marco
del artículo. Una cita mal copiada ya desvió tres experimentos.

Esto además reinterpreta el estudio 68: Pepino obtiene su ganancia de la suma ponderada de
capas **con el encoder congelado**; nosotros descongelamos los 6 bloques superiores y no
ganamos nada (+0.13 pp). Es coherente: si puedes adaptar las capas altas, re-ponderarlas
no añade información. Eso es un **efecto condicional**, no un fracaso, y es una fila de
ablación defendible con una explicación mecánica.

### EL RÉGIMEN DE ENTRENAMIENTO DESCRITO NUNCA SE EJECUTÓ (17 sep 2026)

`create_ablation_optimizer()` en `src/ablation_studies/utils.py` tiene dos ramas. Los
estudios que no están en la tupla `studies_with_separate_lr` caen en la rama `else`, cuyo
propio comentario dice *«Other studies: Standard grouping (fusion only, encoders frozen)»*.
Esa suposición es **falsa** para los estudios 44, 45 y 66-69, que descongelan 6 bloques de
wav2vec2 (y 4 de DistilBERT en el trimodal).

Medido:

| Estudio | Encoder de audio | Tasa realmente aplicada |
|---|---|---|
| 44, 45 (trimodal) | 6 bloques descongelados | **una sola, 5e-6, para todo** |
| 66, 67, 68, 69 | 6 bloques descongelados | **una sola, 5e-6, para todo** |

**`learning_rate_audio: 3.0e-5` y `learning_rate_text: 2.0e-5` nunca se aplicaron en
ninguna de esas corridas.** El número principal del trimodal (estudio 45, 70.73) se
entrenó con todo a 5e-6.

Y explica los tres nulos de forma coherente: a 5e-6 uniforme la cabeza apenas entrena —por
eso quitarle 10 M de parámetros (69) no movió nada— y los 13 logits de capa apenas se
separan del uniforme —por eso el 68 dio +0.13 pp.

**Para el artículo:** la sección de detalles de implementación no puede describir tasas
diferenciadas hasta que se ejecuten. Es la tercera vez en esta auditoría que el texto
describe algo que el código no hace (las otras dos: la cabeza «multiescala temporal» y el
desplazamiento de pitch). Conviene una frase honesta en Limitaciones sobre la verificación
código-texto, y `scripts/report_optimizer.py` deja la comprobación en un comando.

Corregido sin romper nada: se añade la bandera explícita `training.use_param_groups`, que
queda escrita en el `config.json` de cada fold. **No se añaden estudios a la tupla a
posteriori**, porque eso cambiaría en silencio el comportamiento de corridas ya hechas. La
rama de tasa única ahora **avisa** si le llega algún parámetro de encoder entrenable.

### CONFIRMADO: las tasas eran el cuello de botella — estudio 71 (18 sep 2026)

El estudio 66 con **un solo factor cambiado** (`use_param_groups: true`, que hace que se
apliquen las tasas que el config ya declaraba):

| Fold (test) | 66 (todo a 5e-6) | 71 (3e-5 / 5e-6) | Δ | val 66 | val 71 |
|---|---|---|---|---|---|
| S1 | 66.74 | 70.97 | +4.23 | 69.30 | 77.58 |
| S2 | 65.93 | 73.42 | +7.48 | 69.09 | 76.52 |
| S3 | 60.88 | 66.23 | +5.35 | 74.53 | 79.81 |
| S4 | 61.31 | 65.59 | +4.27 | 69.43 | 75.11 |
| S5 | 64.79 | 68.75 | +3.96 | 69.16 | 75.75 |
| **media ± σ** | **63.93 ± 2.68** | **68.99 ± 3.27** | **+5.06 ± 1.46** | 70.30 | 76.95 |

**Pareado: +5.06 ± 1.46 pp, t(4) = 7.76, 5 de 5 folds.** WA: 61.96 → 67.82.

Tras cuatro hipótesis, esta es la explicación. No era el protocolo, ni la selección de
capas, ni el tamaño de la cabeza: era que **el régimen de entrenamiento que el artículo
describe nunca se ejecutó**. Aplicarlo vale 5 pp.

Con esto, la línea base de solo audio queda en **68.99, por encima de los 67.2 de Pepino
et al.** El «hueco de 8.5 pp» que motivó toda esta cadena ya no existe: era mitad cita
equivocada, mitad bug del optimizador.

### LA CONTRIBUCIÓN CENTRAL, MEDIDA BIEN POR PRIMERA VEZ — estudio 72 (18 sep 2026)

Trimodal con las tasas corregidas, protocolo `standard`, contra la línea base de audio
bajo **el mismo régimen y el mismo protocolo** (estudio 71). Pareado por fold:

| Fold (test) | 71 solo audio | 72 trimodal | Δ | val 71 | val 72 |
|---|---|---|---|---|---|
| S1 | 70.97 | 74.70 | +3.73 | 77.58 | 81.12 |
| S2 | 73.42 | 81.22 | +7.81 | 76.52 | 78.68 |
| S3 | 66.23 | 69.86 | +3.62 | 79.81 | 80.74 |
| S4 | 65.59 | 74.60 | +9.01 | 75.11 | 79.78 |
| S5 | 68.75 | 73.20 | +4.45 | 75.75 | 80.20 |
| **media ± σ** | **68.99 ± 3.27** | **74.71 ± 4.13** | **+5.72 ± 2.51** | 76.95 | 80.10 |

**La fusión aporta +5.72 ± 2.51 pp, t(4) = 5.10, 5 de 5 folds.** WA: 67.82 → 73.42.

**La tesis del artículo se sostiene, y con números mejores que los del manuscrito
rechazado**: 74.71 frente a 70.73, contra una línea base que ahora está bien entrenada
(68.99, por encima de los 67.2 de Pepino) en vez de lisiada. La ganancia sale algo menor
que el «+6.71» original, que es lo esperable: aquel comparaba dos ramas frenadas.

Estas son las cifras que van al artículo:

| Valor | Medición |
|---|---|
| CAMF trimodal (sin compuerta, estudio 74) | **75.57 ± 3.06** UA · **74.26** WA |
| CAMF trimodal (con compuerta, estudio 72) | 74.71 ± 4.13 UA · 73.42 ± 3.97 WA |
| Línea base solo audio | **68.99 ± 3.27** UA · 67.82 ± 2.46 WA |
| Línea base solo texto | **67.82 ± 3.20** UA · 66.09 ± 2.96 WA |
| Ganancia sobre solo audio | **+5.72 ± 2.51 pp**, t(4)=5.10, 5/5 folds |
| Ganancia sobre solo texto | **+6.89 ± 1.13 pp**, t(4)=13.58, 5/5 folds |
| Protocolo | LOSO 5-fold, `standard`, semilla 42, corpus completo (5 531) |

### TABLA DE ABLACIÓN CERRADA — estudio 73 (18 sep 2026)

| Fold (test) | 71 solo audio | 73 solo texto | 72 trimodal | 72−71 | 72−73 |
|---|---|---|---|---|---|
| S1 | 70.97 | 66.92 | 74.70 | +3.73 | +7.78 |
| S2 | 73.42 | 73.06 | 81.22 | +7.81 | +8.16 |
| S3 | 66.23 | 64.33 | 69.86 | +3.62 | +5.52 |
| S4 | 65.59 | 67.57 | 74.60 | +9.01 | +7.03 |
| S5 | 68.75 | 67.24 | 73.20 | +4.45 | +5.96 |
| **media ± σ** | **68.99 ± 3.27** | **67.82 ± 3.20** | **74.71 ± 4.13** | **+5.72 ± 2.51** | **+6.89 ± 1.13** |

| Comparación pareada | Ganancia | t(4) | Folds |
|---|---|---|---|
| Fusión vs solo audio | **+5.72 ± 2.51 pp** | 5.10 | 5/5 |
| Fusión vs solo texto | **+6.89 ± 1.13 pp** | **13.58** | 5/5 |

WA: audio 67.82 ± 2.46, texto 66.09 ± 2.96, trimodal 73.42 ± 3.97.

**La fusión gana a las dos ramas por separado, en los cinco folds, con las dos
comparaciones bajo el mismo protocolo y el mismo régimen.** Esta es la tabla que el
manuscrito original no podía construir y que R3 pidió explícitamente.

Detalle que vale para el texto: **la ganancia sobre texto es la más consistente de las dos**
(σ pareada 1.13 frente a 2.51; t = 13.58 frente a 5.10). La rama de audio ayuda al texto de
forma muy estable; el texto ayuda al audio de forma más variable según el fold. Eso es un
resultado analizable, no relleno.

Y algo que ordena la historia entera: en el régimen lisiado, el texto (66.89) parecía
**mejor** que el audio (63.93). Con las tasas corregidas se invierte: audio 68.99, texto
67.82. Aquel orden era un artefacto del optimizador — la rama de audio era la que más
sufría al entrenar a 5e-6, porque es la que más bloques descongelados tiene.

Contexto honesto: 74.71 queda por debajo de MulT
(76.10), Kim (77.70), CTNet (78.30) y UniMSE (79.14) **según las cifras del manuscrito,
que siguen sin verificar contra la fuente primaria** — la única que se ha verificado
resultó estar mal. Verificarlas es ahora barato y obligatorio.

### AUDITORÍA DE LA TABLA DE ESTADO DEL ARTE (18 sep 2026) — ninguna de las cuatro es comparable

Verificadas contra la fuente primaria las cifras con las que el manuscrito se comparaba:

| Cita en el manuscrito | Lo que reporta la fuente | Tarea real |
|---|---|---|
| **Pepino 72.50** (solo audio) | **67.2 ± 0.7** average recall | 4 clases, LOSO 5-fold, encoder congelado — **la única comparable** |
| **MulT 76.10** | Cuatro accuracies **binarias por emoción**: 90.7 / 86.7 / 87.4 / 72.4 (alineado); 84.8 / 77.7 / 73.9 / 62.5 (no alineado) | *«this is a multilabel task (e.g., a person can be sad and angry simultaneously)»* — **no es clasificación 4-way** |
| **UniMSE 79.14** | **70.56 accuracy / 70.66 weighted F1** | **6 clases** (joy, sadness, angry, neutral, excited, frustrated) **+ contexto conversacional de ±2 turnos**; split 5 354 / 528 / 1 650 |
| **CTNet 78.30** | número no descargado (límite de peticiones) | **ERC**: 6 clases, sesiones 1-4 / sesión 5 fija, weighted F1, con contexto conversacional |

Fuentes: `isca-archive.org/interspeech_2021/pepino21_interspeech.pdf`, `aclanthology.org/P19-1656.pdf`,
`arxiv.org/html/2211.11256`, benchmark «IEMOCAP 6-class ERC».

**Tres de cuatro cifras están mal copiadas, y tres de cuatro miden otra tarea.** El
«hueco contra el estado del arte» que motivó los estudios 66-standard, 68 y 69 estaba
construido sobre números que no miden lo mismo que nosotros.

**Lo que esto NO significa.** No significa que CAMF sea estado del arte. 6 clases es más
difícil que 4; la clasificación binaria por emoción vive en otro espacio de salida; los
modelos de ERC reciben contexto que nosotros no usamos. La conclusión honesta es
**«no comparable»**, en ninguna de las dos direcciones. Escribir «superamos a UniMSE»
sería repetir exactamente el error por el que rechazaron el artículo.

**Qué hacer en la reescritura.** Sustituir la tabla de estado del arte por una restringida
a trabajos con **protocolo equiparable** —IEMOCAP, 4 clases, nivel de utterance,
independiente de hablante—, con una columna explícita de protocolo, número de clases,
métrica y si usa contexto conversacional o transcripción humana. Las que no encajen van a
una tabla aparte, con la diferencia de tarea dicha en una frase. Contra Pepino (67.2, solo
audio) nuestra línea base de audio da **68.99** y el trimodal **74.71**, y esa sí es una
comparación que se sostiene.

Esa tabla, bien hecha, es por sí sola una contribución: **la literatura de SER sobre
IEMOCAP mezcla rutinariamente cuatro protocolos distintos en la misma columna de
resultados**, y este artículo puede documentarlo con cuatro casos verificados. Encaja con
lo que R1 premió —la honestidad del reporte— y convierte una debilidad («nuestro número es
más bajo») en el argumento del artículo.

### CRITERIO DE INCLUSIÓN PARA LA TABLA COMPARATIVA (18 sep 2026)

Un trabajo entra en la tabla principal **solo si** cumple las cinco:

1. IEMOCAP, **4 clases** con *excited* fusionado en *happy* (5 531 utterances).
2. **Nivel de utterance**, sin contexto conversacional de turnos vecinos.
3. **Independiente de hablante**: LOSO 5-fold o equivalente. Folds aleatorios **no**.
4. Métrica **UA (= average recall / macro recall)** o WA, declarada cuál.
5. Dice si el texto es transcripción humana o ASR.

Esta definición es exactamente la del benchmark «IEMOCAP 4-class SER»: *«the original happy
and excited labels are merged»*, *«speaker-independent generalization with the five original
dyadic sessions as folds, leaving one complete session out for testing in each fold»*. Vale
citarla para justificar el criterio.

Estado de la verificación:

| Trabajo | Cifra real | ¿Entra? | Por qué |
|---|---|---|---|
| Pepino et al. 2021 | 67.2 ± 0.7 UA (solo audio, encoder congelado) | **Sí** | Cumple las cinco |
| Yoon et al. 2018 (MDRE) | 0.718 WAP | **No** | **5-fold aleatorio 8:0.5:1.5, dependiente de hablante**, y la métrica es *weighted average precision* |
| MulT (Tsai et al. 2019) | binarias por emoción | No | Multietiqueta binaria |
| CTNet (Lian et al. 2021) | — | No | ERC: 6 clases + contexto conversacional |
| UniMSE (Hu et al. 2022) | 70.56 acc / 70.66 W-F1 | No | 6 clases + contexto de ±2 turnos |

**El caso Yoon es el ejemplo más limpio que vamos a encontrar y merece ir en el artículo.**
Usa **exactamente nuestro corpus**: 5 531 utterances, 1636 happy / 1084 sad / 1103 angry /
1708 neutral — los mismos conteos que `verify_iemocap.py`. Mismo dato, mismas clases. La
única diferencia es el reparto: folds aleatorios en vez de LOSO, con lo que los mismos
hablantes aparecen en train y test. Es la demostración de que **el protocolo, y no el
modelo, explica buena parte de la dispersión entre cifras publicadas** — y lo podemos
enseñar sin acusar a nadie de nada, porque son dos elecciones legítimas mal mezcladas en la
misma columna.

Candidatos pendientes de verificar (empezar por aquí, con la revisión sistemática propia
como fuente de la lista): emotion2vec (Ma et al. 2023 — ya está en el repo como encoder del
estudio 57), Morais et al. 2022, Siriwardhana et al. 2020, Makiuchi et al. 2021, Zou et al.
2022, y los benchmarks de fine-tuning de wav2vec2/HuBERT/WavLM.

### Qué queda en pie y qué queda anulado

| Resultado | Estado |
|---|---|
| Fuga léxica de IEMOCAP guionizado (18.3 / 23.3 / 14.8 %) | **En pie.** Es del corpus, no del entrenamiento |
| Terciles de prosodia codifican hablante (η², brecha de género 48.4 → 15.9 pp) | **En pie.** Medido sobre la caché, sin entrenar |
| σ entre folds = ruido de selección de checkpoint (val 1 hablante vs estratificada) | **En pie.** Un solo factor, ambos brazos bajo el mismo régimen |
| El protocolo no mueve el número absoluto (64.02 / 63.63 / 63.93) | **En pie como comparación interna**, pero los tres brazos estaban lisiados: repetir al menos uno con las tasas corregidas antes de publicarlo |
| 68: la suma ponderada de capas no aporta (+0.13 pp) | **Anulado como fila de ablación.** Medido a 5e-6, donde los 13 logits apenas se mueven. Repetir con `use_param_groups` |
| 69: la cabeza de 10 M de parámetros no aporta (+0.73 pp) | **Anulado como fila de ablación.** Misma razón. El hallazgo *estructural* (la cabeza recibe un vector, sus tres «escalas» son copias) **sigue en pie**: es geometría del código, no depende del lr |
| 45: trimodal 70.73 y «+6.71 pp de la fusión» | **Sustituido** por el 72: 74.71 ± 4.13 y +5.72 ± 2.51 pp |
| 66: línea base de audio | **Sustituida** por el 71: 68.99 ± 3.27 |
| 67: línea base de texto | **Sustituida** por el 73: 67.82 ± 3.20 |

### Orden de re-ejecución

1. ~~45 con `use_param_groups` y `standard`~~ — **hecho**: estudio 72, 74.71 ± 4.13.
2. ~~67 (texto) igual~~ — **hecho**: estudio 73, 67.82 ± 3.20. Tabla de ablación cerrada.
3. 68 y 69 igual, si se quieren como filas de ablación.
4. El **70** (replicación de Pepino) baja de prioridad: ya estamos por encima de su cifra,
   así que ha dejado de ser diagnóstico de nada urgente. Conservarlo como comprobación de
   cordura para cuando haya tiempo.

### HALLAZGO: la cabeza «MultiScaleTemporal» no ve ningún eje temporal (16 sep 2026)

Al preparar el estudio 69 salió algo que no es un hiperparámetro mal puesto, sino un
desajuste entre lo que el artículo describe y lo que el código hace.

En `UnimodalBaselineModel.forward` el audio se promedia sobre el tiempo
(`feats.mean(dim=1)`) **antes** de llegar a la cabeza. La cabeza recibe `(B, D)` y hace
`unsqueeze(1)` → `(B, 1, D)`. Sobre una secuencia de longitud 1:

- `mean` sobre dim=1 devuelve el vector de entrada;
- `max` sobre dim=1 devuelve **el mismo vector, idéntico**;
- `attention` con una sola clave tiene softmax trivial (peso 1.0): es una proyección
  lineal de ese mismo vector, no un pooling.

Los tres «pooling types» son tres casi-copias del mismo vector. Ese `(B, 3, D)` de
duplicados alimenta un BiLSTM bidireccional de 2 capas × 512 y una self-attention:

| Componente | Parámetros |
|---|---|
| attn_pool | 263,424 |
| LSTM capa 1 (bi, in=256) | 3,153,920 |
| LSTM capa 2 (bi, in=1024) | 6,299,648 |
| self-attention | 263,680 |
| clasificador final | 8,196 |
| **total** | **9,988,868** |

**El 19 % de los parámetros entrenables de la línea base modela la dinámica temporal de
tres copias de un vector.** De las 768 dimensiones que el pooling manda al `Linear` final,
512 son duplicados exactos.

No es solo de la línea base: el trimodal (estudio 45, vía `Ablation34Model`) instala la
misma cabeza como `self.fusion.classifier`, y en `src/models/fusion.py` se aplica al vector
`fused`. Misma situación.

**Consecuencia para el artículo:** cualquier frase que atribuya a esta cabeza modelado
temporal multiescala hay que retirarla o reescribirla. Verificable en segundos con
`scripts/check_head_69.py`.

**Estudio 69: el 66 con `classification_head_type: "simple"`** —dropout + una densa sobre
el vector agrupado—, un solo factor cambiado. Es además la comparación directa con Pepino
et al.: ellos también colapsan el tiempo con pooling, pero encima ponen una densa, y
reportan 72.50 UA con el mismo encoder frente a nuestros 63.93. Si el 69 sube, el techo lo
ponían esos 10 M de parámetros redundantes y la corrección se aplica igual al trimodal. Si
no sube, toca barrido de lr y regularización, y se escribe en el artículo tal cual: el
número absoluto está limitado por un montaje que no se re-optimizó tras arreglar el corpus,
no por la arquitectura propuesta.


### Incidente de GPU (15 sep 2026) y lo que se cambió para que no vuelva a costar días

`CUDA error: unknown error` a mitad del `backward`, no en la primera iteración. No es un
error de código: los tres folds anteriores corrieron el mismo grafo. Es el runtime de CUDA
cayéndose por debajo —reset del driver por TDR de Windows, la GPU cayéndose del bus,
térmica, o el driver actualizándose en segundo plano. Una vez ocurre, **todas** las
llamadas al runtime siguen fallando hasta reiniciar el proceso.

Dos fallos de ingeniería quedaron al descubierto y están corregidos en
`scripts/run_ablation.py`:

1. El `empty_cache()` del `finally` volvía a lanzar sobre el contexto muerto, la excepción
   escapaba del bucle y **se perdía el resumen de los tres folds que sí habían
   terminado**. Ahora `_clear_cuda_cache()` detecta el contexto muerto, corta el bucle y
   guarda el resumen parcial con `complete: false`.
2. No había forma de reanudar: repetir el estudio significaba repetir los tres folds ya
   pagados. Ahora `--folds 4 5` corre solo esas sesiones y **fusiona** con el resumen en
   disco, descartando los folds cuyo protocolo no coincida. El resumen parcial imprime la
   línea exacta de reanudación.
3. Tampoco había reanudación **dentro** de un fold: cualquier caída lo devolvía a la época
   0, y un fold cuesta 3-4.5 h. Tras la segunda caída (reinicio de Windows a las 19:58 del
   15/09, en la época 23 del mismo fold S4) se añadió `--resume`: `last_model.pt` con
   modelo, optimizador, scheduler, mejor métrica y paciencia, guardado **cada época** con
   escritura atómica. Detalle que habría falseado el número en silencio: al final el fold
   evalúa en test cargando `best_model.pt` de *su* carpeta, así que al reanudar hay que
   copiar allí el mejor checkpoint previo — si no, un fold que reanuda con la paciencia
   avanzada y no vuelve a mejorar se evaluaría con los pesos de la última época.

**Un fold reanudado no es bit-idéntico a uno corrido de un tirón** (el shuffle del
DataLoader arranca de nuevo), así que queda marcado con `resumed: true` y `resume_points`
en su `final_results.json`. El fold S4 del estudio 68 lleva esa marca: se reanudó desde la
época 19 con val UA 67.95 y paciencia 4/8. Si el 68 acaba sosteniendo alguna afirmación del
artículo, volver a correr ese fold limpio antes de publicarlo.

`scripts/rebuild_kfold_summary.py` también tenía un bug: deducía el protocolo solo de
`val_speaker`, así que etiquetaba como `legacy` todas las corridas `standard`. Corregido —
importa, porque la fusión al reanudar descarta por protocolo.

---


## 1. Números medidos, con procedencia

Usar estos y no los del manuscrito anterior.

| Valor | Medición | De dónde sale |
|---|---|---|
| Utterances 4 clases | 5 531 (S1 1085 / S2 1023 / S3 1151 / S4 1031 / S5 1241) | `verify_iemocap.py` |
| Split `standard` (por fold) | 4 sesiones de train, 12 % estratificado a validación, 1 sesión a test | `run_ablation.py --protocol standard` |
| Split `legacy` (histórico) | train 3 259 / val 1 031 / test 1 241 | ídem |
| Distribución de clases | angry 1103, happy 1636, sad 1084, neutral 1708 | ídem |
| Improvisadas / guionizadas | 2 943 / 2 588 | ídem |
| Fuga léxica S1-3 → S5 | 18.3 % global; 23.3 % guion, 14.8 % impro | `verify_iemocap.py --leak-check` |
| σ entre folds LOSO, val de 1 sesión/hablante | 4.1–5.0 pp | `checkpoints/ablation_66/*/camf_loso_*`, `std_*` |
| σ entre folds LOSO, val estratificada | **2.68 pp** (UA) / **0.90 pp** (WA) | `checkpoints/ablation_66/*/std2_*` |
| Solo audio (wav2vec2-base), `standard` | **63.93 ± 2.68** UA · 61.96 ± 0.90 WA | estudio 66 |
| Suma ponderada de las 13 capas del SSL | **+0.13 ± 2.31 pp** (t(4)=0.12, 3/5 folds) — nulo | estudio 68 vs 66 |
| Solo texto (DistilBERT, transcripción humana), `legacy` | 66.89 ± 2.85 UA | estudio 67 |
| Trimodal, `legacy` | 70.73 ± 3.37 UA · 69.65 ± 2.21 WA | estudio 45 |
| Ganancia pareada de la fusión | +6.71 ± 1.68 pp vs audio (5/5 folds, p≈0.0009); +3.85 ± 1.77 vs texto (5/5, p≈0.008) | 45 vs 66/67, `legacy` |
| ViT descongelado | −10.65 pp UA (79.23 → 68.58, un solo factor) | `ablation_45_20260507_065116` |
| η² del hablante | pitch 33.1 %, energía 22.2 %, tasa 3.9 % | caché SpeechCue v4, 5 531 utt. |
| F0 mediana por género | M 149.3 Hz, F 221.2 Hz | ídem |
| Terciles globales de pitch | 154.4 / 219.0 Hz (sesiones 1-3) | ídem |
| Token de pitch, terciles globales | M: 52.7/32.2/15.1 %; F: 4.3/42.5/53.2 % | ídem |
| Token de pitch, normalizado por hablante | M: 41.5/23.0/35.5 %; F: 25.6/41.5/32.9 % | ídem |
| Brecha de género en la proporción `[P_LOW]` | 48.4 pp global → **15.9 pp** normalizado | ídem |
| Tasa silábica | mediana 2.95 sil/s, p90 4.62, máx 11.1 | ídem |
| Utterances sin F0 medible | 0.0 %, plano entre sesiones | ídem |
| Parámetros totales | 263 427 987 (estudio 45) | `scripts/report_finetuning.py` |
| Parámetros entrenables | 87 184 403 (33.1 %) | ídem |
| — audio (wav2vec2-base) | 42 528 768 / 94 371 712 · últimos 6 de 12 bloques, extractor conv congelado | ídem |
| — texto (DistilBERT) | 28 351 488 / 66 362 880 · últimos 4 de 6 bloques | ídem |
| — visión (ViT-B/16) | 0 / 86 389 248 · **completamente congelado** | ídem |
| — fusión + cabeza MST | 16 304 147 / 16 304 147 | ídem |
| SpeechCue | **6 912** parámetros aprendibles (9 × 768). Estudio 64 − estudio 45 = 6 912 exactos | ídem |

---

## 2. Cambios sección por sección

### Abstract
- Fuera «surpassing several previously proposed bimodal models».
- Reportar **media ± σ sobre folds**, no un número único.
- Decir el protocolo: LOSO 5-fold sobre IEMOCAP completo.

### 1. Introducción
- Numeración de referencias del bloque de contribuciones: DCA [36]→[34], IACA [37]→[35],
  M3ER [38]→[36], HBAF [39]→[37], ConflictAwareAH [40]→[38].
- «puente prosódico interpretable [23]» → [23] es Devlin (BERT). Debe ser [19] (Rahman).
- Reformular la contribución 5: la rejilla de 54 configuraciones ya no es el argumento;
  el argumento es el protocolo controlado.

### 2.4 Prosodia en modelos de lenguaje
- Faltan citas: ProsodyBERT, EnCodec, DAC se nombran sin referencia.
- Revisar la comparación con esos trabajos a la luz de lo que SpeechCue hace realmente.

### 3.1 / 3.2 Ramas de audio e imagen
- El régimen de congelado del texto ahora sí coincide con el código y con la Figura 1.
- Añadir conteo de parámetros por rama (`report_finetuning.py`).
- Justificar el ViT congelado con la ablación de −10.65 pp (estudio 65), no con una
  afirmación. **Ojo**: el YAML del estudio 45 tenía `freeze_vision_encoder: false` y había
  además una segunda política de congelado del ViT en `Ablation13Model` que descongelaba
  las últimas 4 capas incluso cuando el flag pedía congelar. Las dos cosas corregidas; la
  política vive ahora solo en `Ablation11Model.apply_finetuning_policy()`.

### 3.3 SpeechCue — reescribir entera
- F0 medio con pYIN sobre frames sonoros, **fmax 300 Hz**, con filtro de salto de octava
  (±1.5× la mediana de la utterance). Explicar por qué: con 400 Hz la media de F0 de los
  hablantes masculinos salía en 165–188 Hz.
- Energía RMS sobre los mismos frames sonoros.
- Tasa silábica sobre la **duración sonora**, no la del fichero (los wavs traen silencio
  de relleno; sin corregir, la mediana bajaba a 2.87 sil/s frente a 4–5 esperables).
- **Normalización por hablante** de log-F0 y log-RMS antes de los terciles, con la tabla
  de η² y la tabla de distribución de tokens por género como justificación. Declarar que
  es adaptación no supervisada al hablante (**transductiva**): usa el audio de los
  hablantes de test, nunca sus etiquetas.
- Terciles ajustados **solo sobre las sesiones de entrenamiento de cada fold**.
- **Normalización de amplitud antes del tracker.** pYIN recibe una copia normalizada por
  pico; la energía se mide sobre la señal original. Sin esto, la sesión 3 (unas 5 veces
  más silenciosa que la 1) daba 34.3 % de utterances sin F0 frente al 13.7 % de la
  sesión 5, y el F0 masculino salía en 171 Hz de mediana. Con la corrección: **0 % sin F0,
  plano entre sesiones**, y mediana masculina de 149.3 Hz, dentro del rango fisiológico.
  Fallback a YIN sobre los frames por encima de la mediana de energía si pYIN falla.
- La tasa silábica se divide entre el tramo hablado (recorte de silencio, `top_db=35`),
  aunque en IEMOCAP eso solo quita el 2 %: los wavs ya vienen segmentados apretados. La
  mediana de 2.95 sil/s es correcta para habla conversacional (~125 palabras/min); no
  confundir con la tasa de articulación (4-5 sil/s), que excluye pausas.
- Nueve tokens añadidos con `add_tokens` + `resize_token_embeddings`, insertados tras
  `[CLS]`. Solo esas nueve filas aprenden (hook de gradiente).

### 3.5 Objetivo de entrenamiento
- **Quitar el warmup**: el log avisa `warmup_ratio is ignored` con scheduler `plateau`.
  O cambiar a `cosine`/`linear` y re-correr.
- Documentar el aumento de forma de onda (ruido SNR 10–30 p=0.3, speed 0.9–1.1 p=0.3,
  pitch ±2 p=0.2, time masking p=0.5) — no se menciona en ningún sitio.

### 4.1 Dataset y protocolo
- 5 531 utterances, LOSO 5-fold. Tabla 1 con los conteos por sesión.
- Reportar el N del test y los *supports* por clase en cada tabla de resultados.
- Describir el reparto exacto: cuatro sesiones de entrenamiento, 12 % estratificado por
  clase para validación y selección de checkpoint, una sesión completa a test. Decir que
  el test es independiente de sesión **y** de hablante.
- Justificar la validación estratificada con la medición: con validación de un solo
  hablante la σ de val UA es 8.00 pp y la de test UA 5.02; con la estratificada bajan a
  2.37 y 2.68 con la misma media de test. Es selección de checkpoint, no capacidad.

### 4.3 Detalles de implementación
- Versiones reales: PyTorch 2.7.1, Transformers 4.57.1 (el artículo dice 2.2 / 4.41).
- Tabla de parámetros totales y entrenables.
- Declarar el determinismo (cuDNN determinista, semillas, `worker_init_fn`).

### 5.1 Ablación — **rehacer la tabla entera**
- **Retirar la fila de la ablación 34.** Sus números venían de
  `checkpoints/phase1_wavlm_gated/`: WavLM-large + RoBERTa + `gated_interactive_attention`
  y **sin rama ViT**. No es un modelo CAMF. La ablación 34 auténtica da 66.30 UA, por
  debajo de la 25 (72.72).
- Escalera nueva, **un factor por fila**: base → +SpeechCue → +cross-attn → +gate → +MST
  → +régimen de entrenamiento. Todas con wav2vec2 + DistilBERT + ViT.
- Filas adicionales que ya tienen sentido: ViT descongelado (−10.65 pp);
  estudio 45 (prompt en lenguaje natural) frente a estudio 64 (SpeechCue real);
  terciles globales frente a normalizados por hablante.

### 5.2 Comparación con el estado del arte
- Marcar qué filas son cifras reportadas por sus autores y cuáles reproducidas por
  nosotros, con el protocolo de cada una.
- Añadir la línea base trivial: **wav2vec2-base solo audio, mismos folds, mismo régimen**.
- Añadir al menos un audio-LLM evaluado por nosotros (petición de R3).
- Test de significancia: bootstrap sobre las predicciones concatenadas de los 5 folds,
  o McNemar contra la mejor ablación. Ninguna afirmación comparativa por debajo de σ.

### 5.3 Análisis por clase
- Matriz de confusión en **conteos absolutos**, no solo normalizada por fila.
- Liberar `predictions.csv` por corrida.

### 5.4 Limitaciones
- La época del checkpoint es la **33**, no la 41 (41 es donde paró el early stopping,
  33 + paciencia 8). La curva de la Fig. 5 muestra **sobreajuste**: `val_loss` toca mínimo
  en la época 24 (0.7710) y sube hasta 0.9204 mientras `train_loss` cae de 0.76 a 0.52 y
  la UA de entrenamiento llega a 0.94 frente a 0.81 en validación. Describirlo así.
- Marcar la época del checkpoint con una línea vertical en la Fig. 5.
- Añadir: fuga léxica de la porción guionizada (18.3 % de las transcripciones de S5
  aparecen literales en S1-3); normalización transductiva por hablante; ausencia de
  evaluación cruzada de corpus.

### Referencias
- Pasar todo a BibTeX. Sin citar hoy: [2] Schuller, [15] Khosla, [22] Vaswani,
  [24] Park (SpecAugment — se discute en 5.1 sin citarse), [25] Ko, [26] Peters,
  [28] Bahdanau. Citar donde corresponda o eliminar.

---

## 3. Afirmaciones que hay que retirar

1. «surpasses CTNet / UniMSE / Kim et al.» — márgenes de 0.09–1.53 pp frente a σ de 4–5 pp.
2. «+18.28 pp de SpeechCue» — medido sobre un corpus con fuga léxica y con un módulo
   que no era el descrito. Volver a medirlo limpio.
3. «+2.73 pp de la compuerta de conflicto» — comparaba arquitecturas distintas.
4. «nine learnable embeddings» — cierto solo a partir de ahora.
5. «10 % linear warmup» — nunca se ejecutó.
6. «checkpointed at this peak (epoch 41)» — el pico está en la 33.
7. «a well regularized model approaching its data-imposed ceiling» — es sobreajuste.
8. Cualquier atribución de **modelado temporal multiescala a la cabeza MST**: recibe un
   vector agrupado, no una secuencia. Sus tres «escalas» son copias del mismo vector.
9. **Las tasas de aprendizaje diferenciadas** (3e-5 audio / 2e-5 texto / 5e-6 fusión):
   nunca se aplicaron en los estudios 44, 45 ni 66-69. Todo entrenó a 5e-6.
10. El **desplazamiento de pitch** de la augmentación: `_apply_pitch_shift` remuestrea y
   vuelve a remuestrear, que es la identidad. Verificado con un tono de 200 Hz:
   correlación 1.0000 con el original, f0 sin cambio.
11. **Toda la tabla de comparación con el estado del arte.** Verificadas las cuatro:
   Pepino es 67.2 (no 72.50); UniMSE es 70.56/70.66 en 6 clases con contexto conversacional
   (no 79.14); MulT reporta multietiqueta binaria por emoción (no 4-way); CTNet es ERC con
   contexto. Ninguna comparación directa con estas tres últimas es válida.

---

## 4. Lo nuevo que sí se puede defender

- **Protocolo limpio**: LOSO 5-fold, multi-semilla, con σ y test de significancia.
- **Buena parte de la σ entre folds que se reporta en LOSO sobre IEMOCAP es ruido en la
  selección del checkpoint, no varianza del hablante.** Medido con un solo factor
  cambiado: validación de un hablante → σ test UA 5.02 pp; validación estratificada del
  12 % → 2.68 pp (WA: 4.49 → 0.90), con la misma media. Hallazgo metodológico
  transferible y barato de reproducir.
- **Cuantificación de la fuga léxica de IEMOCAP guionizado** (18.3 % / 23.3 % / 14.8 %).
  No lo he visto reportado así en la literatura de SER; vale una subsección.
- **Los terciles globales de prosodia codifican identidad de hablante**, con η² y la tabla
  de distribución de tokens por género. Justifica la normalización y es un hallazgo
  metodológico transferible a cualquier trabajo que discretice prosodia.
- **Vista dual del audio** (SSL en tiempo + ViT sobre el mismo espectrograma): R1 y R3 la
  elogiaron explícitamente. Es la contribución conceptual que sobrevive intacta.
- **Caracterización de la compuerta de conflicto**: R1 destacó la transparencia. Si con la
  arquitectura equiparada la compuerta no ayuda, analizar *cuándo* y *por qué* falla es
  mejor artículo que reclamar +2.73 pp.

## 5. Marco del artículo (decidido el 18 sep 2026)

**En una frase:** un estudio controlado de fusión multistream para SER, cuya contribución
es *doble* — una arquitectura que supera de forma medible a una línea base unimodal bien
entrenada, y el aparato de evaluación que hace esa afirmación verificable, porque en este
subcampo las cifras publicadas no son comparables entre sí.

### Los tres pilares

**1. El resultado, medido bien.** Trimodal **74.71 ± 4.13** UA frente a solo-audio
**68.99 ± 3.27**: **+5.72 ± 2.51 pp, t(4)=5.10, 5/5 folds**, mismo protocolo, mismo régimen,
pareado por fold. La línea base supera además a la única referencia comparable publicada
(Pepino, 67.2, solo audio). **Esto es exactamente lo que pidió R3 y lo que el manuscrito
original no podía enseñar.**

**2. El protocolo.** Buena parte de la σ entre folds que la literatura reporta es **ruido de
selección de checkpoint**, no varianza de hablante: con un solo factor cambiado, σ de test
UA 5.02 → 2.68 y de WA 4.49 → 0.90, con la misma media. Más la fuga léxica cuantificada
(18.3 % / 23.3 % / 14.8 %) y la auditoría de la tabla comparativa (4 de 4 no comparables;
Yoon usa **nuestro mismo corpus exacto** con folds aleatorios en vez de LOSO).

**3. La prosodia codifica identidad de hablante.** η² 33.1 % pitch, 22.2 % energía; brecha
de género en `[P_LOW]` 48.4 → 15.9 pp al normalizar por hablante. Transferible a cualquier
trabajo que discretice prosodia.

### LA COMPUERTA DE CONFLICTO NO APORTA — estudio 74 (20 sep 2026)

Un solo factor: `conflict_aware.enabled: false`, que fuerza la máscara a 0 y deja los pesos
uniformes. Misma arquitectura, mismo número de parámetros, mismas formas.

| Fold (test) | 72 CON compuerta | 74 SIN compuerta | Δ |
|---|---|---|---|
| S1 | 74.70 | 76.11 | +1.41 |
| S2 | 81.22 | 80.25 | −0.98 |
| S3 | 69.86 | 71.96 | +2.11 |
| S4 | 74.60 | 75.49 | +0.90 |
| S5 | 73.20 | 74.05 | +0.85 |
| **media ± σ** | **74.71 ± 4.13** | **75.57 ± 3.06** | **+0.86 ± 1.14** |

**Quitar la compuerta: +0.86 ± 1.14 pp, t(4) = 1.68, 4 de 5 folds. IC 95 % [−0.56, +2.28],
incluye el cero.**

**La compuerta no aporta.** El efecto puntual favorece quitarla, pero no se distingue de
cero con cinco folds: lo honesto es decir que **no se detecta ningún beneficio**, no que
«perjudica». Lo que sí es consistente es que **el modelo sin compuerta tiene menos varianza
entre folds** (σ 4.13 → 3.06) y va mejor en 4 de 5 — un componente que añade dispersión sin
añadir señal.

**Consecuencia directa: el título tiene que cambiar.** «Conflict-Aware» no se sostiene como
contribución. Y el modelo que va al artículo es el **74**, no el 72: mejor media, menos σ, y
un componente menos que justificar.

### La tabla de ablación definitiva

| Configuración | Test UA | Test WA |
|---|---|---|
| Solo audio (71) | 68.99 ± 3.27 | 67.82 ± 2.46 |
| Solo texto (73) | 67.82 ± 3.20 | 66.09 ± 2.96 |
| Trimodal con compuerta (72) | 74.71 ± 4.13 | 73.42 ± 3.97 |
| **Trimodal sin compuerta (74)** | **75.57 ± 3.06** | **74.26** |

| Comparación pareada | Ganancia | t(4) | Folds |
|---|---|---|---|
| Fusión vs solo audio | **+6.58 ± 1.97 pp** | 7.46 | 5/5 |
| Fusión vs solo texto | **+7.75 ± 0.91 pp** | **19.04** | 5/5 |
| Quitar la compuerta | +0.86 ± 1.14 pp | 1.68 | 4/5 |

**La contribución que sobrevive, y es sólida: la fusión multistream.** +6.58 pp sobre audio
y +7.75 pp sobre texto, en los cinco folds, con las dos comparaciones bajo el mismo
protocolo y el mismo régimen. El t de 19.04 sobre texto es el resultado más fuerte del
trabajo.

### POR QUÉ NO APORTA: el detector nunca funcionó (20 sep 2026)

`scripts/diagnose_conflict_gate.py` sobre los checkpoints del estudio 72, sin reentrenar:

| Medición | Valor |
|---|---|
| Tasa de disparo de la compuerta | **94.5 %** de las utterances (min 90.2, max 97.8) |
| Voto uniforme en muestras de conflicto | 17.1 % |
| Elegir la rama de máxima confianza | 12.9 % |
| Elegir una rama al azar | 22.3 % |
| Oráculo (alguna rama acierta) | 58.4 % |

**Causa raíz.** Los tres clasificadores por rama que alimentan la detección
(`audio_classifier`, `text_classifier`, `vision_classifier` dentro de
`ConflictAwareCrossAttentionFusion`) se inicializan con `normal(0, std=0.01)` y se
consumen **únicamente** a través del `argmax` de
`_detect_conflict_prediction_divergence`. **El argmax no es diferenciable: nunca reciben
gradiente.** Se quedan en su inicialización aleatoria durante todo el entrenamiento.

Las tres cifras coinciden con lo que predice el modelo «tres clasificadores aleatorios
sobre 4 clases»:

| Cantidad | Predicho si son aleatorios | Medido |
|---|---|---|
| Tasa de conflicto = 1 − 1/16 | 93.75 % | **94.5 %** |
| Oráculo = 1 − (3/4)³ | 57.8 % | **58.4 %** |
| Una rama al azar | 25 % | **22.3 %** |

Tres coincidencias independientes. **La compuerta llevaba toda la vida del proyecto
disparando sobre ruido, en el 94.5 % de las muestras.**

**Reinterpretación del estudio 74.** No comparó «con compuerta vs sin compuerta»: comparó
**«reponderación guiada por ruido vs peso uniforme»**. Que quitarla diera +0.86 pp y bajara
la σ de 4.13 a 3.06 tiene ahora explicación mecánica exacta.

Es el **quinto** desajuste entre lo que el artículo describe y lo que el código hace, y el
más grave: afecta al componente que da nombre al trabajo.

### Estudio 75: la compuerta con el detector arreglado

Añadida `model.conflict_aware.aux_loss_weight` (0 por defecto, no altera nada de lo ya
medido). Con peso > 0 se suma una entropía cruzada de los tres clasificadores por rama
contra la etiqueta, de modo que reciben gradiente y «conflicto» pasa a significar «las
ramas discrepan» en vez de «tres dados salieron distintos».

El peso 0.3 es **una elección inicial, no un valor ajustado** — declararlo así. Verificable
con `scripts/check_aux_loss.py`, que comprueba que sin la auxiliar el gradiente es
exactamente 0 en los tres y con ella no.

Lectura, contra el 74 (75.57 ± 3.06) y el 72 (74.71 ± 4.13), pareado:

- **75 > 74** → la compuerta aporta cuando el detector funciona. El título «Conflict-Aware»
  se recupera y pasa a ser la contribución, ahora con un mecanismo entendido.
- **75 = 74** → el mecanismo no aporta ni con el detector arreglado. Cierra la pregunta con
  autoridad, no por un bug. Sigue siendo el resultado más honesto y el análisis vale igual.

**En los dos casos el artículo mejora**, porque la sección de análisis deja de decir «no
funcionó» y pasa a decir «no funcionaba por esto, lo arreglamos, y esto es lo que pasó».

### El estudio 75 quedó invalidado la primera vez — y por qué (21 sep 2026)

Primera corrida: 74.92 ± 3.68, aparentemente un nulo limpio. **No medía nada.** Los tres
clasificadores auxiliares acabaron en 0.009968 / 0.009839 / 0.010015 — **idénticos a seis
decimales a los del estudio 72**, sin tocar. La tasa de conflicto seguía en 93.1 %.

Causa: `train_epoch` tiene **dos ramas** con su propio bloque de pérdida, una para precisión
mixta y otra sin ella. El término auxiliar se añadió solo a la de precisión mixta, y el
config lleva `mixed_precision: false`. Estuvo en código muerto los cinco folds.

**La lección, y va en el artículo.** `check_aux_loss.py` probaba el módulo de fusión **en
aislamiento**: confirmaba que el mecanismo funciona y nunca comprobaba que estuviera
conectado a la ruta real. Test en verde, componente desconectado. Es la misma forma de
fallo que los cinco desajustes anteriores — verificar la pieza y no la integración.

Corregido en tres capas: el término vive en una función llamada desde las dos ramas;
`train_epoch` **aborta con RuntimeError** si se pide `aux_loss_weight > 0` y no se aplica
en ninguna iteración; y `scripts/check_aux_integration.py` monta el modelo real con el
config real, da un paso de optimización por la ruta real y comprueba que los pesos se
mueven con la auxiliar y no sin ella.

### RESULTADO DEFINITIVO: la compuerta no aporta ni con el detector funcionando (22 sep 2026)

Relanzado tras verificar la integración (los tres clasificadores se mueven 0.0100 con la
auxiliar y exactamente 0.0 sin ella, por la ruta real):

| Fold (test) | 72 compuerta rota | 74 sin compuerta | 75 compuerta entrenada | 75−74 |
|---|---|---|---|---|
| S1 | 74.70 | 76.11 | 75.52 | −0.59 |
| S2 | 81.22 | 80.25 | 81.13 | +0.88 |
| S3 | 69.86 | 71.96 | 73.14 | +1.18 |
| S4 | 74.60 | 75.49 | 74.38 | −1.12 |
| S5 | 73.20 | 74.05 | 72.79 | −1.26 |
| **media ± σ** | **74.71 ± 4.13** | **75.57 ± 3.06** | **75.39 ± 3.38** | **−0.18 ± 1.14** |

| Comparación | Efecto | t(4) | Folds | IC 95 % |
|---|---|---|---|---|
| 75 vs 74 (compuerta vs sin ella) | **−0.18 ± 1.14 pp** | −0.36 | 2/5 | [−1.59, +1.23] |
| 75 vs 72 (arreglada vs rota) | +0.68 ± 1.53 pp | 0.99 | 2/5 | [−1.23, +2.58] |
| 75 vs 71 (fusión sobre solo audio) | **+6.40 ± 2.04 pp** | **7.01** | **5/5** | [+3.86, +8.93] |

**La compuerta de conflicto no aporta, y ahora se sabe que no es por un bug.** El IC 95 %
de −1.59 a +1.23 pp descarta cualquier efecto de tamaño interesante en las dos direcciones.
Arreglar el detector tampoco recupera nada frente a la versión rota (+0.68, IC incluye 0).

**Esta es la versión buena del resultado negativo**, y sostiene una subsección completa:

1. El mecanismo **estaba desconectado**: argmax no diferenciable → gradiente exactamente 0
   en los tres clasificadores → detección entre predictores aleatorios, disparando en el
   94.5 % de las muestras. Cuatro pruebas independientes (estructura del código, tres
   coincidencias numéricas de comportamiento, pesos sin moverse tras entrenar, gradientes
   exactamente nulos).
2. **Se arregló** con una pérdida auxiliar, verificada de extremo a extremo.
3. **Sigue sin aportar.** La pregunta queda cerrada por medición, no por un fallo.

El modelo del artículo sigue siendo el **74**: 75.57 ± 3.06, la mejor media y la menor σ.
Y «Conflict-Aware» sale del título definitivamente.

### DIAGNÓSTICO DEL MECANISMO, y el estudio 76 (22 sep 2026)

Con el detector ya entrenado (estudio 75 válido), `scripts/diagnose_conflict_gate.py`
mide sobre los checkpoints de los cinco folds:

| Métrica | 72 (detector roto) | 75 (detector entrenado) |
|---|---|---|
| Tasa de conflicto | 94.5 % | **81.6 %** |
| Una rama al azar | 22.3 % | 48.9 % |
| Voto uniforme entre ramas | 17.1 % | **70.5 %** |
| Elegir la rama de mayor confianza | 12.9 % | **69.0 %** |
| Oráculo (alguna rama acierta) | 58.4 % | **91.7 %** |

Esto **explica el resultado nulo con precisión mecánica**, y es lo que convierte una
ablación negativa en un análisis publicable:

- El margen entre lo que hace la compuerta desactivada (voto uniforme, 70.5 %) y el techo
  alcanzable (oráculo, 91.7 %) es de **21.2 pp**. Hay señal de sobra que explotar.
- La confianza estimada **sí informa**: +20.1 pp sobre elegir una rama al azar (48.9 %).
- Pero **seleccionar por confianza pierde 1.5 pp contra promediar** (69.0 vs 70.5). Ésa es
  la razón exacta de que activar la compuerta no gane nada. No es que la señal no exista:
  es que la regla de decisión es peor que la media.
- El 91.7 % es un **techo inalcanzable**, no un objetivo. Requiere saber de antemano qué
  rama acierta. Decirlo así en el texto o un revisor lo dirá por nosotros.

**Por qué esto es atacable, y de ahí el estudio 76.** La `confidence_net` emite tres
sigmoides y su única señal de entrenamiento es la pérdida final de clasificación. **Nada
la empuja a predecir qué rama acierta.** Que aun así bata al azar por 20 pp es casi un
accidente. El estudio 76 le pone el objetivo explícito: entropía cruzada binaria de cada
sigmoide contra si el argmax de esa rama coincide con la etiqueta. Peso 0.3, igual que la
auxiliar y **sin ajustar** — declararlo así.

Implementado en `_conf_calibration_loss()` (src/train.py), llamado desde **las dos** ramas
de `train_epoch`, con aborto por `RuntimeError` si se pide y no se aplica. El objetivo se
calcula bajo `no_grad`, así que el término no contamina a los clasificadores por rama.
Verificación previa: `scripts/check_aux_integration.py --study 76`. Para la calibración el
test **no** puede ser de movimiento de pesos — la `confidence_net` ya se mueve por la
pérdida final — así que deriva **solo** ese término y comprueba que el gradiente sobre la
`confidence_net` no es nulo.

**Los dos desenlaces, y los dos se escriben:**

- **76 > 74.** La selección por confianza calibrada captura parte del margen de oráculo.
  Eso es un método, y es la novedad que pide una revista.
- **76 = 74.** El reponderado por confianza no supera al promedio uniforme ni con la
  confianza entrenada. Sigue siendo publicable y el marco no cambia: *no* «probamos tres
  variantes y una ganó», sino **mostramos que el reponderado por confianza no supera al
  promedio uniforme pese a un margen de oráculo de 21 pp, diagnosticamos por qué, y
  medimos qué pasa cuando se entrena la confianza con un objetivo explícito**. Un
  resultado negativo con mecanismo medido y techo cuantificado tiene más peso que un
  +0.8 pp sin explicación.

### RESULTADO DEL ESTUDIO 76: entrenar la confianza tampoco aporta (22 sep 2026)

| Fold (test) | 72 rota | 74 sin compuerta | 75 entrenada | 76 confianza calibrada |
|---|---|---|---|---|
| S1 | 74.70 | 76.11 | 75.52 | 74.91 |
| S2 | 81.22 | 80.25 | 81.13 | 79.86 |
| S3 | 69.86 | 71.96 | 73.14 | 73.69 |
| S4 | 74.60 | 75.49 | 74.38 | 75.17 |
| S5 | 73.20 | 74.05 | 72.79 | 72.28 |
| **media ± σ** | 74.71 ± 4.13 | **75.57 ± 3.06** | 75.39 ± 3.38 | 75.18 ± 2.85 |

| Comparación pareada | Efecto | t(4) | Folds | IC 95 % | p perm. |
|---|---|---|---|---|---|
| 76 vs 74 (calibrada vs sin compuerta) | **−0.39 ± 1.33** | −0.66 | 1/5 | [−2.04, +1.26] | 0.500 |
| 76 vs 75 (calibrada vs solo detector) | −0.21 ± 0.86 | −0.54 | 2/5 | [−1.28, +0.86] | 0.688 |
| 76 vs 72 (calibrada vs rota) | +0.47 ± 2.05 | 0.51 | 3/5 | [−2.07, +3.01] | 0.813 |
| 76 vs 71 (fusión vs solo audio) | **+6.19 ± 2.52** | **5.50** | **5/5** | [+3.07, +9.32] | 0.0625 |

**La compuerta no aporta en ninguna de sus tres formas.** Rota (72), con el detector
entrenado (75), y con el detector *y* la confianza entrenados con objetivos explícitos
(76): las tres empatan con quitarla. El IC 95 % más ancho de los tres, [−2.04, +1.26],
sigue descartando cualquier efecto de tamaño interesante.

Esto cierra la línea. No queda ninguna variante del mecanismo por probar que no sea
adivinar: se arregló el gradiente, se entrenó el detector, se le dio a la confianza el
objetivo que le faltaba, y el resultado no se movió. Lo único que mejora
monótonamente es la **σ**: 4.13 → 3.38 → 2.85. Con n = 5 eso no sostiene una afirmación.

**El modelo del artículo es el 74** (75.57 ± 3.06). «Conflict-Aware» sale del título.

**Y esto es lo que hay que escribir, no esconder.** El valor no está en la cifra final
sino en la cadena: margen de oráculo de 21.2 pp → la confianza informa (+20.1 pp sobre
el azar) → seleccionar por ella pierde contra promediar (69.0 vs 70.5 %) → entrenarla
con un objetivo explícito no cierra esa brecha. Es un resultado negativo **con
mecanismo medido y techo cuantificado**, y responde una pregunta que el campo hace a
menudo sin medirla: si la ponderación adaptativa por modalidad vale la complejidad que
cuesta. La respuesta aquí es que no, y está demostrada en cuatro configuraciones.

**Diagnóstico del 76 (22 sep 2026), y cierra el caso.** Con la confianza entrenada
contra un objetivo explícito:

| Métrica (muestras en conflicto) | 75 detector entrenado | 76 + confianza calibrada |
|---|---|---|
| Tasa de conflicto | 81.6 % | 81.9 % |
| Una rama al azar | 48.9 % | 48.5 % |
| Voto uniforme | 70.5 % | 69.8 % |
| **Elegir por confianza** | **69.0 %** | **69.3 %** |
| Oráculo | 91.7 % | 91.7 % |

El objetivo de calibración movió la selección por confianza **+0.3 pp**. Nada. Y dentro
del propio 76, seleccionar por confianza **sigue sin batir al promedio uniforme**:
−0.54 ± 3.02 pp pareado por fold (t(4) = −0.40, IC 95 % [−4.29, +3.21]; por fold
−2.1, 0.0, −3.7, +4.3, −1.2). La varianza entre folds es mayor que el efecto.

Ése es el resultado que se escribe, y es más fuerte que «no funcionó»: **el techo de
oráculo de ~22 pp es real y estable en las cuatro configuraciones, la confianza
estimada informa sistemáticamente (+20 pp sobre el azar), y aun así ninguna regla de
selección basada en ella supera a promediar las ramas — ni siquiera cuando se la
entrena explícitamente para predecir qué rama acierta.** El margen existe; la
información para explotarlo, en esta formulación, no.

ATENCION: **el veredicto que imprime `diagnose_conflict_gate.py` está obsoleto.** Dice
«merece la pena correr las variantes, correrlas LAS TRES»: es una regla fija escrita
cuando el detector estaba roto y no sabíamos si había techo. Ya se corrieron las
configuraciones que importaban. No leer esa línea como una recomendación nueva.

### Qué hacer con la compuerta en el texto

No se esconde y no se convierte en el titular. Va como **subsección de análisis**, que es
justo lo que R1 elogió del manuscrito original:

- `last_conflict_rate` registra cada cuánto dispara la detección. Reportar esa tasa por
  fold y por clase.
- Analizar **cuándo** detecta conflicto (dos de tres ramas discrepando en su argmax) y por
  qué reponderar por confianza en esos casos no mejora: la hipótesis a contrastar es que
  cuando las ramas discrepan, la confianza estimada no está mejor calibrada que el peso
  uniforme.
- Decirlo en Limitaciones: la detección de conflicto tal como se formula aquí no produce
  beneficio medible; formulaciones alternativas (umbral aprendido, distancia en el espacio
  de embeddings) quedan como trabajo futuro.

Un componente medido honestamente y reportado como nulo es mucho mejor artículo que el
«+2.73 pp» que hubo que retirar. Y el número principal **sube** al quitarlo.

### Títulos candidatos

El título tiene que describir lo que queda en pie. La vista dual del audio es lo que R1 y R3
elogiaron explícitamente, y la fusión multistream es lo que está medido:

- *Multistream Fusion for Speech Emotion Recognition: a Controlled Study*
- *Dual-View Audio and Text Fusion for Speech Emotion Recognition*
- *What Actually Helps in Multistream Fusion for SER? A Protocol-Controlled Study*

### Qué se elimina

- El marco de tabla de clasificación y toda frase de «superamos a X».
- Cualquier atribución de modelado temporal multiescala a la cabeza MST.
- Las cuatro cifras de la tabla de estado del arte, sustituidas por la tabla con criterio
  de inclusión y columnas de protocolo, nº de clases, métrica y tipo de texto.

### Advertencia de dosificación

El artículo **no** debe ser principalmente una auditoría de los demás. Un texto que en el
fondo dice «todo el mundo lo hace mal» es difícil de publicar y se gana enemigos. La
arquitectura con su evaluación rigurosa **es** el artículo; la auditoría es la razón por la
que la evaluación está diseñada así, y vive en una subsección acotada más las columnas de
protocolo de la tabla. Tono: descriptivo, nunca acusatorio — son elecciones legítimas
mezcladas en una misma columna, no mala fe.

## 6. PLAN DE SEGUNDO CORPUS (22 sep 2026) — decidido, con un bloqueo que hay que salvar

Jorge eligió el segundo corpus como la prioridad para la revista. Es la decisión
correcta: un revisor de JCR no acepta una afirmación de generalización sostenida sobre
un solo corpus, y lo demás (multi-semilla, protocolo aleatorio, ASR) fortalece sin
desbloquear.

### El bloqueo que no habíamos visto

**MSP-IMPROV no está en la máquina y requiere licencia.** Procedimiento verificado el
22/09/2026 en <https://lab-msp.com/MSP/MSP-Improv.html> (el laboratorio migró de
`ecs.utdallas.edu`): firma el **director del grupo**, no la oficina jurídica, y se envía
por correo a Carlos Busso desde el correo institucional. Sin costo académico. **El
tiempo de respuesta no está documentado en ninguna fuente** — por eso el trámite se
manda antes de tocar nada de código. Borrador en `docs/SOLICITUD_MSP_IMPROV.md`.

**Y MSP-IMPROV no trae transcripciones.** Esto es crítico para una arquitectura
trimodal y está verificado en una fuente de primera mano: Feng, Kuhn & Provost, NAACL-HLT
2019 (<https://aclanthology.org/N19-3003.pdf>), gente que tenía los datos, dice
literalmente que los diálogos del corpus no contienen transcripciones manuales.

- Las ~1,272 utterances de *target* (652 improvisadas + 620 leídas) sí tienen contenido
  léxico conocido por construcción: son lecturas de 15 frases fijas.
- Las de *other-improvised* (4,381) y *natural* (2,785) — o sea **~92 % del subconjunto
  de 4 clases** — no tienen texto de referencia.

**Consecuencia, y resulta ser una ventaja:** la rama de texto corre sobre Whisper en
MSP-IMPROV. Eso convierte la **condición ASR que pidió R3 sobre IEMOCAP** de tarea
pendiente en el **puente metodológico** del artículo: da la línea base comparable que
hace defendible el texto por ASR en el segundo corpus. Deja de ser un trámite y pasa a
ser parte del diseño.

### Datos del corpus, verificados (Busso et al., IEEE TAFFC 8(1), 2017)

8,438 turnos, 6 sesiones diádicas, 12 actores (6M/6F), >9 h audiovisuales. Cuatro
condiciones de elicitación: target improvisada 652, other-improvisada 4,381, natural
2,785, target leída 620. Etiquetas categóricas primarias H/A/S/N/Other — **no hay
fusión excited→happy** como en IEMOCAP: el filtro de 4 clases es directo.

**Subconjunto estándar de 4 clases: 7,798 utterances**, confirmado de forma independiente
por dos trabajos con distribución idéntica: angry 792, happy 2,644, neutral 3,477,
sad 885. Desbalance severo (neutral 44.6 %, angry 10.2 %) — por eso la literatura de
MSP-IMPROV reporta **UAR/macro-F1** y no WA. Nuestra métrica principal ya es UA, así que
encaja.

**Dos convenciones de partición coexisten**, y hay que declarar cuál: LOSO de 6 sesiones
(el análogo directo de nuestro protocolo) o leave-one-speaker-out de 12 actores.
Reportar la de 6 sesiones como principal por paralelismo con IEMOCAP.

Referencias comparables verificadas: Purohit et al., Interspeech 2023 (7,798, LOSO 6
sesiones, WavLM 54.40 UAR / 64.64 WAR); Chen et al., CTA-RNN, Interspeech 2022 (7,798,
12-fold leave-one-speaker-out, 63.5 UAR); Zhao et al., MMIN, ACL 2021 (subconjunto
DISTINTO de 3,819, 12-fold, F1 ponderada 0.6855). **Rango realista: ~54–64 UAR según
subconjunto y protocolo.** Una cifra de MSP-IMPROV solo es comparable si se declaran las
tres cosas: subconjunto (7,798 vs 3,819), partición (6 sesiones vs 12 hablantes) y
métrica (UAR / WAR / F1 ponderada). Es exactamente el mismo error que auditamos en la
tabla de estado del arte de IEMOCAP: no repetirlo en el corpus nuevo.

### Estructura de MSP-IMPROV en disco: QUE SE SABE Y QUE NO (10 oct 2026)

**Lo verificado** sale del articulo (Busso et al., IEEE TAFFC 8(1):377-390, 2017) y de la
pagina del corpus. **Lo no verificado NO se da por bueno**: es la disposicion de los
ficheros, y solo se puede confirmar con los datos en la mano. Dado lo que paso con
CREMA-D —dos afirmaciones «medidas sobre el arbol de git» resultaron falsas y las cazo el
contrato del cargador, no una lectura mas atenta— aqui se escribe el contrato primero y
se deja fallar ruidosamente.

**Jerarquia logica del corpus (verificada):**

- **6 sesiones diadicas**, cada una una pareja de actores distinta → 6 folds LOSO
  nativos. Mejor que CREMA-D: no hay que inventar grupos, las sesiones ya existen.
- **12 actores** (6M/6F), 2 por sesion.
- **8,438 turnos de habla** en total, repartidos en **cuatro condiciones de elicitacion**:

  | condicion | turnos | que es |
  |---|---|---|
  | target improvisada (P) | 652 | la frase objetivo dentro de la improvisacion |
  | other-improvisada (S) | 4,381 | el resto de la improvisacion |
  | natural (N) | 2,785 | habla espontanea entre grabaciones |
  | target leida (R) | 620 | lecturas de la frase objetivo |

- **15 frases objetivo** efectivamente grabadas (la pagina dice 20 disenadas; el articulo
  dice que la Tabla 1 lista las 15 usadas). Citarlo con cuidado.
- **Video** 1440x1080 a 29.97 fps + audio. **Nosotros no lo necesitamos**: nuestra rama de
  «vision» es una imagen log-mel calculada DENTRO del modelo a partir del audio. Solo
  hace falta el audio.
- **Etiquetas**: categorica primaria de eleccion unica (H/A/S/N/**Other**), categorica
  secundaria multiple, y atributos V/A/D en escalas SAM 1-5 **mas Naturalidad** (1 muy
  actuado → 5 muy natural). Esa cuarta dimension es poco habitual y podria dar material
  propio de analisis.
- Anotacion por crowdsourcing en MTurk: >=5 evaluadores para la mayoria, ~28 de media para
  los 652 turnos de target improvisada.

**Contrato de 4 clases a asertar en el cargador: 7,798 utterances**, angry 792,
happy 2,644, neutral 3,477, sad 885. Sin fusion excited→happy (el esquema primario no
tiene excited). Desbalance severo: neutral 44.6 %, angry 10.2 %.

**LO QUE NO ESTA VERIFICADO y hay que confirmar al recibir los datos:**

1. **La disposicion de carpetas.** Ni la pagina ni el articulo la documentan.
2. **El formato de nombre de fichero.** Inferido de un derivado de terceros en
   HuggingFace (`cairocode/MSPI_Emotion2Vec_Text`) como
   `MSP-IMPROV-S01A-F01-P-FM01.wav`: sesion (S01A), id de frase objetivo (F01), **letra
   de condicion** (P) y pareja de genero + indice de turno (FM01). El mapeo de letras
   (P = target improvisada, R = leida, S = other-improvisada, N = natural) es
   **coherente con el diseno pero NO confirmado por fuente autorizada**.
3. **El nombre del fichero de anotaciones.** El codigo de la comunidad se refiere a
   `Evalution.txt` (con la errata). **Sin confirmar.**

**Checklist al recibir el corpus, antes de escribir una linea de `_collect_records`:**

- [ ] Contar los `.wav`: deben ser 8,438.
- [ ] Verificar el patron de nombre contra TODOS los ficheros, y que el guardarrail
      **nombre** los que no casen (la leccion de `1040_ITH_SAD_X`).
- [ ] Confirmar las letras de condicion contando: P=652, R=620, S=4,381, N=2,785. Si los
      conteos cuadran, el mapeo es correcto; si no, hay que rehacerlo.
- [ ] Localizar el fichero de anotaciones y leer sus columnas de verdad, sin asumir.
- [ ] Filtrar a 4 clases y comprobar **7,798** con la distribucion exacta.
- [ ] Comprobar que las 6 sesiones dan 12 actores y que `speaker_of()` los separa bien
      (hay que sobrescribirlo: el de IEMOCAP no sirve).
- [ ] Decidir y DECLARAR si se usan las cuatro condiciones o solo un subconjunto. La
      literatura comparable (7,798) usa las cuatro.

**Transcripciones: no hay para el 92 % del subconjunto de 4 clases** (verificado en Feng,
Kuhn & Provost, NAACL-HLT 2019). Solo los ~1,272 turnos de target tienen contenido lexico
conocido, por ser lecturas de las 15 frases fijas. El resto va por ASR (Whisper).

**Y esto es ahora lo importante del experimento, no un detalle de implementacion.** Tras
el estudio 78, MSP-IMPROV es el corpus que separa «el aporte es lexico» de «CREMA-D era
actuado y facil» (seccion 11). El texto por ASR es justo lo que hace falta: espontaneo
como IEMOCAP, con texto real aunque ruidoso.

### CREMA-D: el corpus que se puede empezar HOY, y que además dice algo

Descarga abierta, sin licencia ni espera. 7,442 clips, 91 actores (diversidad real de
edad y origen), con vídeo, etiquetas perceptuales crowdsourced, y angry/happy/sad/neutral
entre sus seis clases → subconjunto de 4 clases limpio.

Lo que lo hace valioso no es el tamaño: **las 12 frases son fijas**. Todas las emociones
se expresan sobre el mismo conjunto de frases, así que **la rama de texto es
demostrablemente no informativa por construcción**. Eso no es una limitación que haya que
disculpar: es un experimento controlado que ningún corpus espontáneo permite. La caída de
trimodal a audio+visión mide **cuánto depende la fusión del contenido léxico**, y si la
arquitectura se degrada con elegancia o se rompe cuando una rama no aporta nada.

RAVDESS es la versión extrema de lo mismo (2 frases fijas, 24 actores) y sirve de apoyo;
EmoDB también (10 frases) pero con 535 utterances y en alemán, es anecdótico.

### Los dos corpus hacen trabajos distintos — decirlo así en el artículo

- **MSP-IMPROV** = generalización. Espontáneo, diádico, el compañero estándar de IEMOCAP.
- **CREMA-D** = ablación controlada. Texto ortogonal a la emoción por diseño.

No son sustitutos. Uno responde «¿generaliza?», el otro «¿de qué depende?».

### El paquete `src/data/` estuvo fuera de git (descubierto el 22/09/2026)

La linea 63 del `.gitignore` decia `data/` sin barra inicial. Estaba pensada para un
directorio de datasets, pero tambien casaba con `src/data/`: **el codigo que decide que
datos ve el modelo no tenia historial de versiones**. En un repo donde ya se perdieron
2,943 de 5,531 utterances por un bug silencioso en ese mismo archivo, eso no es
higiene: es la diferencia entre poder auditar un cambio y no poder.

La linea 95, `check_*.py`, ignoraba ademas los seis verificadores -- incluido
`check_aux_integration.py`, el que atrapo que el estudio 75 corria con la perdida en
codigo muerto. Anclado a `/data/` y `/datasets/`, con `!scripts/check_*.py`.
Comprobado que no se cuela ningun `.wav` ni checkpoint.

### Orden de ejecución

0. **Enviar la solicitud de MSP-IMPROV.** Cero GPU, cero código, y el plazo es
   desconocido. Va primero por eso, no por importancia.
1. ~~**Refactor del cargador de datos.**~~ **Hecho el 22/09/2026.**
   `BaseSERDataset` (src/data/base_dataset.py) se queda con lo que no depende del
   corpus; `IEMOCAPDataset` solo con `_collect_records`, `_assert_integrity`, los
   parsers y `EMOTION_MAP`. `src/data/registry.py` mapea `dataset.corpus` del YAML a
   la clase, con `iemocap` por defecto para que ningun config existente cambie de
   comportamiento. Los 9 sitios pasan por `build_dataset(...)`.
   Comprobado a nivel de AST que todos los metodos quedaron identicos salvo
   `_load_data` (que se parte) y `speaker_of` (hook nuevo).
   **Verificado el 23/09/2026** con `scripts/check_dataset_refactor.py`: los nueve
   casos IGUAL (las cinco sesiones por separado, las cinco juntas, el filtro de
   hablante en sus dos direcciones, el subconjunto por indices) y 25 indices de
   `__getitem__` con tensores identicos. Las 5,531 utterances con 2,943 improvisados,
   como debe ser. Falta la prueba aguas abajo: reproducir un fold del 74 (S1 = 76.11).

   **Hallazgo colateral: hay DOS copias del corpus en disco.** `load_base_config()`
   lee el `config.json` del checkpoint 20251029_190659, que apunta a
   `...\archive (1)\IEMOCAP_full_release`; los YAML de ablacion apuntan a
   `...\IEMOCAP_full_release`. Las dos cargan las mismas 5,531 utterances con los
   mismos 2,943 improvisados, asi que hoy son equivalentes. Pero significa que los
   scripts de verificacion (`check_dataset_refactor.py`, `check_random_protocol.py`,
   `fit_speechcue_stats.py`, `smoke_dataset.py`) leen una copia distinta de la que
   usan los experimentos: si alguna vez divergieran, un guardarrail daria verde sobre
   el corpus equivocado. `check_dataset_refactor.py` ya avisa cuando las rutas no
   coinciden y acepta `--data-path`. **Unificar las dos copias es trabajo pendiente**,
   y conviene hacerlo antes de meter un segundo corpus en la mezcla.
2. ~~**CREMA-D**~~ — **descargado y verificado el 02/10/2026**: 4,899 clips de 4 clases
   (1,271 / 1,271 / 1,270 / 1,087), 91 actores en 5 grupos disjuntos balanceados por
   sexo, 16 kHz mono, las doce frases en las cuatro emociones. Listo para el estudio 77.
   Cargador escrito el 23/09/2026 (`src/data/cremad_dataset.py`),
   registrado como `cremad`, con config del estudio 77 y `scripts/check_cremad.py`.
   Falta descargarlo (606 MB de audio con sparse-checkout, no los 3.67 GB del repo
   entero), verificarlo y lanzar. Y falta el estudio 78, sin rama de texto.
3. **Condición ASR (Whisper) sobre IEMOCAP.** La pidió R3 y es el puente para el texto
   por ASR de MSP-IMPROV.
4. **MSP-IMPROV** cuando llegue la licencia.

### CREMA-D: cargador escrito y decisiones declaradas (23 sep 2026)

Todo verificado contra el repositorio (github.com/CheyneyComputerScience/CREMA-D) y
Cao et al. 2014, IEEE TAFFC 5(4), 377-390. **Estas cinco decisiones van en el
artículo**, porque la literatura de CREMA-D suele callarlas y un revisor las pedirá.

**1. Etiqueta intencionada, no percibida.** CREMA-D trae dos verdades posibles y **no
son intercambiables**: la etiqueta percibida por oyentes de solo audio (`VoiceVote`)
coincide con la intencionada en el **41.6 %** de los clips. Peor: para 4 clases,
`VoiceVote` da **70 % neutral** (NEU 3,897 vs HAP 353) — entrenar contra ella sería un
problema de clase mayoritaria disfrazado. Usamos la intencionada, que es además lo que
hace el cargador de TensorFlow Datasets, por donde pasa buena parte de la literatura.
Citar el 41.6 % como justificación de que la elección no es trivial.

**2. 4,899 clips de 4 clases** (angry 1,271 / happy 1,271 / sad 1,270 / neutral 1,087).
Son 4,900 menos `1076_MTI_SAD_XX`, que son 2.57 s de **silencio digital exacto** con
tamaño de fichero normal — ninguna comprobación de tamaño lo detecta. No hay fusión
excited→happy: CREMA-D no tiene excited.

**3. Partición por grupos de actores, declarada como nuestra.** CREMA-D no tiene
sesiones diádicas. Los 91 actores se reparten en 5 grupos disjuntos **estratificados
por sexo**, semilla fija 20260923 e independiente de `config['seed']`. Sigue siendo
independiente de hablante y toda la maquinaria de LOSO funciona sin tocarla.
**No existe partición canónica de CREMA-D**: cada trabajo se inventa la suya. Hay que
decir que ésta es nuestra, no presentarla como estándar. El coste de la
independencia de hablante está medido en la literatura: hasta **7.5 puntos de macro-F1**
(PLOS ONE 10.1371/journal.pone.0355238, mismos modelos, partición aleatoria vs por
actor). Si reportáramos partición aleatoria, un revisor asumiría fuga — con razón.

**4. No hay línea base comparable de 4 clases en el dominio.** Los números que circulan
—SepTr 70.47 %, AuxFormer 71.3 F1, VAVL 0.779 F1— son de **seis** clases, usan
etiquetas distintas entre sí (VAVL declara percibida audiovisual; SepTr no declara
nada) y particiones que a veces ni se dicen (SepTr no afirma independencia de
hablante). Lo único publicado de 4 clases verificable es SELM (Interspeech 2024) y es
**fuera de dominio**, 42.79 UA. **Fijamos línea base propia y lo decimos**; no
insinuamos comparabilidad. Es el mismo criterio que aplicamos a la tabla de IEMOCAP.

**4b. CORRECCIÓN al informe de investigación (02 oct 2026).** La investigación afirmó
que los 7,442 nombres de fichero casan el patrón `ACTOR_FRASE_EMOCION_INTENSIDAD` «sin
excepción», medido sobre el árbol de git. **Es falso, y el guardarraíl del cargador lo
cazó en la primera ejecución sobre los datos reales:** existe
`1040_ITH_SAD_X.wav`, con **una X** en el código de intensidad en vez de dos. Es un
error de nombre del propio repositorio, no de la descarga.

Y explica algo que la investigación había declarado «obsoleto» por error: el cargador de
TensorFlow Datasets lleva `1040_ITH_SAD_XX` en su lista `bad_files` con el comentario
«file names which do not exist in the github». **Tenía razón** — con ese nombre no
existe. TFDS lo descarta; nosotros lo **aceptamos** y normalizamos la intensidad a `XX`,
porque es un clip legítimo de SAD del actor 1040 y no hay motivo para tirarlo.

Con eso las cifras cuadran exactamente con el contrato —sad 1,270, total 4,899—, que es
la confirmación de que ésta es la lectura correcta. **Hay que declarar la diferencia en
el artículo:** nuestro subconjunto tiene un clip más que el de TFDS.

La lección de método: la investigación bibliográfica fue buena pero no sustituye a
ejecutar el cargador contra los bytes reales. Dos afirmaciones medidas «sobre el árbol de
git» resultaron equivocadas, y las dos las detectó un contrato verificable en el código,
no una lectura más atenta del README.

**5. El diseño no está completo:** faltan 20 clips por problemas de extracción (los
actores 1008/WSI, 1009/MTI y 1019/ITH enteros, más 1002/TIE_NEU y 1076/ITS_NEU). Por
eso el cargador no afirma 82 clips por actor en ningún sitio.

**El texto no sale de un ASR ni de un fichero: sale de una tabla.** Las doce frases
están escritas verbatim en `FRASES` (src/data/cremad_dataset.py). Son la verdad por
construcción. `scripts/check_cremad.py` comprueba —y ésta es la comprobación que de
verdad importa— que **las doce frases aparecen en las cuatro emociones**: si alguna
faltara en alguna clase, el texto dejaría de ser ortogonal a la emoción y la rama de
texto podría acertar por el motivo equivocado, que es justo lo que este corpus viene a
descartar.

**MATIZ IMPORTANTE sobre la ortogonalidad, medido el 02/10/2026 al verificar el
corpus.** La frase IEO («It's eleven o'clock») sale **tres veces mas** que las demas en
las clases no neutras, porque es la unica con tres niveles de intensidad (LO/MD/HI)
mientras neutral solo tiene XX: 273 clips por emocion frente a 91, y 91 en neutral. Eso
significa que la identidad de la frase **no es estrictamente independiente de la
etiqueta**: P(neutral | IEO) = 10.0 % frente a P(neutral) = 22.2 %.

Pero el efecto es inofensivo donde importa, y conviene decirlo con el numero exacto:
**un clasificador que SOLO vea la frase acierta 25.94 %, que es exactamente la linea
base de clase mayoritaria.** Cero ganancia. El sesgo de IEO solo desplaza la
probabilidad de «neutral vs resto», no distingue entre angry/happy/sad (ahi los 273
estan empatados a tres bandas).

Asi que la afirmacion correcta para el articulo **no** es «el texto es independiente de
la emocion», sino: **la identidad de la frase no aporta poder discriminativo —clasifica
al nivel de la clase mayoritaria— aunque la frase IEO este sobrerrepresentada en las
clases no neutras.** Declararlo asi y dar el 25.94 %. Si se dijera «independiente» sin
mas, un revisor que mire la tabla de frases encontraria el 273 vs 91 y con razon
desconfiaria del resto.

**Estudio 77** = el modelo del 74 sin un solo cambio de arquitectura, sobre CREMA-D.
Un factor: el corpus. **Estudio 78 (pendiente)** = el mismo sin rama de texto; la caída
77→78 es la medida de cuánto depende la fusión del contenido léxico.

**Licencia ODbL 1.0.** Experimentar y publicar resultados, sin problema, citando a Cao
et al. 2014. Cuidado si alguna vez se publica una base derivada (subconjunto curado,
tabla de características): la ODbL es compartir-igual para bases de datos.

## 7. EL PIPELINE NO ES REPRODUCIBLE CORRIDA A CORRIDA (23 sep 2026)

**Este es el hallazgo mas importante de la semana y condiciona como se reporta todo.**

Al verificar el refactor del cargador se relanzo el fold S1 del estudio 74 con la MISMA
config, la MISMA semilla y el MISMO fold. No dio el mismo numero:

| | 18 sep | 22 sep | delta |
|---|---|---|---|
| Test UA | **76.11** | **74.77** | **−1.33 pp** |
| Test WA | 72.07 | 70.97 | −1.11 |
| best val UA | 78.71 | 79.56 | +0.85 |
| epoca final | 20 | 29 | +9 |

El `config.json` guardado por las dos corridas es **identico en las 0 claves**: no es una
diferencia de configuracion. Y la trayectoria de entrenamiento divergio de verdad —nueve
epocas mas, mejor validacion, peor test—, no es un redondeo.

**Y hay mas evidencia, de las dos corridas del estudio 75:**

| fold | 19 sep | 21 sep | delta |
|---|---|---|---|
| S1 | 76.42 | 75.52 | 0.90 |
| S2 | 80.74 | 81.13 | 0.38 |
| S3 | 72.71 | 73.14 | 0.43 |
| S4 | 72.65 | 74.38 | **1.72** |
| S5 | 72.09 | 72.79 | 0.70 |

Esas dos corridas del 75 SI difieren por un cambio real de codigo (la perdida auxiliar
pasando de rama muerta a rama viva, que hace que tres clasificadores entrenen de cero).
**El efecto de ese cambio real es del mismo tamano que el ruido del 74, que no tuvo
ningun cambio intencionado.** No se pueden separar.

(Las tres corridas del estudio 66 tienen rangos de hasta 3.93 pp, pero son **protocolos
distintos** —camf_loso / std / std2—, asi que no valen como evidencia de ruido. No
usarlas para esto.)

### Por que pasa

`src/train.py` intenta ser determinista (`cudnn.deterministic=True`,
`benchmark=False`, semillas de random/numpy/torch) pero quedan dos agujeros concretos:

1. **`torch.use_deterministic_algorithms(True, warn_only=True)`.** Con `warn_only`, toda
   operacion que no tenga implementacion determinista corre igual, de forma no
   determinista, y solo avisa. En una pila con Wav2Vec2 + atencion + LSTM eso casi
   seguro cubre varias.
2. **`os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')` se ejecuta dentro de
   la funcion de entrenamiento.** Esa variable tiene que estar puesta ANTES de que se
   cree el contexto de CUDA; ponerla ahi puede llegar tarde. Y `setdefault` no pisa un
   valor heredado del entorno.

### Y no se puede saber si ademas cambio el codigo

Entre el 18 y el 22 de septiembre se tocaron `fusion.py`, `ablation_11_model.py`,
`ablation_34_model.py`, `utils.py` y `train.py`. **El ultimo commit del repositorio es
del 10 de septiembre.** Dos semanas enteras de cambios viven en un unico arbol de
trabajo sin confirmar, asi que no hay forma de preguntarle a git que codigo corrio el
18. Ese es el coste real de no commitear, y es el mismo problema de fondo que el
`.gitignore` que dejaba `src/data/` fuera de control de versiones.

### Que sobrevive y que no

**Sobrevive todo lo que depende de un efecto grande:**

- Fusion vs solo audio: **+6.58 ± 1.97 pp**, 5/5 folds. Muy por encima del ruido.
- El arreglo de tasas de aprendizaje (estudio 71): **+5.06 ± 1.46 pp**, 5/5 folds.
- **La compuerta no aporta.** Los nulos se refuerzan, no se debilitan: si al efecto
  medido (−0.18, −0.39) se le suma un ruido de ~1.3 pp, el intervalo se ensancha y el
  cero sigue dentro con mas holgura. Un resultado negativo aguanta el ruido; uno
  positivo pequeno, no.

**No sobrevive la PRECISION que estamos reportando.** Decir "75.57 ± 3.06" sin decir que
volver a correr el mismo fold mueve el numero ~1.3 pp es presentar una precision que no
tenemos. Un revisor que pida una re-ejecucion lo ve al primer intento.

### CONFIRMADO Y CUANTIFICADO con una tercera corrida (28 sep 2026)

Tercera ejecucion del fold S1 del estudio 74, misma config, misma semilla, mismo fold:

| corrida | test UA | val UA | epocas |
|---|---|---|---|
| 18 sep | 76.11 | 78.71 | 20 |
| 22 sep | 74.77 | 79.56 | 29 |
| 28 sep | 75.66 | 79.88 | 20 |
| **media ± σ** | **75.51 ± 0.68** | | |

**Tres valores distintos. La hipotesis del cambio de codigo queda descartada**: el
codigo del 28 es el mismo que el del 22 y no reprodujo su 74.77. Es no determinismo.

**CORRECCION IMPORTANTE a lo que se escribio el 23/09.** Alli se dijo "la banda es
~1.3 pp" y que la media del articulo se mueve esa cantidad. **Es falso, y el error fue
confundir el RANGO de dos muestras con la desviacion tipica.** Con tres corridas:

| cantidad | σ |
|---|---|
| un fold, al re-ejecutarlo | **0.68 pp** |
| la media de 5 folds | **0.30 pp** |
| una diferencia pareada entre dos estudios | **0.43 pp** |
| banda de ±2σ en una diferencia pareada | **±0.86 pp** |

Y se comprueba con las tres medias del estudio 74 medidas hasta hoy —75.57, 75.30,
75.48— cuya σ es 0.14, del orden de la prediccion de 0.30 con n=3.

**La media del articulo es mucho mas estable de lo que dije.** 75.5 es solido a una
decimal. Lo que se mueve es el fold individual, no la media.

### Que sobrevive, ahora con la regla cuantitativa

La regla es: **una diferencia pareada por debajo de ±0.86 pp (2σ) no es atribuible al
modelo.**

| comparacion | efecto | veredicto |
|---|---|---|
| Fusion vs solo audio | +6.58 pp | **15σ. Inamovible.** |
| Tasas de aprendizaje (71 vs 68) | +5.06 pp | **12σ. Inamovible.** |
| 74 vs 72 (quitar la compuerta rota) | +0.86 pp | **justo en el limite de 2σ.** Ya se reportaba como no significativo (t=1.68, IC incluye 0); ahora ademas se sabe que es indistinguible del ruido. Ninguna conclusion dependia de el. |
| 75 vs 74 | −0.18 pp | dentro del ruido. El nulo se refuerza. |
| 76 vs 74 | −0.39 pp | dentro del ruido. El nulo se refuerza. |
| 76 vs 75 | −0.21 pp | dentro del ruido. |

σ con n=3 es una estimacion cruda; la multi-semilla dara una mejor. Pero el orden de
magnitud ya esta fijado y basta para decidir que se puede afirmar.

### Que hacer, y cambia el orden de trabajo

1. **Commitear el arbol YA.** Sin historial no se puede diagnosticar nada de esto.
2. **Un tercer fold S1 del 74** decide entre las dos hipotesis: si sale ~74.77 el
   pipeline es determinista y hay un cambio de codigo que encontrar; si sale un tercer
   valor, es no determinismo y queda confirmado.
3. **La multi-semilla deja de ser un refinamiento y pasa a ser obligatoria**, por delante
   de CREMA-D. No por el test de signos, sino porque es lo que convierte los numeros en
   honestos: 3 semillas x 5 folds permiten reportar la media con su banda de
   variabilidad entre corridas y declararla en el texto.
4. Cerrar los dos agujeros de determinismo (quitar `warn_only`, o al menos registrar que
   operaciones avisan; exportar `CUBLAS_WORKSPACE_CONFIG` desde el lanzador, antes de
   importar torch). Aunque no se consiga determinismo total, **medir** la banda y
   declararla es aceptable; no medirla, no.

### Dos efectos colaterales que hay que saber

**1. El resumen del estudio 74 se sobrescribio.** La re-ejecucion del fold S1 fusiono su
resultado nuevo con los folds 2-5 del resumen anterior, asi que
`kfold_summary_ablation_74.json` ya NO dice 75.57 ± 3.06: dice **75.30 ± 2.74**, con el
fold S1 del 22/09 (74.77) y los demas del 18-19/09.

Las dos son mediciones legitimas del mismo modelo con la misma config. Ninguna es "la
verdadera". Eso ES el hallazgo: **la media del modelo del articulo se mueve 0.27 pp
segun que corrida del fold S1 se use**, que es exactamente 1.33/5. Guardar las dos y
decidir con la multi-semilla.

| medicion del estudio 74 | fold S1 | media ± σ |
|---|---|---|
| resumen original (18-19 sep) | 76.11 | 75.57 ± 3.06 |
| resumen actual (con S1 del 22 sep) | 74.77 | 75.30 ± 2.74 |

**2. El resumen de k-fold NO separaba por semilla — arreglado el 23/09/2026.** El nombre
era `kfold_summary_ablation_NN.json`, sin la semilla, asi que correr 42/123/456 sobre el
mismo estudio habria sobrescrito el resumen anterior en cada pasada: **la multi-semilla
se habria destruido a si misma en silencio**, y encima el rescate de folds del resumen
previo habria mezclado semillas en una misma media. Ahora cada fold lleva su `seed` y su
`experiment_name` como sello de procedencia, el resumen se llama
`..._seedNNN.json` cuando se pasa `--seed`, y el rescate **descarta** folds cuya semilla
no coincida, igual que ya hacia con el protocolo.

**Esto no invalida el trabajo: lo calibra.** La conclusion del articulo —la fusion
trimodal aporta mucho, la compuerta de conflicto no aporta nada, y sabemos por que— es
robusta al ruido medido. Lo que hay que corregir es cuantos decimales nos permitimos.

## 8. MULTI-SEMILLA DEL ESTUDIO 74 (2 oct 2026) — y la sigma de 3 pp no es inestabilidad

Dos semillas completas, cinco folds cada una, protocolo `standard`:

| fold | semilla 42 | semilla 123 | 123−42 |
|---|---|---|---|
| S1 | 75.66 | 75.82 | +0.15 |
| S2 | 80.25 | 80.89 | +0.64 |
| S3 | 71.96 | 74.50 | **+2.54** |
| S4 | 75.49 | 72.71 | **−2.78** |
| S5 | 74.05 | 73.55 | −0.50 |
| **media ± σ** | **75.48 ± 3.05** | **75.49 ± 3.23** | **+0.01** |

**Las dos medias coinciden en 0.01 pp.** Diferencia pareada por fold +0.01 ± 1.93,
t(4) = 0.01, p de permutacion = 1.00. Combinando las 10 observaciones: **75.49 pp**,
σ entre observaciones 2.96, error tipico de la media 0.94.

### Y esto reinterpreta la σ que reportamos

Descomponiendo la varianza con las dos semillas:

| componente | σ | que es |
|---|---|---|
| **entre folds** | **2.99 pp** | la dificultad real de cada sesion de IEMOCAP |
| dentro del fold (semilla + ruido) | 1.29 pp | |
| de los cuales, re-ejecucion con la MISMA semilla | 0.68 pp | medido con 3 corridas del fold S1 |
| luego, atribuible a la semilla | ≈1.10 pp | √(1.29² − 0.68²) |

**La σ de ~3 pp que reportamos es casi toda diferencia de dificultad entre sesiones, no
inestabilidad del metodo.** Eso cambia como se escribe: la σ entre folds en LOSO mide
cuanto varia el problema de hablante a hablante, que es informacion sobre el corpus, no
ruido de nuestra implementacion. Hay que decirlo, porque un revisor que vea ±3 pp puede
leerlo como falta de control.

**La regla operativa se mantiene**, y ahora con dos vias de calculo:
- comparaciones pareadas DENTRO de la misma semilla (que es como se hizo toda la tabla
  de ablacion): σ de la diferencia = √2 × 0.68/√5 = **0.43 pp**, banda 2σ = **±0.86 pp**.
- comparaciones entre semillas distintas: σ = √2 × 1.29/√5 = **0.82 pp**, banda ±1.63 pp.

Todas las conclusiones aguantan: fusion vs solo audio +6.58 pp (15σ), tasas de
aprendizaje +5.06 pp (12σ), y la compuerta de conflicto dentro del ruido en sus tres
variantes.

### Para el articulo, en una frase

«El modelo alcanza 75.49 % UA en LOSO de cinco folds sobre IEMOCAP. Dos semillas
independientes dan 75.48 y 75.49, y la variabilidad de re-ejecucion con semilla fija es
de 0.68 pp por fold; la desviacion de ±3 pp entre folds refleja diferencias de
dificultad entre sesiones, no inestabilidad del entrenamiento.»

Eso es mas de lo que reporta casi cualquier articulo de SER, y responde por adelantado
la pregunta de reproducibilidad.

### Se para aqui: no se corre la semilla 456

Con las medias coincidiendo en 0.01 pp, una tercera semilla no cambia ninguna conclusion.
**Limitacion que hay que declarar:** con n = 2 semillas no se puede poner un ± a la
variabilidad entre semillas; lo que se puede decir, y basta, es que dos semillas
independientes dieron 75.48 y 75.49. La GPU vale mas en CREMA-D.

### Coste operativo, para que quede escrito

La semilla 123 costo tres dias y seis intentos. Tres de cinco folds murieron con
`bad allocation` (memoria de host) y uno despues con `CUDNN_STATUS_INTERNAL_ERROR` en el
LSTM de la cabeza. Se saco adelante con un proceso por fold y `--resume`; los folds S1 y
S5 quedan marcados `resumed: true`. Ese marcado importa: un fold reanudado no es
bit-identico a uno corrido de un tiron, y conviene no esconderlo.

### PRIMER FOLD DE CREMA-D (3 oct 2026): 84.27 UA — y por que NO es «mejor que IEMOCAP»

Estudio 77, fold S1 (grupo de actores 1, 1,026 clips de test): **UA 84.27, WA 84.11,
macro-F1 83.95.** Por clase: angry 92.1, neutral 88.6, sad 83.8, happy 72.6.

**Comprobacion de fuga, hecha antes de interpretar nada.** El `config.json` del fold
confirma `corpus: cremad`, test = grupo 1, train = grupos 2-5, y
`val_fraction: 0.12` con `val_speaker` ausente. Es decir: el protocolo `standard` NO
toco la ruta especifica de IEMOCAP que construye el nombre del hablante de validacion
(`Ses0X_F`), que es lo que podria haber fallado en silencio. Train y test son grupos de
actores **disjuntos**. La validacion sale de un 12 % estratificado de los grupos de
entrenamiento, o sea comparte actores con train —igual que en IEMOCAP— y solo se usa
para elegir checkpoint. No hay fuga hacia test.

**Y AHORA LO IMPORTANTE, que es como NO escribirlo.** 84.27 frente a 75.49 de IEMOCAP
**no significa que el modelo sea mejor en CREMA-D**. Es otra tarea, mas facil, y decir
«generaliza y ademas mejora» seria exactamente el tipo de comparacion inflada que
llevamos semanas quitando de la tabla de estado del arte.

Las razones, concretas:

1. **La etiqueta es la INTENCIONADA, no la percibida.** En CREMA-D el actor recibio la
   instruccion de que emocion representar. Los oyentes humanos de solo audio coinciden
   con esa etiqueta el **41.6 %** de las veces (en el esquema de 6 clases). O sea: la
   etiqueta intencionada no es «lo que se oye», es el objetivo de produccion. Un modelo
   puede aprender la firma de produccion del actor, que es sistematica y aprendible, sin
   resolver el problema perceptual que plantea IEMOCAP.
2. **Emocion actuada y prompteada frente a improvisada.** IEMOCAP es interaccion diadica
   improvisada, con emocion sutil y a veces ambigua; CREMA-D son clips de 1.3-5 s
   (mediana 2.5) con la emocion presente de principio a fin.
3. **Clases casi balanceadas** (25.9 / 25.9 / 25.9 / 22.2) frente al desbalance de
   IEMOCAP (19.9 / 29.6 / 19.6 / 30.9).

**Como se escribe, entonces:** el modelo **no se rompe** al cambiar de corpus —que es lo
que la seccion de generalizacion tiene que demostrar— y los numeros absolutos de los dos
corpus **no son comparables entre si** porque la semantica de la etiqueta y la
dificultad de la tarea difieren. Las dos frases van juntas o ninguna.

**ES UN SOLO FOLD.** Sin sigma no se afirma nada. Faltan los folds 2-5.

## 9. CREMA-D COMPLETO — estudio 77 (7 oct 2026)

Cinco folds, protocolo `standard`, el modelo del 74 sin un solo cambio de arquitectura:

| grupo de actores | UA | WA | epocas |
|---|---|---|---|
| 1 | 84.27 | 84.11 | 29 |
| 2 | 80.46 | 80.00 | 22 |
| 3 | 79.49 | 79.45 | 19 |
| 4 | 86.17 | 86.42 | 28 |
| 5 | 81.36 | 80.94 | 21 |
| **media ± σ** | **82.35 ± 2.78** | 82.18 | |

Ningun fold reanudado: los cinco corrieron de un tiron.

### Recall por clase

| grupo | angry | happy | sad | neutral |
|---|---|---|---|---|
| 1 | 92.11 | 72.56 | 83.83 | 88.60 |
| 2 | 83.40 | 79.25 | 66.29 | 92.92 |
| 3 | 89.06 | 69.43 | 78.87 | 80.62 |
| 4 | 89.87 | 85.23 | 89.87 | 79.70 |
| 5 | 92.86 | 69.75 | 70.17 | 92.65 |
| **media ± σ** | **89.46 ± 3.73** | **75.24 ± 6.84** | **77.81 ± 9.67** | **86.90 ± 6.39** |

Tres afirmaciones que aguantan los cinco folds:

- **`angry` es la clase facil**: 89.46 de media, la mas alta o empatada en 5/5. Coherente
  con arousal alto y actuacion prototipica.
- **`happy` es la mas dificil de media** (75.24) y la mas baja en 3/5 folds, con precision
  alta: el modelo es conservador, cuando dice happy acierta pero se le escapan muchos.
- **`sad` es la mas inestable** (σ 9.67, rango 66.3–89.9). Depende mucho del grupo de
  actores, lo cual tiene sentido: la tristeza actuada con arousal bajo se parece al habla
  neutra, y cuanto la confunde el modelo depende de como la interprete cada actor.

### La σ entre folds refuerza la seccion 8

σ = 2.78 en CREMA-D frente a 3.05 en IEMOCAP. **Dos corpus distintos, con numeros
absolutos muy distintos, dan la misma magnitud de dispersion entre folds.** Eso respalda
la interpretacion de la seccion 8: ese ±3 pp mide dificultad entre grupos de hablantes,
no inestabilidad del metodo. Es un argumento que se puede hacer explicito en el texto.

### Y la comparacion que NO se hace

82.35 frente a 75.49 de IEMOCAP son **+6.87 pp**, y eso **no es una mejora**. Las razones
estan en la seccion de CREMA-D: etiqueta intencionada (los oyentes humanos solo coinciden
con ella el 41.6 % de las veces en 6 clases), emocion actuada y prompteada frente a
improvisada, clips de 1.3–5 s, clases casi balanceadas. Son tareas de dificultad
distinta.

**Lo que SI se afirma, y es lo que la seccion de generalizacion necesita:** la
arquitectura **no se degrada** al cambiar de corpus, y lo hace sobre un corpus donde el
texto no puede aportar informacion emocional. Las dos frases van juntas o ninguna.

## 10. HyCon: por que NO se implementa tal cual (7 oct 2026)

Idea de Jorge: aplicar la perdida contrastiva de HyCon (Mai, Zeng, Zheng, Hu; IEEE TAFFC
14(3):2276-2289, 2022; DOI 10.1109/TAFFC.2022.3172360) para ver si mejora los resultados.
Linea legitima —el repo ya tiene `SupConLoss`, `loss_contrastive` enchufado en las DOS
ramas de `train_epoch`, y los modelos 43/47/48— pero la investigacion dio cuatro motivos
para no portarla como esta, y uno de ellos es decisivo.

### El decisivo: batch 4 rompe justo los dos terminos que aportan

HyCon tiene tres terminos. Los negativos salen **exclusivamente del mini-batch actual**:
no hay memory bank, ni cola, ni encoder de momento, ni estudio de sensibilidad al tamano
de batch en el articulo. Con `batch_size: 4`, K = 4 y por tanto N + M = 3: como maximo
**3 pares por ancla**.

**La acumulacion de gradiente NO ayuda.** El conjunto de negativos lo define lo que esta
en el MISMO forward. Acumular 8 pasos da ocho problemas contrastivos diminutos, no uno de
batch 32. Nuestro batch efectivo de 32 es real para la perdida de tarea y ficticio para
la contrastiva.

Peor con 4 clases: en un batch de 4, **la mayoria de las anclas no tiene ningun positivo
de su clase**, asi que IAMCL e IEMCL quedan indefinidos. Y con muestreo balanceado por
clase es aun mas claro: 4 clases en 4 muestras = una por clase = cero positivos.

Y los propios autores avisan del modo de fallo: la solucion degenerada aparece *"especially
when sufficient amount of negative pairs is unavailable"*. Con M <= 3 la suma de negativos
cae a ~0, el cociente se satura y el gradiente muere. Lo unico que lo sostiene es el
termino de refinamiento, que es un L2 hacia 1 y hacia alfa — es decir, a batch 4 estariamos
entrenando una perdida de centro/margen, no una contrastiva.

**Y subir el batch no es una opcion en esta maquina**: ya se cae con `bad allocation` y
`CUDNN_STATUS_INTERNAL_ERROR` a batch 4 (seccion 7).

### El segundo, y es nuestro: el efecto seria inmedible

La ablacion de HyCon (Tabla 6, MOSI) compara su formulacion contra una perdida
contrastiva clasica: **+0.3 en Acc7**, +1.5 en Acc2. Y en MOSEI —el dataset grande— la
ganancia frente al mejor baseline es +0.9 Acc7 con **cero** mejora en MAE y correlacion.

Nuestra banda de ruido para una diferencia pareada con la misma semilla es **±0.86 pp
(2σ)** (seccion 8). Un efecto de 0.3-0.9 pp **no se puede medir** con cinco folds y una
semilla. Afirmarlo seria exactamente el error que llevamos un mes quitando del articulo.

### El tercero: el articulo publicado no es el de arXiv

arXiv:2109.01797 es una version **anterior y distinta**: la publicada anade un
"pair selection mechanism" para ponderar pares informativos y experimentos de
reconocimiento de emociones que arXiv no tiene. IEEE y ACM DL no fueron accesibles, asi
que implementar desde arXiv seria implementar otro metodo y llamarlo HyCon.

### El cuarto: regimen distinto y sin precedente en SER

- Tarea: **regresion de sentimiento** en [-3,+3] con perdida MAE, no emocion categorica.
- Las ganancias se concentran en **datos pocos** (MOSI, 1,284 de entrenamiento) y se
  desvanecen con datos abundantes (MOSEI, 16,265). Nosotros tenemos 5,531 y 4,899.
- Modalidades con **texto fuerte** (BERT 768-d) y audio/vision de caracteristicas
  artesanales (COVAREP/Facet) fundidas por **suma elemento a elemento**.
- **No existe implementacion oficial** (ni en el articulo ni en paperswithcode).
- **Ningun trabajo aplica HyCon a SER categorico speech-only.** El unico numero de
  HyCon sobre IEMOCAP que se encontro (MMCL, arXiv:2501.12424, Tabla III) usa el
  protocolo binario por emocion de MulT y es trimodal con GloVe: NO comparable con UA de
  4 clases.

### Lo que SI transporta, y conecta con nuestro propio diagnostico

Dos piezas sobreviven a batch 4 y valen la pena:

1. **El margen de modalidad (SCL con alfa = 0.8).** No usa negativos: es un L2 que acerca
   las representaciones cross-modales del MISMO enunciado hasta similitud 0.8, **no 1**.
   La idea es reducir la brecha entre modalidades SIN colapsar la informacion propia de
   cada una. En su ablacion vale 2.4 Acc7 frente a forzar alineacion total — mas que el
   termino de refinamiento. Funciona identico a batch 4.
   **OJO:** su ablacion tambien muestra que quitar SCL *mejora* Acc7 en +0.4 mientras
   empeora Acc2/F1/MAE. Su valor neto es ambiguo. No venderlo como seguro.
2. **Contrastivo supervisado por prototipos**, en lugar de IAMCL/IEMCL: una media movil
   por clase y por modalidad, y se contrasta contra prototipos en vez de contra muestras
   del batch. Elimina la dependencia del tamano de batch y captura la estructura
   inter-clase que su ablacion atribuye a IEMCL (el termino que mas aporta: quitarlo
   cuesta mas que quitar los tres).

**Y aqui esta la conexion honesta con lo nuestro, que es lo que le daria sentido en el
articulo y no seria un anadido oportunista.** El diagnostico de la compuerta (seccion 6)
midio que seleccionar por confianza pierde contra promediar **a pesar de un margen de
oraculo de 21 pp**: las ramas no estan alineadas ni calibradas de forma que una regla de
seleccion pueda explotarlo. Un termino de alineacion cross-modal ataca **ese mismo
problema diagnosticado** desde el otro lado. Eso es una motivacion medida, no una cita
de moda.

### Precondicion antes de gastar GPU, decidida de antemano

Si el objetivo es **afirmar** una mejora de metodo, hace falta multi-semilla desde el
principio: con una semilla no se puede defender nada por debajo de ±0.86 pp. La semilla
123 costo tres dias y seis intentos. Hay que decidir el umbral ANTES de correr: si el
efecto esperado es ~1 pp, el experimento necesita 3 semillas x 5 folds, y hay que querer
pagarlo.

## 11. EL HALLAZGO QUE CAMBIA EL ARTICULO — estudio 78 (10 oct 2026)

Solo audio sobre CREMA-D, cinco folds, el par exacto del 77. **La fusion NO aporta nada
en CREMA-D.**

| | trimodal | solo audio | delta |
|---|---|---|---|
| grupo 1 | 84.27 | 84.85 | −0.58 |
| grupo 2 | 80.46 | 83.60 | −3.14 |
| grupo 3 | 79.49 | 80.03 | −0.53 |
| grupo 4 | 86.17 | 84.71 | **+1.46** |
| grupo 5 | 81.36 | 84.52 | −3.17 |
| **media ± σ** | **82.35 ± 2.78** | **83.54 ± 2.02** | **−1.19 ± 1.97** |

Pareado: **−1.19 ± 1.97 pp, t(4) = −1.35, 1/5 folds, IC 95 % [−3.64, +1.26]**, p de
permutacion 0.3125.

Frente a IEMOCAP, donde el mismo contraste da **+6.49 ± 2.06 pp, t(4) = 7.04, 5/5 folds**:

| corpus | aporte de la fusion sobre solo audio |
|---|---|
| IEMOCAP | **+6.49 ± 2.06** (5/5, ~15σ de la banda de ruido) |
| CREMA-D | **−1.19 ± 1.97** (1/5, el IC incluye el cero) |

**Lectura correcta: en CREMA-D no hay ganancia. NO se afirma que haya perdida** — el IC
incluye el cero y p = 0.31. Lo que queda establecido es que **el +6.49 no se replica.**

### Lo que esto significa, y es mejor articulo que «generaliza»

Esto es una **disociacion controlada entre corpus**, y la predijimos antes de medirla:

- En CREMA-D el texto **no puede** llevar informacion emocional. Lo medimos: las doce
  frases son fijas y la identidad de la frase clasifica al **25.94 %**, que es exactamente
  la linea base de clase mayoritaria (seccion de CREMA-D).
- La rama de vision es una imagen log-mel **derivada del audio**: tampoco anade
  informacion que el audio no tenga ya.
- Luego en CREMA-D «trimodal» = audio + vision redundante + texto vacio. Y rinde igual
  (o un pelo peor) que solo audio.

**La conclusion mecanica: el aporte de la fusion es especificamente LEXICO.** No es que el
modelo sea «mejor reconociendo emociones»; es que usa **lo que se dice** junto con **como
se dice**, y cuando lo que se dice no lleva informacion, no hay nada que sumar.

Eso convierte el claim central de «nuestra fusion trimodal mejora +6.58 pp» en algo mas
fuerte y mas falsable: **mejora +6.49 pp cuando el texto es informativo, y 0 cuando no lo
es, y tenemos los dos lados medidos con el mismo modelo y el mismo protocolo.**

### EL CONFUSOR QUE HAY QUE DECLARAR, y no esconder

CREMA-D no solo difiere en el texto: es **actuado, prompteado y de clips cortos**, frente
a IEMOCAP que es improvisado y espontaneo. Una explicacion alternativa de la no
replicacion es «CREMA-D es mas facil y el audio solo ya satura» (83.54 frente a 68.99 en
IEMOCAP: hay mucho menos margen que recuperar).

**Con estos dos corpus NO se puede separar «texto no informativo» de «actuado y facil».**
Hay que decirlo en el texto. Lo honesto es presentar la explicacion lexica como la mas
parsimoniosa dado que medimos que el texto no informa, y nombrar el confusor.

**Y esto le da un proposito preciso a MSP-IMPROV**: espontaneo (como IEMOCAP) pero con
texto por ASR. Si alli la fusion aporta, la explicacion es lexica; si no aporta, el factor
es espontaneo-vs-actuado. Ese es el experimento que cierra la pregunta, y ahora se sabe
exactamente que decidiria.

### Que se reescribe

1. El claim central pasa de «la fusion aporta» a «la fusion aporta **contenido lexico**,
   y aqui esta medido cuando si y cuando no».
2. La seccion de generalizacion deja de ser «funciona en otro corpus» y pasa a ser **el
   experimento que identifica la fuente del aporte**. Es mas interesante y mas defendible.
3. En Limitaciones: el confusor actuado/espontaneo, y MSP-IMPROV como el experimento que
   lo resolveria.
4. **Audio solo en CREMA-D (83.54 ± 2.02) es el mejor numero de este corpus.** Reportarlo
   como tal, sin maquillarlo. Un revisor que vea trimodal 82.35 presentado como «nuestro
   modelo» y encuentre en la tabla que el unimodal da mas, deja de creerse el resto.

### El experimento de transferencia de protocolo (`--protocol random`)

Idea de Jorge, y es la respuesta directa a «¿puedo superar a esos trabajos?»: en vez de
cambiar la arquitectura para perseguir cifras que miden otra cosa, **correr nuestro modelo
bajo el protocolo de ellos**.

La variante con mejor relación valor/coste es la de Yoon et al. 2018: **folds aleatorios
estratificados sobre las cinco sesiones**, dependiente de hablante. Mismo modelo, mismo
corpus, mismas clases, misma implementación — lo único que cambia es el reparto. Eso mide,
**con un solo factor**, cuánto vale el protocolo por sí solo.

Convierte la sección metodológica de una queja («estos trabajos no son comparables») en un
resultado («y la diferencia vale N puntos, aquí está la medición»). No acusa a nadie:
enseña un número. Y Yoon usa **exactamente nuestras 5 531 utterances con los mismos
conteos por clase**, así que la comparación es limpia.

Implementado como `--protocol random`. Decisiones declaradas, porque el original es ambiguo
al decir «5-fold» con split 8:0.5:1.5: partición estricta en 5 bloques estratificados por
clase, el bloque *k* es test (~20 %), y de lo que queda se aparta un 6 % estratificado para
validación. La semilla del reparto es fija e independiente de `config['seed']`, para que
los cinco folds sean una partición de verdad y no cinco muestreos distintos.

**Cómo reportarlo, y esto no es negociable:** siempre junto al `standard`, nunca solo, y
diciendo explícitamente que **LOSO es el protocolo honesto** y que el aleatorio se muestra
solo para poder compararse con el trabajo previo. Si se reporta solo el que más favorece,
es exactamente el error por el que rechazaron el manuscrito.

Verificación previa: `scripts/check_random_protocol.py` comprueba que los cinco folds son
una partición, que train/val/test son disjuntos y suman 5 531, y que el reparto **es**
dependiente de hablante — si saliera independiente, no estaríamos midiendo lo que creemos.

### Alcance: esto ya no cabe en un artículo

- **Reenvío (corto plazo):** CAMF con evaluación rigurosa, ablaciones internas, la
  compuerta medida, y la transferencia de protocolo. Responde «¿los superas?» sin mentir.
- **Capítulo de tesis / revista:** la generalización entre corpus (MSP-IMPROV como primer
  candidato; MELD si se quiere terreno conversacional). Cada rejilla de cinco folds es uno
  o dos días de GPU, y una corrida suelta no sostiene una afirmación de generalización.
- **Prueba barata y con contenido propio:** un corpus de frases fijas (RAVDESS, CREMA-D),
  donde el texto no lleva información emocional. Es el test de si la fusión **se degrada
  con elegancia o se rompe** cuando una rama no aporta nada. Barato y dice algo real.

### Orden de trabajo

1. ~~Ablación de la compuerta~~ — **hecha**: estudio 74. No aporta (+0.86 ± 1.14,
   t=1.68, IC 95 % incluye 0). Confirmado con el detector arreglado (estudio 75:
   −0.18 ± 1.14, IC 95 % [−1.59, +1.23]). **El título cambia** y el modelo es el 74.
2. Estudio 73 (texto con tasas corregidas) — cierra la tabla de ablación.
3. ~~Multi-semilla~~ — **hecha con 2 semillas el 02/10/2026**: 42 → 75.48, 123 → 75.49.
   Se para ahi; la tercera no aporta (ver seccion 8).
4. ~~Estudio 75~~ — **hecho**: la compuerta con el detector arreglado no aporta
   (−0.18 ± 1.14, IC 95 % [−1.59, +1.23]). «Conflict-Aware» sale del título.
4b. ~~Estudio 76~~ — **hecho y diagnosticado**: la confianza con objetivo de calibración
   tampoco aporta (−0.39 ± 1.33 frente al 74, IC 95 % [−2.04, +1.26]), y dentro del 76
   seleccionar por confianza sigue sin batir al promedio (−0.54 ± 3.02). **La línea de
   la compuerta queda cerrada en cuatro configuraciones. No correr más variantes.**
5. Condición ASR (Whisper) — la pidió R3 y **bajará** el número; reportarla igual.
5. **`--protocol random`** — la fila comparable con Yoon *y* la medición de cuánto vale
   el protocolo. Un día de GPU.
6. Verificar los candidatos de la tabla comparativa con la revisión sistemática propia.
7. Escribir.


---

## 12. MELD: corpus evaluado, auditado y con defectos nuevos (10 oct 2026)

### 12.0 Por qué MELD y no MOSI ni MOSEI

La sección 11 dejó un confound declarado: en CREMA-D la fusión no aporta, y con
IEMOCAP + CREMA-D **no se puede separar «el texto no es informativo» de
«CREMA-D es actuado y fácil»**. Se evaluaron tres candidatos para cerrarlo.

**CMU-MOSI — descartado.** 2,199 clips de opinión, **sentimiento continuo en
[−3,+3]**, sin etiquetas categóricas de emoción. Métricas Acc7 / Acc2 / F1 /
MAE / correlación: ninguna entra en la misma tabla que nuestra UA de 4 clases.
Split fijo por vídeo, no LOSO. Y es el corpus **dominado por texto** por
excelencia (ICCN 39.0 y MAG-BERT 42.9 Acc7 contra MulT 33.6), así que
confirmaría lo que ya sabemos sin resolver el confound. Además la mitad de la
literatura usa Facet (expresión facial real) y nuestra rama de visión es mel
derivado del audio: reportar «trimodal» ahí se leería como que nos falta una
modalidad que el corpus sí tiene.

**CMU-MOSEI — descartado, pero por poco.** A favor: ~23,500 segmentos de ~1,000
hablantes (independencia de hablante genuina, dos órdenes de magnitud más que
IEMOCAP) y **sí tiene etiquetas categóricas**. En contra, y decide: (a) **no hay
clase neutral** — solapa en tres de nuestras cuatro y le falta justo la que más
confusión genera en IEMOCAP; (b) es **multi-etiqueta**, la métrica estándar es
WAcc y F1 **binarios por emoción**, no UA multiclase; (c) desbalance extremo que
obliga a BCE ponderada, es decir otro objetivo de pérdida en el código; (d) la
versión preprocesada estándar trae muestras desalineadas (los papers se quedan
con 20,477 de 23,259) y no coincide del todo con el split oficial — no hay una
única definición del corpus, que es exactamente lo que nos criticaron en
IEMOCAP; (e) el audio es YouTube y hay un resultado conocido de que la modalidad
acústica aporta muy poco. Eso último lo invalida para nuestra pregunta: MOSEI es
el régimen **texto-dominante con audio débil**, así que añade un segundo
confound (calidad de audio) en el extremo opuesto en vez de quitar el primero.
Uso legítimo: otro artículo, con el marco «¿bajo qué condiciones aporta la
fusión?».

**MELD — elegido.** Diálogo de sitcom, 7 clases **con neutral**, transcripciones
humanas de subtítulos (no ASR), descarga libre, y es estándar en SER/ERC.
Cierra el eje con tres puntos medidos:

| corpus | texto | audio | ¿aporta la fusión? |
|---|---|---|---|
| CREMA-D | nulo (12 frases fijas) | suficiente (83.54 solo) | **no** (−1.19 ± 1.97, 1/5 folds) |
| IEMOCAP | informativo | insuficiente | **sí** (+6.49 ± 2.06, 5/5 folds) |
| MELD | **domina** | techo duro | según la literatura, ~+3 WF1 sobre texto solo |

La brecha audio-vs-texto en MELD, en cinco fuentes independientes: bcLSTM
39.08 / 56.44 (−17.4) · DialogueRNN 41.79 / 57.03 (−15.2) · EMART 38.05 / 64.27
(−26.2) · wav2vec2-XLS-R-2B vs DeBERTa 49.3 / 60.9 (−11.6) · speech-LLM vs
text-LLM ~−12 en el mejor caso. El dato más contundente es la clase `joy`
(2,308 ejemplos): audio-only F1 **15.74**, texto **54.31**. El modelo audio-only
de MELD es casi un detector de neutral con ruido alrededor. Escalar el encoder
acústico de openSMILE a un SSL de 2B parámetros compra ~10 puntos y se satura
muy lejos del texto.

### 12.1 El asterisco: el split oficial NO es independiente de hablante

**Verificado contando la columna `Speaker` de los CSVs liberados** (no es una
cita, es nuestra medición):

| hablante | train | dev | test |
|---|---|---|---|
| Joey | 1,509 | 149 | 411 |
| Ross | 1,459 | 217 | 373 |
| Rachel | 1,435 | 164 | 356 |
| Phoebe | 1,321 | 185 | 291 |
| Monica | 1,299 | 137 | 346 |
| Chandler | 1,283 | 101 | 379 |
| **% de los 6 líderes** | **83.2 %** | **85.9 %** | **82.6 %** |

69 de los 100 hablantes de test están también en train. El paper **nunca usa el
término «speaker-independent»**: el split lo heredó de EmotionLines, que dividió
por diálogos sin ninguna restricción de hablante, y todos los diálogos de los
dos corpus salen de *Friends*, así que un split por diálogos **no puede** ser
independiente de hablante.

**Decisión y argumento.** Se usa el split oficial, se declara explícitamente que
es dependiente de hablante, y se defiende así: nuestra comparación es
**pareada** — trimodal vs audio-only sobre el mismo split —, de modo que la
dependencia de hablante es un confound **compartido** por las dos condiciones y
se cancela en la diferencia. Lo que queda inválido es cualquier afirmación sobre
el número absoluto como generalización a hablantes nuevos. Si un revisor
pregunta si el número es independiente de hablante, la respuesta es **no, y es
una propiedad del benchmark, no de nuestro método**. Un split propio
disjunto por hablante es posible, pero entonces el número deja de ser comparable
con toda la literatura: no se hace salvo que un revisor lo exija.

`Speaker` tampoco es un ID limpio: **169 filas** tienen hablante no individual
(`All`, `Both`, `Woman`, `Man`, `Director` y `The Director` sin normalizar, y
conjunciones como `Ross and Rachel`). Esas filas contienen más de una voz y no
son ejemplos válidos de un solo hablante.

### 12.2 Contrato del corpus — verificado contra el disco

Estructura real de la descarga (los nombres de los directorios de medios **no
están documentados**: el README oficial dice literalmente `XXX.tar.gz`):

```
MELD/MELD-RAW/MELD.Raw/
  train/train_sent_emo.csv      train/train_splits/
  dev_sent_emo.csv              dev/dev_splits_complete/
  test_sent_emo.csv             test/output_repeated_splits_test/
```

Encabezado de los tres CSVs, byte-idéntico:
`Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime`

- **`Sr No.` NO es clave**: en train hay 9,989 filas pero solo 9,932 valores
  únicos (57 duplicados) y el máximo es 10,478. La clave es
  **`(Dialogue_ID, Utterance_ID)`**, única en los tres splits.
- `StartTime`/`EndTime` en formato `hh:mm:ss,ms` — **coma antes de los
  milisegundos**, estilo SRT. Un `float()` ingenuo falla.
- Emociones en el CSV en **minúsculas**: `anger, disgust, fear, joy, neutral,
  sadness, surprise`. El README las capitaliza; no escribir el match contra el
  README.
- El orden de índices one-hot oficial **no es alfabético**:
  `{neutral:0, surprise:1, fear:2, sadness:3, joy:4, disgust:5, anger:6}`.
  Fuente frecuente de matrices de confusión silenciosamente mal.
- **El sentimiento es función casi determinista de la emoción** (anger, disgust,
  fear, sadness → 100 % negative; joy → 100 % positive; neutral → 100 % neutral;
  solo `surprise` se reparte). **No presentar emoción y sentimiento como
  evidencia independiente.**
- Conteos y distribución de etiquetas: **cuadran exactamente** con la tabla
  publicada (9,989 / 1,109 / 2,610 filas; las 7 clases celda por celda).
  Verificado por `scripts/check_meld.py --fase estructura`.

**Contrato de 4 clases** (`anger→angry, joy→happy, sadness→sad, neutral`), ya
descontadas las dos utterances sin medio —ambas `neutral`— del punto 12.3:

| split | total | angry | happy | sad | neutral |
|---|---|---|---|---|---|
| train | **8,244** | 1,109 | 1,743 | 683 | 4,709 |
| dev | **896** | 153 | 163 | 111 | 469 |
| test | **2,211** | 345 | 402 | 208 | 1,256 |
| **TOTAL** | **11,351** | 1,607 | 2,308 | 1,002 | 6,434 |

Se descartan 2,355 utterances (disgust 361, fear 358, surprise 1,636). El contrato
de 7 clases utilizables es **9,988 / 1,108 / 2,610 = 13,706**, que es exactamente
lo que contiene el dataset derivado `ajyy/MELD_audio` — confirmación independiente
de la auditoría. Los dos contratos están codificados en
`src/data/meld_dataset.py` (`REF_4CLASS_POR_SPLIT`, `REF_4CLASS_POR_EMOCION`,
`REF_7CLASS_POR_SPLIT`) y los verifica `_assert_integrity` en cada construcción.

**Un resultado de 4 clases NO es comparable con la literatura de MELD**, que
reporta weighted F1 sobre 7 clases; hay que decirlo en el pie de tabla.
**Decidido el 10/10/2026: primero 4 clases** (consistencia con IEMOCAP y CREMA-D,
que es lo que necesita el argumento de la disociación), **y después 7** para tener
la fila comparable.

**Métrica.** `neutral` es el **48 % del test**, así que la accuracy es casi
inútil (una línea base de clase mayoritaria da ~48 %). La métrica de la
literatura es **weighted F1**; el paper original no reporta accuracy. Reportar
además **macro-F1 y F1 por clase** — es lo que refleja las clases minoritarias y
pocos papers de MELD lo hacen.

### 12.3 Defectos — lo publicado y lo que encontramos nosotros

**Ya documentados en el issue tracker oficial, nunca corregidos:**

- `train_splits/dia125_utt3.mp4` **corrupto** (issue #39; confirmado
  independientemente en TelME issue #6). La fila del CSV existe normal. **Está
  en train, no en test** — en test hay otro `dia125_utt3` distinto y sin
  defecto, porque `Dialogue_ID` reinicia en 0 en cada split.
- Clips mal cortados (issue #20): `dia38_utt4` y `dia220_utt0` en test,
  `dia309_utt0` en train.
- Desincronización transcripción-vídeo (issue #9) atribuida por el propio
  mantenedor al alineador **Gentle**; dijo que lo arreglarían «en un futuro
  cercano» y sigue abierto.
- Audio degradado en ~200 clips de train, según el reporter identificables por
  tamaño > 2.5 MB. **No verificado por nosotros.**

**Hallazgos nuestros, no publicados en ninguna parte** (todos reproducibles con
`scripts/check_meld.py`):

1. **El archivo que lee todo el mundo es el incorrecto para 131 utterances de
   test.** El directorio de test contiene **132 archivos con el nombre
   prefijado** `final_videos_testdiaX_uttY.mp4` **además** del `diaX_uttY.mp4`
   correspondiente. No son copias: tienen tamaños y duraciones distintas.
   Arbitrando con la duración del CSV (`EndTime − StartTime`), de las 131
   anotadas **131 coinciden con el archivo prefijado y 0 con el de nombre
   normal** — y coinciden *al milisegundo*, mientras el de nombre normal es más
   largo y **compartido entre utterances vecinas** (p. ej. `dia100_utt5`,
   `utt6` y `utt7` apuntan al mismo archivo de 4.233 s y el mismo tamaño en
   bytes). El nombre del directorio lo dice: *output_repeated_splits_test*.
   Como el `read_meld.py` oficial **construye el nombre desde el CSV**, todo
   pipeline que haga lo mismo está leyendo el corte equivocado en **131 de
   2,610 utterances de test (5.0 %)**. La 132.ª (`dia93_utt6`) no es arbitrable:
   no tiene fila en el CSV.
   **Consecuencia para el loader: en test, si existe el prefijado, se lee el
   prefijado.**
2. **El archivo de dev que falta es `dia110_utt7`.** Hay una fila en
   `dev_sent_emo.csv` sin clip correspondiente. Varias fuentes derivadas
   reportan «un clip de train corrupto y uno de dev ausente» (el dataset de HF
   `ajyy/MELD_audio` ships 9,988 / 1,108 / 2,610) pero **ninguna lo identifica**.
   Es este.
3. **Clips sin anotación**: dev tiene 4 (`dia49_utt4`, `dia49_utt5`,
   `dia66_utt9`, `dia66_utt10`) y test 5 (`dia93_utt5/6/7`, `dia108_utt1/2`).
   Coinciden exactamente con los huecos de `Utterance_ID` del CSV: el clip se
   cortó y la anotación se descartó. **El loader debe iterar sobre las filas del
   CSV, nunca sobre `range(n_utts)` ni sobre el listado de archivos.**
4. **2,060 archivos AppleDouble `._*.mp4`** en el directorio de test, basura de
   compresión en macOS. Inflan cualquier conteo ingenuo de 2,747 a 4,807.
   Hay que filtrarlos.
5. **El audio es AAC a 48 kHz, y el número de canales NO es uniforme**: de los
   13,847 clips legibles, **12,230 son 6 canales (5.1) y 1,617 son estéreo
   (2 canales)**. Nadie lo ha publicado — el paper solo dice que ellos lo
   pasaron a «16-bit PCM WAV» y no libera los WAV. El pipeline de extracción
   tiene que manejar las dos configuraciones, hacer downmix a mono y resample a
   16 kHz explícitamente, y **declararlo**. Un `ffmpeg` que asuma 5.1 falla en
   1,617 clips.
5b. **Todos los clips tienen pista de audio.** `ffprobe` sobre los 13,848
   archivos: **0 sin pista de audio** en los tres splits, y **un solo archivo
   ilegible en todo el corpus** (`train/dia125_utt3.mp4`, el del issue #39).
   Esto **cierra los issues #47 y #49**, abiertos por gente que creyó que el
   split de test no tenía audio: lo tiene, y lo tiene completo. Es un dato
   verificado que no está publicado.
5c. **Desalineación archivo-vs-anotación en todo el corpus.** Comparando la
   duración real de cada `.mp4` con `EndTime − StartTime`, con tolerancia de
   60 ms (el contenedor mp4 redondea 22–42 ms): **train 6, dev 1, test 136**.
   Los 136 de test son exactamente los 131 con gemelo prefijado (punto 1) más
   5 nuevos. Lista completa de los que no están explicados por el punto 1:
   - train: `dia556_utt6` **−8.443 s** (el archivo es 8.4 s *más corto* que su
     anotación — defecto nuevo, no reportado en ningún sitio), `dia338_utt3`
     **+2.516 s** (aquí el roto es el CSV: dice 6 ms y el archivo son 2.52 s),
     `dia437_utt2` +0.274, `dia981_utt3` −0.147, `dia23_utt4` +0.127,
     `dia23_utt5` −0.127.
   - dev: `dia72_utt13` −0.167.
   - test: `dia242_utt2` +1.756, `dia155_utt3` +1.632, `dia151_utt2` +0.847,
     `dia187_utt2` +0.771, `dia233_utt8` +0.409.
   **Matiz importante**: los dos clips largos de test (`dia38_utt4` 304.94 s y
   `dia220_utt0` 235.07 s) **no** aparecen desalineados, porque el archivo sí
   mide lo que dice su anotación. Ahí lo mal cortado es el par
   (timestamp, medio) conjuntamente, no el corte respecto a la anotación: el
   issue #20 reporta que el *contenido* no corresponde a la transcripción. Son
   dos defectos de naturaleza distinta y hay que describirlos por separado.
6. **Duraciones.** El paper reporta media 3.59 / 3.59 / 3.58 s. Medido sobre los
   timestamps liberados: **3.139 / 3.117 / 3.309 s**, mediana 2.46 s en los tres.
   Formulación segura: *«Poria et al. reportan una duración media de 3.59 s; la
   media de `EndTime − StartTime` sobre las anotaciones liberadas es 3.14 s en
   train, con mediana 2.46 s.»* Para los clips sin gemelo prefijado, la duración
   del archivo coincide con la del CSV dentro de **22–42 ms** (redondeo del
   contenedor), así que el corpus está bien cortado en general: la discrepancia
   está en cómo se calculó la cifra publicada, no en los medios.
7. **Duraciones patológicas.** `< 1.5 s`: 2,249 / 255 / 629 (**22.5 % / 23.0 % /
   24.1 %**). `< 1.0 s`: 912 / 105 / 269 (~9–10 %). `< 0.5 s`: 283 / 34 / 89.
   Casos concretos: `train/dia338_utt3` («What?», surprise) con **6 ms**, y
   `test/dia155_utt3` («Oh my God.», surprise) con `StartTime == EndTime`, es
   decir **duración cero**. Hay un suelo de cuantización visible en 0.084 s:
   no son duraciones reales, son fallos del alineador clampados a un mínimo.
   **Si se aplica un filtro de duración mínima hay que declarar el umbral y
   cuántas utterances quita, por split y por clase** — los clips cortos son
   desproporcionadamente interjecciones etiquetadas `surprise`/`neutral`, así
   que el filtro sesga la distribución.
8. **Solapes de timestamps.** Utterances que empiezan antes de que acabe la
   anterior del mismo diálogo: **651 en train (6.5 %), 66 en dev (6.0 %),
   105 en test (4.0 %)**. Su audio extraído comparte contenido con el vecino.
9. **La risa grabada del público no está cuantificada por nadie.** MELD es una
   sitcom con audiencia en estudio y el paper **no menciona «laughter», «music»,
   «noise», «background» ni «overlap» ni una sola vez**. Está casi con seguridad
   correlacionada con los límites de las utterances `joy`/`surprise`. Medirla
   (actividad de un detector de risa por clase, o rendimiento con y sin
   utterances solapadas con risa) sería contribución novedosa. **No afirmarlo
   sin medirlo.**

10. **Los CSV traen mojibake de cp1252 en el 27–28 % de las transcripciones, y
    eso sesga el resultado en la dirección equivocada.** Hallazgo propio, detectado
    el 10/10/2026 al mirar la muestra del smoke test. `Utterance` contiene U+0092
    donde debería haber un apóstrofo tipográfico, más U+0085 (…), U+0097 (—),
    U+0091, U+0093, U+0094 y U+0096: **5,386 ocurrencias en 3,741 utterances**,
    repartidas por igual entre los tres splits (train 26.94 %, dev 27.77 %, test
    28.43 %). El patrón es el clásico: el original estaba en cp1252, alguien lo
    decodificó como latin-1 y lo re-codificó como UTF-8, así que el UTF-8 **es
    válido** y contiene caracteres de control C1 — ninguna comprobación de
    codificación lo detecta.

    **Por qué no es cosmético.** El `BasicTokenizer` de BERT *borra* los caracteres
    de categoría C antes de segmentar. Sin arreglarlo, `company’s` llega al modelo
    como `companys`, `must’ve` como `mustve` y `don’t` como `dont`: en una cuarta
    parte del corpus, **todas las contracciones quedan deformadas**. Como la rama
    de texto es exactamente lo que este artículo mide, el sesgo va en la dirección
    de **subestimar el aporte de la fusión en MELD** — es decir, justo hacia la
    conclusión equivocada sobre la disociación de la sección 11. Un experimento que
    midiera «la fusión aporta poco en MELD» con el texto roto no estaría midiendo
    lo que dice.

    Arreglado en `src/data/meld_dataset.py` con `normalizar_texto()`, que remapea
    solo el rango U+0080–U+009F a través de cp1252 (los cinco bytes que cp1252 no
    define se dejan intactos, y cualquier otro carácter no se toca, así que no puede
    romper puntuación legítima). El conteo se **reporta** en cada construcción.

    **Verificado que esto NO afecta a IEMOCAP ni a CREMA-D.** Los 151 ficheros de
    transcripción de IEMOCAP son UTF-8 válido, usan el apóstrofo ASCII (8,160 en
    una muestra de 200 ficheros) y tienen **un** solo carácter de control en 10,239
    líneas (un tabulador). CREMA-D tampoco, porque su texto sale de la tabla
    `FRASES`. El defecto es exclusivo de MELD, así que ninguna cifra ya publicada
    de este proyecto está contaminada.

Nota negativa útil (arXiv 2509.05634): **el denoising con DEMUCS degrada** el
rendimiento acústico en MELD en todos los modelos, porque elimina
vocalizaciones no verbales que llevan emoción. Si se plantea un front-end de
denoising, hay que citar esto y justificarlo. El mismo trabajo da el coste del
ASR en MELD: DeBERTa saca **60.9 WF1 con transcripción manual y 51.5 con
Whisper** — 9.4 puntos. Argumento fuerte para un artículo centrado en audio: la
ventaja de la rama de texto es en parte un artefacto de disponer de
transcripciones de oro en test, que no existirían en despliegue.

**Advertencia de fuente.** No citar MDPI *Information* 16(7):518 («Speech
Emotion Recognition on MELD and RAVDESS Datasets Using CNN») para ningún
resultado de MELD: reporta **94.0 % de accuracy audio-only**, implausible
(el mejor audio-only publicado ronda 49–55 WF1 y la clase mayoritaria da 48 %),
no usó el split oficial («stratified sampling» propio, con fuga de hablante y de
diálogo), y es internamente inconsistente (intercambia las cifras de MELD y
RAVDESS entre abstract y conclusiones; omite `disgust` de su lista de etiquetas
pero discute conteos de disgust). Si un revisor lo trae como resultado
competidor, estos son los argumentos para descartarlo.

### 12.4 La auditoría como contribución

**Nadie ha publicado una auditoría a nivel de archivo de MELD.** Los diez
puntos de 12.3 son reproducibles desde los CSVs y los medios con un solo script
(`scripts/check_meld.py`, fases `estructura` y `audio`), y cuatro de ellos
—el archivo equivocado en 131 utterances de test, la identidad del clip de dev
ausente, los parámetros reales del audio y el mojibake de cp1252 en el 27 % de
las transcripciones— **no están en ninguna fuente**. Es
exactamente el tipo de rigor cuya ausencia nos criticaron en IEMOCAP, y entra
bien en una subsección de datos y protocolo.

### 12.4.bis Lo que ya está implementado (10 oct 2026)

- **`scripts/check_meld.py`** — auditoría en dos fases. `estructura`: CSVs, conteos,
  diff CSV↔medios, arbitraje de los 131 duplicados, etiquetas contra la tabla
  publicada, hablantes, duraciones y solapes. `audio`: `ffprobe` por clip, paralelo
  y reanudable con checkpoint. Informe en `logs/meld_audit.json`.
- **`scripts/extract_meld_audio.py`** — extracción a `<raiz>/audio_16k/{train,dev,
  test}/diaX_uttY.wav`, 16 kHz mono, `-ac 1` para que el downmix funcione tanto en
  5.1 como en estéreo. **Lee el fichero prefijado cuando existe** (los 131 de test)
  y descarta los dos sin medio. Paralelo y reanudable: salta los wav ya escritos.
- **`src/data/meld_dataset.py`** — `MELDDataset(BaseSERDataset)` con las nueve
  decisiones de 12.3 documentadas en el docstring. Itera siempre sobre las filas del
  CSV (nunca sobre `range(n_utts)` ni sobre el listado de ficheros), `speaker_of()`
  sobrescrito al personaje, `utterance_id` **con prefijo de split** porque
  `dia125_utt3` existe en train y en test siendo utterances distintas, y
  `_assert_integrity()` contra los dos contratos. Registrado en
  `src/data/registry.py` como `meld`.
- **`--protocol fixed`** en `scripts/run_ablation.py` — MELD no tiene LOSO. Corre UN
  solo fold con el reparto del YAML, validando que sea disjunto, y pasa por el mismo
  camino que el k-fold para conservar el sello de semilla, la reanudación y el
  formato del resumen.
- **Estudios 79 y 80** — `ablation_79_meld_trimodal.yaml` (clon del 77) y
  `ablation_80_meld_audio_only.yaml` (clon del 78). Verificados por diff de YAML
  aplanado: difieren **solo** en `dataset.corpus`, `data_path`, las cuatro listas de
  sesiones y las rutas de logging. Registrados en el dispatch y en la lista de
  estudios.
- **`scripts/smoke_dataset.py`** — generalizado a cualquier corpus registrado. El
  guardarraíl de utterances improvisadas se conserva solo para IEMOCAP, que es donde
  detectó el bug de troceado.

Orden de ejecución:

```
python scripts/check_meld.py        --data-path "D:/DOCTORADO/DATASETS/MELD" --fase todo
python scripts/extract_meld_audio.py --data-path "D:/DOCTORADO/DATASETS/MELD"
python scripts/smoke_dataset.py --config config/ablation/ablation_79_meld_trimodal.yaml
python scripts/run_ablation.py --study 79 --protocol fixed --seed 42
python scripts/run_ablation.py --study 80 --protocol fixed --seed 42
```

### 12.6 ESTUDIO 79 — MELD trimodal, 4 clases, semilla 42 (10 oct 2026)

Primer número de MELD. Split oficial fijo (`--protocol fixed`), un solo fold,
parada temprana en la época 14 (mejor val UA 58.61).

| métrica | valor |
|---|---|
| **UA (recall macro)** | **59.41** |
| WA / exactitud | 65.40 |
| F1 ponderada | 66.98 |
| F1 macro | 57.16 |

Por clase (recall / F1):

| clase | recall | F1 | % del test |
|---|---|---|---|
| neutral | 72.29 | 77.41 | 56.8 |
| angry | 62.03 | 55.87 | 15.6 |
| happy | 56.22 | 60.59 | 18.2 |
| sad | **47.12** | **34.75** | 9.4 |

**El modelo NO se colapsó a la clase mayoritaria**, que era el riesgo con neutral al
57 % del train: las cuatro clases están muy por encima del azar (25 %) y `sad`, la
más escasa, se queda en 47.12 de recall. El resultado es interpretable. `sad` es la
clase débil, igual que en la literatura de MELD (sadness F1 24–34 en los baselines
originales).

**OJO CON LA F1 PONDERADA DE 66.98.** Cae dentro del rango del SOTA de 7 clases
(64–70 WF1) y la tentación de leerlo como «estamos en el estado del arte» es
inmediata. **No lo es**: nuestra tarea es de 4 clases y hemos quitado justo las tres
más difíciles y escasas (disgust 361, fear 358, surprise 1,636 en el corpus), que
son las que hunden la F1 de todo el mundo. Comparar esos dos números es la misma
clase de error que ya nos costó la tabla comparativa de IEMOCAP. La fila comparable
sale solo de la condición de 7 clases, y hay que decirlo en el pie de tabla.

**Dos bugs míos corregidos a raíz de esta corrida** (los dos en el camino nuevo de
`--protocol fixed`, ninguno afecta a resultados anteriores):

1. El aviso `[PARCIAL]` comparaba los folds corridos contra `all_sessions` (3) en
   vez de contra las sesiones que el protocolo **rota** (1), así que declaraba
   «1/3 folds, faltan [1, 2]» y aconsejaba `--folds 1 2` — es decir, **entrenar
   usando train y dev como conjuntos de test**. Habría producido números sin
   sentido con aspecto de legítimos. Mismo arreglo en la clave `complete` del
   resumen, que marcaba todo resumen de `fixed` como incompleto para siempre.
2. Al añadir `sesiones_rotadas` al resumen, la clave cayó primero en
   `run_ablation_study_loso`, donde esa variable no existe: habría sido un
   `NameError` en la siguiente corrida LOSO. Detectado con una comprobación de que
   el símbolo solo aparece dentro de `run_ablation_study_kfold`.

**LA BANDA DE DECISIÓN NO ES ±0.86 pp AQUÍ.** Ese valor es 2σ de una *diferencia
pareada de medias de 5 folds* en IEMOCAP. Con un split fijo y una sola semilla por
condición no hay nada pareado: la diferencia 79−80 es la resta de dos estimaciones
puntuales. Tomando la σ de un fold individual de IEMOCAP (0.68 pp), la σ de esa
resta es ≈0.96 pp y la banda a 2σ ≈ **±1.9 pp**. Con dos semillas por condición se
puede estrechar, pero **nunca llega a ±0.86**: esa cifra no se transporta a MELD y
usarla aquí sería inflar la precisión. Declararlo en el artículo.

### 12.7 EL PAR 79/80 CIERRA LA PREDICCIÓN — y da el eje de tres puntos (12 oct 2026)

Estudio 80, MELD solo audio, 4 clases, semilla 42: **UA 46.06**, WA 55.86,
F1 ponderada 55.59, F1 macro 44.49. Parada en la época 13, mejor val UA 49.26.

**Delta 79 − 80 = +13.36 pp de UA.** La predicción, escrita en el encabezado del
config del 80 ANTES de correrlo, era que este delta debía ser **mayor** que el
+6.49 de IEMOCAP. Es más del doble, y unas 7 veces la banda de ±1.9 pp de 12.6.

| métrica | 79 trimodal | 80 solo audio | delta |
|---|---|---|---|
| **UA** | 59.41 | 46.06 | **+13.36** |
| WA | 65.40 | 55.86 | +9.54 |
| F1 ponderada | 66.98 | 55.59 | +11.39 |
| F1 macro | 57.16 | 44.49 | +12.67 |

**El eje de tres puntos, con la misma arquitectura y sin tocar un
hiperparámetro:**

| corpus | el texto es… | delta fusión − solo audio |
|---|---|---|
| CREMA-D | nulo por construcción (12 frases fijas) | **−1.19 ± 1.97** (1/5 folds) |
| IEMOCAP | informativo | **+6.49 ± 2.06** (5/5 folds) |
| MELD | dominante | **+12.80 ± 1.23** (2 semillas, 20.9 σ) |

Es monótono y es la novedad del artículo: no es «la fusión ayuda», es que el
aporte de la fusión **escala con cuánto informa el canal léxico**, medido en tres
corpus que van de texto ortogonal por diseño a texto dominante.

#### La evidencia más fuerte es por clase, no agregada

| clase | 79 recall | 80 recall | delta |
|---|---|---|---|
| **happy** | 56.22 | **23.13** | **+33.08** |
| sad | 47.12 | 37.98 | +9.13 |
| angry | 62.03 | 53.04 | +8.99 |
| neutral | 72.29 | 70.06 | +2.23 |

Todo el delta se concentra en `happy`: +33.08 pp de recall, +30.20 de F1. Las
otras tres clases ganan entre 2 y 9. Y esto **replica de forma independiente el
número más llamativo de la tabla de Poria et al. 2019**: en su propio baseline,
`joy` saca F1 **15.74 con solo audio** y **54.31 con solo texto** — una brecha de
38 puntos, la mayor de su tabla. Nosotros medimos 30.39 → 60.59 en la misma clase,
con otra arquitectura, otro conjunto de etiquetas y otra métrica. Mismo fenómeno.
Un revisor puede discutir nuestra interpretación, pero no que el efecto exista:
está publicado desde 2019 y lo reproducimos sin buscarlo.

Interpretación plausible, y hay que presentarla como tal: la alegría es la clase
cuya marca acústica se confunde más con el habla neutra en una sitcom —
precisamente donde la risa grabada del público solapa (12.3 punto 9) — mientras
que en el texto es trivial. `neutral` casi no gana porque el audio ya la detecta.

#### LA EXPLICACIÓN ALTERNATIVA QUE HAY QUE NOMBRAR, no esconder

El delta grande de MELD admite **dos** lecturas y MELD solo **no puede**
separarlas:

1. el canal léxico es muy informativo, o
2. el canal acústico está degradado — YouTube de sitcom, risa grabada, 22.5 % de
   los clips por debajo de 1.5 s, 6 % con solapamiento de timestamps (12.3).

La literatura apoya la segunda además de la primera: el audio-only de MELD tiene
techo duro en 38–55 WF1 y DEMUCS lo empeora (arXiv 2509.05634).

**Lo que sí separa las dos lecturas es el par CREMA-D / IEMOCAP**, donde el audio
es limpio y elicitado en los dos y lo único que cambia de verdad es si el texto
informa: +0 frente a +6.49. **Ese es el contraste controlado; MELD es el tercer
punto consistente con él, no la prueba.** El artículo tiene que decirlo en ese
orden, o el revisor lo dirá por nosotros.

Y hay que declarar que los tres corpus difieren además en espontaneidad,
dependencia de hablante, balance de clases y protocolo. **«Dosis-respuesta» es una
interpretación, no una manipulación controlada.** La celda que falta —espontáneo,
audio limpio de estudio, texto por ASR— es exactamente MSP-IMPROV.

#### MULTI-SEMILLA DEL 79 — y una advertencia que cambia cómo se reporta (12 oct 2026)

Estudio 79 con semilla 123: UA 58.19, WA 69.29, F1 ponderada 69.34, F1 macro 58.39.

| métrica | semilla 42 | semilla 123 | media | rango |
|---|---|---|---|---|
| **UA** | 59.41 | 58.19 | **58.80** | 1.22 |
| F1 macro | 57.16 | 58.39 | 57.77 | 1.23 |
| F1 ponderada | 66.98 | 69.34 | 68.16 | 2.36 |
| **WA** | 65.40 | 69.29 | 67.35 | **3.89** |

**1. UA es estable en MELD; WA NO lo es.** σ de UA ≈ 1.22/√2 ≈ **0.86 pp**; σ de WA
≈ 3.89/√2 ≈ **2.75 pp**, más de tres veces peor. El motivo es mecánico: `neutral`
es el 56.8 % del test, así que WA está dominada por una sola clase y cualquier
desplazamiento del umbral de decisión la mueve mucho, mientras UA promedia los
cuatro recalls y los desplazamientos se cancelan. **Reportar UA y macro-F1; no usar
WA como cifra principal en MELD.** Coincide con la postura de la propia literatura:
el paper original de MELD no reporta exactitud.

**2. CORRECCIÓN A MI PROPIA ESTIMACIÓN DE LA BANDA.** En 12.6 estimé σ de una
corrida única de MELD a partir de la σ de un fold de IEMOCAP (0.68 pp) y di
±1.9 pp. Medida de verdad sobre MELD, σ de una corrida es **0.86 pp**, así que:
- diferencia de **una** semilla por condición: σ ≈ 1.2 pp → banda 2σ ≈ **±2.4 pp**
- diferencia de **dos** semillas por condición: σ ≈ 0.86 pp → banda 2σ ≈ **±1.7 pp**

Usar la cifra medida, no la prestada de IEMOCAP.

**3. EL RECALL POR CLASE NO ES ESTABLE ENTRE SEMILLAS, y esto obliga a matizar lo
que escribí arriba.** Las dos semillas del 79 se intercambian errores entre clases
de forma masiva:

| clase | semilla 42 | semilla 123 | media | rango |
|---|---|---|---|---|
| neutral | 72.29 | 81.85 | 77.07 | **9.55** |
| sad | 47.12 | 37.02 | 42.07 | **10.10** |
| angry | 62.03 | 54.20 | 58.12 | **7.83** |
| happy | 56.22 | 59.70 | 57.96 | 3.48 |

La semilla 123 desplaza la frontera hacia `neutral` (+9.55 de recall) y lo paga en
`sad` (−10.10) y `angry` (−7.83). UA apenas se mueve porque se cancelan.

**Consecuencia directa: un valor de recall por clase de UNA sola semilla no puede
sostener ningún argumento.** El +33.08 pp de `happy` que anoté arriba viene de la
semilla 42 y hay que recalcularlo con medias: trimodal 57.96 (media de dos
semillas) frente a audio-only 23.13 (una semilla todavía) da ≈ **+34.8 pp**. Con
oscilaciones de ±10 pp por semilla en cada lado, una brecha de ~35 pp sigue siendo
robusta — pero **el número del artículo tiene que ser la media de dos semillas con
su rango**, y las tablas por clase tienen que llevar el rango, no un valor pelado.
`sad` con un rango de 10.10 pp es el ejemplo de por qué: cualquier afirmación sobre
`sad` basada en una corrida es ruido.

#### CERRADO: el par completo con dos semillas por condicion (13 oct 2026)

Estudio 80 con semilla 123: UA 45.94. Con la 42 daba 46.06. **Rango de 0.11 pp.**

| | 79 trimodal | rango | 80 solo audio | rango | **delta** |
|---|---|---|---|---|---|
| **UA** | **58.80** | 1.22 | **46.00** | 0.11 | **+12.80** |
| F1 macro | 57.77 | 1.23 | 43.93 | 1.12 | +13.85 |
| F1 ponderada | 68.16 | 2.36 | 55.06 | 1.05 | +13.10 |

σ de una corrida: 79 = 0.86 pp, 80 = **0.08 pp**. σ de la media de dos semillas:
0.61 y 0.06. σ de la diferencia = 0.61 → **banda 2σ = ±1.23 pp**. El delta de
+12.80 pp es **20.9 σ**. No hay ninguna lectura en la que esto sea ruido.

**Observación lateral que merece una frase:** el modelo solo-audio es **diez veces
más reproducible** que el trimodal (σ 0.08 frente a 0.86). Plausiblemente la fusión
añade varianza de optimización. Es una observación, no una conclusión: habría que
medirlo en los tres corpus antes de afirmarlo.

#### LA CAUTELA DE LA MULTI-SEMILLA SE CONFIRMÓ, Y CAMBIA EL HALLAZGO POR CLASE

| clase | 79 trimodal | 80 solo audio | delta | rango de la clase |
|---|---|---|---|---|
| **happy** | 57.96 | **20.40** | **+37.56** | 3.48 / 5.47 |
| neutral | 77.07 | 69.63 | +7.44 | 9.55 / 0.88 |
| sad | 42.07 | 37.02 | +5.05 | 10.10 / 1.92 |
| **angry** | 58.12 | 56.96 | **+1.16** | **7.83 / 7.83** |

**El +8.99 pp de `angry` que anoté con la semilla 42 era ruido.** Con dos semillas
baja a **+1.16 pp**, muy por debajo de su propio rango entre semillas (7.83 pp en
las dos condiciones). No se puede atribuir al modelo. Esta es exactamente la razón
por la que se escribió la cautela antes de tener el dato, y conviene decirlo así en
el artículo: la disciplina de multi-semilla no es decorativa, aquí eliminó un
hallazgo falso.

**El hallazgo real, ya estable:** en MELD el aporte de la fusión es
**esencialmente `happy` (+37.56 pp)**, con ganancias modestas en `neutral` y `sad`
y **ninguna en `angry`**. Lectura plausible, y hay que presentarla como tal: la ira
tiene una marca acústica fuerte (intensidad, F0) que el audio ya captura, mientras
que la alegría no la tiene — menos aún en una sitcom donde la risa grabada solapa
justo en esas utterances. Es coherente con Poria et al. 2019, cuya mayor brecha
audio-vs-texto también es `joy` (F1 15.74 frente a 54.31).

#### Lo que falta antes de escribir esto

1. **Semilla 123 para 79 y 80.** Con una sola semilla por condición no hay ninguna
   medida de variabilidad. El delta es ~7× la banda, así que no va a desaparecer,
   pero la magnitud puede moverse uno o dos puntos y el número del artículo tiene
   que ser la media de dos semillas, como en IEMOCAP.
2. La condición de **7 clases** para la fila comparable con la literatura.
3. No presentar la F1 ponderada de 66.98 como cercana al SOTA (ver 12.6).

### 12.5 Pendiente en MELD

1. ~~Fase audio del script~~ — **hecha el 10/10/2026** sobre los 13,848 clips.
   Resultados en 12.3 puntos 5, 5b y 5c; informe en `logs/meld_audit.json`.
2. Extraer audio de los 13,847 clips legibles con `ffmpeg` — downmix a mono
   (manejando 5.1 **y** estéreo) y resample a 16 kHz —, leyendo el archivo
   **prefijado** cuando exista. Descartar `train/dia125_utt3`.
3. ~~Decidir 4 clases vs 7~~ — **decidido**: 4 primero, 7 después (ver 12.2).
4. ~~`MELDDataset`~~ — **hecho** (ver 12.4.bis). Nota sobre las 169 filas de
   hablante no individual: **NO se descartan por defecto**, porque descartarlas
   rompe la comparabilidad con la tabla publicada. Se reporta el conteo y existe
   `descartar_hablante_grupal` para la variante. Declararlo en el artículo.
5. Correr 79 y 80 con **dos semillas** (42 y 123, las mismas que IEMOCAP), porque
   con un solo split no hay σ entre folds. Criterio de decisión: la banda de
   ±0.86 pp de la sección 7.
6. Después, la condición de **7 clases** con weighted F1 y macro-F1, para la fila
   comparable con la literatura. Requiere su propio par de configs.
7. Medir la risa grabada (12.3 punto 9) si se quiere convertir en contribución.

---

## 13. EL ENCODER DE AUDIO SE CARGÓ ALEATORIO, Y EL CÓDIGO LO OCULTÓ (20 oct 2026)

**Es el segundo fallo silencioso del repositorio, del mismo tipo que el de
`_find_audio_file`, y la segunda vez que un número plausible salía de un pipeline
roto.** Hay que leerlo junto con la sección 7: ahí el problema era que una corrida
no se reproduce; aquí es que una corrida puede ser **íntegramente ficticia** y
parecer normal.

### Qué pasó

El estudio 85 (RAVDESS trimodal) dio, en su primer fold, **UA 0.2344 — por debajo
del azar** (0.25 con cuatro clases), con colapso de clase detectado en siete de
nueve épocas y parada temprana. En el arranque, entre el ruido del log, una línea:

```
Warning: Could not load audio encoder facebook/wav2vec2-base: Due to a serious
vulnerability issue in `torch.load` ... we now require users to upgrade torch to
at least v2.6 ...
Using default Wav2Vec2 configuration
```

`transformers` 5.8.0 con `torch` 2.5.1 se niega a leer checkpoints `.bin`
([CVE-2025-32434](https://nvd.nist.gov/vuln/detail/CVE-2025-32434)) y exige torch
≥ 2.6. `facebook/wav2vec2-base` tiene **los dos** formatos en caché, pero
`from_pretrained()` sin argumentos elegía el `.bin` y lanzaba `ValueError`.

### El bug es nuestro, no de transformers

En `src/models/conformer.py` había esto:

```python
try:
    self.audio_encoder = AutoModel.from_pretrained(audio_encoder_name)
except Exception as e:
    print(f"Warning: Could not load audio encoder {audio_encoder_name}: {e}")
    print("Using default Wav2Vec2 configuration")
    config = Wav2Vec2Config()
    self.audio_encoder = Wav2Vec2Model(config)   # <-- PESOS ALEATORIOS
```

El `except Exception` se tragaba el error y **sustituía el encoder preentrenado por
uno inicializado al azar**, avisando con `print()` —no con el logger, así que no
entra en el fichero de log— y siguiendo a entrenar. Lo mismo en la rama de texto,
donde además caía a un `bert-base-uncased` sin pesos, que ni es el encoder que pide
el config.

**Y todo lo demás salía normal.** El resumen decía 263 427 987 parámetros totales y
87 184 403 entrenables, las cifras exactas de la sección 1. El informe de congelado
decía `partial: last 6/12 blocks trainable (conv extractor frozen) --
42,528,768/94,371,712 (45.1%)`. El contrato de pérdida, la acumulación de gradiente
y el scheduler, todo correcto. Nada delataba que la rama de audio fuera ruido.

### Por qué se detectó, y es pura suerte

**RAVDESS es el único corpus del eje donde el audio es el único canal informativo**:
dos frases fijas (texto nulo por construcción) y una rama de visión que es un
log-mel derivado del propio audio. Con el encoder en ruido no queda ninguna señal,
así que el modelo se fue por debajo del azar y el detector de colapso gritó.

En IEMOCAP el texto informa y en MELD domina: allí la rama de texto —que sí
cargaba— habría sostenido un número **plausible pero falso**, y no habría salido
ningún aviso. Si el fallo hubiera aparecido una semana antes, con los estudios 79 y
80, habríamos escrito el delta de 13.36 pp del par MELD sobre una rama de audio
aleatoria. **El corpus que montamos como control del eje acabó funcionando como
detector de un fallo de infraestructura.**

### Los resultados anteriores NO están afectados

Comprobado contra los `final_results.json` guardados:

| estudio | UA guardada |
|---|---|
| 77 CREMA-D trimodal | 0.8427 |
| **78 CREMA-D solo audio** | **0.8485** |
| 79 MELD trimodal | 0.5941 |
| 80 MELD solo audio | 0.4606 |

El 78 es **solo audio al 84.85 %**. Con un Wav2Vec2 aleatorio eso es imposible:
estaría en el azar. El 80, solo audio sobre MELD, da 46.06 UA, también muy por
encima. **El fallo es nuevo y no retroactivo**, así que el eje de tres puntos y la
tabla de ablación siguen en pie. Nada que rehacer salvo el 85/86.

### Qué se cambió

1. `use_safetensors=True` en los tres encoders, con reintento sin el argumento para
   los modelos que solo publican `.bin` (`microsoft/wavlm-base-plus` es uno;
   `wav2vec2-base`, `wavlm-base`, `wavlm-large` y `hubert-base-ls960` tienen los dos).
2. **El fallback a pesos aleatorios se eliminó.** `_cargar_encoder_preentrenado()`
   levanta `RuntimeError` con el diagnóstico y las tres salidas posibles (borrar el
   `.bin` de la caché, bajar los safetensors a mano, o subir torch a ≥ 2.6 sabiendo
   que eso cambia el entorno y obliga a rehacer las corridas comparables).

Verificado: `|W_conv0|_fro = 12.8552` con los pesos reales, frente a ~1–2 de una
inicialización aleatoria. Y la ruta de fallo levanta la excepción en vez de devolver
ruido.

### La regla que sale de aquí, y vale para el artículo y para el repo

**Un componente que no carga tiene que parar la corrida, no degradarla.** Un
experimento con el encoder aleatorio no es un experimento peor: es un número
inventado, y eso es estrictamente peor que un fallo, porque un fallo se ve.

Los dos fallos silenciosos del repositorio comparten la forma exacta: algo se
rompía, el código lo absorbía sin avisar de verdad, y el resultado seguía siendo
publicable a la vista. `_find_audio_file` tiró 2 943 de 5 531 utterances durante
meses; esto habría metido una rama aleatoria en la contribución central. **Conviene
decir esto en la sección de limitaciones o de reproducibilidad**: no como anécdota,
sino como la razón de que el repo tenga ahora aserciones de integridad por corpus,
verificadores por corpus y cero fallbacks silenciosos.

**Y revisar si queda algún `except` del mismo patrón.** Este se encontró de rebote.


### 13.1 EL POOLER DEL ViT TAMPOCO TIENE PESOS — pero aquí la conclusión es la contraria

Encontrado de rebote al revisar el arranque del estudio 85 ya arreglado. El informe de
carga del ViT dice:

```
[transformers] ViTModel LOAD REPORT from: google/vit-base-patch16-224
pooler.dense.weight | MISSING
pooler.dense.bias   | MISSING
```

Y la rama visual, en `src/ablation_studies/ablation_11_model.py`, **prefiere** esa
salida:

```python
vision_outputs = self.vision_encoder(pixel_values=vit_images)
if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
    vision_features = vision_outputs.pooler_output      # <-- esta rama
else:
    vision_features = vision_outputs.last_hidden_state[:, 0, :]   # CLS
```

`google/vit-base-patch16-224` es un checkpoint de **clasificación**: trae `classifier.*`
y no trae pooler. `ViTModel` añade uno y lo inicializa de cero. Y como el ViT va
**congelado** (`0/86,389,248 params trainable`), ese `Linear(768,768) + tanh` **nunca se
entrena**. O sea: la representación visual que llega a la fusión pasa por una proyección
aleatoria que no aprende nada. Afecta a **todos** los estudios trimodales: 11, 13, 34,
45, 74, 77, 79, 85.

#### Pero medido, no destruye la señal — y esto es lo que lo distingue del caso del audio

| medida | valor | lectura |
|---|---|---|
| rango de `W` | **767 de 768** | la parte lineal no pierde información |
| sesgo `b` | exactamente **0** | no hay desplazamiento |
| saturación del tanh | **0.0 %** de unidades con \|pre\| > 2; máx 1.734 | opera en régimen cuasi-lineal |
| correlación de distancias por pares, CLS vs pooled | **0.9857** | la fusión ve la misma estructura |

Es una **rotación aleatoria casi isométrica** con un reescalado suave, no ruido.

**La diferencia conceptual con el encoder aleatorio de la sección 13 es la que importa, y
conviene tenerla clara para responder a un revisor:** un *encoder* aleatorio produce
features que son ruido, así que no hay nada que recuperar aguas abajo. Una *proyección*
aleatoria de features buenas conserva la información si es de rango completo, y la
fusión —que sí entrena— puede deshacerla. Lo único que destruiría información sería la
saturación del tanh, y está al 0 %.

#### DECISIÓN: no se cambia. Y el motivo es de comparabilidad, no de elegancia

Pasarlo a `last_hidden_state[:, 0, :]` alteraría **todos** los números trimodales ya
medidos —45, 74, 77, 79, 85— y rompería la comparabilidad del eje de tres puntos
entero, a cambio de recuperar ~1.4 % de estructura geométrica. No compensa.

Si alguna vez se cambia, tiene que ser con una tanda nueva completa, **nunca a mitad del
eje**. El sitio natural sería junto al par 83/84, que todavía no se ha corrido.

#### Lo que SÍ hay que hacer

1. **Declararlo en los detalles de implementación del artículo.** No como anécdota: la
   frase correcta es que la representación del ViT se proyecta con una capa lineal
   **no entrenada** antes de la fusión. Nadie lo asumiría leyendo «ViT congelado».
2. **Corolario para la sección 8, y refina la interpretación de la σ.** `torch.manual_seed`
   se fija en `run_ablation.py` (línea 282) ANTES de construir el modelo (línea 316), así
   que el pooler es **reproducible dentro de una semilla** pero **distinto entre
   semillas**. Luego parte de la σ de 3 pp entre semillas viene de una **proyección
   visual distinta cada vez**, no solo del orden de los datos ni de la no determinación
   del cuDNN. No cambia la banda de ±0.86 pp —se midió empíricamente y ya incluye este
   efecto—, pero sí la explicación de de dónde sale.
3. **No "arreglarlo" en silencio.** Si aparece en un diff sin una tanda nueva detrás, es
   un cambio de resultados disfrazado de limpieza.

#### Por qué esto no entra en la lista de «lo que hay que rehacer»

Porque no es un fallo: es una decisión de implementación que nadie tomó explícitamente y
que resulta ser casi inocua. Queda documentada para que la próxima persona que lea
`pooler_output` en el código no piense que viene del checkpoint. El contraste con la
sección 13 es la lección: **el mismo síntoma —pesos que no están en el checkpoint— puede
ser catastrófico o irrelevante, y lo que decide es si lo aleatorio está antes o después
de donde se extrae la información.** Medirlo cuesta diez minutos; suponerlo, un artículo.

### Pendiente

- Relanzar 85 y 86 desde cero (los checkpoints y logs del 20/10 no sirven).
- Barrer el resto del código buscando `except Exception` que degraden en silencio.
- Declarar el pooler no entrenado del ViT en los detalles de implementación (13.1).
