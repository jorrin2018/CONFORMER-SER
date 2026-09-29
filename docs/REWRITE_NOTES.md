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
2. **CREMA-D.** Cargador **escrito el 23/09/2026** (`src/data/cremad_dataset.py`),
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
3. Multi-semilla 42/123/456 → 15 observaciones pareadas; el test de signos deja de topar en
   p = 0.0625.
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

