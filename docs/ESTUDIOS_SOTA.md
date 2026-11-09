# Estudios 98-103: filas pendientes del artículo y candidatos a superar el estado del arte

Rama `feature/sota-studies` (sobre `feature/expert-gate`). Noviembre 2026.

Todos son **el estudio 74 con un solo cambio**, activado por una clave opt-in. Sin esas
claves el código hace exactamente lo mismo que antes: los logits en modo evaluación de los
estudios 72, 74 y 89 salen idénticos bit a bit antes y después de esta rama.

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
