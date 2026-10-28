# Variantes de la presentación — serie 87-94

Rama `feature/deck-alignment`. Fecha: 28/10/2026.

La presentación `CAMF_presentacion_1.pptx` (diapositivas 13-24) describe CAMF con un
detalle que el código no sigue en varios puntos. Este documento lista **cada
diferencia**, qué se implementó para cerrarla y cómo medir si importa.

Regla de la serie: **todo es opt-in**. Sin las claves nuevas el código hace exactamente lo
mismo que en `main` (verificado: mismos logits del 72 y del 74, mismos aumentos y mismas
muestras del dataset, bit a bit, con semillas fijas). Ninguna corrida anterior cambia.

---

## 1. Diferencias que se corrigen en código (estudios nuevos)

| # | Presentación | Código (`main`) | Clave nueva | Estudio |
|---|---|---|---|---|
| 1 | **SpeechCue**: 3 tokens `[P_x][E_x][R_x]` tras `[CLS]`, terciles **globales** de train, 9 embeddings (Ec. 4-5, diapos 16-17) | El 72 usa `use_prosody_cues: true`: un **prompt en texto libre** `"[SPEECH] … [PROSODY] overall energy is high, …"`. El SpeechCue de tokens solo estaba en el 64, con normalización por hablante | `text.speechcue: {enabled: true, normalize: global}` | **87** |
| 2 | Tasa silábica **desde la alineación forzada** de IEMOCAP a nivel de sílaba (notas diapo 16) | Heurística de grupos vocálicos sobre la transcripción ÷ tramo hablado recortado por energía | `text.speechcue.rate_source: forced_alignment` + cache `cache/speechcue_stats_fa.json` | **88** |
| 3 | `a` = promedio de los **T′ frames** de Wav2Vec2 (Ec. 1) | Todo enunciado se rellena con ceros a 8 s y se promedia sobre los 400 frames, **relleno incluido**. Medido sobre la caché: el **48.2 %** de los frames promediados es relleno (duración media 4.55 s, mediana 3.58 s). Además el ruido (SNR) y la normalización z se calculaban sobre la señal rellenada | `audio.pad_after_preprocessing: true` + `model.audio_masked_pooling: true` | **89** |
| 4 | log-mel con z-score **por enunciado**, redimensionado a 224×224 (Ec. 2-3) | El log-mel se calcula sobre los 8 s: el z-score lo domina el suelo `log(1e-9)` del relleno y el habla ocupa solo la franja izquierda de la imagen | `model.vision_mel_speech_only: true` | **90** |
| 5 | Se proyecta el **[CLS]** del ViT (Ec. 3) | Se usa `pooler_output`: un `Linear+tanh` que el checkpoint no trae, aleatorio y nunca entrenado (REWRITE_NOTES 13.1) | `model.vision_feature: cls` | **91** |
| 6 | ViT con bloques **9-12 + LayerNorm final** afinados (Fig. 2, notas diapo 14, Tabla 13: 28.35 M entrenables) | ViT **completamente congelado** (0 / 86.4 M) | `freeze_vision_encoder: false`, `vision_unfreeze_last_n_layers: 4` | **92** |
| 7 | Aumento con **pitch shift ±2 semitonos** (Tabla 7 y notas diapo 24) | `apply_pitch_shift: true` **no hace nada**: `forward()` nunca llama a `_apply_pitch_shift`, que además es la identidad | `audio.augmentation.pitch_shift_impl: torchaudio` | **93** |
| — | CAMF completo tal como se presenta | — | 1 + 2 + 3 + 4 + 5 + 7 (ViT congelado) | **94** |

Comparación de cada fila, **pareada por fold** y con el mismo protocolo:

| Estudio | Se compara contra | Factor |
|---|---|---|
| 87 | 72 | SpeechCue de tokens en vez del prompt en texto |
| 88 | 87 | fuente de la tasa silábica |
| 89 | 72 | la rama de audio ve solo el enunciado real |
| 90 | 89 | log-mel del ViT sin el relleno |
| 91 | 72 | [CLS] en vez del pooler aleatorio |
| 92 | 91 | 4 bloques superiores del ViT entrenables |
| 93 | 72 | pitch shift real |
| 94 | 72 | todo lo anterior junto (no es una fila de ablación) |

Banda de ruido de re-ejecución: **±0.86 pp** (REWRITE_NOTES sec. 7). Una diferencia
pareada menor no es atribuible al cambio.

### Decisiones tomadas al implementar (revisables)

- **94 deja el ViT congelado.** La presentación se contradice: Fig. 2, notas de la
  diapo 14 y Tabla 13 dicen «bloques 9-12 afinados»; las notas de la diapo 14 («POR QUÉ
  ViT CONGELADO») y la diapo 30 («con ViT frozen») dicen congelado. Se sigue al 72 y se
  mide lo otro aparte en el 92. Para cambiarlo basta con editar dos líneas del YAML.
- **ViT afinado a 5e-6.** La presentación da tres tasas (audio, texto, fusión+cabeza);
  los parámetros del ViT caen en el grupo de fusión.
- **89 junta dos claves en un factor**: `pad_after_preprocessing` (aumento y normalización
  sobre el enunciado real, relleno al final en ceros exactos) y `audio_masked_pooling`.
  Las dos describen lo mismo: que la rama de audio no vea el relleno. Los estudios 89, 90
  y 94 exigen `pad_after_preprocessing` y fallan si falta: con el orden histórico la
  speed perturbation estira el contenido dentro del relleno y la longitud real deja de
  ser exacta.
- **El encoder de Wav2Vec2 sigue sin `attention_mask`.** `wav2vec2-base` se entrenó sin
  ella y Hugging Face recomienda no pasarla; la máscara se aplica solo en el promedio.
- **Pitch shift** con un número entero de semitonos distinto de cero (±1, ±2). Mismo
  algoritmo que `torchaudio.functional.pitch_shift` (vocoder de fase + remuestreo) pero
  con una razón racional de enteros pequeños: ~40 ms por enunciado de 8 s en CPU frente
  a ~6.5 s de la función de torchaudio. Error de la aproximación < 0.01 semitonos.
- **Con `legacy_noop` (por defecto) no se consume ningún número aleatorio** en el paso
  del pitch shift: la secuencia de aumentos de las corridas viejas no cambia. Ahora sí
  sale un `UserWarning` cuando un config pide pitch shift y no se aplica.

### Arreglo colateral

`Ablation11Model` tenía un `except Exception` que, si el ViT no cargaba, seguía con
`ViTModel(ViTConfig())` **aleatorio** sin avisar: el mismo defecto que dejó el audio en
ruido el 20/10 (REWRITE_NOTES sec. 13). Ahora falla. No cambia nada mientras el ViT
cargue, que es lo que pasa en todas las corridas registradas (el `LOAD REPORT` del ViT
aparece en los logs).

---

## 2. Diferencias que se corrigen en la presentación (no en el código)

| Presentación | Lo que hay | Qué hacer |
|---|---|---|
| Clasificadores auxiliares **sin pérdida propia** y «g se activa en el 30.9 % de train» (diapos 18-20) | Sin pérdida no reciben gradiente y quedan en su init: la compuerta dispara en el **94.5 %** (medido el 20/09). Con el detector entrenado (estudio 75) la compuerta **no aporta** (74-76) | Quitar el 30.9 %; describir la compuerta según el resultado del 74-76 |
| Cabeza «MultiScaleTemporal» con tres escalas (diapos 21-22) | Recibe un vector `h`: mean, max y attention pooling son copias; la BiLSTM ve tres copias (REWRITE_NOTES, hallazgo del 16/09). Las dimensiones (768 + 1024 + 256 = 2048) sí coinciden | No llamarla temporal ni multiescala |
| Parámetros: 261.5 M / **85.9 M** entrenables (diapo 13) y **114.23 M** (Tabla 13) | 263 427 987 / **87 184 403** (`report_finetuning.py`, estudio 45). La Tabla 13 cuenta 28.35 M del ViT como entrenables | Usar los del script |
| Fusión 5.27 M + cabeza 9.73 M | Fusión + cabeza = 16.30 M | Ídem |
| Split 1-3 / 4 / 5 y una semilla (diapo 23); 79.23 % UA, etc. | Protocolo `standard` LOSO 5-fold. Las cifras de la presentación son **anteriores** al arreglo del corpus y no valen (AGENTS.md) | Rehacer resultados con LOSO |
| Mínimo del scheduler «1 % del LR inicial de cada grupo» | `max(1 %, 1e-7)`: el grupo de fusión queda en 1e-7 (2 %), no en 5e-8 | Decirlo así o cambiar `scheduler_min_lr` |
| Texto de «máx. 128 tokens + 3 SpeechCue» | `max_length: 128` incluye `[CLS]`, `[SEP]` y los 3 tokens | Decir «128 en total» |
| ViT-B/16 «preentrenado en ImageNet-21k» | `google/vit-base-patch16-224` = ImageNet-21k + ajuste fino en ImageNet-1k | Precisarlo |
| PyTorch 2.2, Transformers 4.41 (Tabla 7) | No coincide con el entorno registrado en `requirements.txt` | Poner las versiones reales |
| No se menciona | El waveform se normaliza (z-score) antes de Wav2Vec2 | Añadirlo a detalles de implementación |

---

## 3. Cómo correrlo (CMD)

### 3.1 Traer la rama

La rama viene en un bundle de git (`deck-alignment.bundle`). Desde la raíz del repo:

```bat
git fetch deck-alignment.bundle feature/deck-alignment:feature/deck-alignment
git checkout feature/deck-alignment
```

### 3.2 Verificar sin GPU (minutos)

```bat
python -m pytest tests\test_deck_variants.py -q
python scripts\check_forced_alignment.py
python scripts\fit_speechcue_stats.py --rate-source forced_alignment
for %s in (87 88 89 90 91 92 93 94) do python scripts\check_deck_variants.py --study %s --with-data
```

- `check_forced_alignment.py` contrasta el parser de `.syseg` con los ficheros reales (se
  escribió sin acceso a IEMOCAP). Tiene que decir **OK**; si dice REVISAR, no correr 88 ni
  94.
- `fit_speechcue_stats.py --rate-source forced_alignment` **no vuelve a correr pYIN**:
  copia F0 y energía de `cache\speechcue_stats.json` y solo cambia la tasa. Escribe
  `cache\speechcue_stats_fa.json` y falla si falta algún `.syseg`.
- `check_deck_variants.py` monta el modelo real con el config real y comprueba cada
  bandera (pooling sobre frames reales, log-mel invariante al relleno, [CLS], bloques del
  ViT, pitch shift sobre un seno, tokens SpeechCue tras `[CLS]` en un batch real).
- Y como siempre: `python scripts\report_optimizer.py --study N`.

### 3.3 Entrenar (GPU)

Un proceso por fold, como se acordó tras los fallos de memoria de host:

```bat
for %f in (1 2 3 4 5) do python scripts\run_ablation.py --study 94 --loso --protocol standard --folds %f --resume --device cuda
```

(En un `.bat`, `%%f` en vez de `%f`.)

Orden sugerido, de más a menos informativo:

1. **94** — ¿el modelo de la presentación difiere del 72?
2. **87** — SpeechCue de tokens frente al prompt en texto (la H2 de la presentación).
3. **89** y **91** — los dos defectos de implementación más grandes (48 % de relleno en el
   promedio; proyección visual aleatoria).
4. **88**, **90**, **93**, **92**.

Para el test de una corrida ya hecha usar `scripts\run_test_only.py` (construye
SpeechCue y las longitudes de audio); `src\evaluate_ensemble.py` no construye SpeechCue.
