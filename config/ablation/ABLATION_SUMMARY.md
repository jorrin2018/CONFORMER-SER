# Resumen de Experimentos de Ablación (Ablation Studies)

Este documento resume las diferencias clave y el objetivo de cada configuración de experimentación ("ablation") en el proyecto.

**Total**: estudios 1-50, 52, 53 y 54 | **Mejor resultado**: Ablation 34 (77.98% UA, gap 2.68%) | **Base para 31-35**: Ablation 25 | **Base para 36-39**: Ablation 34

### Ejecución
```powershell
python scripts/run_ablation.py --study N   # N = 1 a 50, o 52, 53, 54
```

### Datos de audio y rama visual (ViT)

**El dataset IEMOCAP en este flujo no incluye mel-spectrogramas precalculados.** Los dataloaders entregan **waveform** (16 kHz) y **texto** (transcripciones; con SpeechCue cuando la config lo activa). En estudios trimodales, el **log-mel que entra al ViT se obtiene en el propio forward** a partir del **mismo** tensor de audio del batch (p. ej. `MelSpectrogram` de torchaudio, opcional SpecAugment, resize a imagen), como en `Ablation11Model._waveform_to_vit_image` en `src/ablation_studies/ablation_11_model.py`. No es una modalidad leída de disco ni un campo separado del parquet: es una vista **derivada** del audio.

---

## Serie 01-04: Arquitecturas de Clasificación (Classification Heads)
El objetivo de esta serie fue determinar cuál es la mejor arquitectura para procesar las características fusionadas antes de la predicción final.

*   **`ablation_01_unfreeze_encoders.yaml`**: Línea base con encoders descongelados.
*   **`ablation_02_lstm_head.yaml`**: Introduce una cabecera **LSTM** para capturar dependencias temporales en las características fusionadas.
*   **`ablation_03_transformer_head.yaml`**: Prueba una cabecera basada en **Transformer Encoder** en lugar de LSTM.
*   **`ablation_04_cnn1d_head.yaml`**: Prueba una cabecera basada en **Convoluciones 1D** para capturar patrones locales.

## Serie 05-07: Estrategias de Fine-Tuning (Unfreezing)
Experimentos enfocados en qué capas de los encoders pre-entrenados (Wav2Vec2, DistilBERT) se deben entrenar y cuáles congelar.

*   **`ablation_05_unfreeze_lstm.yaml`**: Fine-tuning estándar con cabecera LSTM.
*   **`ablation_06_unfreeze_transformer.yaml`**: Fine-tuning estándar con cabecera Transformer.
*   **`ablation_07_unfreeze_mid_high_lstm.yaml`**: Estrategia específica donde solo se descongelan las capas medias y altas de los encoders, manteniendo las capas bajas (features básicas) congeladas.

## Serie 08-09: Profundidad y Fusión Inicial
*   **`ablation_08_deep_lstm.yaml`**: Aumenta la profundidad de la cabecera LSTM (más capas) para mayor capacidad de modelado.
*   **`ablation_09_gia_lstm.yaml`**: Introduce **Gated Interaction Attention (GIA)** como mecanismo de fusión preliminar.

## Serie 10-12: Integración Multimodal y SpeechCue
Incorporación de la rama visual (log-mel **calculado a partir del waveform** y tratado como imagen para ViT) y texto enriquecido.

*   **`ablation_10_speechcue_internal.yaml`**: Incorporación de **SpeechCue** (texto enriquecido con información prosódica).
*   **`ablation_11_melspec_vit.yaml`**: Incorporación de la rama visual usando **ViT (Vision Transformer)** sobre log-mel generado en el modelo a partir del mismo audio del batch (ver nota arriba).
*   **`ablation_12_speechcue_loso.yaml`**: Validación con estrategia Leave-One-Session-Out (LOSO) usando SpeechCue.

## Serie 13: Refinamiento y Regularización (Baseline Robusto)
Esta serie establece el modelo más fuerte combinando lo mejor de los anteriores y probando estrategias de estabilización.

*   **`ablation_13_melspec_vit_speechcue_unfreeze_vit.yaml`**: Modelo completo Trimodal (Audio + SpeechCue + ViT) descongelando el encoder visual.
*   **`ablation_13a_regularized.yaml`**: Añade técnicas de regularización (Dropout, Weight Decay) para combatir overfitting.
*   **`ablation_13b_no_speechcue.yaml`**: Variante sin SpeechCue para medir su impacto aislado.
*   **`ablation_13c_frozen_vit.yaml`**: Variante con el encoder **ViT congelado** (transfer learning puro para visión).
*   **`ablation_13d_high_reg.yaml`**: Regularización agresiva.
*   **`ablation_13d_kfold.yaml`**: Configuración para validación cruzada (K-Fold).
*   **`ablation_13e_advanced_reg.yaml`**: Regularización avanzada (ej. Label Smoothing, Mixup).
*   **`ablation_13f_focal_loss.yaml`**: Uso de **Focal Loss** en lugar de Cross Entropy para manejar desbalance de clases.

## Serie 20+: Mecanismos de Fusión Avanzados
El foco actual. Se mantiene la arquitectura base (generalmente basada en `13c`) y se cambia **exclusivamente** el módulo de fusión multimodal.

*   **`ablation_21_hierarchical_fusion.yaml`**: **Fusión Jerárquica**. Fusiona primero Audio+Texto, y el resultado se fusiona con Visión.
*   **`ablation_22_tensor_fusion.yaml`**: **Tensor Fusion Network**. Producto exterior de las modalidades para capturar todas las interacciones posibles (costoso computacionalmente).
*   **`ablation_23_bilinear_fusion.yaml`**: **Bilinear Pooling**. Interacciones multiplicativas de segundo orden entre modalidades.
*   **`ablation_24_adaptive_weighted_fusion.yaml`**: **Fusión Ponderada Adaptativa**. El modelo aprende pesos escalares (sigmoide) para dar más o menos importancia a cada modalidad dinámicamente.
*   **`ablation_25_enhanced_cross_attention.yaml`**: **Enhanced Cross-Attention** ⭐. Múltiples capas de atención cruzada donde cada modalidad atiende a las demás. **Mejor fusión base** (77.44% UA). Base para series 31-35.
*   **`ablation_26_gated_fusion.yaml`**: **Gated Multimodal Unit (GMU)**. Usa compuertas aprendibles para controlar el flujo de información de cada modalidad.
*   **`ablation_27_late_fusion.yaml`**: **Late Fusion**. Procesamiento independiente hasta el final, donde se promedian o concatenan los logits/probabilidades.
*   **`ablation_28_glu_fusion.yaml`**: **GLU Fusion**. Uso de Gated Linear Units para una fusión no lineal eficiente.

## Serie 30: Dirichlet Evidence Learning
*   **`ablation_30_dirichlet.yaml`**: **Dirichlet Evidence Learning**. Sustituye softmax por distribución Dirichlet para modelar incertidumbre epistémica. Mejor calibración de confianza. Basado en Ablation 25.

## Serie 31-35: Innovaciones Multimodales Avanzadas
Nuevas técnicas para mejorar el procesamiento multimodal, alineación temporal, robustez y manejo de conflictos.
**Base: Ablation 25 (Enhanced Cross-Attention)** - Se utiliza como base por haber obtenido los mejores resultados.

*   **`ablation_31_temporal_alignment.yaml`**: **Temporal Alignment Learning (Adaptive DTW)**. Módulo insertado antes de la fusión que aprende a "estirar" o "comprimir" temporalmente los features de audio/texto para que coincidan evento-a-evento antes de fusionarlos. Resuelve la asincronía real entre modalidades. Basado en Ablation 25.
*   **`ablation_32_temporal_bucketing.yaml`**: **Temporal-Aligned Multimodal Bucketing**. Divide el audio en "buckets" (ventanas) de ~400ms. Fusiona modalidades dentro de cada bucket y luego usa una LSTM para conectar la secuencia de buckets. Mejora la precisión local evitando que el ruido de un segmento largo contamine la decisión emocional de una frase corta. Basado en Ablation 25.
*   **`ablation_33_noise_aware_speechcue.yaml`**: **Noise-Aware SpeechCue**. Modifica el generador de prompts para que detecte ruido de fondo. Si el audio es ruidoso, inyecta un token [UNCERTAIN] al texto, avisando a BERT que "desconfíe" de la prosodia. Aporta robustez en mundo real. Basado en Ablation 25.
*   **`ablation_34_conflict_aware_attention.yaml`**: **Conflict-Aware Cross-Attention** 🏆. Detecta contradicciones entre modalidades (ej. texto feliz + audio enojado) y pondera por confianza. **Mejor resultado global** (77.98% UA, gap 2.68%). Base para series 36-39.
    *   **Conflict Detection Methods**: En `model.conflict_aware.conflict_detection_method` se puede elegir: `prediction_divergence` (por defecto: conflicto cuando 2+ modalidades predicen distinta clase), `feature_distance` (conflicto cuando la distancia coseno media entre embeddings > umbral), o `learned` (MLP que predice conflicto). Ver [docs/CONFLICT_DETECTION.md](../../docs/CONFLICT_DETECTION.md).
*   **`ablation_35_modal_stratified_al.yaml`**: **Modal-Stratified Active Learning**. Estrategia de entrenamiento donde el modelo identifica qué modalidad le confunde más y puede solicitar etiquetas adicionales solo para esa modalidad. Eficiencia de datos y reducción de costos de anotación. Basado en Ablation 25.

## Serie 36-39: Conflict-Aware + Dirichlet y SOTA
**Base: Ablation 34 (Conflict-Aware Cross-Attention)** - Combina Conflict-Aware con Dirichlet, Focal Loss y mejoras state-of-the-art.

*   **`ablation_36a_dirichlet_conflict.yaml`**: **Conflict-Aware + Dirichlet (Pure)**. Combina Conflict-Aware Fusion con Dirichlet Classification Head. Regularización: Dropout 0.5, Weight decay 0.1.
*   **`ablation_36b_focal_conflict.yaml`**: **Conflict-Aware + Focal Loss**. Focal Loss con pesos aumentados para neutral (2.0x), gamma=2.5.
*   **`ablation_36c_hybrid_conflict.yaml`**: **Conflict-Aware + Dirichlet Hybrid**. Dirichlet con regularización adicional para neutral (DirichletNeutralRegularizedLoss).
*   **`ablation_37_dirichlet_conflict_aware.yaml`**: **Dirichlet + Conflict-Aware**. Configuración específica que combina Dirichlet Evidence Loss con Conflict-Aware Cross-Attention.
*   **`ablation_38_opensmile_residual.yaml`**: **OpenSMILE + Attention Gates + Residual**. Rama OpenSMILE (eGeMAPSv02, 88 features), attention gates, cross-attention Wav2Vec2↔OpenSMILE, conexiones residuales.
*   **`ablation_39_sota_improvements.yaml`**: **ACN + MIC + BCAF**. Audio Context Network, Modal Invariance Constraint (SKL), Bidirectional Connection Attention Fusion con autoencoder.

## Ablation 40: WavLM-base + Conflict-Aware (Trimodal)
*   **`ablation_40_wavlm_conflict_aware.yaml`**: **WavLM-base + Conflict-Aware Cross-Attention**. Misma arquitectura que Ablation 34 (trimodal, Conflict-Aware, ViT, DistilBERT, LSTM head) pero con encoder de audio **microsoft/wavlm-base** en lugar de Wav2Vec2-base. Objetivo: evaluar si WavLM mejora el rendimiento por mejor modelado de ruido y speaker identity. Para cambiar el método de detección de conflicto (`prediction_divergence` | `feature_distance` | `learned`), ver [docs/CONFLICT_DETECTION.md](../../docs/CONFLICT_DETECTION.md).

## Ablation 41 (Study 41): WavLM-base + ViT + Conflict-Aware, K-Fold
*   **`ablation_41_wavlm_base_vit_kfold.yaml`**: **Ablation 41 = WavLM-base + ViT + Conflict-Aware, K-Fold**. Misma arquitectura que Ablation 40 (WavLM-base + DistilBERT + ViT + Conflict-Aware trimodal) con validación cruzada 5 folds (leave-one-session-out).

## Ablation 42 (Study 42): WavLM-base + ViT + Conflict-Aware, single split
*   **`ablation_42_wavlm_base_vit.yaml`**: **WavLM-base + ViT + Conflict-Aware, single split**. Mismo diseño que 34/40 con modelo más pequeño y rama visual; sin K-Fold (train 1,2,3 / val 4 / test 5). Evalúa si añadir ViT con WavLM-base mejora respecto a 34 bimodal grande.

## Ablation 43 (Study 43): Conflict-Aware + mejoras de entrenamiento
*   **`ablation_43_training_improvements.yaml`**: **Conflict-Aware + Focal Loss + Supervised Contrastive Loss + augmentación**. Extiende Ablation 34 con una receta de entrenamiento más fuerte: focal loss para clases difíciles, componente contrastivo supervisado y augmentaciones orientadas a mejorar separación entre clases y robustez. Implementado por `Ablation43Model`.

## Ablation 44 (Study 44): Conflict-Aware + Transformer Head
*   **`ablation_44_transformer_head.yaml`**: **Conflict-Aware Cross-Attention (trimodal) + TransformerClassificationHead**. Misma configuración que Ablation 34 (Wav2Vec2-base + DistilBERT + ViT, Conflict-Aware) pero sustituyendo la cabeza LSTM por una cabeza basada en **TransformerEncoder** sobre el vector fusionado.

## Ablation 45 (Study 45): Conflict-Aware + MultiScaleTemporal Head
*   **`ablation_45_multiscale_head.yaml`**: **Conflict-Aware Cross-Attention (trimodal) + MultiScaleTemporalClassificationHead**. Igual que Ablation 34, pero usando una cabeza **MultiScaleTemporal** (pooling mean/max/attention + LSTM + self-attention) aplicada al output de la fusión para capturar representaciones más ricas antes de clasificar.

## Ablation 46 (Study 46): Conflict-Aware + CNN1D Head
*   **`ablation_46_cnn1d_head.yaml`**: **Conflict-Aware Cross-Attention (trimodal) + CNN1DClassificationHead**. Misma arquitectura base que 34, pero reemplazando la cabeza LSTM por una cabeza **CNN 1D** que aplica convoluciones 1D y pooling global sobre el vector fusionado antes de la capa lineal final.

## Ablation 47 (Study 47): Conflict-Aware + SupCon (pre-fusión, 2 vistas)
*   **`ablation_47_supcon.yaml`**: **Conflict-Aware Cross-Attention (trimodal) + Supervised Contrastive Loss (SupCon)**. Misma arquitectura que Ablation 34 (Wav2Vec2-base + DistilBERT + ViT, Conflict-Aware), pero añade una pérdida contrastiva supervisada sobre los embeddings **audio/texto** normalizados antes de la fusión Conflict-Aware. La pérdida total combina Cross Entropy sobre los logits finales con SupCon ponderada por `contrastive_weight`.

## Ablation 48 (Study 48): Conflict-Aware + SupCon (pre-fusión, 3 vistas)
*   **`ablation_48_supcon_3views.yaml`**: **Conflict-Aware + SupCon (audio + texto + visión)**. Igual que Ablation 47 pero la pérdida contrastiva supervisada se aplica sobre **tres vistas**: audio, texto y visión (en lugar de solo audio y texto). Objetivo: alinear las tres modalidades en el espacio de embeddings antes de la fusión Conflict-Aware.

## Ablation 49 (Study 49): Conflict-Aware + SupCon (3 vistas) + WavLM-base
*   **`ablation_49_supcon_wavlm.yaml`**: **Conflict-Aware + SupCon (audio + texto + visión) con encoder de audio WavLM-base**. Misma arquitectura que Ablation 48 (SupCon con 3 vistas y Conflict-Aware trimodal), pero sustituyendo Wav2Vec2-base por **microsoft/wavlm-base** como encoder de audio para explotar mejor robustez al ruido y modelado de locutor.

## Ablation 50 (Study 50): Conflict-Aware + SupCon (3 vistas) + Emotion2Vec
*   **`ablation_50_emotion2vec.yaml`**: **Conflict-Aware + SupCon (audio + texto + visión) con encoder de audio Emotion2Vec**. Misma arquitectura que Ablation 48 (SupCon con 3 vistas y Conflict-Aware trimodal), pero el encoder de audio es **Emotion2Vec** (emotion2vec/emotion2vec_base), cargado vía FunASR. Requiere `pip install funasr` para ejecutar el estudio 50.

## Ablation 52 (Study 52): Igual que 49 + sin augment de waveform + SpecAugment sobre log-mel del ViT (derivado del waveform)
*   **`ablation_52_supcon_wavlm_specaug_vit.yaml`**: **Misma base que Ablation 49** (WavLM + SupCon 3 vistas + Conflict-Aware). Diferencias: `audio.apply_waveform_augment: false` (el preprocessor sigue aplicando **z-score** al waveform para escala estable); en la rama visual, `model.vision_mel_spec_augment: true` activa **SpecAugment** (FrequencyMasking / TimeMasking de torchaudio) sobre el **log-mel calculado desde ese waveform** antes del resize a imagen ViT, solo en entrenamiento (`model.training`). Cualquier estudio trimodal puede activar lo mismo con `vision_mel_spec_augment` y opcionalmente `vision_spec_augment.freq_mask_param` / `time_mask_param`.

## Ablation 53 (Study 53): Run tipo 174035 de 52 + WavLM suma ponderada por capas
*   **`ablation_53_wavlm_weighted_hidden_states.yaml`**: **Misma receta operativa que el checkpoint `ablation_52_20260320_174035`** (p. ej. `freeze_text_encoder: false`, `freeze_vision_encoder: true`, sin augment de waveform, SpecAugment sobre log-mel del ViT derivado del waveform, WavLM-base + SupCon 3 vistas + Conflict-Aware). **Novedad**: `model.wavlm_weighted_hidden_states: true` activa en `Ablation48Model` una combinación **softmax-aprendida** de todos los `hidden_states` de WavLM (embeddings + capas del transformer) antes del pooling temporal, alineado con la recomendación de explotar varias capas de WavLM.

## Ablation 54 (Study 54): Igual que 53 con WavLM-Base+
*   **`ablation_54_wavlm_base_plus_weighted.yaml`**: **Misma configuración que el estudio 53** (weighted hidden states, SupCon 3 vistas, Conflict-Aware, etc.) sustituyendo el encoder de audio por **`microsoft/wavlm-base-plus`**. Sirve para comparar Base vs Base+ bajo el mismo protocolo de entrenamiento.

## Post-45 optimization (OpenSpec `mejoras-post-ablation-45`)

*   **`ablation_45_sweep_v1.yaml`**: sweep acotado (`conflict_threshold: 0.25`) sobre la misma receta que Ablation 45 (MultiScaleTemporal + Conflict-Aware); mismo split de sesiones **1–3 / 4 / 5**. Baseline de resultados: `checkpoints/ablation_45/post_fix_conflict_chain_study_45/`.
*   **`ablation_phase2_49_wavlm_multiscale_head_FUTURE.yaml`**: esqueleto **no ejecutable** para combinar receta tipo estudio 49 (WavLM + SupCon) con cabeza MultiScale del 45; requiere validar compatibilidad en `scripts/run_ablation.py` (estudios 45 usan `Ablation34Model`; 48/49 usan `Ablation48Model`).
*   **Contrato `training.loss`**: comentarios en `src/utils/losses.py` y mensajes de `log_loss_contract_traceability` al arrancar `src/train.py` / `run_ablation.py` (``gamma`` no aplica a `cross_entropy` plano).

---

## Mapeo Study → Config (run_ablation.py)

| Study | Config | Modelo |
|-------|--------|--------|
| 1-11 | ablation_01...ablation_11 | Ablation01Model...Ablation11Model |
| 12 | ablation_12_speechcue_loso | Ablation10Model (LOSO) |
| 13-20 | ablation_13...ablation_13d_kfold | Ablation13Model |
| 21-28 | ablation_21...ablation_28 | Ablation21Model...Ablation28Model |
| 29 | ablation_25_kfold | Ablation25Model |
| 30 | ablation_30_dirichlet | Ablation30Model |
| 31-35 | ablation_31...ablation_35 | Ablation31Model...Ablation35Model |
| 36 | ablation_36a_dirichlet_conflict | Ablation36AModel |
| 37 | ablation_37_dirichlet_conflict_aware | Ablation36AModel |
| 38 | ablation_38_opensmile_residual | Ablation38Model |
| 39 | ablation_36c_hybrid_conflict | Ablation39Model |
| 40 | ablation_40_wavlm_conflict_aware | Ablation34Model |
| 41 | ablation_41_wavlm_base_vit_kfold | Ablation34Model |
| 42 | ablation_42_wavlm_base_vit | Ablation34Model |
| 43 | ablation_43_training_improvements | Ablation43Model |
| 44 | ablation_44_transformer_head | Ablation34Model |
| 45 | ablation_45_multiscale_head | Ablation34Model |
| 46 | ablation_46_cnn1d_head | Ablation34Model |
| 47 | ablation_47_supcon | Ablation47Model |
| 48 | ablation_48_supcon_3views | Ablation48Model |
| 49 | ablation_49_supcon_wavlm | Ablation48Model |
| 50 | ablation_50_emotion2vec | Ablation48Model |
| 52 | ablation_52_supcon_wavlm_specaug_vit | Ablation48Model |
| 53 | ablation_53_wavlm_weighted_hidden_states | Ablation48Model |
| 54 | ablation_54_wavlm_base_plus_weighted | Ablation48Model |

*Última actualización: Marzo 2026*
