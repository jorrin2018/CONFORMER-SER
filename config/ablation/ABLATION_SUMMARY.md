# Resumen de Experimentos de Ablación (Ablation Studies)

Este documento resume las diferencias clave y el objetivo de cada configuración de experimentación ("ablation") en el proyecto.

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
Incorporación de características visuales (Espectrogramas como imágenes) y texto enriquecido.

*   **`ablation_10_speechcue_internal.yaml`**: Incorporación de **SpeechCue** (texto enriquecido con información prosódica).
*   **`ablation_11_melspec_vit.yaml`**: Incorporación de la rama visual usando **ViT (Vision Transformer)** procesando Mel-Spectrograms como imágenes.
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
*   **`ablation_25_enhanced_cross_attention.yaml`**: **Enhanced Cross-Attention**. Múltiples capas de atención cruzada donde cada modalidad atiende a las demás. (Configuración actual en análisis).
*   **`ablation_26_gated_fusion.yaml`**: **Gated Multimodal Unit (GMU)**. Usa compuertas aprendibles para controlar el flujo de información de cada modalidad.
*   **`ablation_27_late_fusion.yaml`**: **Late Fusion**. Procesamiento independiente hasta el final, donde se promedian o concatenan los logits/probabilidades.
*   **`ablation_28_glu_fusion.yaml`**: **GLU Fusion**. Uso de Gated Linear Units para una fusión no lineal eficiente.

## Serie 31-35: Innovaciones Multimodales Avanzadas
Nuevas técnicas para mejorar el procesamiento multimodal, alineación temporal, robustez y manejo de conflictos.
**Base: Ablation 25 (Enhanced Cross-Attention)** - Se utiliza como base por haber obtenido los mejores resultados.

*   **`ablation_31_temporal_alignment.yaml`**: **Temporal Alignment Learning (Adaptive DTW)**. Módulo insertado antes de la fusión que aprende a "estirar" o "comprimir" temporalmente los features de audio/texto para que coincidan evento-a-evento antes de fusionarlos. Resuelve la asincronía real entre modalidades. Basado en Ablation 25.
*   **`ablation_32_temporal_bucketing.yaml`**: **Temporal-Aligned Multimodal Bucketing**. Divide el audio en "buckets" (ventanas) de ~400ms. Fusiona modalidades dentro de cada bucket y luego usa una LSTM para conectar la secuencia de buckets. Mejora la precisión local evitando que el ruido de un segmento largo contamine la decisión emocional de una frase corta. Basado en Ablation 25.
*   **`ablation_33_noise_aware_speechcue.yaml`**: **Noise-Aware SpeechCue**. Modifica el generador de prompts para que detecte ruido de fondo. Si el audio es ruidoso, inyecta un token [UNCERTAIN] al texto, avisando a BERT que "desconfíe" de la prosodia. Aporta robustez en mundo real. Basado en Ablation 25.
*   **`ablation_34_conflict_aware_attention.yaml`**: **Conflict-Aware Cross-Attention**. Mecanismo de atención que detecta explícitamente cuando dos modalidades se contradicen (ej. Texto feliz + Audio enojado) y asigna pesos basados en "confianza" en lugar de simplemente promediar. Maneja ironía y sarcasmo. Basado en Ablation 25.
*   **`ablation_35_modal_stratified_al.yaml`**: **Modal-Stratified Active Learning**. Estrategia de entrenamiento donde el modelo identifica qué modalidad le confunde más y puede solicitar etiquetas adicionales solo para esa modalidad. Eficiencia de datos y reducción de costos de anotación. Basado en Ablation 25.

