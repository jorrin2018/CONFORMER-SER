# Compuerta con expertos congelados

Rama `feature/expert-gate` (sobre `feature/deck-alignment`). Fecha: 04/11/2026.

## La idea

Los clasificadores que deciden el conflicto **no** se entrenan junto con la fusión. Se
entrenan **antes y por separado**, uno por modalidad, y luego se congelan. En el
entrenamiento global solo aportan sus probabilidades:

- `g` = 1 si los tres expertos no predicen la misma clase;
- con `g` = 1, pesos `w_e = c_e / Σ c`, donde `c_e` es la confianza del experto (su
  softmax máxima). Sustituyen a la `confidence_net`.

### En qué se diferencia del estudio 75

| | 75 (detector entrenado) | Expertos congelados |
|---|---|---|
| Clasificador | lineal sobre el embedding compartido `a`, `m`, `t` | modelo unimodal completo (encoder afinado + cabeza) |
| Entrenamiento | a la vez que la fusión (pérdida auxiliar 0.3) | antes, solo, con su propia pérdida |
| Durante el global | co-adapta con la fusión | fijo |

### Por qué los expertos leen su propio encoder

Si solo se congelaran las cabezas, seguirían leyendo los embeddings del modelo global,
que cambian al afinar los encoders: sus predicciones se degradarían. Cada experto lleva
su encoder congelado. Como no cambian, sus predicciones se **precalculan una vez por
fold** y se leen de caché: la etapa 2 cuesta lo mismo que el 74 y no ocupa más memoria
de GPU.

## Etapa 1: los expertos

| Experto | Origen | Coste |
|---|---|---|
| Audio | estudio **71** (wav2vec2-base, LOSO `standard`, 68.99 % UA) | ya entrenado |
| Texto | estudio **73** (DistilBERT, LOSO `standard`, 67.82 % UA) | ya entrenado |
| Imagen | regresión logística sobre el **[CLS] del ViT congelado** (log-mel del 74), `C` elegido por UA de validación | segundos en CPU |

Los tres ven el mismo reparto que el 74 por fold: cuatro sesiones de entrenamiento, 12 %
estratificado para validación con semilla 42 y la sesión restante como test. Los
checkpoints se toman de la corrida que figura en cada `kfold_summary_ablation_N.json`.

`scripts\precompute_experts.py` reproduce el UA de test de cada experto y lo compara con
el `final_results.json` de su corrida: si difieren en más de 0.5 pp, **para**. Un
checkpoint mal cargado daría predicciones plausibles y falsas.

## Antes de la etapa 2: prueba sin entrenar

`scripts\expert_gate_offline.py` compara, por fold y sobre test, el promedio de los
expertos con la compuerta y con otras reglas (ponderado sin compuerta, selección del más
seguro, voto). También da el oráculo como techo. Reporta además la exactitud dentro de
las muestras en conflicto y el 74 como referencia.

**Regla de decisión, fijada antes de ver los números:** la etapa 2 se corre solo si la
compuerta le gana al promedio por **más de 0.86 pp** de media pareada (la banda de ruido
de re-ejecución) **y en al menos 4 de 5 folds**.

### Sobreconfianza en train

Los expertos se entrenaron con esas mismas utterances: en train aciertan más y discrepan
menos que en test. El script mide la tasa de `g` en train, validación y test. Si en train
dispara mucho menos (más de 10 pp de diferencia), la etapa 2 se entrenaría con una
compuerta casi apagada y se evaluaría con otra que dispara a menudo. En ese caso las
predicciones de train tienen que salir de *cross-fitting*: expertos entrenados sin esas
utterances, lo que cuesta GPU adicional.

## Cómo correrlo (PowerShell, desde la raíz del repo)

```powershell
python -m pytest tests\test_expert_gate.py -q
python scripts\precompute_experts.py --device cuda
python scripts\expert_gate_offline.py
```

`precompute_experts.py` escribe `cache\experts\fold_S1.json` … `fold_S5.json` (ignorados
por git) y tarda unos minutos: es inferencia de dos modelos por fold más una pasada del
ViT. `expert_gate_offline.py` imprime las tablas y el veredicto, y guarda
`cache\experts\offline_summary.json`.

## Fusión por incertidumbre: ECE y Brier

Propuesta de la asesora (04/11/2026). Las dos son medidas de **calibración**: dicen si la
confianza de un modelo corresponde a su tasa real de acierto.

- **ECE** (Expected Calibration Error, 15 bins de la confianza máxima):
  `Σ_b (n_b/N) · |acierto_b − confianza_b|`. Vale 0 si el modelo está bien calibrado.
- **Brier** multiclase: `media de Σ_c (p_c − 1[y = c])²`, entre 0 y 2. Mezcla
  calibración y capacidad de discriminar.

Con las predicciones de los expertos ya guardadas se usan de dos maneras, sin GPU.

**1. Para fusionar.** Cada experto se calibra con escalado de temperatura ajustado en
validación (minimizando la NLL) y después:

| Regla | Peso de cada experto |
|---|---|
| `promedio_cal` | igual, sobre las probabilidades calibradas |
| `compuerta_cal` | la compuerta con las confianzas calibradas |
| `brier` | fijo por fold, `w_e ∝ 1 / Brier_e` en validación |
| `fiabilidad` | por muestra: el acierto esperado del experto a esa confianza, leído de su curva de fiabilidad en validación (la curva que resume la ECE) |
| `producto` | producto de expertos calibrados dividido entre `prior^(E−1)` (fusión bayesiana) |

La regla de decisión es la misma de antes: más de 0.86 pp sobre el promedio y al menos
4 de 5 folds. Como se prueban varias reglas a la vez, una que pase por poco hay que
confirmarla con otra semilla.

**2. Para medir.** Se calculan la ECE, el Brier y la NLL en test de cada experto (crudo y
calibrado) y de cada fusión. También de la fusión con una temperatura aplicada **después**
de fusionar: no cambia el UA y corrige que el promedio de expertos que discrepan quede
infraconfiado. Si `precompute_experts.py` se corre con `--ref-study 74`, sale también la
del modelo trimodal, para compararlo con la fusión tardía de expertos.

**Limitación que hay que declarar.** En el protocolo `standard`, la validación sale de las
mismas sesiones, y por tanto de los mismos hablantes, que el train. Una temperatura
ajustada en validación subestima la sobreconfianza ante hablantes nuevos: la ECE de los
expertos en validación (0.03 a 0.08) es menor que en test.

```powershell
python scripts\precompute_experts.py --device cuda --ref-study 74
python scripts\expert_gate_offline.py
```

`precompute_experts.py` ya no recalcula los folds que están en disco: reanuda donde se
quedó y, con `--ref-study`, solo añade el 74.

## Etapa 2

Se implementa **solo si la prueba sin entrenar sale a favor**. Será un estudio nuevo
sobre el 74: la fusión recibe `g` y `w` de la caché de expertos en lugar de calcularlos
con los clasificadores auxiliares y la `confidence_net`. El resto del modelo y el régimen
de entrenamiento no cambian.

## Cambio colateral

`UnimodalBaselineModel` cargaba sus encoders con `AutoModel.from_pretrained(name)`. Ahora
usa el mismo cargador que el trimodal: pide safetensors primero y falla en vez de seguir
con pesos aleatorios. Con transformers ≥ 5 y torch < 2.6, la versión anterior elegía el
`.bin` de la caché y fallaba. Cuando el encoder carga, los pesos son los mismos.
