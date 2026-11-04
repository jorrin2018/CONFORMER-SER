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

## Plan por pasos (decidido el 05/11/2026)

**Paso 1: expertos entrenados por separado y sin fuga.** Es lo que hay que correr ahora.

**Paso 2: la compuerta original de la presentación, alimentada por los expertos.** Un
estudio nuevo sobre el 72 (CAMF con la compuerta original): `g` sale de la discrepancia
de los expertos congelados, en lugar de los clasificadores auxiliares, y los pesos `w`
siguen saliendo de la `confidence_net`, como en la Ec. 9. Se compara pareado por fold con
el 72, el 74 (sin compuerta) y el 75 (auxiliares entrenados junto con la fusión). Se
escribe después del paso 1, cuando existan las predicciones fuera de muestra.

Este paso no lo decide la prueba sin entrenar: esa prueba evalúa reglas que pesan con la
confianza **del experto**, y la `confidence_net` solo se puede medir entrenándola.

### Paso 1: cross-fitting de los expertos

Para entrenar con la compuerta, cada utterance de **train** necesita la predicción de un
experto que **no la vio**. Los checkpoints del 71 y del 73 se entrenaron con ellas:
medido en S1-S3, dan 95-99 % de UA en train y 66-73 % en test, y la compuerta
dispara en el 37 % de train frente al 66 % de test. Entrenar así sería entrenar con una
compuerta casi apagada y evaluar con otra encendida.

Solución, por fold externo `S` (la sesión de test de CAMF):

1. Se excluye `S` y se hace LOSO sobre las otras cuatro: cada modelo interno entrena con
   3 sesiones (más su 12 % de validación) y predice la cuarta.
2. La predicción de cada utterance de train sale del modelo interno que dejó fuera **su**
   sesión.
3. Validación y test del fold externo siguen usando el experto completo (71 / 73), que
   tampoco los vio.

| Experto | Estudio | Corridas | Coste aprox. (RTX 3070) |
|---|---|---|---|
| Audio | **95** = el 71 sin cambios | 4 por fold externo, 20 en total | ~2 h cada una, ~40 h |
| Texto | **96** = el 73 sin cambios | 20 | ~17 min cada una, ~6 h |
| Imagen | logística sobre el [CLS] del ViT | se ajusta por sesión dentro de `precompute_experts.py` | segundos |

`run_ablation.py --outer-test S` hace el LOSO interno y lo guarda en
`checkpoints\ablation_9X\outer_S<S>\`. La sesión externa no entra nunca: ni en train, ni
en validación, ni en el test interno (hay un test que lo comprueba).

```powershell
# Paso 1a: expertos internos. Un proceso por fold interno, como siempre.
foreach ($S in 1..5) {
  foreach ($s in (1..5 | Where-Object { $_ -ne $S })) {
    python scripts\run_ablation.py --study 96 --outer-test $S --protocol standard --folds $s --resume --device cuda
    python scripts\run_ablation.py --study 95 --outer-test $S --protocol standard --folds $s --resume --device cuda
  }
}
# Paso 1b: predicciones fuera de muestra (inferencia; minutos)
python scripts\precompute_experts.py --crossfit --device cuda
# Paso 1c: comprobar que la compuerta dispara igual en train que en test
python scripts\expert_gate_offline.py
```

Conviene empezar con un solo fold externo (`$S = 1`: ~9 h) y revisar el paso 1c antes de
lanzar los otros cuatro. `precompute_experts.py --crossfit` reproduce el UA de test de
cada modelo interno contra su `final_results.json` y para si no coincide, igual que con
los expertos completos.

## Cambio colateral

`UnimodalBaselineModel` cargaba sus encoders con `AutoModel.from_pretrained(name)`. Ahora
usa el mismo cargador que el trimodal: pide safetensors primero y falla en vez de seguir
con pesos aleatorios. Con transformers ≥ 5 y torch < 2.6, la versión anterior elegía el
`.bin` de la caché y fallaba. Cuando el encoder carga, los pesos son los mismos.
