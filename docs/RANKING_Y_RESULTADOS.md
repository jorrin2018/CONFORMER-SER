# Ranking y analisis de resultados (checkpoints locales)

**Generado**: 2026-04-27 19:00 UTC ejecutando `python scripts/aggregate_ablation_results.py --write-docs`.

## Criterios

- **WA (weighted accuracy)**: accuracy global ponderada por soporte de clase; suele alinearse con lo que muchos papers citan como accuracy.
- **UA (unweighted accuracy)**: media de accuracy por clase; mas justa con clases minoritarias.
- **gap_ua**: `best_val_ua - UA_test`. Indica cuanto cae el modelo del mejor punto de validacion UA al test. **Valores mas bajos** suelen indicar mejor transferencia (menos sobreajuste a val), siempre comparando runs comparables.
- **Agregacion por estudio**: si hay varios `final_results.json` en carpetas `fold_*`, se usa **media** en test (K-Fold). Si hay varios runs en un solo split, se elige el run con **mayor WA** (`mejor_WA`).

## Aviso sobre el estudio 34 historico

En `config/ablation/ABLATION_SUMMARY.md` se cita un run canonico de Ablation 34 con ~77.98% UA por su bajo gap. Eso **no sustituye** al ranking de esta tabla: aqui solo entran runs que existen hoy bajo `checkpoints/` con `final_results.json`. Un run concreto puede superar al 34 en WA/UA segun configuracion y semilla.

## Top 3 por WA (test)

| Pos | Estudio | WA | UA | gap_ua | Nota |
|-----|---------|----|----|--------|------|
| 1 | 45 | 0.7961 | 0.7923 | 0.0291 | unico |
| 2 | 52 | 0.7824 | 0.7682 | 0.0302 | unico |
| 3 | 48 | 0.7804 | 0.7728 | 0.0497 | unico |

## Top 3 por UA (test)

| Pos | Estudio | UA | WA | gap_ua | Nota |
|-----|---------|----|----|--------|------|
| 1 | 45 | 0.7923 | 0.7961 | 0.0291 | unico |
| 2 | 48 | 0.7728 | 0.7804 | 0.0497 | unico |
| 3 | 52 | 0.7682 | 0.7824 | 0.0302 | unico |

## Mejor generalizacion (menor gap_ua con UA test > 0.70)

Orden: gap creciente (mejor primero), desempate por UA mayor.

| Pos | Estudio | gap_ua | UA | WA |
|-----|---------|--------|----|----|
| 1 | 29 | -0.0024 | 0.7571 | 0.7546 |
| 2 | 40 | 0.0018 | 0.7032 | 0.7157 |
| 3 | 12 | 0.0230 | 0.7085 | 0.7017 |
| 4 | 45 | 0.0291 | 0.7923 | 0.7961 |
| 5 | 52 | 0.0302 | 0.7682 | 0.7824 |

## Interpretacion (accuracy vs generalizacion)

Con los numeros agregados de `checkpoints/` en la fecha del documento:

- **Mejor accuracy en test (WA y UA)**: el estudio **45** encabeza ambos rankings. Es el candidato principal si el objetivo es maximizar metricas globales en el split estandar (sesiones 1-3 / 4 / 5) con la cadena `post_fix_conflict_chain`.
- **Segundo y tercer puesto por WA/UA**: **52** y **48**, en el orden mostrado arriba.
- **Gap de generalizacion (`gap_ua`)**: valores bajos indican poca caida entre el mejor UA de validacion y el UA de test. El estudio **29** (media K-Fold) aparece con gap negativo en esta agregacion; conviene interpretarlo con cuidado porque el protocolo K-Fold no es identico al split unico y la media puede no ser directamente comparable con runs `unico`.
- **Equilibrio fuerte**: el estudio **45** combina UA de test **0.7923** con **gap_ua 0.0291**, es decir alta accuracy y un gap pequeno en el mismo criterio que el resumen historico del 34.
- **Sobre el estudio 34**: en esta tabla el 34 se agregara con el run de **mayor WA** entre los disponibles. Eso puede no coincidir con el run historico del resumen de ablaciones si ese checkpoint no esta en el disco o si otro run supera en WA.

Al generar este informe pueden faltar estudios sin `final_results.json` bajo `checkpoints/`; en el snapshot actual no aparecian resultados agregados para **38, 46, 50, 51** ni **53** (revisar si aun no se evaluo en test o la carpeta no contiene el JSON).

## Tabla completa (todos los estudios agregados)

| Estudio | WA | UA | F1 macro | gap_ua | n | Nota |
|---------|-----|-----|----------|--------|---|------|
| 1 | 0.6404 | 0.5531 | 0.6340 | -0.0034 | 1 | unico |
| 2 | 0.6287 | 0.5266 | 0.6247 | -0.0136 | 1 | unico |
| 3 | 0.6038 | 0.5036 | 0.6025 | -0.0145 | 1 | unico |
| 4 | 0.2544 | 0.1429 | 0.0676 | 0.0160 | 1 | unico |
| 5 | 0.6404 | 0.5531 | 0.5435 | 0.0092 | 1 | unico |
| 6 | 0.6345 | 0.5431 | 0.6275 | 0.0136 | 1 | unico |
| 7 | 0.7353 | 0.7272 | 0.7276 | 0.0923 | 2 | mejor_WA |
| 8 | 0.6389 | 0.5392 | 0.6266 | 0.0207 | 1 | unico |
| 9 | 0.6433 | 0.5467 | 0.5468 | 0.0044 | 2 | mejor_WA |
| 10 | 0.7451 | 0.7359 | 0.7347 | 0.0747 | 4 | mejor_WA |
| 11 | 0.7471 | 0.7369 | 0.7370 | 0.0694 | 2 | mejor_WA |
| 12 | 0.7017 | 0.7085 | 0.6949 | 0.0230 | 3 | media_kfold |
| 13 | 0.7490 | 0.7446 | 0.7418 | 0.0759 | 1 | unico |
| 14 | 0.7647 | 0.7669 | 0.7608 | 0.0442 | 1 | unico |
| 15 | 0.7451 | 0.7410 | 0.7394 | 0.0763 | 1 | unico |
| 16 | 0.7608 | 0.7606 | 0.7544 | 0.0602 | 1 | unico |
| 17 | 0.7608 | 0.7508 | 0.7487 | 0.0750 | 1 | unico |
| 18 | 0.7431 | 0.7381 | 0.7362 | 0.0590 | 1 | unico |
| 19 | 0.5176 | 0.5438 | 0.5066 | -0.0145 | 1 | unico |
| 20 | 0.7026 | 0.6990 | 0.6874 | 0.0053 | 5 | media_kfold |
| 21 | 0.7510 | 0.7464 | 0.7437 | 0.0694 | 1 | unico |
| 22 | 0.6784 | 0.6626 | 0.6595 | 0.0236 | 1 | unico |
| 23 | 0.7412 | 0.7354 | 0.7315 | 0.0616 | 1 | unico |
| 24 | 0.7471 | 0.7377 | 0.7378 | 0.0496 | 1 | unico |
| 25 | 0.7392 | 0.7272 | 0.7240 | 0.0859 | 1 | unico |
| 26 | 0.7686 | 0.7642 | 0.7609 | 0.0606 | 1 | unico |
| 27 | 0.7667 | 0.7617 | 0.7581 | 0.0715 | 1 | unico |
| 28 | 0.7353 | 0.7215 | 0.7172 | 0.0937 | 1 | unico |
| 29 | 0.7546 | 0.7571 | 0.7487 | -0.0024 | 5 | media_kfold |
| 30 | 0.2725 | 0.2500 | 0.1071 | 0.0000 | 1 | unico |
| 31 | 0.7392 | 0.7296 | 0.7293 | 0.0878 | 1 | unico |
| 32 | 0.7451 | 0.7380 | 0.7376 | 0.0891 | 1 | unico |
| 33 | 0.7431 | 0.7336 | 0.7328 | 0.0778 | 1 | unico |
| 34 | 0.7608 | 0.7545 | 0.7525 | 0.0684 | 3 | mejor_WA |
| 35 | 0.7471 | 0.7405 | 0.7374 | 0.0632 | 1 | unico |
| 36 | 0.2196 | 0.2500 | 0.0900 | -0.0000 | 1 | unico |
| 37 | 0.2196 | 0.2500 | 0.0900 | -0.0000 | 1 | unico |
| 39 | 0.3863 | 0.4252 | 0.2646 | -0.0147 | 2 | mejor_WA |
| 40 | 0.7157 | 0.7032 | 0.7012 | 0.0018 | 1 | unico |
| 41 | 0.6109 | 0.6016 | 0.5895 | 0.0170 | 5 | media_kfold |
| 42 | 0.6941 | 0.6833 | 0.6788 | -0.0191 | 1 | unico |
| 43 | 0.5549 | 0.5683 | 0.5175 | -0.0127 | 1 | unico |
| 44 | 0.4529 | 0.4773 | 0.4124 | 0.0059 | 1 | unico |
| 45 | 0.7961 | 0.7923 | 0.7868 | 0.0291 | 1 | unico |
| 47 | 0.6216 | 0.6300 | 0.6029 | 0.0440 | 1 | unico |
| 48 | 0.7804 | 0.7728 | 0.7696 | 0.0497 | 1 | unico |
| 49 | 0.7549 | 0.7478 | 0.7475 | 0.0667 | 1 | unico |
| 52 | 0.7824 | 0.7682 | 0.7653 | 0.0302 | 1 | unico |
| 54 | 0.7647 | 0.7574 | 0.7562 | 0.0864 | 1 | unico |

## Como regenerar

```powershell
cd d:\DOCTORADO\PROYECTOS\CONFORMER-SER
python scripts\aggregate_ablation_results.py --write-docs
```

