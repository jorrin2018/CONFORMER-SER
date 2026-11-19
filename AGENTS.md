# AGENTS.md

Operating manual for AI coding assistants working in this repository. Human-facing docs:
[README.md](README.md) · [quickstart.md](quickstart.md) · [docs/README.md](docs/README.md).

---

## 1. Project at a glance

| | |
|---|---|
| **Domain** | Multimodal Speech Emotion Recognition (SER) on IEMOCAP |
| **Classes** | 4 — `angry`, `happy`, `sad`, `neutral` |
| **Stack** | PyTorch + HuggingFace Transformers |
| **Primary environment** | Windows / PowerShell (paths in docs use `\`) |
| **Python** | 3.8+ ; CUDA GPU strongly recommended for training |
| **Split** | Speaker-independent: sessions 1–3 train, 4 val, 5 test |

The research contribution is **CAMF** — a Conflict-Aware Multimodal Fusion framework combining
audio, text and vision streams. Everything else in the repo exists to train, ablate and evaluate it.

---

## 2. Repository layout

```text
CONFORMER-SER/
├── config/
│   ├── config.yaml                 # baseline config (set dataset.data_path here)
│   ├── config_audio_only.yaml
│   ├── experiment_lightweight.yaml
│   └── ablation/                   # 68 ablation YAMLs + ABLATION_SUMMARY.md (canonical catalog)
├── src/
│   ├── train.py                    # baseline training entry point
│   ├── evaluate.py                 # baseline evaluation ONLY
│   ├── evaluate_ensemble.py        # ablation + ensemble evaluation
│   ├── inference.py                # single-file inference (baseline model)
│   ├── data/                       # dataset, preprocessing, augmentation, prosody cues
│   ├── models/                     # baseline model, fusion mechanisms, encoders
│   ├── ablation_studies/           # per-study model classes + classification heads
│   ├── experiments/                # lightweight-encoder experiment track
│   └── utils/                      # losses, metrics, logging
├── scripts/
│   ├── run_ablation.py             # ablation training entry point (source of truth for study→model)
│   ├── run_test_only.py            # test-set evaluation of an existing run
│   ├── aggregate_ablation_results.py
│   ├── export_poster_*.ps1
│   └── windows/                    # .bat / .ps1 convenience wrappers (see §7)
├── docs/                           # versioned documentation (index: docs/README.md)
│   ├── experimentos/               # per-series experiment logs
│   └── diagrams/                   # Draw.io / PlantUML / Mermaid sources
├── paper/                          # publication artifacts (see §8)
├── tests/
├── notebooks/
├── setup_check.py                  # environment + dataset verification
└── requirements.txt
```

**Generated, gitignored:** `checkpoints/`, `logs/`, `evaluation_results/`, `data/`,
`node_modules/`, `.cache/`, `venv/`, `.venv/`, `openspec/`.

---

## 3. The #1 mistake: baseline vs ablation are separate paths

There are two independent training/evaluation pipelines. Using the wrong script for a checkpoint
is the most common failure in this repo.

| Goal | Command | Model source |
|------|---------|--------------|
| Train baseline | `python src\train.py --config config\config.yaml --experiment_name <name> --device cuda` | `ConformerMultimodalModel` / `AudioOnlyModel` |
| Train ablation | `python scripts\run_ablation.py --study N --device cuda` | `create_ablation_model()` in `scripts/run_ablation.py` |
| Evaluate **baseline** checkpoint | `python src\evaluate.py --checkpoint <best_model.pt> --config config\config.yaml --split test --device cuda` | Rebuilds baseline model |
| Evaluate **ablation** checkpoint | `python src\evaluate_ensemble.py --checkpoints <ckpt> --configs <yaml> --study_numbers N --weights 1.0 --split test --device cuda` | Loads the ablation model class |
| Test-only on a saved run | `python scripts\run_test_only.py --checkpoint_dir <run_dir> --study N --device cuda` | Reads `config.json` from the run dir |

> **Critical:** `src/evaluate.py` does **not** know about the classes in `src/ablation_studies/`.
> Never point it at an ablation checkpoint — use `evaluate_ensemble.py` or `run_test_only.py`.

---

## 4. Ablation studies

- **Valid study numbers: `1–50` and `52–65`.** There is **no study 51**.
  The range is enforced by `create_ablation_model()` in `scripts/run_ablation.py`.
- Many studies **reuse the same model class** with a different YAML. `create_ablation_model()` is
  the single source of truth for the study number → model class mapping — read it before assuming
  a study has its own class.
- Configs live in `config/ablation/ablation_*.yaml`. Human-readable catalog:
  [`config/ablation/ABLATION_SUMMARY.md`](config/ablation/ABLATION_SUMMARY.md).
- **Orphan configs exist.** `ablation_65_unfrozen_vit.yaml` and
  `ablation_phase2_49_wavlm_multiscale_head_FUTURE.yaml` have **no code path** in
  `run_ablation.py`. A YAML file existing does *not* mean the study is runnable.
- Studies **20, 29, 41** run K-Fold / LOSO cross-validation instead of the standard split.

### Recent series (58–63)

| Study | Focus | Model class |
|-------|-------|-------------|
| 58 | MultiScaleTemporal head + SupCon (3 views) + Wav2Vec2 | `Ablation48Model` |
| 59 | MultiScaleTemporal + WavLM + SpecAugment on ViT (no waveform aug) | `Ablation34Model` |
| 60 | SOTA combo — WavLM + MultiScaleTemporal + SupCon + SpecAugment | `Ablation48Model` |
| 61 | SOTA combo + CompositeLoss with neutral-class focus | `Ablation48Model` |
| 62 | SOTA combo + attention-based audio pooling | `Ablation34Model` |
| 63 | Fine-tuned study 45 (lower dropout, lower label smoothing) | `Ablation34Model` |

---

## 5. Architecture notes

- **Audio encoders:** Wav2Vec2, WavLM (base / base-plus, optional weighted hidden states),
  HuBERT, Emotion2Vec.
- **Text encoders:** BERT / RoBERTa / DistilBERT, plus **SpeechCue** prosodic prompt injection
  in specific studies (`src/data/prosody_cues.py`).
- **Vision branch:** ViT over a **log-mel spectrogram computed inside the forward pass** from the
  *same* batch waveform (torchaudio `MelSpectrogram` → image). **IEMOCAP in this pipeline ships raw
  audio only — there is no precomputed mel feature stored on disk.** This trips people up
  repeatedly; do not add a "load mel features" path.
- **Fusion mechanisms** (`src/models/fusion.py`): Enhanced Cross-Attention, Conflict-Aware
  Cross-Attention, Gated Interactive Attention, Tensor Fusion, Bilinear, GLU, Late, Hierarchical,
  Adaptive Weighted.
- **Classification heads** (`src/ablation_studies/classification_heads.py`): MLP, LSTM,
  Transformer, CNN1D, MultiScaleTemporal.
- **Losses** (`src/utils/losses.py`): weighted Cross Entropy, Focal, Dirichlet Evidence Learning,
  SupCon, and composite combinations. `log_loss_contract_traceability()` prints the resolved loss
  contract at startup — note that `gamma` does **not** apply to plain `cross_entropy`.
- **Optimizer groups** (`src/ablation_studies/utils.py`): separate learning rates for encoders,
  fusion and classifier.

---

## 6. Run directory layout

```text
checkpoints/ablation_XX/ablation_XX_YYYYMMDD_HHMMSS/
├── best_model.pt              # gitignored
├── config.json                # merged config actually used — read this, not the YAML
├── best_metrics.json
├── final_results.json         # written by run_test_only.py
├── confusion_matrix_test.png
├── training_curves.png
└── test_eval.log
```

When reproducing or diagnosing a run, trust `config.json` inside the run directory over the YAML
in `config/ablation/` — the YAML may have changed since the run.

Aggregate results across runs:

```powershell
python scripts\aggregate_ablation_results.py --write-docs
```

This regenerates [`docs/RANKING_Y_RESULTADOS.md`](docs/RANKING_Y_RESULTADOS.md) from all
`final_results.json` files found under `checkpoints/`.

---

## 7. Windows wrappers

All `.bat` / `.ps1` convenience scripts live in `scripts/windows/`. Each `.bat` starts with
`pushd "%~dp0..\.."`, so it works both from the repo root and by double-click.

```powershell
.\scripts\windows\setup.bat              # pip install + setup_check.py
.\scripts\windows\train.bat              # baseline training (forwards all args)
.\scripts\windows\start_tensorboard.bat  # TensorBoard on logs/
.\scripts\windows\EVALUAR_RAPIDO.bat     # interactive evaluation menu
```

Their hardcoded checkpoint paths point at historical runs and are frequently stale — treat them as
templates, not as working commands.

---

## 8. `paper/` — publication artifacts

```text
paper/
├── manuscripts/    # CIARP 2025/2026 .docx drafts (v9, v10, v11)
├── figures/        # fig1–fig6 SVG + graphical_abstract.pdf
├── posters/        # poster PDFs/PNGs + poster_conflict_aware_print.html + assets/
└── presentations/  # .pptx deep-dive deck
```

Posters are **generated**: edit `paper/posters/poster_conflict_aware_print.html`, then run
`.\scripts\export_poster_18x24.ps1 -Format pdf` (or `-Format png -Dpi 300`). The scripts write
back into `paper/posters/`. Do not hand-edit the generated PDFs/PNGs.

---

## 9. Conventions for agents

- **Language:** code, identifiers and commit messages in English; user-facing docs (`README.md`,
  `quickstart.md`, `docs/`) are in **Spanish** — keep them that way. `AGENTS.md` and `CLAUDE.md`
  are in English.
- **Paths in docs:** Windows style (`config\config.yaml`), PowerShell line continuation is a
  backtick `` ` ``.
- **Before claiming a study range, model mapping or CLI flag**, verify it against
  `scripts/run_ablation.py` — the docs have drifted from the code before.
- **Never commit** checkpoints, datasets, HuggingFace caches or `node_modules/`. They are
  gitignored; if `git status` shows them, something is wrong with `.gitignore`.
- **Do not launch training runs** unless explicitly asked — they take hours on a single GPU.
- **`openspec/` is gitignored**: change proposals stay local.

---

## 10. Key files to read first

| File | Why |
|------|-----|
| `scripts/run_ablation.py` | Source of truth for study → model → config mapping |
| `config/ablation/ABLATION_SUMMARY.md` | Human-readable catalog of every study |
| `docs/ARCHITECTURE_AND_WORKFLOWS.md` | Detailed code map and data flow |
| `src/models/fusion.py` | All fusion mechanisms |
| `src/ablation_studies/classification_heads.py` | All classification heads |
| `src/utils/losses.py` | Loss configuration contract |
| `src/models/conformer.py` | Baseline model |

---

## Post-CIARP-2026 audit — read before touching results or the paper

The CIARP 2026 submission was rejected. A full audit (Sept 2026) found ten issues; the
data-loading bug behind most of them is fixed, but **every number in the previous
manuscript and in `docs/ANALISIS_RESULTADOS_EXPERIMENTOS.md` predates the fix and must
not be reused.**

| Read this | For |
|---|---|
| `docs/REWRITE_NOTES.md` | Everything the paper rewrite has to change, section by section, with verified numbers and their provenance |
| `docs/dataset_integrity_patch.md` | The dataset contract and why it exists |
| `scripts/verify_iemocap.py` | Run before any training run. 5 531 four-class utterances, `impro > 0` in all five sessions |
| `scripts/report_finetuning.py` | Which blocks actually train, per study |

Non-negotiables for anything that goes in a paper:

- **LOSO 5-fold** (`--loso --protocol standard`), report mean ± sigma. `standard` trains
  on four sessions and validates on a class-stratified 12 % of them; `speaker` validates
  on one held-out speaker instead (measured: val UA sigma 8.00 pp, too noisy to select a
  checkpoint); `legacy` (3+1+1) is kept only to reproduce runs from before 12/09/2026 and
  is not comparable with published numbers. The protocol is recorded in every
  `kfold_summary_*.json` and in every fold's `config.json`.
- **The protocol is not a knob for the absolute number.** Measured on study 66 (audio
  only, same seed): legacy 64.02 +- 4.89, speaker 63.63 +- 5.02, standard 63.93 +- 2.68.
  What `standard` buys is honest variance (test WA sigma 3.74 -> 0.90), not accuracy.
  Do not re-litigate this.
- Pair ablations **by fold**: the paired sigma is about half the absolute between-fold
  sigma (1.7 pp vs 3.4 pp measured), which is what makes a 4-7 pp effect detectable at
  all. With five folds the exact sign test bottoms out at p = 0.0625, so report the
  paired t-test alongside it and say so.
- **One factor per ablation row.** Never compare runs whose `config.json` differs in
  encoder, branches or fusion type.
- Report the test N and per-class support in every results table; confusion matrices in
  absolute counts.
- The description of the training regime must match the released `config.json`.
- **Verify the optimizer before every run that matters**: `python scripts/report_optimizer.py
  --study N`. `create_ablation_optimizer()` has a single-LR branch whose comment assumes
  "encoders frozen"; studies not listed in `studies_with_separate_lr` fall into it and train
  EVERYTHING at `learning_rate_fusion`, ignoring `learning_rate_audio`/`_text`. That is how
  studies 44, 45 and 66-69 trained their unfrozen encoders at 5e-6. Applying the declared
  rates (`training.use_param_groups: true`) is worth **+5.06 +- 1.46 pp, t(4)=7.76, 5/5
  folds** (study 71 vs 66). Every number produced before 18/09/2026 by a study in that
  branch is void.
- **Verify what the paper claims against what the code does.** Three mismatches found in
  this audit: the "MultiScaleTemporal" head receives a pooled vector, not a sequence
  (`scripts/check_head_69.py`); `_apply_pitch_shift` resamples and resamples back, which is
  the identity; and the differential learning rates were never applied
  (`scripts/report_optimizer.py`).
- **Verify every literature number against the primary source before citing it.** "Pepino
  72.50" in the manuscript's SOTA table does not exist; the paper reports 67.2 +- 0.7. That
  one miscitation drove three experiments.

**When a k-fold run dies mid-way** (GPU fault, reboot, `CUDA error: unknown error`):
never restart the whole study. The folds that finished are on disk.

```powershell
# 1. reconstruir el resumen desde los folds que si terminaron
python scripts\rebuild_kfold_summary.py --study 68 --experiment std2

# 2. correr SOLO lo que falta, reanudando el fold a medias desde su checkpoint
python scripts\run_ablation.py --study 68 --loso --protocol standard --folds 4 5 --resume
```

`--resume` retoma cada fold desde `last_model.pt` (modelo, optimizador, scheduler,
mejor metrica y contador de paciencia, guardados cada epoca con escritura atomica).
Si solo hay `best_model.pt` de una corrida anterior a este cambio, reanuda desde ahi y
recuenta la paciencia leyendo `metrics.json`. Encuentra el checkpoint aunque el nombre
del experimento haya cambiado —busca en las carpetas hermanas del mismo fold— pero solo
acepta una cuyo `config.json` coincida en sesiones, modelo y semilla, y que no tenga ya
`final_results.json`. Tambien copia el mejor checkpoint previo a la carpeta de la corrida
nueva: sin eso, un fold que reanuda con la paciencia avanzada y no vuelve a mejorar
evaluaria en test con los pesos de la ultima epoca en vez de los mejores, y el numero
saldria peor sin ningun aviso.

**Un fold reanudado no es bit-identico a uno corrido de un tiron**: el orden de shuffle
del DataLoader arranca de nuevo, asi que la semilla ya no lo reproduce exactamente. Queda
marcado con `resumed: true` y `resume_points` en su `final_results.json`. Usar `--resume`
para no perder horas de GPU, y volver a correr limpio ese fold si el numero acaba siendo
decisivo para una afirmacion del articulo.

`--folds` keeps the global fold index, so the per-fold split is identical to what a full
run would have produced. Folds recovered from the previous summary whose `protocol` does
not match the one requested are discarded with a warning — mixing protocols in one mean
is how a table becomes unreproducible.

**`train_epoch` tiene DOS ramas de perdida** — precision mixta y sin ella — y los
configs de ablacion corren con `mixed_precision: false`. El 21/09/2026 el estudio 75 se
corrio entero, los cinco folds, con un termino puesto solo en la rama de precision mixta:
nunca se ejecuto, los tres clasificadores acabaron con los MISMOS pesos a seis decimales
que sin el arreglo, y el resultado no media nada. Cualquier termino nuevo de perdida va
en una funcion propia (`_aux_branch_loss`, `_conf_calibration_loss`), se llama desde las
dos ramas, y lleva una guardia que aborta con `RuntimeError` al final de la epoca si se
pidio y no se aplico.

Y el test que lo verifica tiene que ejercitar **la ruta real**: `check_aux_loss.py`
probaba el modulo de fusion en aislamiento y daba verde mientras el entrenamiento no
aplicaba nada. `check_aux_integration.py` monta el modelo real con el config real y da un
paso de optimizacion. Ojo con el criterio: para la auxiliar vale mirar si los pesos se
mueven (sin ella reciben gradiente exactamente cero), pero para la calibracion **no**
—la `confidence_net` ya se mueve por la perdida final— asi que hay que derivar SOLO ese
termino y comprobar que su gradiente sobre la `confidence_net` no es nulo.

**Los corpus se registran, no se instancian a mano** (refactor del 22/09/2026).
`src/data/base_dataset.py` tiene `BaseSERDataset` con todo lo que no depende del
corpus -- `__getitem__`, recorte de audio, prosodia, SpeechCue, pesos de clase, filtro
de hablante, subconjunto por indices -- y cada corpus implementa `_collect_records()`,
`_assert_integrity()` y, casi siempre, `speaker_of()`. `src/data/registry.py` mapea
`dataset.corpus` del YAML a la clase; ausente significa `iemocap`, que es lo que hacen
todos los configs que ya existen. Nadie llama `IEMOCAPDataset(...)` directamente: se
usa `build_dataset(corpus=corpus_from_config(config), ...)`.

Dos trampas al anadir un corpus. Una: `speaker_of()` por defecto parsea el formato de
IEMOCAP, y si no se sobrescribe el filtro de hablante agrupa mal **sin lanzar ninguna
excepcion** -- train y validacion acaban compartiendo locutor y el numero sube solo.
Dos: `_assert_integrity()` no es opcional en la practica. En IEMOCAP fue lo que
detecto que un troceado de cadena tiraba 2,943 de 5,531 utterances en silencio. Un
corpus sin contrato verificable puede mentir durante meses.

El refactor se verifica con `scripts/check_dataset_refactor.py`, que carga
`tests/fixtures/dataset_pre_refactor.py.frozen` -- la copia del archivo anterior -- por
ruta, como modulo aparte, y compara registro a registro. Esa copia no se borra: es la
unica referencia de que el cargador sigue leyendo lo mismo.

**El pipeline NO es reproducible corrida a corrida.** Medido el 23/09/2026: el fold S1
del estudio 74, misma config, misma semilla, mismo fold, dio 76.11 el 18/09 y 74.77 el
22/09, con `config.json` identico y trayectorias distintas (20 vs 29 epocas). Las dos
corridas del estudio 75 difieren hasta 1.72 pp por fold. **Cualquier diferencia pareada
por debajo de ~1.5 pp esta dentro del ruido de re-ejecucion y no se puede atribuir al
modelo.** Los resultados NULOS aguantan (el ruido ensancha el intervalo y el cero sigue
dentro); los positivos pequenos, no.

Dos agujeros concretos: `torch.use_deterministic_algorithms(True, warn_only=True)` deja
correr sin aviso de error toda operacion sin implementacion determinista, y
`CUBLAS_WORKSPACE_CONFIG` se pone con `setdefault` DENTRO de la funcion de
entrenamiento, cuando tiene que estar en el entorno antes de crear el contexto de CUDA.

Consecuencia practica: antes de afirmar que un cambio mejora algo, comprobar que el
efecto es mayor que esa banda, o correr multi-semilla. Y **commitear**: el ultimo commit
era del 10/09 mientras dos semanas de cambios vivian sin confirmar, asi que no hubo
forma de preguntarle a git si ademas del ruido habia cambiado el codigo.

**NO correr git en este repo desde el puente de Claude.** Ese shell no puede borrar
archivos en las carpetas conectadas, asi que git crea sus `.lock` con normalidad pero
falla al eliminarlos al terminar: cada comando deja un candado que bloquea al
siguiente, y el repositorio acaba inutilizable desde Windows ("cannot lock ref 'HEAD'")
hasta que alguien los borra a mano. Paso el 29/09/2026 al hacer los dos commits de la
auditoria; los commits salieron bien, pero dejaron seis `.lock` y 226 `tmp_obj_*`
sueltos que tuvo que limpiar Jorge.

Los commits se preparan aqui y se le pasan a Jorge como comando para que los corra el,
igual que los entrenamientos. Y ojo: **Jorge usa CMD, no PowerShell** — `del /f /q`,
no `Remove-Item`.

**`save_top_k` del YAML no hace nada.** Aparece en todos los configs de ablacion y NO
se lee en ningun sitio del codigo (comprobado el 01/10/2026 con un grep en src y
scripts). Solo se guardan dos checkpoints por fold: `best_model.pt` y `last_model.pt`,
los dos con el estado del optimizador dentro, **1.75 GB cada uno**. O sea 3.5 GB por
fold. No recomendar bajar `save_top_k` para ahorrar memoria: es una clave muerta.

**Los fallos de entrenamiento que REVIENTAN en esta maquina son de recursos, no de
codigo** -- pero ojo, eso vale para los que revientan. El del 20/10/2026 no reventaba:
entrenaba entero con la rama de audio en ruido (ver la trampa del encoder, mas abajo).
No usar este diagnostico para descartar un resultado raro que termino bien.

Tres errores distintos en el mismo estudio 74 con el mismo codigo que ya completo cinco
folds dos veces: `CUDA error: unknown error` (15/09), `bad allocation` -- memoria de HOST, no
CUDA -- (29/09, tres folds de cinco), y `CUDNN_STATUS_INTERNAL_ERROR` dentro del LSTM de
la cabeza (01/10). cuDNN reporta la falta de memoria como INTERNAL_ERROR, asi que los
tres apuntan al mismo sitio.

Lo que NO arregla esto: tocar la retencion de checkpoints (clave muerta) ni el tamano de
`best_model.pt`, porque los fallos ocurren en el forward y el backward, no al guardar.
Lo que si: agrandar el archivo de paginacion de Windows, cerrar lo demas, y **un proceso
por fold** con `--folds N --resume`. Reducir memoria de GPU de verdad exigiria tocar
batch size o modelo, y eso SI cambia el experimento.

**UN ENCODER QUE NO CARGA TIENE QUE PARAR LA CORRIDA, NO DEGRADARLA.** El 20/10/2026 el
estudio 85 entreno cinco folds enteros con `facebook/wav2vec2-base` inicializado AL AZAR.
`transformers` 5.8.0 con `torch` 2.5.1 se niega a leer checkpoints `.bin`
(CVE-2025-32434) y `from_pretrained()` sin argumentos elegia el `.bin` aunque el
`model.safetensors` estuviera en la cache. Hasta aqui es un problema de entorno; el bug
era nuestro: un `except Exception` en `src/models/conformer.py` sustituia el encoder por
`Wav2Vec2Model(Wav2Vec2Config())` y seguia, avisando con `print()` -- que NO llega al
fichero de log.

Lo que hace esto peligroso es que **todo lo demas sale normal**: 263,427,987 parametros
totales y 87,184,403 entrenables (las cifras correctas), el informe de congelado
correcto, el contrato de perdida correcto. Solo se detecto porque RAVDESS es el unico
corpus donde el audio es el UNICO canal informativo, asi que el modelo cayo por debajo
del azar y el detector de colapso grito. En IEMOCAP o MELD la rama de texto habria
sostenido un numero plausible y falso.

Ya esta arreglado -- `use_safetensors=True` y el fallback eliminado, asi que ahora
levanta `RuntimeError` --, pero la regla se queda:

- Al revisar un resultado raro, mirar el ARRANQUE del log, no solo las metricas. Si
  aparece `Using default ... configuration` o `Could not load ... encoder`, la corrida
  no vale. En las corridas nuevas eso ya no puede pasar sin fallar.
- `Loading weights: 100%` y un `LOAD REPORT` por encoder es la senal de que cargo de
  verdad. Tres encoders, tres informes.
- **No anadir `except` que degraden en silencio.** Un numero inventado es peor que un
  fallo, porque un fallo se ve. Es el segundo caso de esta forma en el repo: el primero
  fue `_find_audio_file`, que tiro 2,943 de 5,531 utterances durante meses.
- Detalle de los encoders en cache: `wav2vec2-base`, `wavlm-base`, `wavlm-large` y
  `hubert-base-ls960` tienen `.bin` Y safetensors; `wavlm-base-plus` SOLO `.bin`, asi
  que para ese la carga reintenta sin `use_safetensors`. Si algun dia hace falta, la
  salida limpia es bajar los safetensors, no subir torch: subir torch cambia el entorno
  y obliga a rehacer las corridas que se quieran comparar entre si.

Postmortem completo: `docs/REWRITE_NOTES.md` seccion 13.
