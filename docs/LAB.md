# Correr los estudios 123-126 en la PC del laboratorio

Rama: `lab/revision-noviembre`. Contiene todo lo de `feature/sota-studies` más
`scripts\correr_lab.ps1`, la opción `--iemocap-path` y la comprobación previa de 123-126.

| Estudio | Qué es | Se compara con |
|---|---|---|
| 123 | audio + texto, sin rama mel (fusión de dos tokens, pesos fijos) | 98 |
| 124 | 123 + atención transmodal de MulT | 123 |
| 125 | 123 + detector de conflicto con cross-fitting | 123 |
| 126 | MulT + detector con cross-fitting | 124 y 123 |

Cada uno con 3 semillas (42, 123, 456) y LOSO de 5 folds: 12 corridas.

## 1. Traer el código

**Con GitHub** (desde la PC principal, una vez):

```powershell
git fetch .\lab-revision-noviembre.bundle lab/revision-noviembre:lab/revision-noviembre
git push -u origin lab/revision-noviembre
```

y en la PC del laboratorio:

```powershell
git clone https://github.com/jorrin2018/CONFORMER-SER.git
cd CONFORMER-SER
git checkout lab/revision-noviembre
```

**Sin GitHub** (con una memoria USB): copia `lab-revision-noviembre-completo.bundle`, que
trae toda la historia, y en la PC del laboratorio:

```powershell
git clone -b lab/revision-noviembre .\lab-revision-noviembre-completo.bundle CONFORMER-SER
```

## 2. Entorno

```powershell
cd CONFORMER-SER
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

La primera corrida descarga wav2vec2-base y distilbert-base-uncased de Hugging Face (el
ViT ya no se usa en estos estudios).

## 3. Copiar lo que no está en git

- **IEMOCAP** (`IEMOCAP_full_release`) en cualquier carpeta; su ruta se pasa con
  `-IemocapPath`.
- **Rasgos de los expertos** para 125 y 126: copia de la PC principal los cinco archivos
  `cache\experts\fold_S1.json` … `fold_S5.json` (unos 17 MB) a la misma ruta en el repo del
  laboratorio. Sin ellos el script se detiene antes de empezar.

## 4. Correr

```powershell
.\scripts\correr_lab.ps1 -IemocapPath "E:\DATASETS\IEMOCAP\IEMOCAP_full_release" -SoloListar
.\scripts\correr_lab.ps1 -IemocapPath "E:\DATASETS\IEMOCAP\IEMOCAP_full_release"
```

Antes de entrenar, el script comprueba Python, torch, la GPU, IEMOCAP y los rasgos de los
expertos, y pasa la comprobación previa de los cuatro estudios en el fold 1 (incluye el
pico de memoria). Es reanudable: si se corta, se lanza igual y continúa.

**RTX 3090 Ti (24 GB): dos corridas a la vez.** Una corrida usa bastante menos de 12 GB.
En dos ventanas de PowerShell:

```powershell
.\scripts\correr_lab.ps1 -IemocapPath "..." -Filtro "s42|s123"
.\scripts\correr_lab.ps1 -IemocapPath "..." -Filtro "s456" -SinComprobar
```

Si alguna se cae por memoria, cierra una y deja la otra: la cola retoma sola.

**No cambiar `batch_size` ni `gradient_accumulation_steps`** aunque la GPU tenga más
memoria: el 98 se entrenó con batch 4 × 8 de acumulación, y cambiarlo sería otro factor.

## 5. Traer los resultados

Copia a la PC principal, a la misma ruta, las carpetas `checkpoints\ablation_123` …
`ablation_126` (basta con los `kfold_summary_*.json` y los `final_results.json`) y compara:

```powershell
python scripts\comparar_estudios.py 124 123 --semillas 42 123 456
python scripts\comparar_estudios.py 125 123 --semillas 42 123 456
python scripts\comparar_estudios.py 126 124 --semillas 42 123 456
python scripts\comparar_estudios.py 123 98  --semillas 42 123 456
```

**Ojo con 123 − 98.** El 98 se corrió en la RTX 3070 y el 123 en la 3090 Ti. Con la misma
semilla, dos GPUs distintas no dan exactamente lo mismo, así que esa diferencia mezcla el
cambio de modelo con el de máquina. Las comparaciones limpias son entre estudios corridos
en el laboratorio (124, 125 y 126 contra 123).
