<#
correr_lab.ps1 -- estudios 123-126 (revision de noviembre) en la PC del laboratorio.

Comprueba el entorno, pasa la comprobacion previa de los cuatro estudios en el fold 1 y
lanza el bloque H de la cola (4 estudios x 3 semillas). Es reanudable: si se corta, se
vuelve a lanzar igual y se salta lo que ya tiene resumen completo.

Uso, desde la raiz del repo con el venv activado:
    .\scripts\correr_lab.ps1 -IemocapPath "E:\DATASETS\IEMOCAP\IEMOCAP_full_release" -SoloListar
    .\scripts\correr_lab.ps1 -IemocapPath "E:\DATASETS\IEMOCAP\IEMOCAP_full_release"

Dos corridas a la vez en una GPU de 24 GB (RTX 3090 Ti), en dos ventanas:
    .\scripts\correr_lab.ps1 -IemocapPath "..." -Filtro "s42|s123"
    .\scripts\correr_lab.ps1 -IemocapPath "..." -Filtro "s456" -SinComprobar

Detalle completo: docs\LAB.md
#>
param(
    [Parameter(Mandatory = $true)][string]$IemocapPath,
    [string]$Device = "cuda",
    [string]$Filtro = "",
    [switch]$SoloListar,
    [switch]$SinComprobar
)

$ErrorActionPreference = "Continue"
function Paro($msg) { Write-Host "PARO: $msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- entorno
if (-not (Test-Path "scripts\run_ablation.py")) { Paro "lanza este script desde la raiz del repo" }
if (-not (Test-Path $IemocapPath)) { Paro "no existe la carpeta de IEMOCAP: $IemocapPath" }
if (-not (Test-Path (Join-Path $IemocapPath "Session1"))) { Paro "$IemocapPath no parece IEMOCAP_full_release (falta Session1)" }
$env:CAMF_IEMOCAP_PATH = $IemocapPath

python -c "import sys, torch; print('Python', sys.version.split()[0], '| torch', torch.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NINGUNA')"
if ($LASTEXITCODE -ne 0) { Paro "python o torch no responden: activa el venv" }
if ($Device -eq "cuda") {
    python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) { Paro "torch no ve la GPU (instala la version CUDA de torch)" }
}

# Rasgos de los expertos para el detector con cross-fitting (125 y 126). No estan en git:
# se copian de la PC de Jorge (cache\experts\fold_S1..5.json).
foreach ($k in 1..5) {
    if (-not (Test-Path "cache\experts\fold_S$k.json")) {
        Paro "falta cache\experts\fold_S$k.json: copialo de la PC principal (docs\LAB.md, paso 3)"
    }
}
Write-Host "Entorno OK. IEMOCAP: $IemocapPath" -ForegroundColor Green

# ---------------------------------------------------------- comprobacion previa
if (-not $SoloListar -and -not $SinComprobar) {
    foreach ($n in 123, 124, 125, 126) {
        python scripts\check_sota_studies.py --study $n --fold 1 --device $Device
        if ($LASTEXITCODE -ne 0) { Paro "la comprobacion previa del estudio $n fallo" }
    }
}

# ---------------------------------------------------------------- corridas
$argumentos = @{ Bloques = @("H"); Device = $Device }
if ($Filtro) { $argumentos.Filtro = $Filtro }
if ($SoloListar) { $argumentos.SoloListar = $true }
& .\scripts\cola_articulo.ps1 @argumentos
