<#
.SYNOPSIS
    Reentrena en GPU los estudios de ablación marcados como no comparables tras las correcciones.

.DESCRIPTION
    Ejecuta scripts/run_ablation.py con --device cuda y nombres de experimento alineados con
    docs/POST_FIX_RETRAIN.md: Fase A (34,36-50,52-54) con prefijo post_fix_conflict_chain;
    Fase B: 28 post_fix_glu; 25 y 30 post_fix_baseline. Requiere Python en PATH y GPU disponible.

.PARAMETER RepoRoot
    Raíz del repositorio (por defecto: padre de scripts/).

.PARAMETER SkipPhaseA
    No ejecuta la Fase A.

.PARAMETER SkipPhaseB
    No ejecuta la Fase B.

.PARAMETER Python
    Ejecutable de Python (por defecto: python).

.EXAMPLE
    .\scripts\retrain_post_fix_gpu.ps1
.EXAMPLE
    .\scripts\retrain_post_fix_gpu.ps1 -Python "D:\venvs\ser\Scripts\python.exe"
#>
[CmdletBinding()]
param(
    [string] $RepoRoot = "",
    [switch] $SkipPhaseA,
    [switch] $SkipPhaseB,
    [string] $Python = "python"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
    $RepoRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
}

Set-Location -LiteralPath $RepoRoot

function Invoke-Study {
    param(
        [int] $Study,
        [string] $ExperimentPrefix
    )
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "Study $Study | experiment prefix: $ExperimentPrefix" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
    & $Python scripts/run_ablation.py --study $Study --device cuda --experiment_name $ExperimentPrefix
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo en estudio $Study (código $LASTEXITCODE)."
    }
}

# Fase A: conflict-aware y cadena
$phaseA = @(34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54)
$expA = "post_fix_conflict_chain"

# Fase B: GLU + baselines
$phaseB28 = @{ Study = 28; Prefix = "post_fix_glu" }
$phaseBBaseline = @(
    @{ Study = 25; Prefix = "post_fix_baseline" },
    @{ Study = 30; Prefix = "post_fix_baseline" }
)

if (-not $SkipPhaseA) {
    foreach ($s in $phaseA) {
        Invoke-Study -Study $s -ExperimentPrefix $expA
    }
}

if (-not $SkipPhaseB) {
    Invoke-Study -Study $phaseB28.Study -ExperimentPrefix $phaseB28.Prefix
    foreach ($item in $phaseBBaseline) {
        Invoke-Study -Study $item.Study -ExperimentPrefix $item.Prefix
    }
}

Write-Host ""
Write-Host "Reentrenamiento completado. Checkpoints bajo ./checkpoints/ablation_XX/<experiment_name>_study_<N>" -ForegroundColor Green
