<#
.SYNOPSIS
    Elimina artefactos de checkpoint de ablaciones afectadas por correcciones previas
    (scheduler, gradient accumulation, fusion conflict-aware, GLU).

.DESCRIPTION
    Borra recursivamente las carpetas ./checkpoints/ablation_XX definidas en POST_FIX_RETRAIN.md
    (Fase A: conflict-aware y cadena; Fase B: GLU 28 y baselines 25, 30). Esto incluye subcarpetas
    de K-Fold/LOSO bajo el mismo raíz (p. ej. ablation_41/fold_*). No toca logs fuera de
    checkpoint_dir salvo que compartan ruta.

.PARAMETER RepoRoot
    Raíz del repositorio CONFORMER-SER (por defecto: padre de scripts/).

.PARAMETER WhatIf
    Solo muestra qué rutas se eliminarían (no borra).

.EXAMPLE
    .\scripts\cleanup_pre_fix_checkpoints.ps1
.EXAMPLE
    .\scripts\cleanup_pre_fix_checkpoints.ps1 -WhatIf
#>
[CmdletBinding()]
param(
    [string] $RepoRoot = "",
    [switch] $WhatIf
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
    $RepoRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
}

# Estudios documentados en docs/POST_FIX_RETRAIN.md (Fase A + Fase B)
$studyNumbers = @(
    25, 28, 30,
    34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54
) | Sort-Object -Unique

$checkpointRoot = Join-Path $RepoRoot "checkpoints"
if (-not (Test-Path -LiteralPath $checkpointRoot)) {
    Write-Host "No existe $checkpointRoot ; nada que borrar."
    exit 0
}

foreach ($n in $studyNumbers) {
    $dirName = "ablation_{0:D2}" -f $n
    $target = Join-Path $checkpointRoot $dirName
    if (Test-Path -LiteralPath $target) {
        if ($WhatIf) {
            Write-Host "[WhatIf] Se eliminaría: $target"
        }
        else {
            Remove-Item -LiteralPath $target -Recurse -Force
            Write-Host "Eliminado: $target"
        }
    }
    else {
        Write-Host "Omitido (no existe): $target"
    }
}

Write-Host "Listo."
