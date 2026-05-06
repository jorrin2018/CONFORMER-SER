#requires -Version 5.1
<#
.SYNOPSIS
    Overnight screening batch for Studies 59-62.
    Runs sequentially: 59 -> 60 -> 61 -> 62 with seed 42.
    Designed to run overnight and complete all 4 screenings.
    
.EXAMPLE
    .\run_screening_overnight.ps1
#>
param(
    [string]$Device = "cuda",
    [string]$Seed = "42",
    [string]$PythonPath = ".\venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$studies = @(59, 60, 61, 62)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "OVERNIGHT SCREENING BATCH" -ForegroundColor Cyan
Write-Host "Studies: $($studies -join ', ')" -ForegroundColor Cyan
Write-Host "Seed: $Seed | Device: $Device" -ForegroundColor Cyan
Write-Host "Python: $PythonPath" -ForegroundColor Cyan
Write-Host "Start time: $(Get-Date)" -ForegroundColor Cyan
Write-Host "Estimated total: ~24-32 hours" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Verify Python exists
if (-not (Test-Path $PythonPath)) {
    Write-Host "ERROR: Python not found at $PythonPath" -ForegroundColor Red
    exit 1
}

foreach ($study in $studies) {
    $startTime = Get-Date
    Write-Host ""
    Write-Host ">>> [$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] STARTING STUDY $study" -ForegroundColor Green
    Write-Host ""
    
    try {
        & $PythonPath scripts\run_ablation.py `
            --study $study `
            --device $Device `
            --seed $Seed `
            --experiment_name "screening_seed_${Seed}"
        
        $elapsed = (Get-Date) - $startTime
        Write-Host ""
        Write-Host ">>> [$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] STUDY $study COMPLETED" -ForegroundColor Green
        Write-Host "    Duration: $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
    }
    catch {
        Write-Host ""
        Write-Host ">>> [$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] STUDY $study FAILED: $_" -ForegroundColor Red
        # Continue with next study even if one fails
    }
    
    # Small pause between runs to let GPU cool down
    Write-Host "    Waiting 30s before next study..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "OVERNIGHT BATCH COMPLETED" -ForegroundColor Cyan
Write-Host "End time: $(Get-Date)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Check results in checkpoints\ablation_59, 60, 61, 62" -ForegroundColor Yellow
Write-Host "2. Run: python scripts\aggregate_ablation_results.py" -ForegroundColor Yellow
Write-Host "3. Compare final_results.json across all studies" -ForegroundColor Yellow
Write-Host "4. Pick top 2 for multi-seed (3 seeds each)" -ForegroundColor Yellow
