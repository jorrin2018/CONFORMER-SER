#requires -Version 5.1
<#
.SYNOPSIS
    Run ablation study with multiple random seeds for reproducible evaluation.
    
.DESCRIPTION
    This script automates running the same ablation study with 3 different seeds
    (42, 123, 456) to compute mean ± std for publication-ready results.
    
.PARAMETER Study
    Ablation study number (e.g., 45, 55, 58)
    
.PARAMETER Device
    Device to use (cuda or cpu)
    
.PARAMETER Seeds
    Comma-separated list of seeds (default: 42,123,456)
    
.EXAMPLE
    .\scripts\windows\run_seeds.ps1 -Study 58
    .\scripts\windows\run_seeds.ps1 -Study 58 -Seeds "42,123,456" -Device cuda
#>
param(
    [Parameter(Mandatory=$true)]
    [int]$Study,
    
    [string]$Device = "cuda",
    
    [string]$Seeds = "42,123,456"
)

$ErrorActionPreference = "Stop"
$seedList = $Seeds -split ","

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Running Ablation Study $Study with $($seedList.Count) seeds" -ForegroundColor Cyan
Write-Host "Seeds: $Seeds" -ForegroundColor Cyan
Write-Host "Device: $Device" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

foreach ($seed in $seedList) {
    $seed = $seed.Trim()
    $expName = "study_${Study}_seed_${seed}"
    
    Write-Host ""
    Write-Host ">>> Starting run with seed $seed (experiment: $expName)" -ForegroundColor Green
    Write-Host ""
    
    try {
        python scripts\run_ablation.py `
            --study $Study `
            --device $Device `
            --experiment_name $expName `
            --seed $seed
        
        Write-Host ""
        Write-Host ">>> Run with seed $seed completed successfully" -ForegroundColor Green
    }
    catch {
        Write-Host ""
        Write-Host ">>> ERROR in run with seed $seed : $_" -ForegroundColor Red
        continue
    }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "All seeds completed for Study $Study" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Check results in checkpoints\ablation_$Study\" -ForegroundColor Yellow
Write-Host "2. Run: python scripts\aggregate_ablation_results.py" -ForegroundColor Yellow
Write-Host "3. Compare final_results.json across seeds" -ForegroundColor Yellow
