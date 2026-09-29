<#
.SYNOPSIS
Exporta el poster HTML Conflict-Aware SER a PDF o PNG de alta resolucion.

.DESCRIPTION
Script PowerShell compatible con Windows. Usa Chrome o Microsoft Edge.
La ruta mas confiable para impresion desde este HTML es PNG de alta resolucion:
- poster_A0_300dpi.png para imagen raster A0 landscape a DPI configurable.

PDF se conserva como opcion auxiliar, pero para imprenta se recomienda exportar PNG
o abrir el HTML en el navegador y usar imprimir manualmente con A0 horizontal.

.PARAMETER Format
Formato de salida: pdf o png.

.PARAMETER Dpi
DPI objetivo para PNG. A0 horizontal a 300 DPI equivale a 14043 x 9933 px.

.EXAMPLE
.\scripts\export_poster_a0.ps1 -Format pdf

.EXAMPLE
.\scripts\export_poster_a0.ps1 -Format png -Dpi 300
#>

param(
    [ValidateSet("pdf", "png")]
    [string]$Format = "pdf",

    [ValidateRange(72, 600)]
    [int]$Dpi = 300
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$HtmlPath = Join-Path $RepoRoot "paper\posters\poster_conflict_aware_print.html"

if (-not (Test-Path $HtmlPath)) {
    throw "No se encontro el poster HTML: $HtmlPath"
}

$BrowserCandidates = @(
    "C:\Program Files\Google\Chrome\Application\chrome.exe",
    "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
)

$Browser = $BrowserCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $Browser) {
    throw "No se encontro Chrome ni Edge. Instala uno de ellos o ajusta la ruta en este script."
}

$HtmlUri = (New-Object System.Uri($HtmlPath)).AbsoluteUri

if ($Format -eq "pdf") {
    $OutputPath = Join-Path $RepoRoot "paper\posters\poster_A0.pdf"
    & $Browser `
        --headless `
        --disable-gpu `
        --allow-file-access-from-files `
        --run-all-compositor-stages-before-draw `
        --no-pdf-header-footer `
        --print-to-pdf="$OutputPath" `
        "$HtmlUri"

    Write-Host "PDF generado: $OutputPath"
    Write-Host "Nota: si el PDF aparece cortado, exporta PNG: .\scripts\export_poster_a0.ps1 -Format png -Dpi 300"
    exit 0
}

# A0 landscape: 1189 mm x 841 mm. px = mm / 25.4 * dpi.
$WidthPx = [math]::Round(1189 / 25.4 * $Dpi)
$HeightPx = [math]::Round(841 / 25.4 * $Dpi)
$OutputPath = Join-Path $RepoRoot ("paper\posters\poster_A0_{0}dpi.png" -f $Dpi)

# El HTML esta maquetado en un viewport base de 1600 x 1132 px.
# Para alta resolucion se mantiene el layout y se aumenta deviceScaleFactor.
$BaseWidth = 1600
$BaseHeight = [math]::Round($BaseWidth * 841 / 1189)
$ScaleFactor = [math]::Round($WidthPx / $BaseWidth, 4)

& $Browser `
    --headless `
    --disable-gpu `
    --allow-file-access-from-files `
    --hide-scrollbars `
    --run-all-compositor-stages-before-draw `
    "--force-device-scale-factor=$ScaleFactor" `
    "--window-size=$BaseWidth,$BaseHeight" `
    "--screenshot=$OutputPath" `
    "$HtmlUri"

Start-Sleep -Seconds 1

if (-not (Test-Path $OutputPath)) {
    throw "Chrome no escribio el PNG. Prueba con un DPI menor, por ejemplo: .\scripts\export_poster_a0.ps1 -Format png -Dpi 120"
}

Write-Host "PNG generado: $OutputPath"
Write-Host "Resolucion objetivo: $WidthPx x $HeightPx px"
Write-Host "Viewport base: $BaseWidth x $BaseHeight px; escala: $ScaleFactor"
