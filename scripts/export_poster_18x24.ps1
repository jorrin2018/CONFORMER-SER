<#
.SYNOPSIS
Exporta el poster vertical 18x24 pulgadas a PDF o PNG.

.DESCRIPTION
Script PowerShell compatible con Windows. Usa Chrome o Microsoft Edge en modo
headless para abrir paper/posters/poster_conflict_aware_print.html y generar archivos
listos para revisión o impresión.

Tamaño físico: 18 in x 24 in, orientación vertical.
Resoluciones comunes:
- 150 DPI: 2700 x 3600 px
- 200 DPI: 3600 x 4800 px
- 300 DPI: 5400 x 7200 px

.PARAMETER Format
Formato de salida: pdf o png.

.PARAMETER Dpi
DPI objetivo para PNG. Si -Format es png y se usa -AutoHr, se ignora (se prueba 300, 200, 150).

.PARAMETER AutoHr
Solo con -Format png: genera <repo>\paper\posters\poster_18x24_alta_resolucion.png probando DPI hasta que Chrome escriba el archivo.

.EXAMPLE
.\scripts\export_poster_18x24.ps1 -Format png -Dpi 300

.EXAMPLE
.\scripts\export_poster_18x24.ps1 -Format png -AutoHr

.EXAMPLE
.\scripts\export_poster_18x24.ps1 -Format pdf
#>

param(
    [ValidateSet("pdf", "png")]
    [string]$Format = "png",

    [ValidateRange(72, 600)]
    [int]$Dpi = 300,

    [switch]$AutoHr
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
    $OutputPath = Join-Path $RepoRoot "paper\posters\poster_18x24.pdf"
    & $Browser `
        --headless `
        --disable-gpu `
        --allow-file-access-from-files `
        --run-all-compositor-stages-before-draw `
        --no-pdf-header-footer `
        --print-to-pdf="$OutputPath" `
        "$HtmlUri"

    Start-Sleep -Seconds 1
    if (-not (Test-Path $OutputPath)) {
        throw "Chrome no escribio el PDF."
    }

    Write-Host "PDF generado: $OutputPath"
    exit 0
}

# El HTML esta maquetado en un viewport base de 900 x 1200 px.
# Para PNG de alta resolucion se mantiene el layout y se aumenta deviceScaleFactor.
$BaseWidth = 900
$BaseHeight = 1200

function Export-PosterPng {
    param(
        [int]$DpiIn,
        [string]$OutPath
    )
    $WidthPx = [math]::Round(18 * $DpiIn)
    $HeightPx = [math]::Round(24 * $DpiIn)
    $ScaleFactor = [math]::Round($WidthPx / $BaseWidth, 4)
    if (Test-Path $OutPath) { Remove-Item -Force $OutPath }
    & $Browser `
        --headless `
        --disable-gpu `
        --allow-file-access-from-files `
        --hide-scrollbars `
        --run-all-compositor-stages-before-draw `
        "--force-device-scale-factor=$ScaleFactor" `
        "--window-size=$BaseWidth,$BaseHeight" `
        "--screenshot=$OutPath" `
        "$HtmlUri"
    $FileReady = $false
    $maxWait = if ($ScaleFactor -ge 5) { 60 } else { 25 }
    for ($i = 0; $i -lt $maxWait; $i++) {
        Start-Sleep -Seconds 1
        if (Test-Path $OutPath) {
            $p = Get-Item $OutPath
            if ($p.Length -gt 5000) { $FileReady = $true; break }
        }
    }
    return [PSCustomObject]@{
        Ok          = $FileReady
        Path        = $OutPath
        Dpi         = $DpiIn
        WidthPx     = $WidthPx
        HeightPx    = $HeightPx
        ScaleFactor = $ScaleFactor
    }
}

if ($AutoHr) {
    $DpiList = @(300, 200, 150)
    $Last = $null
    foreach ($d in $DpiList) {
        $Out = Join-Path $RepoRoot ("paper\posters\poster_18x24_{0}dpi.png" -f $d)
        Write-Host "Intentando export PNG a $d DPI -> $Out"
        $Last = Export-PosterPng -DpiIn $d -OutPath $Out
        if ($Last.Ok) {
            $Hr = Join-Path $RepoRoot "paper\posters\poster_18x24_alta_resolucion.png"
            Copy-Item -Path $Out -Destination $Hr -Force
            Write-Host "Listo. Archivo principal (copia): $Hr"
            Write-Host "Tambien: $Out  ($($Last.WidthPx) x $($Last.HeightPx) px, escala $($Last.ScaleFactor))"
            exit 0
        }
        Write-Host "Fallo a $d DPI, reintentando con menor resolucion..."
    }
    throw "No se pudo generar PNG ni a 300, 200 u 150 DPI. Cierra otras instancias de Chrome o baja el DPI manualmente: -Dpi 96"
}

$WidthPx = [math]::Round(18 * $Dpi)
$HeightPx = [math]::Round(24 * $Dpi)
$OutputPath = Join-Path $RepoRoot ("paper\posters\poster_18x24_{0}dpi.png" -f $Dpi)

$Last = Export-PosterPng -DpiIn $Dpi -OutPath $OutputPath
if (-not $Last.Ok) {
    throw "Chrome no escribio el PNG. Prueba: .\scripts\export_poster_18x24.ps1 -Format png -AutoHr"
}
$Hr = Join-Path $RepoRoot "paper\posters\poster_18x24_alta_resolucion.png"
Copy-Item -Path $OutputPath -Destination $Hr -Force
Write-Host "PNG generado: $OutputPath"
Write-Host "Copia entrega HR: $Hr"
Write-Host "Resolucion objetivo: $WidthPx x $HeightPx px"
Write-Host "Viewport base: $BaseWidth x $BaseHeight px; escala: $($Last.ScaleFactor)"
