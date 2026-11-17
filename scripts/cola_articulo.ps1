<#
cola_articulo.ps1 -- corridas que faltan para el articulo, en orden y reanudable.

Bloques (detalle y lecturas en docs\ESTUDIOS_SOTA.md, seccion 9):
  A  numero del articulo: 5 semillas del modelo final (98), como las 25 corridas de Zhao
  B  ablaciones del modelo final: 107 prosodia, 109 fusion, 108 aumentacion
  C  lineas base unimodales con varias semillas (71 audio, 73 texto)
  D  protocolos de los trabajos relacionados: Yoon (random), Pepino (70),
     Chen & Rudnicky (99), Wu (100, ASR), MELD 7 clases (110 / 82), CREMA-D random
  E  opcional: 103 (RoBERTa-base)

Cada tarea se salta si su resumen ya existe y esta completo. Si encuentra un resumen
PARCIAL se detiene y dice como reanudarlo, en vez de adivinar.

Uso, desde la raiz del repo con el venv activado:
    .\scripts\cola_articulo.ps1 -SoloListar      # que falta, sin correr nada
    .\scripts\cola_articulo.ps1                  # corre todo lo que falta
    .\scripts\cola_articulo.ps1 -Bloques A,B     # solo esos bloques

Ojo: en PowerShell las variables no distinguen mayusculas ($S y $s son la misma).
#>
param(
    [switch]$SoloListar,
    [string[]]$Bloques = @("A", "B", "C", "D"),
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Continue"
$ck = "checkpoints"

function Resumen([int]$estudio, [string]$sufijo) {
    return Join-Path $ck ("ablation_{0}\kfold_summary_ablation_{0}{1}.json" -f $estudio, $sufijo)
}

$tareas = @(
    # ---- A: numero del articulo ----------------------------------------------------
    @{ B = "A"; N = "98 semilla 789";  R = (Resumen 98 "_seed789");  A = "--study 98 --loso --seed 789" },
    @{ B = "A"; N = "98 semilla 1011"; R = (Resumen 98 "_seed1011"); A = "--study 98 --loso --seed 1011" },
    # ---- B: ablaciones del modelo final ---------------------------------------------
    @{ B = "B"; N = "107 sin prompt de prosodia";    R = (Resumen 107 ""); A = "--study 107 --loso"; C = 107 },
    @{ B = "B"; N = "109 fusion por concatenacion";  R = (Resumen 109 ""); A = "--study 109 --loso"; C = 109 },
    @{ B = "B"; N = "108 sin aumentacion de audio";  R = (Resumen 108 ""); A = "--study 108 --loso"; C = 108 },
    # ---- C: lineas base unimodales con semillas -------------------------------------
    @{ B = "C"; N = "73 texto semilla 123"; R = (Resumen 73 "_seed123"); A = "--study 73 --loso --seed 123" },
    @{ B = "C"; N = "73 texto semilla 456"; R = (Resumen 73 "_seed456"); A = "--study 73 --loso --seed 456" },
    @{ B = "C"; N = "71 audio semilla 123"; R = (Resumen 71 "_seed123"); A = "--study 71 --loso --seed 123" },
    @{ B = "C"; N = "71 audio semilla 456"; R = (Resumen 71 "_seed456"); A = "--study 71 --loso --seed 456" },
    # ---- D: protocolos de los trabajos relacionados ---------------------------------
    @{ B = "D"; N = "Yoon: 98 con folds aleatorios"; R = (Resumen 98 "_random"); A = "--study 98 --loso --protocol random" },
    @{ B = "D"; N = "Yoon: 71 con folds aleatorios"; R = (Resumen 71 "_random"); A = "--study 71 --loso --protocol random" },
    @{ B = "D"; N = "Pepino: 70 (wav2vec2 congelado)"; R = (Resumen 70 ""); A = "--study 70 --loso" },
    @{ B = "D"; N = "Chen & Rudnicky: 99 (epocas fijas)"; R = (Resumen 99 ""); A = "--study 99 --loso"; C = 99 },
    @{ B = "D"; N = "Wu: 100 (texto de ASR)"; R = (Resumen 100 ""); A = "--study 100 --loso"; C = 100; ASR = $true },
    @{ B = "D"; N = "MELD 7 clases: 110 semilla 42";  R = (Resumen 110 "_seed42");  A = "--study 110 --protocol fixed --seed 42"; C = 110 },
    @{ B = "D"; N = "MELD 7 clases: 110 semilla 123"; R = (Resumen 110 "_seed123"); A = "--study 110 --protocol fixed --seed 123" },
    @{ B = "D"; N = "MELD 7 clases: 82 semilla 42";   R = (Resumen 82 "_seed42");   A = "--study 82 --protocol fixed --seed 42" },
    @{ B = "D"; N = "MELD 7 clases: 82 semilla 123";  R = (Resumen 82 "_seed123");  A = "--study 82 --protocol fixed --seed 123" },
    @{ B = "D"; N = "CREMA-D aleatorio: 105"; R = (Resumen 105 "_random"); A = "--study 105 --loso --protocol random" },
    @{ B = "D"; N = "CREMA-D aleatorio: 78";  R = (Resumen 78 "_random");  A = "--study 78 --loso --protocol random" },
    # ---- E: opcional ----------------------------------------------------------------
    @{ B = "E"; N = "103 RoBERTa-base"; R = (Resumen 103 ""); A = "--study 103 --loso"; C = 103 }
)

function Estado($t) {
    if (-not (Test-Path $t.R)) { return "falta" }
    try { $j = Get-Content $t.R -Raw | ConvertFrom-Json } catch { return "ilegible" }
    if ($j.complete) { return "hecho" } else { return "parcial" }
}

foreach ($t in $tareas) {
    if ($Bloques -notcontains $t.B) { continue }
    $estado = Estado $t
    if ($estado -eq "hecho") { Write-Host "[hecho]   $($t.B)  $($t.N)"; continue }
    if ($estado -ne "falta") {
        Write-Host "[$estado] $($t.B)  $($t.N): $($t.R)" -ForegroundColor Yellow
        Write-Host "          Reanuda los folds que faltan a mano (el script imprime el comando --folds) y vuelve a lanzar la cola."
        break
    }
    if ($SoloListar) { Write-Host "[falta]   $($t.B)  $($t.N)"; continue }

    Write-Host "`n=== $($t.B)  $($t.N) ===" -ForegroundColor Cyan
    if ($t.ASR) {
        $cache = "cache\asr\iemocap_whisper-small.en.json"
        if (-not (Test-Path $cache)) {
            python scripts\transcribe_asr.py --config config\ablation\ablation_100_asr_transcripts.yaml --device $Device
            if ($LASTEXITCODE -ne 0) { Write-Host "Fallo la transcripcion; se detiene la cola." -ForegroundColor Red; break }
        }
    }
    if ($t.C) {
        python scripts\check_sota_studies.py --study $t.C --fold 1 --device $Device
        if ($LASTEXITCODE -ne 0) { Write-Host "La comprobacion previa fallo; se detiene la cola." -ForegroundColor Red; break }
    }
    $argumentos = $t.A.Split(" ") + @("--device", $Device)
    python scripts\run_ablation.py @argumentos
    if ((Estado $t) -ne "hecho") {
        Write-Host "El resumen no quedo completo ($($t.R)); se detiene la cola." -ForegroundColor Red
        break
    }
}
