<#
cola_articulo.ps1 -- corridas que faltan para el articulo, en orden y reanudable.

Bloques (detalle y lecturas en docs\ESTUDIOS_SOTA.md, secciones 9, 11, 12 y 13):
  A  numero del articulo: 5 semillas del modelo final (98), como las 25 corridas de Zhao
  B  ablaciones del modelo final: 107 prosodia, 109 fusion, 108 aumentacion
  C  lineas base unimodales con varias semillas (71 audio, 73 texto)
  D  protocolos de los trabajos relacionados: Yoon (random), Pepino (70),
     Chen & Rudnicky (99), Wu (100, ASR), MELD 7 clases (110 / 82), CREMA-D random
  E  opcional: 103 (RoBERTa-base)
  F  familia clasica (eGeMAPS + GloVe, misma fusion): 111-119. Opcional, se pide con
     -Bloques F. La primera vez necesita -Glove <glove.840B.300d.zip> para los caches
     (docs\ESTUDIOS_SOTA.md, seccion 11).
  G  alineacion contrastiva audio-texto sobre el 98: 120, 121, 122 con 3 semillas.
     Opcional, con -Bloques G. Antes del 122 revisa la nota de la seccion 12.
  H  revision de noviembre: 123 sin rama mel, 124 MulT, 125 detector con cross-fitting,
     126 los dos; 3 semillas cada uno. Con -Bloques H.

Cada tarea se salta si su resumen ya existe y esta completo. Si encuentra un resumen
PARCIAL se detiene y dice como reanudarlo, en vez de adivinar.

Uso, desde la raiz del repo con el venv activado:
    .\scripts\cola_articulo.ps1 -SoloListar      # que falta, sin correr nada
    .\scripts\cola_articulo.ps1                  # corre todo lo que falta
    .\scripts\cola_articulo.ps1 -Bloques A,B     # solo esos bloques
    .\scripts\cola_articulo.ps1 -Bloques F -Glove D:\DOCTORADO\RECURSOS\glove.840B.300d.zip

Ojo: en PowerShell las variables no distinguen mayusculas ($S y $s son la misma).
#>
param(
    [switch]$SoloListar,
    [string[]]$Bloques = @("A", "B", "C", "D"),
    [string]$Device = "cuda",
    [string]$Glove = ""
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
    @{ B = "E"; N = "103 RoBERTa-base"; R = (Resumen 103 ""); A = "--study 103 --loso"; C = 103 },
    # ---- F: familia clasica (X = estudio con el que se extraen los caches del corpus) --
    @{ B = "F"; N = "113 clasico IEMOCAP audio+texto"; R = (Resumen 113 ""); A = "--study 113 --loso"; X = 113 },
    @{ B = "F"; N = "111 clasico IEMOCAP solo audio";  R = (Resumen 111 ""); A = "--study 111 --loso"; X = 113 },
    @{ B = "F"; N = "112 clasico IEMOCAP solo texto";  R = (Resumen 112 ""); A = "--study 112 --loso"; X = 113 },
    @{ B = "F"; N = "116 clasico CREMA-D audio+texto"; R = (Resumen 116 ""); A = "--study 116 --loso"; X = 116 },
    @{ B = "F"; N = "117 clasico CREMA-D solo audio";  R = (Resumen 117 ""); A = "--study 117 --loso"; X = 116 },
    @{ B = "F"; N = "118 clasico MELD audio+texto s42";  R = (Resumen 118 "_seed42");  A = "--study 118 --protocol fixed --seed 42";  X = 118 },
    @{ B = "F"; N = "118 clasico MELD audio+texto s123"; R = (Resumen 118 "_seed123"); A = "--study 118 --protocol fixed --seed 123"; X = 118 },
    @{ B = "F"; N = "119 clasico MELD solo audio s42";   R = (Resumen 119 "_seed42");  A = "--study 119 --protocol fixed --seed 42";  X = 118 },
    @{ B = "F"; N = "119 clasico MELD solo audio s123";  R = (Resumen 119 "_seed123"); A = "--study 119 --protocol fixed --seed 123"; X = 118 },
    @{ B = "F"; N = "114 clasico IEMOCAP + compuerta"; R = (Resumen 114 ""); A = "--study 114 --loso"; X = 113 },
    @{ B = "F"; N = "115 clasico IEMOCAP + ViT";       R = (Resumen 115 ""); A = "--study 115 --loso"; X = 113 },
    # ---- G: alineacion contrastiva (3 semillas cada uno, como el 98) ------------------
    @{ B = "G"; N = "120 contrastiva con memoria s42";    R = (Resumen 120 "");         A = "--study 120 --loso" },
    @{ B = "G"; N = "121 contrastiva con prototipos s42"; R = (Resumen 121 "");         A = "--study 121 --loso" },
    @{ B = "G"; N = "120 contrastiva con memoria s123";   R = (Resumen 120 "_seed123"); A = "--study 120 --loso --seed 123" },
    @{ B = "G"; N = "121 contrastiva con prototipos s123"; R = (Resumen 121 "_seed123"); A = "--study 121 --loso --seed 123" },
    @{ B = "G"; N = "120 contrastiva con memoria s456";   R = (Resumen 120 "_seed456"); A = "--study 120 --loso --seed 456" },
    @{ B = "G"; N = "121 contrastiva con prototipos s456"; R = (Resumen 121 "_seed456"); A = "--study 121 --loso --seed 456" },
    @{ B = "G"; N = "122 contrastiva + co-atencion s42";  R = (Resumen 122 "");         A = "--study 122 --loso" },
    @{ B = "G"; N = "122 contrastiva + co-atencion s123"; R = (Resumen 122 "_seed123"); A = "--study 122 --loso --seed 123" },
    @{ B = "G"; N = "122 contrastiva + co-atencion s456"; R = (Resumen 122 "_seed456"); A = "--study 122 --loso --seed 456" },
    # ---- H: revision de noviembre (3 semillas cada uno) --------------------------------
    @{ B = "H"; N = "123 audio+texto sin rama mel s42";  R = (Resumen 123 "");         A = "--study 123 --loso" },
    @{ B = "H"; N = "124 + MulT s42";                    R = (Resumen 124 "");         A = "--study 124 --loso" },
    @{ B = "H"; N = "125 + detector cross-fitting s42";  R = (Resumen 125 "");         A = "--study 125 --loso" },
    @{ B = "H"; N = "126 MulT + detector s42";           R = (Resumen 126 "");         A = "--study 126 --loso" },
    @{ B = "H"; N = "123 audio+texto sin rama mel s123"; R = (Resumen 123 "_seed123"); A = "--study 123 --loso --seed 123" },
    @{ B = "H"; N = "124 + MulT s123";                   R = (Resumen 124 "_seed123"); A = "--study 124 --loso --seed 123" },
    @{ B = "H"; N = "125 + detector cross-fitting s123"; R = (Resumen 125 "_seed123"); A = "--study 125 --loso --seed 123" },
    @{ B = "H"; N = "126 MulT + detector s123";          R = (Resumen 126 "_seed123"); A = "--study 126 --loso --seed 123" },
    @{ B = "H"; N = "123 audio+texto sin rama mel s456"; R = (Resumen 123 "_seed456"); A = "--study 123 --loso --seed 456" },
    @{ B = "H"; N = "124 + MulT s456";                   R = (Resumen 124 "_seed456"); A = "--study 124 --loso --seed 456" },
    @{ B = "H"; N = "125 + detector cross-fitting s456"; R = (Resumen 125 "_seed456"); A = "--study 125 --loso --seed 456" },
    @{ B = "H"; N = "126 MulT + detector s456";          R = (Resumen 126 "_seed456"); A = "--study 126 --loso --seed 456" }
)

$extraidos = @{}

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
    if ($t.X -and -not $extraidos.ContainsKey($t.X)) {
        # Caches de la familia clasica; incremental: si ya existen no se rehacen.
        $ex = @("scripts\extraer_clasicas.py", "--study", $t.X)
        if ($Glove) { $ex += @("--glove", $Glove) }
        python @ex
        if ($LASTEXITCODE -ne 0) { Write-Host "Fallo la extraccion de caracteristicas clasicas; se detiene la cola." -ForegroundColor Red; break }
        $extraidos[$t.X] = $true
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
