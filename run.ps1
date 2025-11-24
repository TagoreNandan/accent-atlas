Param(
  [string]$Port = "5050",
  [string]$ModelPath = "results/presentation_imbalance/mfcc_prosody_model.joblib"
)
$env:PORT = $Port
$env:MFCC_MODEL_PATH = $ModelPath
$env:CHUNK_SEC = $env:CHUNK_SEC -ne $null ? $env:CHUNK_SEC : 0.8
$env:CHUNK_HOP = $env:CHUNK_HOP -ne $null ? $env:CHUNK_HOP : 0.4
$env:MIN_CHUNKS = $env:MIN_CHUNKS -ne $null ? $env:MIN_CHUNKS : 1
$env:CALIB_TEMPERATURE = $env:CALIB_TEMPERATURE -ne $null ? $env:CALIB_TEMPERATURE : 1.3
$env:MIN_DURATION_SEC = $env:MIN_DURATION_SEC -ne $null ? $env:MIN_DURATION_SEC : 1.2
$env:MIN_RMS = $env:MIN_RMS -ne $null ? $env:MIN_RMS : 0.003

if (-Not (Test-Path .venv)) {
  Write-Host "[run] Creating virtualenv"
  python -m venv .venv
}
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip | Out-Null
python -m pip install -r requirements.txt | Out-Null

Write-Host "[run] Starting server on port $Port with model=$ModelPath"
python server/app.py
