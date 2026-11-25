# One-click launcher for the Large-Scale Weather Data Analysis project.
# Usage (from the project folder):
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1            # start backend + dashboard (uses existing DB)
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Pipeline  # re-run Spark ETL + ML training first
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Fetch -Pipeline  # also re-download raw data
param([switch]$Fetch, [switch]$Pipeline)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py   = "C:\Users\harsh\.venvs\wbd\Scripts\python.exe"
$env:JAVA_HOME = "C:\Program Files\Java\jre1.8.0_461"
$env:PYSPARK_PYTHON = $Py
$env:PYSPARK_DRIVER_PYTHON = $Py
Set-Location $Root

if ($Fetch) {
    Write-Host "==> [1/4] Downloading raw hourly weather data (Open-Meteo ERA5)" -ForegroundColor Cyan
    & $Py spark_jobs\fetch_data.py
}
if ($Pipeline) {
    Write-Host "==> [2/4] Spark ETL: raw CSV -> aggregations -> SQLite" -ForegroundColor Cyan
    & $Py spark_jobs\etl.py
    Write-Host "==> [3/4] Spark MLlib training -> metrics in SQLite" -ForegroundColor Cyan
    & $Py spark_jobs\train.py
}

Write-Host "==> [4/4] Starting FastAPI backend (http://127.0.0.1:8000) and dashboard (http://127.0.0.1:5173)" -ForegroundColor Cyan
Start-Process -FilePath $Py -ArgumentList "-m","uvicorn","backend.app:app","--host","127.0.0.1","--port","8000" -WorkingDirectory $Root
if (-not (Test-Path "$Root\dashboard\node_modules")) { Push-Location "$Root\dashboard"; npm install; Pop-Location }
Start-Process -FilePath "cmd.exe" -ArgumentList "/c","npm run dev -- --host 127.0.0.1 --port 5173" -WorkingDirectory "$Root\dashboard"

Write-Host ""
Write-Host "Backend starts instantly for analytics; the Spark ML models train in-process (~1-2 min) before live predictions appear." -ForegroundColor Yellow
Write-Host "Open the dashboard: http://127.0.0.1:5173" -ForegroundColor Green
Start-Sleep -Seconds 6
Start-Process "http://127.0.0.1:5173"
