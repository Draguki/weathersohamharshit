# Start only the FastAPI backend on http://127.0.0.1:8000 (foreground; Ctrl+C to stop).
# Analytics endpoints work immediately; Spark MLlib models train in-process (~1-3 min) before
# /api/live and /api/predict become available (watch /api/health -> spark_ready).
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py   = "C:\Users\harsh\.venvs\wbd\Scripts\python.exe"
$env:JAVA_HOME = "C:\Program Files\Java\jre1.8.0_461"
$env:PYSPARK_PYTHON = $Py
$env:PYSPARK_DRIVER_PYTHON = $Py
Remove-Item Env:WBD_FAKE_MODEL -ErrorAction SilentlyContinue
Remove-Item Env:WBD_DB_PATH -ErrorAction SilentlyContinue
Set-Location $Root
& $Py -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
