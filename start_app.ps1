$ErrorActionPreference = 'Stop'
$frontendPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $frontendPython)) {
    throw 'Install the frontend environment in .venv first.'
}
Set-Location -LiteralPath $PSScriptRoot
& $frontendPython -m streamlit run (Join-Path $PSScriptRoot 'app.py') --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
