$ErrorActionPreference = 'Stop'

$root = 'c:\Users\Jobbie\c++Workspace\Comp301 project\dash'
$venv = Join-Path $root '.venv'
$python = Join-Path $venv 'Scripts/python.exe'

if (!(Test-Path $python)) {
    Write-Error "dash/.venv not found at $venv. Create it first."
    exit 1
}

# Ensure UTF-8 I/O and deterministic Python text behavior
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

# Launch the Dash app via the venv interpreter
& $python (Join-Path $root 'app.py')
