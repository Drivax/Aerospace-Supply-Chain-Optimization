param(
    [string[]]$Args
)

Set-Location -Path $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    & $venvPython -m pytest -q @Args
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python -m pytest -q @Args
} else {
    throw "Python was not found. Activate a virtual environment or install Python."
}
