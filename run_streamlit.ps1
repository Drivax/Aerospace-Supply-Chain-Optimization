param(
    [string[]]$Args
)

Set-Location -Path $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    & $venvPython -m streamlit run app.py @Args
} elseif (Get-Command streamlit -ErrorAction SilentlyContinue) {
    streamlit run app.py @Args
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python -m streamlit run app.py @Args
} else {
    throw "Python or Streamlit was not found. Activate a virtual environment or install dependencies."
}
