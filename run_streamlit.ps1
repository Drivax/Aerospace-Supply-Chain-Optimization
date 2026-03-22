param(
    [string[]]$Args
)

Set-Location -Path $PSScriptRoot

if (Get-Command streamlit -ErrorAction SilentlyContinue) {
    streamlit run app.py @Args
} else {
    python -m streamlit run app.py @Args
}
