$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pandocDir = Join-Path $projectRoot ".venv\Lib\site-packages\pypandoc\files"
$miktexDir = Join-Path $PSScriptRoot "MiKTeX\miktex\bin\x64"
$userMiktexDir = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64"
$pathParts = @($pandocDir, $miktexDir, $userMiktexDir) | Where-Object { Test-Path $_ }
$env:PATH = (($pathParts + $env:PATH) -join ";")

Push-Location $projectRoot
try {
    .\.venv\Scripts\jupyter-nbconvert.exe --to latex REPORT.ipynb --output REPORT --output-dir latex\nbconvert_tex

    $texPath = Join-Path $PSScriptRoot "nbconvert_tex\REPORT.tex"
    $tex = Get-Content $texPath -Raw -Encoding UTF8
    if ($tex -notmatch "\\newcounter\{none\}") {
        $tex = $tex -replace "\\documentclass\[11pt\]\{article\}", "\\documentclass[11pt]{article}`n\\newcounter{none}"
        Set-Content $texPath $tex -Encoding UTF8
    }

    Push-Location (Join-Path $PSScriptRoot "nbconvert_tex")
    try {
        xelatex -interaction=nonstopmode REPORT.tex
        xelatex -interaction=nonstopmode REPORT.tex
    }
    finally {
        Pop-Location
    }

    Copy-Item (Join-Path $PSScriptRoot "nbconvert_tex\REPORT.pdf") (Join-Path $projectRoot "REPORT.pdf") -Force
    Copy-Item (Join-Path $PSScriptRoot "nbconvert_tex\REPORT.tex") (Join-Path $PSScriptRoot "REPORT.tex") -Force
}
finally {
    Pop-Location
}
