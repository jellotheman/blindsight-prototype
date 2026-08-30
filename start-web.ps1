[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8000,

    [switch]$Phone
)

$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot
$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
$frontendDir = Join-Path $repoRoot 'frontend'

Push-Location $repoRoot
try {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        Write-Host 'Creating the local Python environment...'
        py -3.12 -m venv .venv
    }

    & $venvPython -c 'import fastapi, uvicorn, blindsight' 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Installing the local server dependencies...'
        & $venvPython -m pip install -e . 'uvicorn>=0.30'
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    Push-Location $frontendDir
    try {
        if (-not (Test-Path -LiteralPath (Join-Path $frontendDir 'node_modules'))) {
            Write-Host 'Installing the Expo dependencies...'
            npm ci
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }

        Write-Host 'Building the Expo web client...'
        npm run build:web
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    finally {
        Pop-Location
    }

    if ($Phone) {
        Write-Host 'Starting the web client with a phone-accessible HTTPS tunnel...'
        & $venvPython -m tools.local_dev --port $Port --generate-api-key
    }
    else {
        Write-Host 'Starting the web client for this computer only...'
        & $venvPython -m tools.local_dev --port $Port --no-tunnel --generate-api-key
    }
}
finally {
    Pop-Location
}
