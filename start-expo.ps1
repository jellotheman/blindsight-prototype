[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$ApiPort = 8000
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

    $ErrorActionPreference = 'Continue'
    & $venvPython -c 'import fastapi, uvicorn, blindsight' 2>$null
    $dependencyProbeExitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($dependencyProbeExitCode -ne 0) {
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

        # The backend serves this export even while the native Expo client is under development.
        npm run build:web
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    finally {
        Pop-Location
    }

    $shellPath = (Get-Process -Id $PID).Path
    $backendCommand = "& `"$venvPython`" -m tools.local_dev --port $ApiPort --generate-api-key"
    Start-Process -FilePath $shellPath -WorkingDirectory $repoRoot -ArgumentList @(
            '-NoExit',
            '-ExecutionPolicy', 'Bypass',
            '-Command', $backendCommand
        )

    Write-Host 'The API is starting in a second window.'
    Write-Host 'Enter its HTTPS URL and temporary key in BlindSight Settings.'
    Write-Host 'Starting Expo for a physical device...'
    Push-Location $frontendDir
    try {
        npm start -- --tunnel
    }
    finally {
        Pop-Location
    }
}
finally {
    Pop-Location
}
