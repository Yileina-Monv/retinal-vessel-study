param([string]$BasePython = 'python')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$temporaryPath = Join-Path $projectRoot '.runtime\tmp'
New-Item -ItemType Directory -Force -Path $temporaryPath | Out-Null
$oldTemp = $env:TEMP
$oldTmp = $env:TMP
$oldUtf8 = $env:PYTHONUTF8
try {
    $env:TEMP = $temporaryPath
    $env:TMP = $temporaryPath
    $env:PYTHONUTF8 = '1'
    if (-not (Test-Path -LiteralPath $environmentPython)) {
        & $BasePython -c 'import sys; assert sys.version_info[:2] == (3, 13), "Use Python 3.13 for this locked Windows environment"'
        if ($LASTEXITCODE -ne 0) { throw 'Python version check failed' }
        & $BasePython -m venv (Join-Path $projectRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
    }
    & $environmentPython -c 'import sys; assert sys.version_info[:2] == (3, 13), "The existing environment must use Python 3.13"'
    if ($LASTEXITCODE -ne 0) { throw 'Environment version check failed' }
    & $environmentPython -m pip --version
    if ($LASTEXITCODE -ne 0) {
        & $environmentPython -m ensurepip --upgrade --default-pip
        if ($LASTEXITCODE -ne 0) { throw 'pip initialization failed' }
    }
    & $environmentPython -m pip install --only-binary=:all: --no-cache-dir --disable-pip-version-check --index-url https://pypi.org/simple -r (Join-Path $PSScriptRoot 'requirements-runtime.lock.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Runtime dependency installation failed' }
    & $environmentPython -m pip install --only-binary=:all: --no-deps --no-cache-dir --disable-pip-version-check --index-url https://download.pytorch.org/whl/cu130 -r (Join-Path $PSScriptRoot 'requirements-torch.lock.txt')
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
    & $environmentPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed' }
    & $environmentPython (Join-Path $PSScriptRoot 'verify_environment.py')
    if ($LASTEXITCODE -ne 0) { throw 'Environment verification failed' }
} finally {
    $env:TEMP = $oldTemp
    $env:TMP = $oldTmp
    $env:PYTHONUTF8 = $oldUtf8
}
