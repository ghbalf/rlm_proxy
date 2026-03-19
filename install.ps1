# ──────────────────────────────────────────────────────────────────────────
# RLM Proxy Installer (Windows)
#
# Downloads the latest release from GitHub and installs RLM Proxy.
# Optionally sets up a Windows Service (requires NSSM) or Docker container.
#
# Usage (PowerShell, run as Administrator for --Service):
#   irm https://raw.githubusercontent.com/ghbalf/rlm_proxy/main/install.ps1 | iex
#   .\install.ps1 [-Service] [-Docker] [-Dir C:\rlm-proxy] [-Uninstall]
# ──────────────────────────────────────────────────────────────────────────

param(
    [switch]$Service,
    [switch]$Docker,
    [string]$Dir = "$env:LOCALAPPDATA\rlm-proxy",
    [switch]$Uninstall,
    [switch]$Help
)

$ErrorActionPreference = "Stop"
$Repo = "ghbalf/rlm_proxy"
$ServiceName = "rlm-proxy"

function Write-Info($msg)  { Write-Host "[rlm-proxy] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)    { Write-Host "[rlm-proxy] $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "[rlm-proxy] $msg" -ForegroundColor Yellow }
function Write-Err($msg)   { Write-Host "[rlm-proxy] $msg" -ForegroundColor Red }

if ($Help) {
    Write-Host @"
Usage: install.ps1 [OPTIONS]

Options:
  -Service     Set up as Windows Service (requires NSSM and Administrator)
  -Docker      Set up Docker container instead
  -Dir PATH    Install to PATH (default: $env:LOCALAPPDATA\rlm-proxy)
  -Uninstall   Remove RLM Proxy
  -Help        Show this help
"@
    exit 0
}

# ── Uninstall ─────────────────────────────────────────────────────────────

if ($Uninstall) {
    Write-Info "Uninstalling RLM Proxy..."

    # Stop Windows Service
    $svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if ($svc) {
        if ($svc.Status -eq "Running") {
            Write-Info "Stopping service..."
            Stop-Service -Name $ServiceName -Force
        }
        if (Get-Command nssm -ErrorAction SilentlyContinue) {
            nssm remove $ServiceName confirm 2>$null
        } else {
            sc.exe delete $ServiceName 2>$null
        }
        Write-Ok "Service removed"
    }

    # Stop Docker
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        $container = docker ps -a --format "{{.Names}}" 2>$null | Where-Object { $_ -eq $ServiceName }
        if ($container) {
            docker stop $ServiceName 2>$null
            docker rm $ServiceName 2>$null
            Write-Ok "Docker container removed"
        }
    }

    # Remove directory
    if (Test-Path $Dir) {
        Remove-Item -Recurse -Force $Dir
        Write-Ok "Removed $Dir"
    }

    Write-Ok "RLM Proxy uninstalled"
    exit 0
}

# ── Check prerequisites ──────────────────────────────────────────────────

Write-Info "Checking prerequisites..."

if ($Docker) {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Err "Docker is not installed. Install Docker Desktop: https://docs.docker.com/desktop/install/windows-install/"
        exit 1
    }
    Write-Ok "Docker found"
} else {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) {
        $py = Get-Command python3 -ErrorAction SilentlyContinue
    }
    if (-not $py) {
        Write-Err "Python not found. Install Python 3.11+: https://www.python.org/downloads/"
        Write-Err "Make sure to check 'Add Python to PATH' during installation."
        exit 1
    }
    $PythonCmd = $py.Name
    $ver = & $PythonCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    $major, $minor = $ver -split '\.'
    if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 11)) {
        Write-Err "Python 3.11+ required, found $ver"
        exit 1
    }
    Write-Ok "Python $ver ($PythonCmd)"
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Err "git is not installed. Install: https://git-scm.com/download/win"
    exit 1
}

# ── Download ──────────────────────────────────────────────────────────────

Write-Info "Downloading RLM Proxy..."

try {
    $release = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest" -ErrorAction Stop
    $LatestTag = $release.tag_name
    Write-Info "Latest release: $LatestTag"
} catch {
    Write-Warn "No releases found, using main branch"
    $LatestTag = ""
}

if (Test-Path $Dir) {
    Write-Info "Updating existing installation..."
    Push-Location $Dir
    git fetch --all --tags 2>$null
    if ($LatestTag) { git checkout $LatestTag 2>$null }
    git pull 2>$null
    Pop-Location
} else {
    Write-Info "Cloning to $Dir..."
    $parent = Split-Path $Dir -Parent
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    git clone "https://github.com/$Repo.git" $Dir
    Push-Location $Dir
    if ($LatestTag) { git checkout $LatestTag 2>$null }
    Pop-Location
}

Write-Ok "Downloaded to $Dir"

# ── Install (Python) ─────────────────────────────────────────────────────

if (-not $Docker) {
    Write-Info "Setting up Python environment..."
    Push-Location $Dir

    & $PythonCmd -m venv .venv
    & .\.venv\Scripts\pip install --quiet --upgrade pip
    & .\.venv\Scripts\pip install --quiet -e ".[test]"

    if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
        Copy-Item ".env.example" ".env"
        Write-Info "Created .env from .env.example"
    }

    Pop-Location
    Write-Ok "Python dependencies installed"
}

# ── Windows Service ──────────────────────────────────────────────────────

if ($Service) {
    Write-Info "Setting up Windows Service..."

    # Check for NSSM
    if (-not (Get-Command nssm -ErrorAction SilentlyContinue)) {
        Write-Warn "NSSM not found. Installing via winget..."
        try {
            winget install nssm --silent 2>$null
        } catch {
            Write-Err "Could not install NSSM. Install manually: https://nssm.cc/download"
            Write-Err "Then run this installer again with --Service"
            exit 1
        }
    }

    # Remove existing service if present
    $svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if ($svc) {
        if ($svc.Status -eq "Running") { Stop-Service -Name $ServiceName -Force }
        nssm remove $ServiceName confirm 2>$null
    }

    $pythonExe = Join-Path $Dir ".venv\Scripts\python.exe"
    $mainPy = Join-Path $Dir "main.py"

    nssm install $ServiceName $pythonExe $mainPy
    nssm set $ServiceName AppDirectory $Dir
    nssm set $ServiceName DisplayName "RLM Proxy"
    nssm set $ServiceName Description "Recursive Language Model proxy for LLMs"
    nssm set $ServiceName AppStdout (Join-Path $Dir "logs\stdout.log")
    nssm set $ServiceName AppStderr (Join-Path $Dir "logs\stderr.log")
    nssm set $ServiceName AppRotateFiles 1
    nssm set $ServiceName AppRotateBytes 1048576

    New-Item -ItemType Directory -Path (Join-Path $Dir "logs") -Force | Out-Null

    Start-Service -Name $ServiceName
    Write-Ok "Windows Service installed and started"
    Write-Host ""
    Write-Info "Manage with:"
    Write-Host "  Get-Service $ServiceName"
    Write-Host "  Restart-Service $ServiceName"
    Write-Host "  Stop-Service $ServiceName"
    Write-Host "  nssm edit $ServiceName"
}

# ── Docker ────────────────────────────────────────────────────────────────

if ($Docker) {
    Write-Info "Building Docker image..."
    Push-Location $Dir

    if (-not (Test-Path "Dockerfile")) {
        @"
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .
EXPOSE 8881
CMD ["python", "main.py"]
"@ | Set-Content -Path "Dockerfile"
    }

    docker build -t $ServiceName .
    docker stop $ServiceName 2>$null
    docker rm $ServiceName 2>$null

    $dockerArgs = @(
        "run", "-d",
        "--name", $ServiceName,
        "-p", "8881:8881",
        "--restart", "unless-stopped",
        "-e", "OLLAMA_BASE_URL=http://host.docker.internal:11434"
    )

    if (Test-Path "config.json") {
        $configPath = (Resolve-Path "config.json").Path
        $dockerArgs += @("-v", "${configPath}:/app/config.json")
    }

    $dockerArgs += $ServiceName
    & docker @dockerArgs

    Pop-Location
    Write-Ok "Docker container started"
    Write-Host ""
    Write-Info "Manage with:"
    Write-Host "  docker logs -f $ServiceName"
    Write-Host "  docker restart $ServiceName"
    Write-Host "  docker stop $ServiceName"
}

# ── Done ──────────────────────────────────────────────────────────────────

Write-Host ""
Write-Ok "RLM Proxy installed successfully!"
Write-Host ""
Write-Info "Admin dashboard: http://localhost:8881/admin"
Write-Info "Health check:    http://localhost:8881/health"
Write-Info "API docs:        http://localhost:8881/docs"
Write-Host ""

if (-not $Service -and -not $Docker) {
    Write-Info "To start manually:"
    Write-Host "  cd $Dir"
    Write-Host "  .\.venv\Scripts\activate"
    Write-Host "  python main.py"
    Write-Host ""
    Write-Info "To install as a service:"
    Write-Host "  .\install.ps1 -Service"
}
