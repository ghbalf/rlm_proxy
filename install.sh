#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# RLM Proxy Installer (Linux + macOS)
#
# Downloads the latest release from GitHub and installs RLM Proxy.
# Optionally sets up a background service (systemd on Linux, launchd on macOS)
# or a Docker container.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/ghbalf/rlm_proxy/main/install.sh | bash
#   ./install.sh [--service] [--docker] [--dir /custom/path] [--uninstall]
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="ghbalf/rlm_proxy"
SERVICE_NAME="rlm-proxy"
SETUP_SERVICE=false
SETUP_DOCKER=false
UNINSTALL=false

# Platform detection
OS="$(uname -s)"
case "$OS" in
  Linux*)  PLATFORM="linux";  DEFAULT_DIR="/opt/rlm-proxy" ;;
  Darwin*) PLATFORM="macos";  DEFAULT_DIR="$HOME/.rlm-proxy" ;;
  *)       echo "Unsupported OS: $OS (use install.ps1 for Windows)"; exit 1 ;;
esac

INSTALL_DIR="$DEFAULT_DIR"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[rlm-proxy]${NC} $1"; }
ok()    { echo -e "${GREEN}[rlm-proxy]${NC} $1"; }
warn()  { echo -e "${YELLOW}[rlm-proxy]${NC} $1"; }
error() { echo -e "${RED}[rlm-proxy]${NC} $1" >&2; }

# ── Parse args ────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case "$1" in
    --service)   SETUP_SERVICE=true; shift ;;
    --docker)    SETUP_DOCKER=true; shift ;;
    --dir)       INSTALL_DIR="$2"; shift 2 ;;
    --uninstall) UNINSTALL=true; shift ;;
    --help|-h)
      echo "Usage: install.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --service     Set up as background service (systemd on Linux, launchd on macOS)"
      echo "  --docker      Set up Docker container instead"
      echo "  --dir PATH    Install to PATH (default: $DEFAULT_DIR)"
      echo "  --uninstall   Remove RLM Proxy"
      echo "  --help        Show this help"
      exit 0
      ;;
    *) error "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Helper: sudo if needed ────────────────────────────────────────────────

maybe_sudo() {
  if [[ -w "$(dirname "$1" 2>/dev/null || echo "$1")" ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

# ── Uninstall ─────────────────────────────────────────────────────────────

if $UNINSTALL; then
  info "Uninstalling RLM Proxy..."

  # Stop service
  if [[ "$PLATFORM" == "linux" ]]; then
    if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
      sudo systemctl stop "$SERVICE_NAME"
      sudo systemctl disable "$SERVICE_NAME"
    fi
    if [[ -f "/etc/systemd/system/${SERVICE_NAME}.service" ]]; then
      sudo rm "/etc/systemd/system/${SERVICE_NAME}.service"
      sudo systemctl daemon-reload
      ok "Systemd service removed"
    fi
  elif [[ "$PLATFORM" == "macos" ]]; then
    PLIST="$HOME/Library/LaunchAgents/com.rlm-proxy.plist"
    if launchctl list 2>/dev/null | grep -q "$SERVICE_NAME"; then
      launchctl unload "$PLIST" 2>/dev/null || true
    fi
    if [[ -f "$PLIST" ]]; then
      rm "$PLIST"
      ok "Launchd service removed"
    fi
  fi

  # Stop Docker
  if command -v docker &>/dev/null; then
    if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${SERVICE_NAME}$"; then
      docker stop "$SERVICE_NAME" 2>/dev/null || true
      docker rm "$SERVICE_NAME" 2>/dev/null || true
      ok "Docker container removed"
    fi
  fi

  # Remove directory
  if [[ -d "$INSTALL_DIR" ]]; then
    maybe_sudo rm -rf "$INSTALL_DIR"
    ok "Removed $INSTALL_DIR"
  fi

  ok "RLM Proxy uninstalled"
  exit 0
fi

# ── Check prerequisites ──────────────────────────────────────────────────

info "Checking prerequisites ($PLATFORM)..."

if $SETUP_DOCKER; then
  if ! command -v docker &>/dev/null; then
    error "Docker is not installed. Install Docker first: https://docs.docker.com/get-docker/"
    exit 1
  fi
  ok "Docker found"
else
  if ! command -v python3 &>/dev/null; then
    if [[ "$PLATFORM" == "macos" ]]; then
      error "Python 3 not found. Install with: brew install python@3.12"
    else
      error "Python 3 not found. Install with: sudo apt install python3 python3-venv"
    fi
    exit 1
  fi

  PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
  PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
  PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
  if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 11 ]]; then
    error "Python 3.11+ required, found $PYTHON_VERSION"
    exit 1
  fi
  ok "Python $PYTHON_VERSION"
fi

if ! command -v git &>/dev/null; then
  error "git is not installed"
  exit 1
fi

# ── Download ──────────────────────────────────────────────────────────────

info "Downloading RLM Proxy..."

LATEST_TAG=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null | grep '"tag_name"' | cut -d'"' -f4 || echo "")

if [[ -z "$LATEST_TAG" ]]; then
  warn "No releases found, using main branch"
  BRANCH="main"
else
  info "Latest release: $LATEST_TAG"
  BRANCH="$LATEST_TAG"
fi

if [[ -d "$INSTALL_DIR" ]]; then
  info "Updating existing installation..."
  cd "$INSTALL_DIR"
  git fetch --all --tags
  git checkout "$BRANCH" 2>/dev/null || git checkout "origin/$BRANCH" 2>/dev/null || true
  git pull 2>/dev/null || true
else
  info "Installing to $INSTALL_DIR..."
  maybe_sudo mkdir -p "$(dirname "$INSTALL_DIR")"
  if [[ -w "$(dirname "$INSTALL_DIR")" ]]; then
    git clone "https://github.com/${REPO}.git" "$INSTALL_DIR"
  else
    sudo git clone "https://github.com/${REPO}.git" "$INSTALL_DIR"
    sudo chown -R "$(whoami)" "$INSTALL_DIR"
  fi
  cd "$INSTALL_DIR"
  if [[ -n "$LATEST_TAG" ]]; then
    git checkout "$LATEST_TAG"
  fi
fi

ok "Downloaded to $INSTALL_DIR"

# ── Install (Python) ─────────────────────────────────────────────────────

if ! $SETUP_DOCKER; then
  info "Setting up Python environment..."
  cd "$INSTALL_DIR"
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -e ".[test]"
  ok "Python dependencies installed"

  if [[ ! -f "$INSTALL_DIR/.env" ]] && [[ -f "$INSTALL_DIR/.env.example" ]]; then
    cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
    info "Created .env from .env.example"
  fi
fi

# ── Systemd service (Linux) ──────────────────────────────────────────────

if $SETUP_SERVICE && [[ "$PLATFORM" == "linux" ]]; then
  info "Setting up systemd service..."
  CURRENT_USER=$(whoami)

  sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" > /dev/null <<EOF
[Unit]
Description=RLM Proxy — Recursive Language Model proxy for LLMs
After=network.target ollama.service
Documentation=https://github.com/${REPO}

[Service]
Type=simple
User=${CURRENT_USER}
WorkingDirectory=${INSTALL_DIR}
Environment=PATH=${INSTALL_DIR}/.venv/bin:/usr/bin:/bin
EnvironmentFile=-${INSTALL_DIR}/.env
ExecStart=${INSTALL_DIR}/.venv/bin/python main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

  sudo systemctl daemon-reload
  sudo systemctl enable "$SERVICE_NAME"
  sudo systemctl start "$SERVICE_NAME"
  ok "Systemd service installed and started"
  echo ""
  info "Manage with:"
  echo "  sudo systemctl status $SERVICE_NAME"
  echo "  sudo systemctl restart $SERVICE_NAME"
  echo "  journalctl -u $SERVICE_NAME -f"
fi

# ── Launchd service (macOS) ──────────────────────────────────────────────

if $SETUP_SERVICE && [[ "$PLATFORM" == "macos" ]]; then
  info "Setting up launchd service..."
  PLIST_DIR="$HOME/Library/LaunchAgents"
  PLIST="$PLIST_DIR/com.rlm-proxy.plist"
  LOG_DIR="$HOME/Library/Logs/rlm-proxy"
  mkdir -p "$PLIST_DIR" "$LOG_DIR"

  cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.rlm-proxy</string>
    <key>ProgramArguments</key>
    <array>
        <string>${INSTALL_DIR}/.venv/bin/python</string>
        <string>${INSTALL_DIR}/main.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${INSTALL_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>${INSTALL_DIR}/.venv/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF

  launchctl unload "$PLIST" 2>/dev/null || true
  launchctl load "$PLIST"
  ok "Launchd service installed and started"
  echo ""
  info "Manage with:"
  echo "  launchctl list | grep rlm-proxy"
  echo "  launchctl unload $PLIST   # stop"
  echo "  launchctl load $PLIST     # start"
  echo "  tail -f $LOG_DIR/stderr.log"
fi

# ── Docker ────────────────────────────────────────────────────────────────

if $SETUP_DOCKER; then
  info "Building Docker image..."
  cd "$INSTALL_DIR"

  if [[ ! -f "Dockerfile" ]]; then
    cat > Dockerfile <<'DEOF'
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .
EXPOSE 8881
CMD ["python", "main.py"]
DEOF
  fi

  docker build -t "$SERVICE_NAME" .

  docker stop "$SERVICE_NAME" 2>/dev/null || true
  docker rm "$SERVICE_NAME" 2>/dev/null || true

  DOCKER_ARGS="-d --name $SERVICE_NAME -p 8881:8881 --restart unless-stopped"

  [[ -f "$INSTALL_DIR/config.json" ]] && DOCKER_ARGS="$DOCKER_ARGS -v ${INSTALL_DIR}/config.json:/app/config.json"
  [[ -f "$INSTALL_DIR/.env" ]] && DOCKER_ARGS="$DOCKER_ARGS --env-file ${INSTALL_DIR}/.env"

  if [[ "$PLATFORM" == "linux" ]]; then
    DOCKER_ARGS="$DOCKER_ARGS --network host"
    warn "Using host networking — Ollama at localhost:11434 will be accessible"
  else
    DOCKER_ARGS="$DOCKER_ARGS -e OLLAMA_BASE_URL=http://host.docker.internal:11434"
  fi

  eval docker run $DOCKER_ARGS "$SERVICE_NAME"
  ok "Docker container started"
  echo ""
  info "Manage with:"
  echo "  docker logs -f $SERVICE_NAME"
  echo "  docker restart $SERVICE_NAME"
  echo "  docker stop $SERVICE_NAME"
fi

# ── Done ──────────────────────────────────────────────────────────────────

echo ""
ok "RLM Proxy installed successfully!"
echo ""
info "Admin dashboard: http://localhost:8881/admin"
info "Health check:    http://localhost:8881/health"
info "API docs:        http://localhost:8881/docs"
echo ""

if ! $SETUP_SERVICE && ! $SETUP_DOCKER; then
  info "To start manually:"
  echo "  cd $INSTALL_DIR"
  echo "  source .venv/bin/activate"
  echo "  python main.py"
  echo ""
  info "To install as a service:"
  echo "  ${INSTALL_DIR}/install.sh --dir ${INSTALL_DIR} --service"
fi
