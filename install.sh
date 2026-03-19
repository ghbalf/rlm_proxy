#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# RLM Proxy Installer
#
# Downloads the latest release from GitHub, installs to /opt/rlm-proxy,
# and optionally sets up a systemd service or Docker container.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/ghbalf/rlm_proxy/main/install.sh | bash
#   # or
#   ./install.sh [--service] [--docker] [--dir /custom/path] [--uninstall]
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="ghbalf/rlm_proxy"
INSTALL_DIR="/opt/rlm-proxy"
SERVICE_NAME="rlm-proxy"
SETUP_SERVICE=false
SETUP_DOCKER=false
UNINSTALL=false

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
    --service)  SETUP_SERVICE=true; shift ;;
    --docker)   SETUP_DOCKER=true; shift ;;
    --dir)      INSTALL_DIR="$2"; shift 2 ;;
    --uninstall) UNINSTALL=true; shift ;;
    --help|-h)
      echo "Usage: install.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --service     Set up systemd service"
      echo "  --docker      Set up Docker container instead"
      echo "  --dir PATH    Install to PATH (default: /opt/rlm-proxy)"
      echo "  --uninstall   Remove RLM Proxy"
      echo "  --help        Show this help"
      exit 0
      ;;
    *) error "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Uninstall ─────────────────────────────────────────────────────────────

if $UNINSTALL; then
  info "Uninstalling RLM Proxy..."

  # Stop and disable service
  if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
    info "Stopping service..."
    sudo systemctl stop "$SERVICE_NAME"
    sudo systemctl disable "$SERVICE_NAME"
  fi
  if [[ -f "/etc/systemd/system/${SERVICE_NAME}.service" ]]; then
    sudo rm "/etc/systemd/system/${SERVICE_NAME}.service"
    sudo systemctl daemon-reload
    ok "Service removed"
  fi

  # Stop Docker container
  if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${SERVICE_NAME}$"; then
    info "Stopping Docker container..."
    docker stop "$SERVICE_NAME" 2>/dev/null || true
    docker rm "$SERVICE_NAME" 2>/dev/null || true
    ok "Docker container removed"
  fi

  # Remove install directory
  if [[ -d "$INSTALL_DIR" ]]; then
    info "Removing $INSTALL_DIR..."
    sudo rm -rf "$INSTALL_DIR"
    ok "Directory removed"
  fi

  ok "RLM Proxy uninstalled"
  exit 0
fi

# ── Check prerequisites ──────────────────────────────────────────────────

info "Checking prerequisites..."

if $SETUP_DOCKER; then
  if ! command -v docker &>/dev/null; then
    error "Docker is not installed. Install Docker first: https://docs.docker.com/get-docker/"
    exit 1
  fi
  ok "Docker found"
else
  if ! command -v python3 &>/dev/null; then
    error "Python 3 is not installed. Install Python 3.11+ first."
    exit 1
  fi

  PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
  PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
  PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
  if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 11 ]]; then
    error "Python 3.11+ required, found $PYTHON_VERSION"
    exit 1
  fi
  ok "Python $PYTHON_VERSION found"
fi

if ! command -v git &>/dev/null; then
  error "git is not installed"
  exit 1
fi

# ── Download ──────────────────────────────────────────────────────────────

info "Downloading RLM Proxy..."

# Get latest release tag (fall back to main if no releases)
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
  git checkout "$BRANCH" 2>/dev/null || git checkout "origin/$BRANCH"
  git pull origin "$BRANCH" 2>/dev/null || true
else
  info "Cloning to $INSTALL_DIR..."
  sudo mkdir -p "$(dirname "$INSTALL_DIR")"
  sudo git clone "https://github.com/${REPO}.git" "$INSTALL_DIR"
  sudo chown -R "$(whoami)" "$INSTALL_DIR"
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

  # Copy example config if none exists
  if [[ ! -f "$INSTALL_DIR/.env" ]] && [[ -f "$INSTALL_DIR/.env.example" ]]; then
    cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
    info "Created .env from .env.example — edit it with your settings"
  fi
fi

# ── Systemd service ──────────────────────────────────────────────────────

if $SETUP_SERVICE; then
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
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

  sudo systemctl daemon-reload
  sudo systemctl enable "$SERVICE_NAME"
  sudo systemctl start "$SERVICE_NAME"

  ok "Service installed and started"
  echo ""
  info "Manage with:"
  echo "  sudo systemctl status $SERVICE_NAME"
  echo "  sudo systemctl restart $SERVICE_NAME"
  echo "  journalctl -u $SERVICE_NAME -f"
fi

# ── Docker ────────────────────────────────────────────────────────────────

if $SETUP_DOCKER; then
  info "Building Docker image..."

  cd "$INSTALL_DIR"

  # Create Dockerfile if it doesn't exist
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

  # Stop existing container if running
  docker stop "$SERVICE_NAME" 2>/dev/null || true
  docker rm "$SERVICE_NAME" 2>/dev/null || true

  # Run
  DOCKER_ARGS="-d --name $SERVICE_NAME -p 8881:8881 --restart unless-stopped"

  # Mount config if it exists
  if [[ -f "$INSTALL_DIR/config.json" ]]; then
    DOCKER_ARGS="$DOCKER_ARGS -v ${INSTALL_DIR}/config.json:/app/config.json"
  fi

  # Mount .env if it exists
  if [[ -f "$INSTALL_DIR/.env" ]]; then
    DOCKER_ARGS="$DOCKER_ARGS --env-file ${INSTALL_DIR}/.env"
  fi

  # Add host networking on Linux for Ollama access
  if [[ "$(uname)" == "Linux" ]]; then
    DOCKER_ARGS="$DOCKER_ARGS --network host"
    warn "Using host networking (Linux) — Ollama at localhost:11434 will be accessible"
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
  echo "  $0 --service"
  echo ""
  info "To run with Docker:"
  echo "  $0 --docker"
fi
