#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# RLM Proxy Uninstaller
#
# Stops the service/container, removes files, cleans up.
#
# Usage:
#   ./uninstall.sh [--dir /custom/path] [--keep-config]
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

INSTALL_DIR="/opt/rlm-proxy"
SERVICE_NAME="rlm-proxy"
KEEP_CONFIG=false

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[rlm-proxy]${NC} $1"; }
ok()    { echo -e "${GREEN}[rlm-proxy]${NC} $1"; }
warn()  { echo -e "${RED}[rlm-proxy]${NC} $1"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)         INSTALL_DIR="$2"; shift 2 ;;
    --keep-config) KEEP_CONFIG=true; shift ;;
    --help|-h)
      echo "Usage: uninstall.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --dir PATH      Install directory (default: /opt/rlm-proxy)"
      echo "  --keep-config   Keep config.json and .env files"
      echo "  --help          Show this help"
      exit 0
      ;;
    *) warn "Unknown option: $1"; exit 1 ;;
  esac
done

echo ""
info "Uninstalling RLM Proxy..."
echo ""

# ── Stop systemd service ─────────────────────────────────────────────────

if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
  info "Stopping systemd service..."
  sudo systemctl stop "$SERVICE_NAME"
fi

if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
  sudo systemctl disable "$SERVICE_NAME"
fi

if [[ -f "/etc/systemd/system/${SERVICE_NAME}.service" ]]; then
  sudo rm "/etc/systemd/system/${SERVICE_NAME}.service"
  sudo systemctl daemon-reload
  ok "Systemd service removed"
fi

# ── Stop Docker container ────────────────────────────────────────────────

if command -v docker &>/dev/null; then
  if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${SERVICE_NAME}$"; then
    info "Stopping Docker container..."
    docker stop "$SERVICE_NAME" 2>/dev/null || true
    docker rm "$SERVICE_NAME" 2>/dev/null || true
    ok "Docker container removed"
  fi

  if docker images --format '{{.Repository}}' 2>/dev/null | grep -q "^${SERVICE_NAME}$"; then
    info "Removing Docker image..."
    docker rmi "$SERVICE_NAME" 2>/dev/null || true
    ok "Docker image removed"
  fi
fi

# ── Save config if requested ─────────────────────────────────────────────

if $KEEP_CONFIG && [[ -d "$INSTALL_DIR" ]]; then
  BACKUP_DIR="$HOME/.rlm-proxy-backup"
  mkdir -p "$BACKUP_DIR"
  for f in config.json .env; do
    if [[ -f "$INSTALL_DIR/$f" ]]; then
      cp "$INSTALL_DIR/$f" "$BACKUP_DIR/$f"
      info "Saved $f to $BACKUP_DIR/"
    fi
  done
fi

# ── Remove install directory ─────────────────────────────────────────────

if [[ -d "$INSTALL_DIR" ]]; then
  info "Removing $INSTALL_DIR..."
  sudo rm -rf "$INSTALL_DIR"
  ok "Directory removed"
else
  info "Install directory not found at $INSTALL_DIR (already removed?)"
fi

# ── Done ──────────────────────────────────────────────────────────────────

echo ""
ok "RLM Proxy uninstalled"

if $KEEP_CONFIG; then
  info "Config files saved to $BACKUP_DIR/"
  info "To restore later: cp $BACKUP_DIR/* /path/to/new/install/"
fi

echo ""
