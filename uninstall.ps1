# ──────────────────────────────────────────────────────────────────────────
# RLM Proxy Uninstaller (Windows)
# Delegates to install.ps1 -Uninstall
# ──────────────────────────────────────────────────────────────────────────
param([string]$Dir = "$env:LOCALAPPDATA\rlm-proxy")
& "$PSScriptRoot\install.ps1" -Uninstall -Dir $Dir
