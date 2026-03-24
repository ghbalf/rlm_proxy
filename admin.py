"""Admin configuration UI and API endpoints."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from config import get_all_settings, load_settings, save_settings, update_settings
from dispatcher import dispatcher
from providers import ProviderConfig

log = logging.getLogger("rlm_proxy.admin")

router = APIRouter()

_ADMIN_HTML = Path(__file__).parent / "admin.html"


@router.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """Serve the admin dashboard."""
    if not _ADMIN_HTML.exists():
        raise HTTPException(500, detail="admin.html not found")
    return HTMLResponse(_ADMIN_HTML.read_text())


# ── Config API ──────────────────────────────────────────────────────────────


@router.get("/v1/rlm/config/all")
async def config_all():
    """Return all current settings."""
    return get_all_settings()


@router.put("/v1/rlm/config")
async def config_update(body: dict):
    """Update settings in memory (partial dict)."""
    if not body:
        raise HTTPException(400, detail="Empty body")
    errors = update_settings(body)
    result = {"updated": {k: v for k, v in body.items() if k not in errors}}
    if errors:
        result["errors"] = errors
    return result


@router.post("/v1/rlm/config/save")
async def config_save():
    """Write current settings and providers to config.json."""
    path = save_settings(providers=dispatcher.get_provider_configs())
    log.info("Settings saved to %s", path)
    return {"saved": path}


@router.post("/v1/rlm/config/reset")
async def config_reset():
    """Reload settings from env vars + config.json."""
    load_settings()
    log.info("Settings reset from env + config.json")
    return {"status": "reset", "settings": get_all_settings()}


# ── Provider management ───────────────────────────────────────────────────


@router.post("/v1/rlm/providers/add")
async def provider_add(body: dict):
    """Add a new provider (ollama or openai)."""
    name = body.get("name", "").strip()
    api_type = body.get("api_type", "ollama").strip()
    url = body.get("url", "").strip()
    api_key = body.get("api_key", "").strip()
    if not name:
        raise HTTPException(400, detail="name is required")
    if not url:
        raise HTTPException(400, detail="url is required")
    params = body.get("params", {})
    config = ProviderConfig(name=name, api_type=api_type, url=url, api_key=api_key, params=params)
    result = await dispatcher.add_provider(config)
    if "error" in result:
        raise HTTPException(400, detail=result["error"])
    return result


@router.post("/v1/rlm/providers/remove")
async def provider_remove(body: dict):
    """Remove a provider by name."""
    name = body.get("name", body.get("url", "")).strip()
    if not name:
        raise HTTPException(400, detail="name is required")
    result = await dispatcher.remove_provider(name)
    if "error" in result:
        raise HTTPException(400, detail=result["error"])
    return result


@router.put("/v1/rlm/providers/params")
async def provider_params_update(body: dict):
    """Update params for an existing provider. Body: {"name": "...", "params": {...}}"""
    name = body.get("name", "").strip()
    params = body.get("params")
    if not name:
        raise HTTPException(400, detail="name is required")
    if params is None:
        raise HTTPException(400, detail="params is required")
    ps = dispatcher._providers.get(name)
    if ps is None:
        raise HTTPException(404, detail=f"Provider '{name}' not found")
    ps.provider.config.params = params
    return {"name": name, "params": params}


# Backward compat — old host endpoints delegate to provider endpoints
@router.post("/v1/rlm/hosts/add")
async def host_add(body: dict):
    """Add a new Ollama host (backward compat)."""
    if "name" not in body:
        body["name"] = f"ollama-{body.get('url', 'new').split('/')[-1].split(':')[0]}"
    body.setdefault("api_type", "ollama")
    return await provider_add(body)


@router.post("/v1/rlm/hosts/remove")
async def host_remove(body: dict):
    """Remove a host (backward compat)."""
    return await provider_remove(body)


@router.post("/v1/rlm/hosts/refresh")
async def host_refresh():
    """Re-probe all providers for available models."""
    await dispatcher.refresh()
    return dispatcher.snapshot()
