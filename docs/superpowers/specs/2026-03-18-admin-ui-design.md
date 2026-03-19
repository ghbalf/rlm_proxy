# RLM Proxy: Admin Configuration UI

**Date:** 2026-03-18
**Status:** Approved

## Overview

A self-contained admin dashboard served at `GET /admin` for dynamically managing Ollama hosts, RLM settings, performance knobs, and auth — without restarting the server.

## Architecture

- Single HTML file with embedded CSS + vanilla JS, served by FastAPI via `HTMLResponse`
- Communicates with the proxy via `fetch()` to its own API endpoints
- No build step, no external dependencies
- Sidebar navigation layout with 6 sections

## New API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/admin` | Serves the admin HTML page |
| `GET` | `/v1/rlm/config/all` | Returns all current settings as JSON |
| `PUT` | `/v1/rlm/config` | Update settings in memory (partial dict) |
| `POST` | `/v1/rlm/config/save` | Write current settings to `config.json` |
| `POST` | `/v1/rlm/config/reset` | Reload settings from env vars + `config.json` |
| `POST` | `/v1/rlm/hosts/add` | Add a new Ollama host URL |
| `POST` | `/v1/rlm/hosts/remove` | Remove an Ollama host URL |
| `POST` | `/v1/rlm/hosts/refresh` | Re-probe all hosts for models |

## Sidebar Sections

1. **Hosts** — add/remove/edit host URLs, health status, active requests, model count, refresh button
2. **Models** — deduplicated model list with host mapping (read-only)
3. **RLM Settings** — threshold, max iterations, max sub-calls, exec timeout, compact threshold, depth limit, token budget
4. **Performance** — concurrent sessions, queue size, cache size, concurrent sub-calls, retry config, circuit breaker
5. **Auth** — API key (masked display, editable)
6. **Metrics** — live dashboard with auto-refresh (5s): request counts, active sessions, queue depth, latency percentiles, circuit breaker state

## Config Mutation Flow

- Edits update in-memory settings immediately (session-only by default)
- "Save to Disk" writes `config.json` to project root
- "Reset" reloads from env vars + `config.json`
- On startup: env vars → overlay `config.json` if exists → active config
- UI shows "unsaved changes" indicator when in-memory differs from saved

## Files

| File | Type | Purpose |
|---|---|---|
| `admin.py` | New | Config API endpoints + HTML serving |
| `admin.html` | New | Self-contained admin dashboard |
| `config.py` | Modified | Runtime mutation, save/load `config.json` |
| `dispatcher.py` | Modified | Dynamic host add/remove methods |
| `main.py` | Modified | Mount admin router |
