# Installation Guide

## Prerequisites

- **Python 3.11+** — check with `python3 --version`
- **At least one LLM provider:**
  - [Ollama](https://ollama.com) running locally or on another machine, and/or
  - An OpenAI-compatible API key (OpenAI, Groq, OpenRouter, Together, etc.)

### Installing Ollama (if using local models)

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve

# Pull a model with strong coding ability (needed for RLM root model)
ollama pull qwen3-coder-next
```

Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

## Installation

### 1. Clone and set up

```bash
git clone <your-repo-url> rlm_proxy
cd rlm_proxy

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with test dependencies
pip install -e ".[test]"
```

### 2. Configure

There are three ways to configure the proxy (all optional — defaults work for a single local Ollama):

#### Option A: Environment variables (simplest)

```bash
cp .env.example .env
# Edit .env with your settings
```

#### Option B: config.json (recommended for multi-provider)

Create `config.json` in the project root:

```json
{
  "providers": [
    {"name": "local", "api_type": "ollama", "url": "http://localhost:11434"}
  ],
  "root_model": "local/qwen3-coder-next"
}
```

#### Option C: Admin UI (runtime)

Start the server first, then configure everything at `http://localhost:8881/admin`.

### 3. Start the server

```bash
source .venv/bin/activate
python main.py
```

The server starts at `http://localhost:8881`. Open `http://localhost:8881/admin` to verify.

### 4. Verify

```bash
# Health check
curl http://localhost:8881/health

# Run integration tests
./test_proxy.sh

# Run unit tests
python -m pytest test_unit.py -v
```

## Quick Start Configurations

### Single local Ollama (zero config)

Just start the server — it connects to `http://localhost:11434` by default:

```bash
python main.py
```

### Two local Ollama machines

```json
{
  "providers": [
    {"name": "local", "api_type": "ollama", "url": "http://localhost:11434"},
    {"name": "gpu-box", "api_type": "ollama", "url": "http://192.168.1.50:11434"}
  ],
  "root_model": "local/qwen3-coder-next"
}
```

### Local Ollama + cloud APIs

```json
{
  "providers": [
    {"name": "local", "api_type": "ollama", "url": "http://localhost:11434"},
    {"name": "groq", "api_type": "openai", "url": "https://api.groq.com/openai/v1", "api_key": "gsk-..."},
    {"name": "openai", "api_type": "openai", "url": "https://api.openai.com/v1", "api_key": "sk-..."},
    {"name": "openrouter", "api_type": "openai", "url": "https://openrouter.ai/api/v1", "api_key": "sk-or-..."}
  ],
  "root_model": "local/qwen3-coder-next"
}
```

Usage:
```bash
# Uses local Ollama
curl http://localhost:8881/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "local/qwen3-coder-next", "messages": [{"role": "user", "content": "Hello"}]}'

# Uses Groq
curl http://localhost:8881/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "groq/llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "Hello"}]}'

# Uses OpenAI
curl http://localhost:8881/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}'
```

### With authentication

Set `RLM_API_KEY` to require a bearer token on all `/v1/*` endpoints:

```bash
RLM_API_KEY=my-secret-key python main.py
```

Clients must include `Authorization: Bearer my-secret-key` in their requests. The `/admin`, `/health`, and `/api/*` (Ollama-native) endpoints remain open.

## Running as a Service

### systemd (Linux)

Create `/etc/systemd/system/rlm-proxy.service`:

```ini
[Unit]
Description=RLM Proxy
After=network.target ollama.service

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/rlm_proxy
Environment=PATH=/path/to/rlm_proxy/.venv/bin:/usr/bin
ExecStart=/path/to/rlm_proxy/.venv/bin/python main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable rlm-proxy
sudo systemctl start rlm-proxy

# Check status
sudo systemctl status rlm-proxy
journalctl -u rlm-proxy -f
```

## Docker

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e .

EXPOSE 8881

CMD ["python", "main.py"]
```

### Build and run

```bash
docker build -t rlm-proxy .
docker run -d \
  --name rlm-proxy \
  -p 8881:8881 \
  -v $(pwd)/config.json:/app/config.json \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  rlm-proxy
```

Note: use `host.docker.internal` (macOS/Windows) or `--network host` (Linux) to reach Ollama on the host machine.

### Docker Compose

```yaml
version: '3.8'
services:
  rlm-proxy:
    build: .
    ports:
      - "8881:8881"
    volumes:
      - ./config.json:/app/config.json
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama

volumes:
  ollama-data:
```

```bash
docker compose up -d
```

## Connecting Apps

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8881/v1",
    api_key="your-key-here",  # or "unused" if no auth
)

response = client.chat.completions.create(
    model="local/qwen3-coder-next",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Ollama Python SDK

```python
import ollama

client = ollama.Client(host="http://localhost:8881")

response = client.chat(
    model="local/qwen3-coder-next",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response["message"]["content"])
```

### curl (OpenAI format)

```bash
curl http://localhost:8881/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-key-here" \
  -d '{"model": "local/qwen3-coder-next", "messages": [{"role": "user", "content": "Hello"}]}'
```

### curl (Ollama format)

```bash
curl http://localhost:8881/api/chat \
  -d '{"model": "local/qwen3-coder-next", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Any app that supports custom OpenAI base URL

Set these environment variables in the app:
```bash
OPENAI_API_BASE=http://localhost:8881/v1
OPENAI_API_KEY=your-key-here
```

### Any app that supports custom Ollama host

```bash
OLLAMA_HOST=http://localhost:8881
```

## Troubleshooting

### Server won't start

```
ModuleNotFoundError: No module named 'uvicorn'
```
You're not in the venv. Run `source .venv/bin/activate` first.

### Can't reach Ollama

```
Could not reach Ollama: [ConnectError]
```
Check that Ollama is running (`ollama serve`) and the URL is correct. If Ollama is on another machine, make sure `OLLAMA_HOST=0.0.0.0` is set on that machine.

### RLM mode fails / model writes bad code

The root model must be good at Python code generation. Try a stronger coding model:
```bash
ollama pull qwen3-coder-next
```
Or set a different root model:
```bash
RLM_ROOT_MODEL=local/qwen3-coder-next python main.py
```

### Provider probe fails

```
Provider 'groq' added but probe failed
```
Check the URL and API key. For OpenAI-compatible providers, the URL should end with `/v1` (e.g. `https://api.groq.com/openai/v1`).

### Circuit breaker opens

```
Circuit breaker is open. Retry after 30s.
```
A provider had too many consecutive failures. It will auto-recover after the timeout. Check the provider's status in the admin UI at `/admin`.

### Config changes don't persist

Changes via the admin UI or `PUT /v1/rlm/config` are in-memory only. Click **"Save to Disk"** in the admin UI or call `POST /v1/rlm/config/save` to persist to `config.json`.
