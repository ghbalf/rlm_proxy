"""Provider abstraction for different LLM backends.

Each provider handles its own API format and returns Ollama-format
responses so that calling code doesn't need to change.

Designed for extensibility — adding Anthropic later means adding
one new class that implements the Provider interface.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

log = logging.getLogger("rlm_proxy.providers")


@dataclass
class ProviderConfig:
    """Configuration for a single provider instance.

    ``params`` lets you override, add, or suppress any API parameter
    on a per-provider basis::

        {"temperature": 1}       → always send temperature=1
        {"temperature": null}    → never send temperature
        {"top_k": 50}            → add top_k=50 to every request
    """
    name: str  # user-chosen label, used as prefix in "name/model" routing
    api_type: str  # "ollama" or "openai"
    url: str
    api_key: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.name

    def to_dict(self) -> dict:
        d = {"name": self.name, "api_type": self.api_type, "url": self.url}
        if self.api_key:
            d["api_key"] = self.api_key
        if self.params:
            d["params"] = self.params
        return d


def parse_model_string(model: str) -> tuple[str | None, str]:
    """Parse 'provider/model' into (provider_name, model_name).

    If no '/' prefix, returns (None, model) for auto-dispatch.
    """
    if "/" in model:
        provider_name, _, model_name = model.partition("/")
        return provider_name, model_name
    return None, model


class Provider(ABC):
    """Base class for LLM API providers."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client: httpx.AsyncClient | None = None

    @property
    def id(self) -> str:
        return self.config.name

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def url(self) -> str:
        return self.config.url

    @property
    def api_type(self) -> str:
        return self.config.api_type

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.config.url,
                timeout=httpx.Timeout(300.0, connect=10.0),
                headers=headers,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @abstractmethod
    async def chat(
        self, model: str, messages: list[dict], *,
        temperature: float = 0.7, top_p: float = 0.9,
        max_tokens: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Chat completion. Returns Ollama-format response."""
        ...

    @abstractmethod
    async def chat_stream(
        self, model: str, messages: list[dict], *,
        temperature: float = 0.7, top_p: float = 0.9,
        max_tokens: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Streaming chat. Yields content string chunks."""
        ...

    @abstractmethod
    async def embed(self, model: str, input_texts: list[str]) -> dict[str, Any]:
        """Generate embeddings. Returns Ollama-format response."""
        ...

    @abstractmethod
    async def list_models(self) -> list[dict[str, Any]]:
        """List available models. Returns list of {"name": "..."} dicts."""
        ...

    @abstractmethod
    async def show_model(self, model: str) -> dict[str, Any]:
        """Get model details."""
        ...


class OllamaProvider(Provider):
    """Provider for Ollama API (native format)."""

    async def chat(self, model, messages, *, temperature=0.7, top_p=0.9, max_tokens=None, extra_params=None):
        client = await self.get_client()
        payload: dict[str, Any] = {
            "model": model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "top_p": top_p},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        log.debug("Ollama chat → %s  model=%s", self.url, model)
        resp = await client.post("/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def chat_stream(self, model, messages, *, temperature=0.7, top_p=0.9, max_tokens=None, extra_params=None):
        client = await self.get_client()
        payload: dict[str, Any] = {
            "model": model, "messages": messages, "stream": True,
            "options": {"temperature": temperature, "top_p": top_p},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        async with client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done", False):
                    return

    async def embed(self, model, input_texts):
        client = await self.get_client()
        payload = {"model": model, "input": input_texts}
        resp = await client.post("/api/embed", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def list_models(self):
        client = await self.get_client()
        resp = await client.get("/api/tags")
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return [{"name": m.get("name", m.get("model", "unknown"))} for m in models]

    async def show_model(self, model):
        client = await self.get_client()
        resp = await client.post("/api/show", json={"model": model})
        resp.raise_for_status()
        return resp.json()

    async def generate(self, model, prompt, *, system=None, temperature=0.7, max_tokens=None):
        """Ollama-specific generate endpoint."""
        client = await self.get_client()
        payload: dict[str, Any] = {
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        resp = await client.post("/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "")


class OpenAIProvider(Provider):
    """Provider for OpenAI-compatible APIs.

    Translates to/from OpenAI format, returns Ollama-format responses
    so calling code doesn't change.
    """

    # Matches "invalid <param_name>" in API error messages.
    _PARAM_REJECTED_RE = re.compile(r"invalid (\w+)", re.IGNORECASE)

    def _build_payload(
        self, model: str, messages: list[dict], *,
        temperature: float, top_p: float,
        max_tokens: int | None, stream: bool = False,
        drop_params: set[str] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if stream:
            payload["stream"] = True
        # Defaults from caller
        payload["temperature"] = temperature
        payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        # Forward extra fields from the client request (tools, tool_choice, etc.)
        if extra_params:
            payload.update(extra_params)
        # Apply provider params: override values, add new ones, null = suppress
        for key, value in self.config.params.items():
            if value is None:
                payload.pop(key, None)
            else:
                payload[key] = value
        # Drop params that were rejected on a previous attempt
        for key in (drop_params or set()):
            payload.pop(key, None)
        return payload

    def _rejected_param(self, error_body: str, payload: dict) -> str | None:
        """Parse a 400 error to find which payload param was rejected."""
        m = self._PARAM_REJECTED_RE.search(error_body)
        if m:
            param = m.group(1)
            # Only act on it if we actually sent that param
            if param in payload:
                return param
        return None

    async def chat(self, model, messages, *, temperature=0.7, top_p=0.9, max_tokens=None, extra_params=None):
        client = await self.get_client()
        drop_params: set[str] = set()

        for attempt in range(3):
            payload = self._build_payload(
                model, messages, temperature=temperature, top_p=top_p,
                max_tokens=max_tokens, drop_params=drop_params,
                extra_params=extra_params,
            )
            log.debug("OpenAI chat → %s  model=%s  attempt=%d", self.url, model, attempt + 1)
            resp = await client.post("/chat/completions", json=payload)

            if resp.status_code == 400:
                rejected = self._rejected_param(resp.text, payload)
                if rejected:
                    log.warning("Model %s rejected %s — retrying without it", model, rejected)
                    drop_params.add(rejected)
                    continue
                log.error("OpenAI API error 400: %s", resp.text[:500])

            if resp.status_code >= 400:
                log.error("OpenAI API error %d: %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()

            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            return {
                "model": data.get("model", model),
                "message": {"role": "assistant", "content": content},
                "done": True,
                "created_at": data.get("created", ""),
                "eval_count": usage.get("completion_tokens", 0),
                "prompt_eval_count": usage.get("prompt_tokens", 0),
            }

    async def chat_stream(self, model, messages, *, temperature=0.7, top_p=0.9, max_tokens=None, extra_params=None):
        client = await self.get_client()
        drop_params: set[str] = set()

        for attempt in range(3):
            payload = self._build_payload(
                model, messages, temperature=temperature, top_p=top_p,
                max_tokens=max_tokens, stream=True, drop_params=drop_params,
                extra_params=extra_params,
            )
            log.debug("OpenAI stream → %s  model=%s  attempt=%d", self.url, model, attempt + 1)

            # Non-streaming probe: detect param rejection before opening SSE
            probe = {k: v for k, v in payload.items() if k != "stream"}
            probe["max_tokens"] = 1
            resp = await client.post("/chat/completions", json=probe)
            if resp.status_code == 400:
                rejected = self._rejected_param(resp.text, payload)
                if rejected:
                    log.warning("Model %s rejected %s — retrying without it", model, rejected)
                    drop_params.add(rejected)
                    continue

            # Probe passed — do the real streaming request
            async with client.stream("POST", "/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if content:
                        yield content
                return

    async def embed(self, model, input_texts):
        client = await self.get_client()
        payload = {"model": model, "input": input_texts}
        resp = await client.post("/embeddings", json=payload)
        resp.raise_for_status()
        data = resp.json()
        # Translate OpenAI → Ollama format
        embeddings = [item["embedding"] for item in data.get("data", [])]
        return {"model": data.get("model", model), "embeddings": embeddings}

    async def list_models(self):
        client = await self.get_client()
        resp = await client.get("/models")
        resp.raise_for_status()
        data = resp.json()
        return [{"name": m.get("id", "unknown")} for m in data.get("data", [])]

    async def show_model(self, model):
        # OpenAI doesn't have a detailed show — return basic info
        client = await self.get_client()
        resp = await client.get(f"/models/{model}")
        resp.raise_for_status()
        data = resp.json()
        return {
            "model": model,
            "details": {"family": data.get("owned_by", ""), "format": "openai"},
        }


# ── Factory ──────────────────────────────────────────────────────────────

API_TYPES = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
}


def create_provider(config: ProviderConfig) -> Provider:
    """Create a provider instance from config."""
    cls = API_TYPES.get(config.api_type)
    if cls is None:
        raise ValueError(f"Unknown api_type: {config.api_type}. Available: {list(API_TYPES.keys())}")
    return cls(config)
