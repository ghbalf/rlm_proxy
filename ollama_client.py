"""Unified LLM client with multi-provider dispatch, retry, and circuit breaker."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from circuit_breaker import breaker, CircuitOpenError
from config import settings
from dispatcher import dispatcher

log = logging.getLogger("rlm_proxy.ollama")

_RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})


async def _with_retry(fn, *args, **kwargs):
    """Call *fn* with exponential-backoff retry on transient errors."""
    await breaker.check()
    last_exc = None
    for attempt in range(settings.ollama_max_retries):
        try:
            result = await fn(*args, **kwargs)
            await breaker.record_success()
            return result
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout) as exc:
            last_exc = exc
            if attempt < settings.ollama_max_retries - 1:
                delay = settings.ollama_retry_base_delay * (2 ** attempt)
                log.warning("Call failed (attempt %d/%d): %s — retrying in %.1fs",
                            attempt + 1, settings.ollama_max_retries, exc, delay)
                await asyncio.sleep(delay)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in _RETRYABLE_STATUS_CODES:
                last_exc = exc
                if attempt < settings.ollama_max_retries - 1:
                    delay = settings.ollama_retry_base_delay * (2 ** attempt)
                    log.warning("Server returned %d (attempt %d/%d) — retrying in %.1fs",
                                exc.response.status_code, attempt + 1, settings.ollama_max_retries, delay)
                    await asyncio.sleep(delay)
            else:
                await breaker.record_failure()
                raise
    await breaker.record_failure()
    raise last_exc


def _pick(model: str, pin_url: str | None = None) -> tuple:
    """Pick a provider for a model, optionally pinned to a URL.

    Returns (provider, actual_model_name) with any provider/ prefix stripped.
    """
    return dispatcher.pick_provider(model, pin_url=pin_url)


async def close_client() -> None:
    """Close all provider clients."""
    await dispatcher.stop()


# ── Chat ─────────────────────────────────────────────────────────────────


async def chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stream: bool = False,
    provider_url: str | None = None,
    extra_params: dict | None = None,
) -> dict[str, Any]:
    """Chat completion via the best available provider."""

    async def _do():
        provider, actual_model = _pick(model, provider_url)
        dispatcher.acquire(provider.name)
        try:
            return await provider.chat(actual_model, messages,
                                       temperature=temperature, top_p=top_p,
                                       max_tokens=max_tokens,
                                       extra_params=extra_params)
        except Exception:
            dispatcher.record_error(provider.name)
            raise
        finally:
            dispatcher.release(provider.name)

    return await _with_retry(_do)


async def chat_stream(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    provider_url: str | None = None,
    extra_params: dict | None = None,
):
    """Streaming chat via the best available provider."""
    await breaker.check()
    last_exc = None
    for attempt in range(settings.ollama_max_retries):
        provider, actual_model = _pick(model, provider_url)
        try:
            dispatcher.acquire(provider.name)
            try:
                async for chunk in provider.chat_stream(
                    actual_model, messages,
                    temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                    extra_params=extra_params,
                ):
                    yield chunk
                await breaker.record_success()
                return
            finally:
                dispatcher.release(provider.name)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            dispatcher.record_error(provider.name)
            last_exc = exc
            if attempt < settings.ollama_max_retries - 1:
                delay = settings.ollama_retry_base_delay * (2 ** attempt)
                log.warning("Stream failed on %s (attempt %d/%d) — retrying in %.1fs",
                            provider.name, attempt + 1, settings.ollama_max_retries, delay)
                await asyncio.sleep(delay)
        except httpx.HTTPStatusError as exc:
            dispatcher.record_error(provider.url)
            if exc.response.status_code in _RETRYABLE_STATUS_CODES:
                last_exc = exc
                if attempt < settings.ollama_max_retries - 1:
                    delay = settings.ollama_retry_base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
            else:
                await breaker.record_failure()
                raise
    await breaker.record_failure()
    raise last_exc


async def chat_batch(
    model: str,
    prompts_with_messages: list[list[dict[str, str]]],
    *,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    max_concurrent: int | None = None,
) -> list[dict[str, Any]]:
    """Multiple chat requests concurrently, dispatched across providers."""
    limit = max_concurrent or settings.max_concurrent_sub_calls
    sem = asyncio.Semaphore(limit)

    async def _one(messages):
        async with sem:
            return await chat(model=model, messages=messages,
                              temperature=temperature, max_tokens=max_tokens)

    return await asyncio.gather(*[_one(msgs) for msgs in prompts_with_messages])


# ── Embeddings ───────────────────────────────────────────────────────────


async def embed(model: str, input_texts: list[str]) -> dict[str, Any]:
    """Generate embeddings via the best available provider."""

    async def _do():
        provider, actual_model = _pick(model)
        dispatcher.acquire(provider.name)
        try:
            return await provider.embed(actual_model, input_texts)
        except Exception:
            dispatcher.record_error(provider.name)
            raise
        finally:
            dispatcher.release(provider.name)

    return await _with_retry(_do)


# ── Generate (Ollama-specific) ───────────────────────────────────────────


async def generate(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> str:
    """Ollama generate endpoint. Falls back to chat for non-Ollama providers."""

    async def _do():
        provider, actual_model = _pick(model)
        dispatcher.acquire(provider.name)
        try:
            if hasattr(provider, 'generate'):
                return await provider.generate(actual_model, prompt,
                                                system=system, temperature=temperature,
                                                max_tokens=max_tokens)
            # Fallback for non-Ollama providers: use chat
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = await provider.chat(actual_model, messages, temperature=temperature, max_tokens=max_tokens)
            return resp.get("message", {}).get("content", "")
        except Exception:
            dispatcher.record_error(provider.name)
            raise
        finally:
            dispatcher.release(provider.name)

    return await _with_retry(_do)


# ── Model info ───────────────────────────────────────────────────────────


async def list_models() -> list[dict[str, Any]]:
    """List models from all providers (deduplicated)."""
    if dispatcher.is_multi_host:
        return [{"name": m["name"]} for m in dispatcher.all_models()]

    async def _do():
        provider, _ = _pick("")
        return await provider.list_models()

    return await _with_retry(_do)


async def show_model(model: str) -> dict[str, Any]:
    """Get model details from the appropriate provider."""

    async def _do():
        provider, actual_model = _pick(model)
        return await provider.show_model(actual_model)

    return await _with_retry(_do)
