"""RLM Proxy — OpenAI-compatible API that wraps Ollama with a Recursive Language Model scaffold."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import re
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

import ollama_client
from admin import router as admin_router
from auth import APIKeyAuthMiddleware
from circuit_breaker import CircuitOpenError, breaker
from config import settings
from dispatcher import dispatcher
from metrics import collector as metrics
from request_queue import queue as rlm_queue
import json as _json

from fastapi.responses import StreamingResponse

from rlm_engine import RLMResult, passthrough_chat, run_rlm, run_rlm_streaming
from schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
    ModelDetail,
    ModelInfo,
    ModelList,
    StreamChoice,
    StreamDelta,
    Usage,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

from log_buffer import buffer_handler
buffer_handler.setFormatter(logging.Formatter("%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S"))
logging.getLogger().addHandler(buffer_handler)

log = logging.getLogger("rlm_proxy")

# URL pattern to strip from error messages (prevents leaking internal infra)
_URL_RE = re.compile(r"https?://[^\s'\"]+")


def _sanitize_error(exc: Exception) -> str:
    """Return a user-facing error message with internal URLs stripped."""
    msg = str(exc)
    # httpx errors often contain "for url '...'" — strip the URL
    sanitized = _URL_RE.sub("<upstream>", msg)
    return sanitized


async def _parse_json_body(request: Request) -> dict:
    """Parse JSON body tolerantly — works even without Content-Type header.

    Real Ollama accepts JSON without Content-Type; FastAPI's dict parsing requires it.
    This helper reads the raw body and parses it as JSON regardless of headers.
    """
    body = await request.body()
    if not body:
        raise HTTPException(400, detail="Request body is empty")
    try:
        return _json.loads(body)
    except _json.JSONDecodeError as exc:
        raise HTTPException(400, detail=f"Invalid JSON: {exc}") from exc


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if dispatcher.is_multi_host:
        log.info("RLM Proxy starting — %d Ollama hosts configured", dispatcher.host_count)
    else:
        log.info("RLM Proxy starting — Ollama @ %s", settings.ollama_base_url)
    log.info("Root model: %s  |  Sub model: from request", settings.root_model)
    log.info("RLM threshold: %s chars  |  Max iterations: %d", f"{settings.rlm_threshold_chars:,}", settings.max_iterations)
    await dispatcher.start_refresh_loop()
    yield
    await ollama_client.close_client()
    log.info("RLM Proxy shut down.")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RLM Proxy",
    description="Recursive Language Model proxy for local Ollama models. "
    "Implements the RLM inference paradigm (Zhang, Kraska, Khattab 2025) "
    "to extend effective context length of local models.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(APIKeyAuthMiddleware)
app.include_router(admin_router)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _raise_if_model_not_found(exc: Exception, model: str) -> None:
    """If the upstream returned 404, raise a clean 'model not found' error."""
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
        raise HTTPException(404, detail=f"Model '{model}' not found") from exc


def _total_content_length(messages: list[ChatMessage]) -> int:
    return sum(len(m.content) for m in messages)


def _should_use_rlm(req: ChatCompletionRequest) -> bool:
    # Explicit overrides take priority
    if getattr(req, 'force_passthrough', False):
        return False
    if req.force_rlm:
        return True
    if not settings.passthrough_short:
        return True

    total_chars = _total_content_length(req.messages)

    # Token-based threshold
    estimated_tokens = total_chars / settings.token_estimate_ratio
    if estimated_tokens >= settings.rlm_threshold_chars / settings.token_estimate_ratio:
        return True

    # Multi-message heuristic: if there are multiple substantial user messages
    # with context, prefer RLM even below threshold
    user_msgs = [m for m in req.messages if m.role == "user"]
    if len(user_msgs) > 2 and total_chars > settings.rlm_threshold_chars * 0.5:
        return True

    # Check if structured context was provided
    if getattr(req, 'context', None) is not None:
        return True

    return False


def _extract_query_and_context(messages: list[ChatMessage]) -> tuple[str, str]:
    """Split messages into a query (last user message) and context (everything else)."""
    # Find last user message as the query
    query = ""
    context_parts: list[str] = []

    for i, msg in enumerate(messages):
        if msg.role == "user" and i == len(messages) - 1:
            query = msg.content
        elif msg.role == "system":
            context_parts.append(f"[System instruction]: {msg.content}")
        elif msg.role == "user":
            context_parts.append(f"[User message]: {msg.content}")
        elif msg.role == "assistant":
            context_parts.append(f"[Assistant response]: {msg.content}")

    # If last message isn't user, treat entire conversation as context
    if not query:
        query = messages[-1].content if messages else ""

    context = "\n\n".join(context_parts) if context_parts else query
    # If the query itself is very long, treat most of it as context
    if len(query) > settings.rlm_threshold_chars and not context_parts:
        # Heuristic: first line(s) up to a delimiter are the query, rest is context
        lines = query.split("\n", 1)
        if len(lines) == 2 and len(lines[0]) < 2000:
            query = lines[0]
            context = lines[1]

    return query, context


def _sse_chunk(model: str, content: str, finish_reason: str | None = None) -> str:
    chunk = ChatCompletionChunk(
        model=model,
        choices=[StreamChoice(delta=StreamDelta(content=content), finish_reason=finish_reason)],
    )
    return f"data: {chunk.model_dump_json()}\n\n"


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    resp = {"status": "ok", "mode": "rlm_proxy"}
    if dispatcher.is_multi_host:
        resp["ollama_hosts"] = dispatcher.host_count
        resp["healthy_hosts"] = sum(1 for h in dispatcher._hosts.values() if h.healthy)
    else:
        resp["ollama_url"] = settings.ollama_base_url
    return resp


@app.get("/v1/models")
async def list_models():
    try:
        models = await ollama_client.list_models()
    except Exception as exc:
        raise HTTPException(502, detail=f"Could not reach Ollama: {_sanitize_error(exc)}") from exc

    return ModelList(
        data=[
            ModelInfo(id=m.get("name", m.get("model", "unknown")))
            for m in models
        ]
    )


# ── Embeddings ──────────────────────────────────────────────────────────────

@app.post("/v1/embeddings")
async def openai_embeddings(req: EmbeddingRequest):
    """OpenAI-compatible embeddings endpoint."""
    input_texts = [req.input] if isinstance(req.input, str) else req.input
    try:
        result = await ollama_client.embed(model=req.model, input_texts=input_texts)
    except Exception as exc:
        log.error("Embeddings failed: %s", exc)
        raise HTTPException(502, detail=_sanitize_error(exc)) from exc

    embeddings = result.get("embeddings", [])
    return EmbeddingResponse(
        model=req.model,
        data=[
            EmbeddingData(index=i, embedding=emb)
            for i, emb in enumerate(embeddings)
        ],
        usage=Usage(
            prompt_tokens=sum(len(t) for t in input_texts) // 4,
            total_tokens=sum(len(t) for t in input_texts) // 4,
        ),
    )


@app.post("/api/embed")
async def ollama_embed(request: Request):
    """Ollama-native /api/embed passthrough with dispatch."""
    req = await _parse_json_body(request)
    model = req.get("model", "")
    if not model:
        raise HTTPException(400, detail="model is required")
    raw_input = req.get("input", "")
    input_texts = [raw_input] if isinstance(raw_input, str) else raw_input
    try:
        result = await ollama_client.embed(model=model, input_texts=input_texts)
    except Exception as exc:
        log.error("Ollama embed failed: %s", exc)
        raise HTTPException(502, detail=_sanitize_error(exc)) from exc
    return result


# ── Model detail ────────────────────────────────────────────────────────────

@app.get("/v1/models/{model_id:path}")
async def openai_model_detail(model_id: str):
    """OpenAI-compatible model detail endpoint."""
    try:
        info = await ollama_client.show_model(model_id)
    except Exception as exc:
        log.error("Model show failed: %s", exc)
        raise HTTPException(502, detail=_sanitize_error(exc)) from exc

    details = info.get("details", {})
    model_info = info.get("model_info", {})
    parameters = {}
    if details:
        parameters["family"] = details.get("family", "")
        parameters["parameter_size"] = details.get("parameter_size", "")
        parameters["quantization_level"] = details.get("quantization_level", "")
        parameters["format"] = details.get("format", "")
    if model_info:
        for k, v in model_info.items():
            if "context" in k.lower():
                parameters["context_length"] = v

    return ModelDetail(
        id=model_id,
        parameters=parameters or None,
        template=info.get("template"),
        license=info.get("license"),
    )


@app.post("/api/show")
async def ollama_show(request: Request):
    """Ollama-native /api/show passthrough with dispatch."""
    req = await _parse_json_body(request)
    model = req.get("model", "")
    if not model:
        raise HTTPException(400, detail="model is required")
    try:
        result = await ollama_client.show_model(model)
    except Exception as exc:
        log.error("Ollama show failed: %s", exc)
        raise HTTPException(502, detail=_sanitize_error(exc)) from exc
    return result


# ── Ollama-native chat + generate passthroughs ─────────────────────────────

@app.post("/api/chat")
async def ollama_chat(request: Request):
    """Ollama-native /api/chat passthrough with dispatch."""
    req = await _parse_json_body(request)
    model = req.get("model", "")
    messages = req.get("messages", [])
    if not model or not messages:
        raise HTTPException(400, detail="model and messages are required")
    try:
        result = await ollama_client.chat(
            model=model,
            messages=messages,
            temperature=req.get("options", {}).get("temperature", 0.7),
            top_p=req.get("options", {}).get("top_p", 0.9),
            max_tokens=req.get("options", {}).get("num_predict"),
        )
    except Exception as exc:
        _raise_if_model_not_found(exc, model)
        log.error("Ollama chat failed: %s", exc)
        raise HTTPException(502, detail=_sanitize_error(exc)) from exc
    return result


@app.post("/api/generate")
async def ollama_generate(request: Request):
    """Ollama-native /api/generate passthrough with dispatch."""
    req = await _parse_json_body(request)
    model = req.get("model", "")
    prompt = req.get("prompt", "")
    if not model:
        raise HTTPException(400, detail="model is required")
    try:
        result = await ollama_client.generate(
            model=model,
            prompt=prompt,
            system=req.get("system"),
            temperature=req.get("options", {}).get("temperature", 0.7),
            max_tokens=req.get("options", {}).get("num_predict"),
        )
    except Exception as exc:
        log.error("Ollama generate failed: %s", exc)
        raise HTTPException(502, detail=_sanitize_error(exc)) from exc
    return {"model": model, "response": result}


@app.get("/api/tags")
async def ollama_tags():
    """Ollama-native /api/tags passthrough with dispatch."""
    try:
        models = await ollama_client.list_models()
    except Exception as exc:
        raise HTTPException(502, detail=_sanitize_error(exc)) from exc
    return {"models": models}


# ── Dispatch info ───────────────────────────────────────────────────────────

@app.get("/v1/rlm/dispatch")
async def dispatch_info():
    """Show the dispatcher routing table and host status."""
    return dispatcher.snapshot()


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        raise HTTPException(400, detail="messages cannot be empty")
    if not req.model:
        raise HTTPException(400, detail="model is required")

    use_rlm = _should_use_rlm(req)
    total_chars = _total_content_length(req.messages)
    log.info(
        "POST /v1/chat/completions  model=%s  total_chars=%s  rlm=%s  stream=%s",
        req.model, f"{total_chars:,}", use_rlm, req.stream,
    )

    metrics.record_request(rlm=use_rlm, stream=bool(req.stream))

    if req.stream and not use_rlm:
        async def _stream_passthrough():
            ollama_msgs = [{"role": m.role, "content": m.content} for m in req.messages]
            async for chunk in ollama_client.chat_stream(
                model=req.model, messages=ollama_msgs,
                temperature=req.temperature, top_p=req.top_p,
                max_tokens=req.max_tokens,
            ):
                yield _sse_chunk(req.model, chunk)
            yield _sse_chunk(req.model, "", finish_reason="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _stream_passthrough(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if req.stream and use_rlm:
        try:
            wait_time = await rlm_queue.acquire()
        except RuntimeError as exc:
            metrics.record_error()
            raise HTTPException(429, detail=_sanitize_error(exc), headers={"Retry-After": "30"}) from exc

        async def _stream_rlm():
            try:
                query, context = _extract_query_and_context(req.messages)
                async for chunk in run_rlm_streaming(
                    query=query, context=context,
                    model=req.model, temperature=req.temperature,
                ):
                    yield _sse_chunk(req.model, chunk)
                yield _sse_chunk(req.model, "", finish_reason="stop")
                yield "data: [DONE]\n\n"
            finally:
                rlm_queue.release()

        return StreamingResponse(
            _stream_rlm(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if not use_rlm:
        # ── Direct passthrough ──
        ollama_msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        t_pass = time.perf_counter()
        try:
            resp = await passthrough_chat(
                ollama_msgs,
                model=req.model,
                temperature=req.temperature,
                top_p=req.top_p,
                max_tokens=req.max_tokens,
            )
        except CircuitOpenError as exc:
            metrics.record_error()
            raise HTTPException(
                status_code=503,
                detail=_sanitize_error(exc),
                headers={"Retry-After": str(int(exc.retry_after))},
            ) from exc
        except Exception as exc:
            _raise_if_model_not_found(exc, req.model)
            metrics.record_error()
            log.error("Ollama passthrough failed: %s", exc)
            raise HTTPException(502, detail=_sanitize_error(exc)) from exc
        metrics.record_passthrough_duration(time.perf_counter() - t_pass)

        answer = resp.get("message", {}).get("content", "")
        prompt_tokens = resp.get("prompt_eval_count", 0) or 0
        completion_tokens = resp.get("eval_count", 0) or 0

        return ChatCompletionResponse(
            model=req.model,
            choices=[Choice(message=ChatMessage(role="assistant", content=answer))],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    # ── RLM mode ──
    query, text_context = _extract_query_and_context(req.messages)

    # Merge explicit context with message-extracted context
    if getattr(req, 'context', None) is not None:
        if isinstance(req.context, dict):
            context = {**req.context, "_conversation": text_context} if text_context else req.context
        elif isinstance(req.context, list):
            context = req.context if not text_context else req.context + [text_context]
        else:
            # String context — concatenate
            context = f"{req.context}\n\n{text_context}" if text_context else req.context
    else:
        context = text_context

    context_desc = f"{len(context):,}" if isinstance(context, str) else f"{type(context).__name__} with {len(context)} entries"
    log.info("RLM activated — query len=%d  context=%s", len(query), context_desc)

    try:
        wait_time = await rlm_queue.acquire()
    except RuntimeError as exc:
        metrics.record_error()
        raise HTTPException(
            status_code=429,
            detail=_sanitize_error(exc),
            headers={"Retry-After": "30"},
        ) from exc

    metrics.session_start()
    try:
        result: RLMResult = await run_rlm(
            query=query,
            context=context,
            model=req.model,
            temperature=req.temperature,
        )
    except CircuitOpenError as exc:
        metrics.record_error()
        metrics.session_end()
        raise HTTPException(
            status_code=503,
            detail=_sanitize_error(exc),
            headers={"Retry-After": str(int(exc.retry_after))},
        ) from exc
    except Exception as exc:
        metrics.record_error()
        metrics.session_end()
        log.error("RLM engine failed: %s", exc, exc_info=True)
        raise HTTPException(500, detail=f"RLM engine error: {_sanitize_error(exc)}") from exc
    finally:
        rlm_queue.release()

    metrics.record_rlm_result(
        iterations=result.iterations,
        sub_calls=result.sub_calls,
        cache_hits=result.cache_hits,
        duration=result.elapsed_seconds,
    )
    metrics.session_end()

    log.info(
        "RLM complete — iterations=%d  sub_calls=%d  elapsed=%.1fs",
        result.iterations, result.sub_calls, result.elapsed_seconds,
    )

    return ChatCompletionResponse(
        model=req.model,
        choices=[Choice(message=ChatMessage(role="assistant", content=result.answer))],
        usage=Usage(
            prompt_tokens=total_chars // 4,  # rough estimate
            completion_tokens=len(result.answer) // 4,
            total_tokens=(total_chars + len(result.answer)) // 4,
        ),
    )


# ── RLM status endpoint ─────────────────────────────────────────────────────

@app.get("/v1/rlm/metrics")
async def rlm_metrics():
    data = metrics.snapshot()
    data["queue"] = rlm_queue.snapshot()
    data["circuit_breaker"] = breaker.snapshot()
    data["dispatcher"] = dispatcher.snapshot()
    return data


@app.get("/v1/rlm/config")
async def rlm_config():
    return {
        "root_model": settings.root_model,
        "rlm_threshold_chars": settings.rlm_threshold_chars,
        "max_iterations": settings.max_iterations,
        "max_sub_calls": settings.max_sub_calls,
        "sub_call_max_chars": settings.sub_call_max_chars,
        "passthrough_short": settings.passthrough_short,
    }


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
