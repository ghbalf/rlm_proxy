"""OpenAI-compatible request / response schemas."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ── Requests ─────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int | None = None
    stream: bool = False
    # RLM-specific: force RLM mode regardless of length
    force_rlm: bool = False
    # Explicit passthrough bypass
    force_passthrough: bool = False
    # Structured context
    context: dict | list | str | None = None


# ── Responses ────────────────────────────────────────────────────────────────

class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[Choice]
    usage: Usage = Usage()


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "ollama-local"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class ErrorResponse(BaseModel):
    error: dict[str, Any]


# ── Streaming ───────────────────────────────────────────────────────────────

class StreamDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: StreamDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[StreamChoice]


# ── Embeddings ─────────────────────────────────────────────────────────────

class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""
    model: str
    input: str | list[str]
    encoding_format: str = "float"


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int = 0
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str = ""
    usage: Usage = Usage()


# ── Ollama-native embedding request ───────────────────────────────────────

class OllamaEmbedRequest(BaseModel):
    """Ollama-native /api/embed request."""
    model: str
    input: str | list[str]


# ── Model detail ──────────────────────────────────────────────────────────

class ModelDetail(BaseModel):
    """Extended model info (from /api/show)."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "ollama-local"
    parameters: dict[str, Any] | None = None
    template: str | None = None
    modelfile: str | None = None
    license: str | None = None
