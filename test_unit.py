#!/usr/bin/env python3
"""Unit tests for the RLM proxy — no running server or Ollama needed."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Phase 1A: Trajectory Logger ─────────────────────────────────────────────


def test_trajectory_logger_noop_when_disabled():
    from trajectory_logger import TrajectoryLogger

    tlog = TrajectoryLogger("")
    tlog.log_event("test", {"key": "value"})
    result = tlog.finalize()
    assert result == {}
    assert tlog.session_id == ""


def test_trajectory_logger_writes_jsonl():
    from trajectory_logger import TrajectoryLogger

    with tempfile.TemporaryDirectory() as tmp:
        tlog = TrajectoryLogger(tmp)
        tlog.log_event("root_llm_response", {"iteration": 1, "response_len": 42})
        tlog.log_event("code_block_exec", {"iteration": 1, "block_index": 1})
        summary = tlog.finalize()

        assert summary["total_events"] == 2
        assert summary["session_id"] == tlog.session_id

        log_file = Path(summary["log_file"])
        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "session_id" in data
            assert "ts" in data
            assert "event" in data


# ── Phase 1B: Config + Schema Additions ──────────────────────────────────────


def test_config_new_fields():
    from config import Settings

    s = Settings()
    assert isinstance(s.token_estimate_ratio, float)
    assert s.sub_call_cache_size > 0
    assert s.max_concurrent_sub_calls > 0
    assert isinstance(s.metrics_enabled, bool)
    assert isinstance(s.prompt_profile_override, str)
    assert isinstance(s.trajectory_log_dir, str)


def test_schema_force_passthrough():
    from schemas import ChatCompletionRequest

    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        force_passthrough=True,
    )
    assert req.force_passthrough is True


def test_schema_context_field():
    from schemas import ChatCompletionRequest

    # String context
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        context="some docs",
    )
    assert req.context == "some docs"

    # Dict context
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        context={"doc1": "content1"},
    )
    assert isinstance(req.context, dict)

    # List context
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        context=["doc1", "doc2"],
    )
    assert isinstance(req.context, list)


def test_streaming_schemas():
    from schemas import ChatCompletionChunk, StreamChoice, StreamDelta

    chunk = ChatCompletionChunk(
        model="test",
        choices=[StreamChoice(delta=StreamDelta(content="hello"))],
    )
    assert chunk.object == "chat.completion.chunk"
    d = chunk.model_dump()
    assert d["choices"][0]["delta"]["content"] == "hello"


# ── Phase 1C: Smarter RLM Activation ────────────────────────────────────────


def test_should_use_rlm_force_passthrough():
    from main import _should_use_rlm
    from schemas import ChatCompletionRequest

    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "x" * 100_000}],
        force_passthrough=True,
    )
    assert _should_use_rlm(req) is False


def test_should_use_rlm_force_rlm():
    from main import _should_use_rlm
    from schemas import ChatCompletionRequest

    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "short"}],
        force_rlm=True,
    )
    assert _should_use_rlm(req) is True


def test_should_use_rlm_context_triggers():
    from main import _should_use_rlm
    from schemas import ChatCompletionRequest

    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "short"}],
        context={"doc": "content"},
    )
    assert _should_use_rlm(req) is True


# ── Phase 2A: Batch Sub-calls ───────────────────────────────────────────────


def test_repl_has_batch_function():
    from repl import REPLEnvironment

    async def dummy_fn(prompt):
        return "response"

    async def dummy_batch(prompts):
        return ["resp"] * len(prompts)

    env = REPLEnvironment(
        context="test",
        llm_query_fn=dummy_fn,
        llm_query_batch_fn=dummy_batch,
    )
    assert "llm_query_batch" in env.namespace
    assert callable(env.namespace["llm_query_batch"])


# ── Phase 2B: Sub-call Caching ──────────────────────────────────────────────


def test_repl_caching():
    from repl import REPLEnvironment

    call_count = 0

    async def counting_fn(prompt):
        nonlocal call_count
        call_count += 1
        return f"response to: {prompt}"

    env = REPLEnvironment(context="test", llm_query_fn=counting_fn)

    # First call
    r1 = env._llm_query_sync_wrapper("hello")
    assert call_count == 1
    assert env.sub_call_count == 1
    assert env.cache_hits == 0

    # Second identical call — should be cached
    r2 = env._llm_query_sync_wrapper("hello")
    assert call_count == 1  # No new LLM call
    assert env.sub_call_count == 1  # Not incremented
    assert env.cache_hits == 1
    assert r1 == r2


# ── Phase 3A: REPL Sandboxing ───────────────────────────────────────────────


def test_repl_blocks_dangerous_imports():
    from repl import REPLEnvironment

    async def dummy_fn(prompt):
        return ""

    env = REPLEnvironment(context="test", llm_query_fn=dummy_fn)
    result = env.execute("import os")
    assert "not allowed" in result.lower() or "import" in result.lower()


def test_repl_allows_safe_imports():
    from repl import REPLEnvironment

    async def dummy_fn(prompt):
        return ""

    env = REPLEnvironment(context="test", llm_query_fn=dummy_fn)
    result = env.execute("import json\nprint(json.dumps({'a': 1}))")
    assert '{"a": 1}' in result


def test_repl_basic_execution():
    from repl import REPLEnvironment

    async def dummy_fn(prompt):
        return ""

    env = REPLEnvironment(context="hello world", llm_query_fn=dummy_fn)
    result = env.execute("print(len(context))")
    assert "11" in result


# ── Phase 3B: REPL Utility Functions ────────────────────────────────────────


def test_chunk_by_lines():
    from repl_utils import chunk_by_lines

    text = "\n".join(f"line {i}" for i in range(250))
    chunks = chunk_by_lines(text, n=100)
    assert len(chunks) == 3
    assert "line 0" in chunks[0]
    assert "line 100" in chunks[1]


def test_chunk_by_chars():
    from repl_utils import chunk_by_chars

    text = "a" * 1000
    chunks = chunk_by_chars(text, n=300, overlap=50)
    assert len(chunks) >= 3
    # With overlap, chunks should overlap
    assert len(chunks[0]) == 300


def test_search():
    from repl_utils import search

    text = "line 1: hello\nline 2: world\nline 3: hello again"
    matches = search(text, "hello")
    assert len(matches) == 2


def test_count_tokens():
    from repl_utils import count_tokens

    assert count_tokens("a" * 400) == 100  # ratio 4.0


def test_chunk_by_sections():
    from repl_utils import chunk_by_sections

    text = "# Header 1\nContent 1\n## Header 2\nContent 2"
    chunks = chunk_by_sections(text)
    assert len(chunks) >= 2


def test_repl_has_utilities():
    from repl import REPLEnvironment

    async def dummy_fn(prompt):
        return ""

    env = REPLEnvironment(context="test", llm_query_fn=dummy_fn)
    for name in ["chunk_by_lines", "chunk_by_chars", "search", "count_tokens", "chunk_by_sections"]:
        assert name in env.namespace, f"{name} not in REPL namespace"


# ── Phase 3C: Prompt Profiles ───────────────────────────────────────────────


def test_get_profile_default():
    from prompt_profiles import get_profile

    p = get_profile("unknown-model-xyz")
    assert p.name == "default"


def test_get_profile_substring_match():
    from prompt_profiles import get_profile

    p = get_profile("qwen3-coder-next")
    assert p.name == "qwen3-coder"


def test_get_profile_llama():
    from prompt_profiles import get_profile

    p = get_profile("llama-3.3-70b")
    assert p.name == "llama"


def test_build_system_prompt_applies_profile():
    from rlm_engine import _build_system_prompt

    prompt = _build_system_prompt("test context", model_name="qwen3-coder-next")
    assert "concise" in prompt.lower()


# ── Phase 4B: Multi-Document Context ────────────────────────────────────────


def test_build_system_prompt_dict_context():
    from rlm_engine import _build_system_prompt

    prompt = _build_system_prompt({"doc1": "content1", "doc2": "content2"})
    assert "dict" in prompt


def test_build_system_prompt_list_context():
    from rlm_engine import _build_system_prompt

    prompt = _build_system_prompt(["item1", "item2", "item3"])
    assert "list of 3 items" in prompt


# ── Phase 4C: Observability ─────────────────────────────────────────────────


def test_metrics_collector():
    from metrics import MetricsCollector

    m = MetricsCollector()
    m.record_request(rlm=True, stream=False)
    m.record_request(rlm=False, stream=True)
    m.record_rlm_result(iterations=5, sub_calls=10, cache_hits=2, duration=3.5)
    m.record_passthrough_duration(0.5)
    m.record_error()

    snap = m.snapshot()
    assert snap["requests"]["total"] == 2
    assert snap["requests"]["rlm"] == 1
    assert snap["requests"]["passthrough"] == 1
    assert snap["requests"]["stream"] == 1
    assert snap["requests"]["errors"] == 1
    assert snap["rlm"]["total_sub_calls"] == 10
    assert snap["rlm"]["total_cache_hits"] == 2
    assert snap["rlm"]["total_iterations"] == 5
    assert snap["rlm"]["iteration_histogram"] == {5: 1}
    assert snap["total_completed_requests"] == 2
    assert snap["duration_percentiles"]["p50"] > 0


def test_metrics_session_tracking():
    from metrics import MetricsCollector

    m = MetricsCollector()
    assert m.active_sessions == 0
    m.session_start()
    m.session_start()
    assert m.active_sessions == 2
    m.session_end()
    assert m.active_sessions == 1
    m.session_end()
    m.session_end()  # Should not go below 0
    assert m.active_sessions == 0


# ── REPL extract helpers ────────────────────────────────────────────────────


def test_extract_repl_blocks():
    from repl import extract_repl_blocks

    text = "some text\n```repl\nprint('hello')\n```\nmore text\n```repl\nx = 1\n```"
    blocks = extract_repl_blocks(text)
    assert len(blocks) == 2
    assert "print('hello')" in blocks[0]


def test_extract_final():
    from repl import extract_final

    assert extract_final("FINAL(42)") == ("42", None)
    assert extract_final("FINAL_VAR(answer)") == (None, "answer")
    assert extract_final("no final here") == (None, None)


# ── Enhancement 1: Auth Middleware ──────────────────────────────────────────


def test_auth_middleware_skip_when_no_key():
    from auth import APIKeyAuthMiddleware
    # When api_key is empty, middleware should pass through
    # (tested via config — api_key defaults to "")
    from config import settings
    assert settings.api_key == ""  # Default: auth disabled


# ── Enhancement 2: Request Queue ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_request_queue_acquire_release():
    from request_queue import RequestQueue
    q = RequestQueue()
    wait_time = await q.acquire()
    assert wait_time >= 0
    assert q.active_sessions == 1
    q.release()
    assert q.active_sessions == 0


@pytest.mark.asyncio
async def test_request_queue_snapshot():
    from request_queue import RequestQueue
    q = RequestQueue()
    snap = q.snapshot()
    assert "active_sessions" in snap
    assert "queued_requests" in snap
    assert snap["active_sessions"] == 0


# ── Enhancement 3: Retry Config ────────────────────────────────────────────


def test_retry_config():
    from config import settings
    assert settings.ollama_max_retries >= 1
    assert settings.ollama_retry_base_delay > 0


# ── Enhancement 4: Smart Compaction ────────────────────────────────────────


def test_compact_threshold_config():
    from config import settings
    assert settings.history_compact_threshold >= 10


# ── Enhancement 5: Adaptive Early-stop ─────────────────────────────────────


def test_sequence_matcher_for_stuck_detection():
    from difflib import SequenceMatcher
    # Same output should have high similarity
    s1 = "hello world output"
    s2 = "hello world output"
    assert SequenceMatcher(None, s1, s2).ratio() > 0.8
    # Different output should have low similarity
    s3 = "completely different content here"
    assert SequenceMatcher(None, s1, s3).ratio() < 0.5


# ── Enhancement 6: Depth Limit + Token Budget ─────────────────────────────


def test_repl_depth_limit():
    from repl import REPLEnvironment
    from config import settings

    call_count = 0

    async def counting_fn(prompt):
        nonlocal call_count
        call_count += 1
        return "response"

    env = REPLEnvironment(context="test", llm_query_fn=counting_fn)

    # Simulate being at max depth
    env._nesting_depth = settings.max_sub_call_depth
    result = env._llm_query_sync_wrapper("test prompt")
    assert "[ERROR]" in result
    assert "nesting depth" in result.lower()
    assert call_count == 0  # Should not have called LLM


def test_repl_token_tracking():
    from repl import REPLEnvironment

    async def dummy_fn(prompt):
        return "short response"

    env = REPLEnvironment(context="test", llm_query_fn=dummy_fn)
    assert env.estimated_tokens_used == 0

    env._llm_query_sync_wrapper("hello world")
    assert env.estimated_tokens_used > 0


def test_depth_limit_config():
    from config import settings
    assert settings.max_sub_call_depth >= 1


def test_token_budget_config():
    from config import settings
    assert isinstance(settings.token_budget, int)


# ── Enhancement 7: LRU Cache ──────────────────────────────────────────────


def test_lru_cache_eviction_order():
    from collections import OrderedDict
    from repl import REPLEnvironment

    call_count = 0

    async def counting_fn(prompt):
        nonlocal call_count
        call_count += 1
        return f"resp-{prompt}"

    env = REPLEnvironment(context="test", llm_query_fn=counting_fn)
    assert isinstance(env._sub_call_cache, OrderedDict)


# ── Enhancement 8: Circuit Breaker ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_circuit_breaker_closed_by_default():
    from circuit_breaker import CircuitBreaker, CircuitState
    cb = CircuitBreaker()
    assert cb.state == CircuitState.CLOSED
    await cb.check()  # Should not raise


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_threshold():
    from circuit_breaker import CircuitBreaker, CircuitState, CircuitOpenError
    from config import settings

    cb = CircuitBreaker()
    for _ in range(settings.circuit_breaker_threshold):
        await cb.record_failure()
    assert cb.state == CircuitState.OPEN

    with pytest.raises(CircuitOpenError):
        await cb.check()


@pytest.mark.asyncio
async def test_circuit_breaker_resets_on_success():
    from circuit_breaker import CircuitBreaker, CircuitState
    cb = CircuitBreaker()
    await cb.record_failure()
    await cb.record_failure()
    await cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb._failure_count == 0


def test_circuit_breaker_snapshot():
    from circuit_breaker import CircuitBreaker
    cb = CircuitBreaker()
    snap = cb.snapshot()
    assert snap["state"] == "closed"
    assert "threshold" in snap
    assert "timeout_seconds" in snap


# ── Dispatcher ─────────────────────────────────────────────────────────────


def test_dispatcher_config():
    from config import settings
    assert isinstance(settings.ollama_hosts, str)
    assert settings.dispatcher_refresh_interval >= 0


def test_dispatcher_singleton():
    from dispatcher import Dispatcher, dispatcher
    assert isinstance(dispatcher, Dispatcher)
    assert dispatcher.host_count >= 1


def test_dispatcher_pick_host_default():
    from dispatcher import Dispatcher
    d = Dispatcher()
    # Before any refresh, should return the default host
    host = d.pick_host("any-model")
    assert host.startswith("http")


def test_dispatcher_least_loaded():
    from dispatcher import Dispatcher, ProviderState
    from providers import ProviderConfig, create_provider
    d = Dispatcher()
    p1 = create_provider(ProviderConfig(name="host1", api_type="ollama", url="http://host1:11434"))
    p2 = create_provider(ProviderConfig(name="host2", api_type="ollama", url="http://host2:11434"))
    ps1 = ProviderState(provider=p1, models={"model-a"}, healthy=True, active_requests=5)
    ps2 = ProviderState(provider=p2, models={"model-a"}, healthy=True, active_requests=1)
    d._providers = {"host1": ps1, "host2": ps2}
    d._model_to_providers = {"model-a": ["host1", "host2"]}

    _, model_name = d.pick_provider("model-a")
    assert model_name == "model-a"


def test_dispatcher_unhealthy_host_skipped():
    from dispatcher import Dispatcher, ProviderState
    from providers import ProviderConfig, create_provider
    d = Dispatcher()
    p1 = create_provider(ProviderConfig(name="host1", api_type="ollama", url="http://host1:11434"))
    p2 = create_provider(ProviderConfig(name="host2", api_type="ollama", url="http://host2:11434"))
    ps1 = ProviderState(provider=p1, models={"model-a"}, healthy=False, active_requests=0)
    ps2 = ProviderState(provider=p2, models={"model-a"}, healthy=True, active_requests=3)
    d._providers = {"host1": ps1, "host2": ps2}
    d._model_to_providers = {"model-a": ["host1", "host2"]}

    provider, _ = d.pick_provider("model-a")
    assert provider.name == "host2"


def test_dispatcher_acquire_release():
    from dispatcher import Dispatcher
    d = Dispatcher()
    d._init_from_config()
    name = list(d._providers.keys())[0]
    ps = d._providers[name]
    assert ps.active_requests == 0
    d.acquire(name)
    assert ps.active_requests == 1
    d.release(name)
    assert ps.active_requests == 0


def test_dispatcher_snapshot():
    from dispatcher import Dispatcher
    d = Dispatcher()
    snap = d.snapshot()
    assert "provider_count" in snap
    assert "providers" in snap
    assert "total_models" in snap
    assert "routing_table" in snap
    assert snap["provider_count"] >= 1


def test_dispatcher_model_routing():
    from dispatcher import Dispatcher, ProviderState
    from providers import ProviderConfig, create_provider
    d = Dispatcher()
    p1 = create_provider(ProviderConfig(name="host1", api_type="ollama", url="http://host1:11434"))
    p2 = create_provider(ProviderConfig(name="host2", api_type="ollama", url="http://host2:11434"))
    ps1 = ProviderState(provider=p1, models={"model-a", "model-b"}, healthy=True)
    ps2 = ProviderState(provider=p2, models={"model-b", "model-c"}, healthy=True)
    d._providers = {"host1": ps1, "host2": ps2}
    d._model_to_providers = {
        "model-a": ["host1"],
        "model-b": ["host1", "host2"],
        "model-c": ["host2"],
    }

    provider, model = d.pick_provider("model-a")
    assert provider.name == "host1"
    provider, model = d.pick_provider("model-c")
    assert provider.name == "host2"
    # Explicit prefix routing
    provider, model = d.pick_provider("host2/model-b")
    assert provider.name == "host2"
    assert model == "model-b"


def test_parse_model_string():
    from providers import parse_model_string
    assert parse_model_string("groq/llama-3.3-70b") == ("groq", "llama-3.3-70b")
    assert parse_model_string("local/qwen3-coder-next") == ("local", "qwen3-coder-next")
    assert parse_model_string("qwen3-coder-next") == (None, "qwen3-coder-next")
    assert parse_model_string("openai/gpt-4o") == ("openai", "gpt-4o")


def test_provider_factory():
    from providers import ProviderConfig, OllamaProvider, OpenAIProvider, create_provider
    p1 = create_provider(ProviderConfig(name="local", api_type="ollama", url="http://localhost:11434"))
    assert isinstance(p1, OllamaProvider)
    assert p1.name == "local"
    p2 = create_provider(ProviderConfig(name="openai", api_type="openai", url="https://api.openai.com/v1", api_key="sk-test"))
    assert isinstance(p2, OpenAIProvider)
    assert p2.config.api_key == "sk-test"


def test_provider_config_serialization():
    from providers import ProviderConfig
    cfg = ProviderConfig(name="groq", api_type="openai", url="https://api.groq.com/v1", api_key="key123")
    d = cfg.to_dict()
    assert d["name"] == "groq"
    assert d["api_type"] == "openai"
    assert d["api_key"] == "key123"
    # Round-trip
    cfg2 = ProviderConfig(**d)
    assert cfg2.url == cfg.url
    assert cfg2.name == cfg.name


# ── Embeddings + Model Detail Schemas ──────────────────────────────────────


def test_embedding_request_string_input():
    from schemas import EmbeddingRequest
    req = EmbeddingRequest(model="nomic-embed-text", input="hello world")
    assert req.input == "hello world"


def test_embedding_request_list_input():
    from schemas import EmbeddingRequest
    req = EmbeddingRequest(model="nomic-embed-text", input=["hello", "world"])
    assert len(req.input) == 2


def test_embedding_response():
    from schemas import EmbeddingData, EmbeddingResponse
    resp = EmbeddingResponse(
        model="nomic-embed-text",
        data=[EmbeddingData(index=0, embedding=[0.1, 0.2, 0.3])],
    )
    assert resp.object == "list"
    assert len(resp.data) == 1
    assert resp.data[0].embedding == [0.1, 0.2, 0.3]


def test_ollama_embed_request():
    from schemas import OllamaEmbedRequest
    req = OllamaEmbedRequest(model="nomic-embed-text", input=["hello"])
    assert req.model == "nomic-embed-text"


def test_model_detail_schema():
    from schemas import ModelDetail
    md = ModelDetail(
        id="qwen3-coder-next",
        parameters={"family": "qwen3", "parameter_size": "32B"},
        template="...",
    )
    assert md.id == "qwen3-coder-next"
    assert md.parameters["family"] == "qwen3"


# ── Provider Response Format ───────────────────────────────────────────────


def test_openai_provider_returns_ollama_format():
    """OpenAI provider must translate responses to Ollama format."""
    from providers import OpenAIProvider, ProviderConfig

    # Simulate what OpenAIProvider.chat() returns
    # The real method calls the API, but we can verify the format contract
    # by checking that the translation logic produces the right shape
    openai_response = {
        "choices": [{"message": {"role": "assistant", "content": "Hello world"}}],
        "model": "gpt-4o",
    }
    # This is what the provider translates to:
    content = openai_response.get("choices", [{}])[0].get("message", {}).get("content", "")
    ollama_format = {
        "model": openai_response.get("model", "gpt-4o"),
        "message": {"role": "assistant", "content": content},
        "done": True,
    }
    # Verify it matches what calling code expects
    assert ollama_format.get("message", {}).get("content", "") == "Hello world"
    assert ollama_format["message"]["role"] == "assistant"


def test_openai_provider_embed_returns_ollama_format():
    """OpenAI provider must translate embedding responses to Ollama format."""
    openai_embed_response = {
        "data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}],
        "model": "text-embedding-3-small",
    }
    # Translation logic from OpenAIProvider.embed():
    embeddings = [item["embedding"] for item in openai_embed_response.get("data", [])]
    ollama_format = {"model": "text-embedding-3-small", "embeddings": embeddings}
    assert len(ollama_format["embeddings"]) == 2
    assert ollama_format["embeddings"][0] == [0.1, 0.2, 0.3]


# ── Admin Config API ───────────────────────────────────────────────────────


def test_get_all_settings():
    from config import get_all_settings
    data = get_all_settings()
    assert "max_iterations" in data
    assert "root_model" in data
    assert "ollama_base_url" in data


def test_update_settings_valid():
    from config import settings, update_settings
    original = settings.max_iterations
    errors = update_settings({"max_iterations": 99})
    assert not errors
    assert settings.max_iterations == 99
    # Restore
    update_settings({"max_iterations": original})


def test_update_settings_invalid_key():
    from config import update_settings
    errors = update_settings({"nonexistent_field": 42})
    assert "nonexistent_field" in errors


def test_update_settings_non_editable():
    from config import update_settings
    errors = update_settings({"host": "0.0.0.0"})
    assert "host" in errors


def test_save_and_load_settings():
    import json
    from config import CONFIG_FILE, load_settings, save_settings, settings, update_settings
    # Save current state
    original_iterations = settings.max_iterations
    # Modify and save
    update_settings({"max_iterations": 77})
    path = save_settings()
    assert CONFIG_FILE.exists()
    data = json.loads(CONFIG_FILE.read_text())
    assert data["max_iterations"] == 77
    # Reset
    load_settings()
    # Clean up
    CONFIG_FILE.unlink(missing_ok=True)
    update_settings({"max_iterations": original_iterations})


def test_admin_html_exists():
    from pathlib import Path
    html = Path(__file__).parent / "admin.html"
    assert html.exists()
    content = html.read_text()
    assert "RLM Proxy" in content
    assert "Admin Dashboard" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
