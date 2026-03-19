# RLM Proxy: Production Hardening + Smarter Reasoning

**Date:** 2026-03-16
**Status:** Approved
**Context:** Single user, multiple client apps on 2 machines hitting one proxy concurrently.

## 1. API Key Auth Middleware

**New file:** `auth.py`
**Config:** `RLM_API_KEY` (empty = disabled)

FastAPI middleware on all `/v1/*` routes. Checks `Authorization: Bearer <key>` header. Returns 401 if invalid. `/health` and `/v1/rlm/metrics` stay open for monitoring. Backwards compatible — no key configured means no auth.

## 2. Request Queue + Session Concurrency Limit

**New file:** `request_queue.py`
**Config:** `RLM_MAX_CONCURRENT_SESSIONS` (default 2), `RLM_MAX_QUEUE_SIZE` (default 10)

`asyncio.Semaphore` for concurrency, `asyncio.Queue` for waiting requests. Only RLM sessions are queued (passthrough is cheap, flows freely). Queue full → 429 with `Retry-After` header. Tracks position in queue for logging.

## 3. Retry with Exponential Backoff

**Modified file:** `ollama_client.py`
**Config:** `RLM_OLLAMA_MAX_RETRIES` (default 3), `RLM_OLLAMA_RETRY_BASE_DELAY` (default 1.0)

`@with_retry` decorator on `chat()`, `chat_batch()`, `chat_stream()`. Retries on: `httpx.ConnectError`, `httpx.TimeoutException`, HTTP 502/503/504. Does NOT retry on 4xx or successful-but-empty. Backoff: base * 2^attempt. Logs each retry.

## 4. Smart History Compaction

**Modified file:** `rlm_engine.py`
**Config:** `RLM_HISTORY_COMPACT_THRESHOLD` (default 30)

When history exceeds threshold, instead of discarding middle turns:
1. Extract turns being removed
2. Format them as a summary prompt
3. Call sub-LLM to produce a "reasoning so far" summary
4. Replace removed turns with single summary message

Format: `[Compacted: iterations 1-N summary] <LLM-generated summary>`

Uses sub-model at temperature 0.2. One extra sub-call per compaction (cheap vs. losing context). Falls back to current discard strategy if sub-call fails.

## 5. Adaptive Early-Stop

**Modified file:** `rlm_engine.py`

**Premature convergence detection:** If model outputs FINAL within first 2 iterations AND context is >100K chars, inject verification prompt before accepting. "Verify your answer by checking at least 3 different sections."

**Stuck detection:** Track per-iteration state (new variables count, sub_call delta, output similarity via SequenceMatcher ratio). If 3 consecutive iterations show <20% change:
1. First stuck: inject redirect with alternative strategy suggestion
2. Still stuck after 2 more: extract best variable from namespace, force FINAL

## 6. Sub-call Depth Limit + Token Budget

**Modified files:** `repl.py`, `config.py`
**Config:** `RLM_MAX_SUB_CALL_DEPTH` (default 3), `RLM_TOKEN_BUDGET` (default 0 = unlimited)

**Depth:** `_nesting_depth` counter on REPLEnvironment. Incremented before LLM call, decremented after. Exceeding depth returns error string, not exception.

**Budget:** `TokenBudget` dataclass tracking estimated tokens (chars / ratio) for root calls and sub-calls. When exceeded, next iteration gets "[TOKEN BUDGET EXHAUSTED] Provide your best answer now with FINAL()." Budget status shown in REPL feedback line.

## 7. True LRU Cache

**Modified file:** `repl.py`

Replace `dict` with `OrderedDict`. On hit: `move_to_end(key)`. On eviction: `popitem(last=False)`. Applied to both single and batch wrappers. Same SHA256 keying, same size limit.

## 8. Ollama Circuit Breaker

**New file:** `circuit_breaker.py`
**Config:** `RLM_CIRCUIT_BREAKER_THRESHOLD` (default 5), `RLM_CIRCUIT_BREAKER_TIMEOUT` (default 30)

Three states: Closed → Open → Half-Open → Closed.
- **Closed:** requests flow, failures counted. N consecutive failures → Open.
- **Open:** all calls fail immediately with `CircuitOpenError`. After timeout → Half-Open.
- **Half-Open:** one probe request allowed. Success → Closed. Failure → Open.

Wraps all `ollama_client` outbound calls. Thread-safe (asyncio.Lock). Logged state transitions. Exposed in `/v1/rlm/metrics` as `circuit_breaker_state`.

## New Files

| File | Purpose |
|------|---------|
| `auth.py` | Bearer token auth middleware |
| `request_queue.py` | Session concurrency + request queuing |
| `circuit_breaker.py` | Ollama circuit breaker |

## Modified Files

| File | Changes |
|------|---------|
| `config.py` | 7 new settings |
| `ollama_client.py` | Retry decorator, circuit breaker integration |
| `rlm_engine.py` | Smart compaction, adaptive early-stop, token budget |
| `repl.py` | LRU cache, depth tracking, budget tracking |
| `main.py` | Auth middleware, request queue, updated metrics |
| `metrics.py` | Circuit breaker state, queue depth, budget stats |
| `test_unit.py` | Tests for all 8 features |
