"""Sandboxed Python REPL for RLM code execution.

Provides a persistent namespace where the LLM can execute code across
multiple iterations, with a built-in `llm_query` function for sub-calls.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import re
import time
import traceback
from collections import OrderedDict
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Callable

from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Eval import default_guarded_getiter, default_guarded_getitem
from RestrictedPython.Guards import guarded_unpack_sequence, safer_getattr
from RestrictedPython.PrintCollector import PrintCollector

from config import settings
from repl_utils import chunk_by_lines, chunk_by_chars, search, count_tokens, chunk_by_sections

log = logging.getLogger("rlm_proxy.repl")

_SAFE_MODULES = frozenset({
    "re", "json", "math", "string", "collections", "itertools",
    "functools", "operator", "textwrap", "difflib", "unicodedata",
    "datetime", "hashlib", "base64", "copy", "statistics",
    "bisect", "heapq", "random", "csv", "io",
})


def _safe_import(name, *args, **kwargs):
    """Import guard that only allows whitelisted modules."""
    if name.split(".")[0] not in _SAFE_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed in the RLM REPL. "
                          f"Allowed modules: {', '.join(sorted(_SAFE_MODULES))}")
    return __import__(name, *args, **kwargs)

# Maximum characters of stdout we return per execution
MAX_STDOUT_CHARS = 8000


class REPLEnvironment:
    """A persistent Python REPL with a shared namespace."""

    def __init__(
        self,
        context: str | list[str],
        llm_query_fn: Callable[[str], str],
        on_sub_call: Callable[[str, str, float], None] | None = None,
        llm_query_batch_fn: Callable[[list[str]], list[str]] | None = None,
    ) -> None:
        self.sub_call_count = 0
        self._sub_call_cache: OrderedDict[str, str] = OrderedDict()
        self.cache_hits = 0
        self._nesting_depth = 0
        self.estimated_tokens_used = 0  # token budget tracking
        self._llm_query_fn = llm_query_fn
        self._on_sub_call = on_sub_call
        self._llm_query_batch_fn = llm_query_batch_fn

        # Build restricted builtins
        import builtins as _builtins_mod
        _safe_builtins = {
            k: getattr(_builtins_mod, k) for k in [
                "True", "False", "None", "abs", "all", "any", "bool", "bytes",
                "chr", "complex", "dict", "divmod", "enumerate", "filter",
                "float", "format", "frozenset", "getattr", "hasattr", "hash",
                "hex", "id", "int", "isinstance", "issubclass", "iter", "len",
                "list", "map", "max", "min", "next", "oct", "ord", "pow",
                "print", "range", "repr", "reversed", "round", "set", "slice",
                "sorted", "str", "sum", "tuple", "type", "zip",
                "ValueError", "TypeError", "KeyError", "IndexError",
                "AttributeError", "RuntimeError", "StopIteration", "Exception",
            ] if hasattr(_builtins_mod, k)
        }
        _safe_builtins["__import__"] = _safe_import

        self.namespace: dict[str, Any] = {
            "__builtins__": _safe_builtins,
            "_getiter_": default_guarded_getiter,
            "_getitem_": default_guarded_getitem,
            "_getattr_": safer_getattr,
            "_unpack_sequence_": guarded_unpack_sequence,
            "_write_": lambda obj: obj,  # allow attribute writes
            "_inplacevar_": lambda op, x, y: op(x, y),
            "_print_": PrintCollector,
            "_getattr_": safer_getattr,
            "context": context,
            "llm_query": self._llm_query_sync_wrapper,
            "llm_query_batch": self._llm_query_batch_sync_wrapper,
            "print": print,
            "chunk_by_lines": chunk_by_lines,
            "chunk_by_chars": chunk_by_chars,
            "search": search,
            "count_tokens": count_tokens,
            "chunk_by_sections": chunk_by_sections,
        }

    def _llm_query_sync_wrapper(self, prompt: str) -> str:
        """Synchronous wrapper so LLM-generated code can call llm_query()."""
        # Depth limit check
        if self._nesting_depth >= settings.max_sub_call_depth:
            return f"[ERROR] Maximum sub-call nesting depth ({settings.max_sub_call_depth}) reached."

        # Truncate overly long prompts
        if len(prompt) > settings.sub_call_max_chars:
            prompt = prompt[: settings.sub_call_max_chars] + "\n...[truncated]"

        # Check cache before counting against sub_call_count
        cache_key = hashlib.sha256(prompt.encode()).hexdigest()
        if cache_key in self._sub_call_cache:
            self.cache_hits += 1
            self._sub_call_cache.move_to_end(cache_key)  # LRU: mark as recently used
            log.info("Sub-call cache HIT #%d  (prompt len=%d)", self.cache_hits, len(prompt))
            return self._sub_call_cache[cache_key]

        # Token budget check
        if settings.token_budget > 0:
            prompt_tokens = int(len(prompt) / settings.token_estimate_ratio)
            if self.estimated_tokens_used + prompt_tokens > settings.token_budget:
                return f"[ERROR] Token budget exhausted ({self.estimated_tokens_used:,}/{settings.token_budget:,} tokens used)."

        if self.sub_call_count >= settings.max_sub_calls:
            return f"[ERROR] Maximum sub-call limit ({settings.max_sub_calls}) reached."

        self.sub_call_count += 1
        self._nesting_depth += 1
        log.info("Sub-call #%d  (prompt len=%d, depth=%d)", self.sub_call_count, len(prompt), self._nesting_depth)

        # Run the async llm call in the current event loop
        t_start = time.perf_counter()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context — use a thread to avoid deadlock
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._llm_query_fn(prompt))
                response = future.result(timeout=settings.repl_exec_timeout)
        else:
            response = asyncio.run(self._llm_query_fn(prompt))

        elapsed = time.perf_counter() - t_start
        self._nesting_depth -= 1

        # Track token usage
        self.estimated_tokens_used += int((len(prompt) + len(response)) / settings.token_estimate_ratio)

        # Store in cache (LRU eviction if over limit)
        if len(self._sub_call_cache) >= settings.sub_call_cache_size:
            self._sub_call_cache.popitem(last=False)  # LRU: evict least recently used
        self._sub_call_cache[cache_key] = response

        if self._on_sub_call is not None:
            self._on_sub_call(prompt, response, elapsed)

        return response

    def _llm_query_batch_sync_wrapper(self, prompts: list[str]) -> list[str]:
        """Synchronous wrapper for parallel batch sub-calls with caching."""
        # Truncate overly long prompts
        truncated = []
        for p in prompts:
            if len(p) > settings.sub_call_max_chars:
                p = p[: settings.sub_call_max_chars] + "\n...[truncated]"
            truncated.append(p)

        # Separate cached from uncached
        results: list[str | None] = [None] * len(truncated)
        uncached_indices: list[int] = []
        uncached_prompts: list[str] = []

        for i, p in enumerate(truncated):
            cache_key = hashlib.sha256(p.encode()).hexdigest()
            if cache_key in self._sub_call_cache:
                self.cache_hits += 1
                self._sub_call_cache.move_to_end(cache_key)  # LRU: mark as recently used
                results[i] = self._sub_call_cache[cache_key]
            else:
                uncached_indices.append(i)
                uncached_prompts.append(p)

        if uncached_indices:
            log.info(
                "Batch sub-call: %d prompts (%d cached, %d to fetch)",
                len(truncated), len(truncated) - len(uncached_prompts), len(uncached_prompts),
            )
        else:
            log.info("Batch sub-call: %d prompts (all cached)", len(truncated))
            return [r for r in results if r is not None]

        # Check sub-call limit for uncached prompts only
        remaining = settings.max_sub_calls - self.sub_call_count
        if remaining <= 0:
            for i in uncached_indices:
                results[i] = f"[ERROR] Maximum sub-call limit ({settings.max_sub_calls}) reached."
            return [r for r in results if r is not None]

        if len(uncached_prompts) > remaining:
            # Fill excess with error, only send what fits
            for i in uncached_indices[remaining:]:
                results[i] = f"[ERROR] Maximum sub-call limit ({settings.max_sub_calls}) reached."
            uncached_indices = uncached_indices[:remaining]
            uncached_prompts = uncached_prompts[:remaining]

        self.sub_call_count += len(uncached_prompts)

        if self._llm_query_batch_fn is None:
            # Fallback to sequential calls (bypass cache since we already checked)
            for idx, p in zip(uncached_indices, uncached_prompts):
                results[idx] = self._llm_query_sync_wrapper(p)
            return [r for r in results if r is not None]

        t_start = time.perf_counter()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._llm_query_batch_fn(uncached_prompts))
                responses = future.result(timeout=settings.repl_exec_timeout * len(uncached_prompts))
        else:
            responses = asyncio.run(self._llm_query_batch_fn(uncached_prompts))

        elapsed = time.perf_counter() - t_start

        # Store responses in cache and results
        for idx, p, resp in zip(uncached_indices, uncached_prompts, responses):
            cache_key = hashlib.sha256(p.encode()).hexdigest()
            if len(self._sub_call_cache) >= settings.sub_call_cache_size:
                self._sub_call_cache.popitem(last=False)  # LRU: evict least recently used
            self._sub_call_cache[cache_key] = resp
            results[idx] = resp

        if self._on_sub_call is not None:
            per_call_elapsed = elapsed / max(len(uncached_prompts), 1)
            for p, resp in zip(uncached_prompts, responses):
                self._on_sub_call(p, resp, per_call_elapsed)

        return [r for r in results if r is not None]

    def execute(self, code: str) -> str:
        """Execute code in the restricted REPL and return captured stdout + stderr."""
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            compiled = compile_restricted(code, filename="<repl>", mode="exec")
            if getattr(compiled, "errors", None):
                return "[RESTRICTED] " + "; ".join(compiled.errors)
        except SyntaxError as e:
            return f"[SYNTAX ERROR] {e}"

        # Execute with timeout
        import concurrent.futures

        def _run():
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compiled, self.namespace)  # noqa: S102 — compiled via compile_restricted

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run)
                future.result(timeout=settings.repl_exec_timeout)
        except concurrent.futures.TimeoutError:
            stderr_buf.write(f"[TIMEOUT] Code execution exceeded {settings.repl_exec_timeout}s limit")
        except Exception:
            stderr_buf.write(traceback.format_exc())

        stdout = stdout_buf.getvalue()
        stderr = stderr_buf.getvalue()

        # Collect output from RestrictedPython's PrintCollector
        printed = self.namespace.get("_print")
        if printed is not None and hasattr(printed, "__call__"):
            # PrintCollector was used; extract its output
            try:
                printed_text = printed()
                if printed_text:
                    stdout = printed_text + ("\n" + stdout if stdout else "")
            except Exception:
                pass
            # Reset for next execution
            del self.namespace["_print"]

        output_parts = []
        if stdout:
            output_parts.append(stdout)
        if stderr:
            output_parts.append(f"[STDERR]\n{stderr}")

        result = "\n".join(output_parts) if output_parts else "(no output)"

        if len(result) > MAX_STDOUT_CHARS:
            result = result[: MAX_STDOUT_CHARS] + f"\n...[output truncated at {MAX_STDOUT_CHARS} chars]"

        return result

    def get_variable(self, name: str) -> Any:
        """Retrieve a variable from the REPL namespace."""
        return self.namespace.get(name)


def extract_repl_blocks(text: str) -> list[str]:
    """Extract ```repl ... ``` code blocks from LLM output."""
    pattern = r"```repl\s*\n(.*?)```"
    blocks = re.findall(pattern, text, re.DOTALL)
    return blocks


def extract_final(text: str) -> tuple[str | None, str | None]:
    """Check for FINAL(answer) or FINAL_VAR(varname) in text.

    Returns (final_answer, final_var_name) — at most one will be set.
    """
    # FINAL_VAR(varname)
    m = re.search(r"FINAL_VAR\((\w+)\)", text)
    if m:
        return None, m.group(1)

    # FINAL(answer) — can span multiple lines
    m = re.search(r"FINAL\((.*?)\)\s*$", text, re.DOTALL)
    if m:
        return m.group(1).strip(), None

    # Also check for single-line FINAL(...)
    m = re.search(r"FINAL\((.+?)\)", text)
    if m:
        return m.group(1).strip(), None

    return None, None
