"""Core RLM engine — orchestrates the recursive language model loop."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import ollama_client
from config import settings
from prompt_profiles import get_profile
from repl import REPLEnvironment, extract_final, extract_repl_blocks
from system_prompts import RLM_ROOT_SYSTEM_PROMPT, RLM_SUB_CALL_SYSTEM_PROMPT
from trajectory_logger import TrajectoryLogger

log = logging.getLogger("rlm_proxy.engine")


@dataclass
class RLMResult:
    answer: str
    iterations: int
    sub_calls: int
    elapsed_seconds: float
    cache_hits: int = 0
    trajectory: list[dict] = field(default_factory=list)


def _build_system_prompt(context: str | dict | list, model_name: str = "") -> str:
    """Fill in the RLM system prompt template with context metadata."""
    if isinstance(context, dict):
        context_type = "dict"
        context_len = sum(len(str(v)) for v in context.values())
    elif isinstance(context, list):
        context_type = f"list of {len(context)} items"
        context_len = sum(len(str(item)) for item in context)
    else:
        context_type = "string"
        context_len = len(context)
    # Estimate chunk lengths (the REPL loads the full context as one variable)
    chunk_size = 50_000
    n_chunks = max(1, context_len // chunk_size)
    chunk_lengths = ", ".join(
        [str(min(chunk_size, context_len - i * chunk_size)) for i in range(n_chunks)]
    )

    sub_call_max = settings.sub_call_max_chars
    sub_batch = min(sub_call_max, 200_000)

    prompt = RLM_ROOT_SYSTEM_PROMPT.format(
        context_type=context_type,
        context_total_length=f"{context_len:,}",
        context_lengths=chunk_lengths,
        sub_call_max_chars=f"{sub_call_max:,}",
        sub_call_batch_chars=f"{sub_batch:,}",
    )

    profile = get_profile(model_name)
    if profile.extra_instructions:
        prompt += f"\n\n{profile.extra_instructions}"
    if profile.code_style_hint:
        prompt += f"\nPreferred code style: {profile.code_style_hint}"
    return prompt


def _make_sub_query_fns(sub_model_name: str):
    """Create sub-call functions bound to a specific model."""

    async def sub_llm_query(prompt: str) -> str:
        messages = [
            {"role": "system", "content": RLM_SUB_CALL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        resp = await ollama_client.chat(
            model=sub_model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
        )
        return resp.get("message", {}).get("content", "")

    async def sub_llm_query_batch(prompts: list[str]) -> list[str]:
        messages_list = [
            [
                {"role": "system", "content": RLM_SUB_CALL_SYSTEM_PROMPT},
                {"role": "user", "content": p},
            ]
            for p in prompts
        ]
        results = await ollama_client.chat_batch(
            model=sub_model_name,
            prompts_with_messages=messages_list,
            temperature=0.3,
            max_tokens=4096,
        )
        return [r.get("message", {}).get("content", "") for r in results]

    return sub_llm_query, sub_llm_query_batch


async def run_rlm(
    query: str,
    context: str | dict | list,
    *,
    model: str | None = None,
    temperature: float = 0.7,
) -> RLMResult:
    """Run the full RLM loop.

    Parameters
    ----------
    query:
        The user's question.
    context:
        The input context to reason over. Can be a string, dict (keyed by
        document name), or list of documents/items.
    model:
        Override root model. Defaults to settings.root_model.
    temperature:
        Sampling temperature for the root model.
    """
    root_model = settings.root_model  # may include provider prefix e.g. "local/qwen3-coder-next"
    sub_model_name = model or root_model
    t0 = time.perf_counter()
    trajectory: list[dict] = []

    log.info("Models — root: %s  sub: %s", root_model, sub_model_name)

    # Trajectory logger (no-op when log dir is not configured)
    tlog = TrajectoryLogger(settings.trajectory_log_dir)

    def _on_sub_call(prompt: str, response: str, elapsed: float) -> None:
        tlog.log_event("sub_call", {
            "prompt_len": len(prompt),
            "response_len": len(response),
            "elapsed_seconds": round(elapsed, 4),
            "prompt_preview": prompt[:200],
            "response_preview": response[:200],
        })

    # Create sub-call functions bound to the user's requested model
    _sub_query, _sub_query_batch = _make_sub_query_fns(sub_model_name)

    # Initialise the REPL
    repl = REPLEnvironment(
        context=context,
        llm_query_fn=_sub_query,
        on_sub_call=_on_sub_call,
        llm_query_batch_fn=_sub_query_batch,
    )

    # Build the system prompt with context metadata
    system_prompt = _build_system_prompt(context, model_name=root_model)

    # History for the root LLM (iterative conversation)
    history: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {query}\n\nThe context has been loaded into the `context` variable in your REPL. Begin by exploring it."},
    ]

    # Adaptive early-stop tracking
    _prev_outputs: list[str] = []  # last 3 REPL outputs for stuck detection
    _prev_sub_call_counts: list[int] = []  # sub_call_count at end of last 3 iterations
    _prev_namespace_sizes: list[int] = []  # namespace variable count at end of last 3 iterations
    _stuck_redirected = False  # whether we've already injected a redirect
    _stuck_redirect_iteration = 0

    for iteration in range(1, settings.max_iterations + 1):
        log.info("═══ RLM iteration %d ═══", iteration)

        # Call the root LLM (provider prefix in model name handles routing)
        t_llm = time.perf_counter()
        resp = await ollama_client.chat(
            model=root_model,
            messages=history,
            temperature=temperature,
        )
        assistant_text = resp.get("message", {}).get("content", "")
        tlog.log_event("root_llm_response", {
            "iteration": iteration,
            "response_len": len(assistant_text),
            "elapsed_seconds": round(time.perf_counter() - t_llm, 4),
        })
        history.append({"role": "assistant", "content": assistant_text})
        trajectory.append({"iteration": iteration, "role": "assistant", "content": assistant_text})

        log.debug("Root LLM output (first 500 chars): %s", assistant_text[:500])

        # Check for FINAL answer before executing code
        final_answer, final_var = extract_final(assistant_text)
        if final_answer is not None:
            # Premature convergence detection: if FINAL in first 2 iterations
            # and context is large, ask the model to verify first
            context_len = len(context) if isinstance(context, str) else sum(len(str(v)) for v in (context.values() if isinstance(context, dict) else context))
            if iteration <= 2 and context_len > 100_000 and repl.sub_call_count < 2:
                log.info("Premature FINAL at iteration %d with large context — requesting verification", iteration)
                tlog.log_event("premature_final_blocked", {"iteration": iteration})
                verify_msg = (
                    f"[REPL] You provided a FINAL answer after only {iteration} iteration(s), "
                    f"but the context is {context_len:,} characters and you've only made "
                    f"{repl.sub_call_count} sub-call(s). Please verify your answer by checking "
                    "at least 3 different sections of the context before answering. "
                    "Use the REPL to explore more of the context."
                )
                history.append({"role": "user", "content": verify_msg})
                trajectory.append({"iteration": iteration, "role": "system", "content": verify_msg})
                continue

            log.info("FINAL answer received at iteration %d", iteration)
            tlog.log_event("final_answer", {"iteration": iteration, "answer_len": len(final_answer)})
            tlog.finalize()
            return RLMResult(
                answer=final_answer,
                iterations=iteration,
                sub_calls=repl.sub_call_count,
                elapsed_seconds=time.perf_counter() - t0,
                cache_hits=repl.cache_hits,
                trajectory=trajectory,
            )
        if final_var is not None:
            val = repl.get_variable(final_var)
            if val is not None:
                log.info("FINAL_VAR(%s) received at iteration %d", final_var, iteration)
                tlog.log_event("final_var", {"iteration": iteration, "var_name": final_var})
                tlog.finalize()
                return RLMResult(
                    answer=str(val),
                    iterations=iteration,
                    sub_calls=repl.sub_call_count,
                    elapsed_seconds=time.perf_counter() - t0,
                    cache_hits=repl.cache_hits,
                    trajectory=trajectory,
                )
            else:
                # Variable not found — tell the model
                feedback = f"[REPL] Variable '{final_var}' is not defined. Please set it first or use FINAL(answer) directly."
                history.append({"role": "user", "content": feedback})
                trajectory.append({"iteration": iteration, "role": "system", "content": feedback})
                continue

        # Extract and execute code blocks
        code_blocks = extract_repl_blocks(assistant_text)
        if not code_blocks:
            # No code and no FINAL — nudge the model
            nudge = (
                "[REPL] No code block detected and no FINAL answer provided. "
                "Please write code in ```repl ... ``` blocks to explore the context, "
                "or provide your answer with FINAL(your answer)."
            )
            history.append({"role": "user", "content": nudge})
            trajectory.append({"iteration": iteration, "role": "system", "content": nudge})
            continue

        # Execute each code block
        all_outputs: list[str] = []
        for i, code in enumerate(code_blocks):
            log.info("Executing code block %d/%d  (%d chars)", i + 1, len(code_blocks), len(code))
            tlog.log_event("code_block_exec", {
                "iteration": iteration,
                "block_index": i + 1,
                "code_len": len(code),
            })
            output = repl.execute(code)
            tlog.log_event("code_block_output", {
                "iteration": iteration,
                "block_index": i + 1,
                "output_len": len(output),
                "output_preview": output[:300],
            })
            all_outputs.append(f"[Code block {i + 1}]\n{output}")

        combined = "\n\n".join(all_outputs)
        budget_str = ""
        if settings.token_budget > 0:
            budget_str = f"  tokens: {repl.estimated_tokens_used:,}/{settings.token_budget:,}"
        repl_feedback = f"[REPL stdout — sub_calls_used: {repl.sub_call_count}/{settings.max_sub_calls}  cache_hits: {repl.cache_hits}{budget_str}]\n{combined}"

        # Token budget exhaustion — force model to wrap up
        if settings.token_budget > 0 and repl.estimated_tokens_used >= settings.token_budget:
            repl_feedback += (
                "\n\n[TOKEN BUDGET EXHAUSTED] You have used your full token budget. "
                "Provide your best answer now with FINAL(). No more sub-calls are available."
            )

        # ── Stuck detection ──
        cur_ns_size = len([k for k in repl.namespace if not k.startswith("_")])
        _prev_outputs.append(combined)
        _prev_sub_call_counts.append(repl.sub_call_count)
        _prev_namespace_sizes.append(cur_ns_size)

        # Keep only last 3
        if len(_prev_outputs) > 3:
            _prev_outputs.pop(0)
            _prev_sub_call_counts.pop(0)
            _prev_namespace_sizes.pop(0)

        if len(_prev_outputs) >= 3:
            # Check: sub_call count unchanged, namespace unchanged, output similar
            sub_calls_stale = _prev_sub_call_counts[-1] == _prev_sub_call_counts[-3]
            ns_stale = _prev_namespace_sizes[-1] == _prev_namespace_sizes[-3]
            output_similarity = SequenceMatcher(
                None, _prev_outputs[-1][:2000], _prev_outputs[-3][:2000]
            ).ratio()
            is_stuck = sub_calls_stale and ns_stale and output_similarity > 0.8

            if is_stuck:
                if not _stuck_redirected:
                    log.warning("Stuck detected at iteration %d (similarity=%.2f) — injecting redirect", iteration, output_similarity)
                    tlog.log_event("stuck_detected", {"iteration": iteration, "similarity": round(output_similarity, 3)})
                    redirect = (
                        "[REPL] You appear to be stuck — the last 3 iterations produced very similar output "
                        "with no new variables or sub-calls. Try a different approach:\n"
                        "- Use llm_query_batch() to process multiple sections in parallel\n"
                        "- Try a different chunking strategy (by headers, by paragraphs, etc.)\n"
                        "- Search for specific keywords with search(context, 'keyword')\n"
                        "- If you have enough information, provide your answer with FINAL()"
                    )
                    repl_feedback += f"\n\n{redirect}"
                    _stuck_redirected = True
                    _stuck_redirect_iteration = iteration
                elif iteration >= _stuck_redirect_iteration + 2:
                    # Still stuck after redirect — force FINAL with best available answer
                    log.warning("Still stuck after redirect — forcing FINAL at iteration %d", iteration)
                    tlog.log_event("forced_final", {"iteration": iteration})
                    # Try to find the best answer variable in namespace
                    best_answer = None
                    for var_name in ("final_answer", "answer", "result", "summary", "output"):
                        val = repl.get_variable(var_name)
                        if val is not None and str(val).strip():
                            best_answer = str(val)
                            break
                    if best_answer is None:
                        best_answer = combined[:4000] if combined else "[No answer produced — model was stuck]"
                    tlog.finalize()
                    return RLMResult(
                        answer=best_answer,
                        iterations=iteration,
                        sub_calls=repl.sub_call_count,
                        elapsed_seconds=time.perf_counter() - t0,
                        cache_hits=repl.cache_hits,
                        trajectory=trajectory,
                    )

        # Compact history if it's getting very long
        await _maybe_compact_history(history)

        history.append({"role": "user", "content": repl_feedback})
        trajectory.append({"iteration": iteration, "role": "repl", "content": repl_feedback})

    # Ran out of iterations
    log.warning("RLM reached max iterations (%d) without FINAL", settings.max_iterations)
    tlog.log_event("max_iterations_reached", {"iterations": settings.max_iterations})
    tlog.finalize()
    return RLMResult(
        answer="[RLM exhausted maximum iterations without producing a final answer. Last output shown above.]",
        iterations=settings.max_iterations,
        sub_calls=repl.sub_call_count,
        elapsed_seconds=time.perf_counter() - t0,
        cache_hits=repl.cache_hits,
        trajectory=trajectory,
    )


async def run_rlm_streaming(
    query: str,
    context: str | dict | list,
    *,
    model: str | None = None,
    temperature: float = 0.7,
):
    """Run the RLM loop, yielding progress strings for SSE streaming.

    Yields status updates like "[RLM: iteration 1, exploring context...]"
    and the final answer content at the end.
    """
    root_model = settings.root_model
    sub_model_name = model or root_model
    t0 = time.perf_counter()
    tlog = TrajectoryLogger(settings.trajectory_log_dir)

    def _on_sub_call(prompt, response, elapsed):
        tlog.log_event("sub_call", {
            "prompt_len": len(prompt), "response_len": len(response),
            "elapsed_seconds": round(elapsed, 4),
        })

    _sub_query, _sub_query_batch = _make_sub_query_fns(sub_model_name)
    repl = REPLEnvironment(
        context=context, llm_query_fn=_sub_query, on_sub_call=_on_sub_call,
        llm_query_batch_fn=_sub_query_batch,
    )
    system_prompt = _build_system_prompt(context, model_name=root_model)
    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {query}\n\nThe context has been loaded into the `context` variable in your REPL. Begin by exploring it."},
    ]

    for iteration in range(1, settings.max_iterations + 1):
        yield f"[RLM: iteration {iteration}/{settings.max_iterations}]\n"

        resp = await ollama_client.chat(model=root_model, messages=history, temperature=temperature)
        assistant_text = resp.get("message", {}).get("content", "")
        tlog.log_event("root_llm_response", {"iteration": iteration, "response_len": len(assistant_text)})
        history.append({"role": "assistant", "content": assistant_text})

        final_answer, final_var = extract_final(assistant_text)
        if final_answer is not None:
            tlog.log_event("final_answer", {"iteration": iteration})
            tlog.finalize()
            yield final_answer
            return
        if final_var is not None:
            val = repl.get_variable(final_var)
            if val is not None:
                tlog.finalize()
                yield str(val)
                return
            feedback = f"[REPL] Variable '{final_var}' is not defined."
            history.append({"role": "user", "content": feedback})
            continue

        code_blocks = extract_repl_blocks(assistant_text)
        if not code_blocks:
            nudge = "[REPL] No code block detected and no FINAL answer provided."
            history.append({"role": "user", "content": nudge})
            yield f"[RLM: no code block, nudging...]\n"
            continue

        yield f"[RLM: executing {len(code_blocks)} code block(s)...]\n"
        all_outputs = []
        for i, code in enumerate(code_blocks):
            output = repl.execute(code)
            all_outputs.append(f"[Code block {i + 1}]\n{output}")

        combined = "\n\n".join(all_outputs)
        budget_str = ""
        if settings.token_budget > 0:
            budget_str = f"  tokens: {repl.estimated_tokens_used:,}/{settings.token_budget:,}"
        repl_feedback = f"[REPL stdout — sub_calls_used: {repl.sub_call_count}/{settings.max_sub_calls}  cache_hits: {repl.cache_hits}{budget_str}]\n{combined}"
        if settings.token_budget > 0 and repl.estimated_tokens_used >= settings.token_budget:
            repl_feedback += "\n\n[TOKEN BUDGET EXHAUSTED] Provide your best answer now with FINAL()."
        await _maybe_compact_history(history)
        history.append({"role": "user", "content": repl_feedback})

        yield f"[RLM: sub_calls={repl.sub_call_count}, cache_hits={repl.cache_hits}]\n"

    tlog.finalize()
    yield "[RLM exhausted maximum iterations without producing a final answer.]"


async def _maybe_compact_history(history: list[dict[str, str]]) -> None:
    """If history grows beyond threshold, summarize early turns instead of discarding.

    Uses a sub-LLM call to produce a compact summary of the removed turns,
    preserving the chain of reasoning.
    """
    if len(history) <= settings.history_compact_threshold:
        return

    keep_tail = settings.history_compact_threshold - 4

    # Extract turns to be summarized (everything between system+first_user and keep_tail)
    turns_to_summarize = history[2:-keep_tail]
    if not turns_to_summarize:
        return

    # Format turns for summarization
    summary_parts = []
    for turn in turns_to_summarize:
        role = turn["role"]
        content = turn["content"][:2000]  # Truncate each turn for the summary prompt
        summary_parts.append(f"[{role}]: {content}")

    summary_prompt = (
        "Summarize the following conversation turns from an RLM (Recursive Language Model) session. "
        "Focus on: key findings discovered, variables created and their values, "
        "approaches tried (successful and failed), and any intermediate conclusions. "
        "Be concise but preserve all important information.\n\n"
        + "\n\n".join(summary_parts)
    )

    # Try to summarize via the root model's provider; fall back to discard on failure
    try:
        resp = await ollama_client.chat(
            model=settings.root_model,
            messages=[
                {"role": "system", "content": "Summarize concisely. Preserve key facts and variable names."},
                {"role": "user", "content": summary_prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        summary = resp.get("message", {}).get("content", "")
        summary_msg = f"[Compacted reasoning from {len(turns_to_summarize)} earlier turns]\n{summary}"
    except Exception as exc:
        log.warning("Smart compaction failed (%s), falling back to discard", exc)
        summary_msg = "[Earlier conversation turns have been compacted to save context. Continue from here.]"

    compacted = history[:2]  # system + initial user
    compacted.append({"role": "user", "content": summary_msg})
    compacted.extend(history[-keep_tail:])
    history.clear()
    history.extend(compacted)
    log.info("History compacted to %d messages (summarized %d turns)", len(history), len(turns_to_summarize))


async def passthrough_chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    extra_params: dict | None = None,
) -> dict:
    """Direct passthrough to Ollama for short prompts (no RLM scaffold).

    Returns the full Ollama response dict so callers can extract tokens.
    """
    return await ollama_client.chat(
        model=model or settings.root_model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_params=extra_params,
    )
