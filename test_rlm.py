#!/usr/bin/env python3
"""Quick smoke test for the RLM proxy."""

import httpx
import json
import sys

BASE = "http://localhost:8881"


def test_health():
    r = httpx.get(f"{BASE}/health", timeout=5)
    print(f"Health: {r.json()}")
    assert r.status_code == 200


def test_models():
    r = httpx.get(f"{BASE}/v1/models", timeout=10)
    data = r.json()
    print(f"Models: {len(data['data'])} available")
    for m in data["data"][:5]:
        print(f"  - {m['id']}")


def test_short_passthrough():
    """Short prompt — should passthrough directly to Ollama."""
    r = httpx.post(
        f"{BASE}/v1/chat/completions",
        json={
            "model": "qwen3-coder-next",
            "messages": [{"role": "user", "content": "What is 2 + 2? Reply with just the number."}],
            "temperature": 0.1,
        },
        timeout=120,
    )
    data = r.json()
    answer = data["choices"][0]["message"]["content"]
    print(f"Short passthrough: {answer[:200]}")


def test_rlm_forced():
    """Force RLM mode on a task that benefits from code execution."""
    # Create a synthetic "long" context with hidden info
    lines = [f"Line {i}: This is filler text with no useful information." for i in range(500)]
    lines[247] = "Line 247: The secret password is 'rainbow-unicorn-42'."
    lines[389] = "Line 389: The launch date is March 15, 2026."
    context = "\n".join(lines)

    r = httpx.post(
        f"{BASE}/v1/chat/completions",
        json={
            "model": "qwen3-coder-next",
            "messages": [
                {"role": "user", "content": f"Find the secret password hidden in this document:\n\n{context}"},
            ],
            "force_rlm": True,
            "temperature": 0.3,
        },
        timeout=600,
    )
    data = r.json()
    answer = data["choices"][0]["message"]["content"]
    print(f"RLM forced result: {answer[:500]}")
    if "rainbow-unicorn-42" in answer.lower():
        print("  ✓ Password found correctly!")
    else:
        print("  ✗ Password not found in answer")


if __name__ == "__main__":
    tests = [test_health, test_models, test_short_passthrough, test_rlm_forced]

    if len(sys.argv) > 1:
        name = sys.argv[1]
        fn = globals().get(f"test_{name}")
        if fn:
            fn()
        else:
            print(f"Unknown test: {name}")
            sys.exit(1)
    else:
        for t in tests:
            print(f"\n{'─'*60}")
            print(f"Running {t.__name__}...")
            try:
                t()
                print(f"  ✓ {t.__name__} passed")
            except Exception as e:
                print(f"  ✗ {t.__name__} failed: {e}")
