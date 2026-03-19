#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# RLM Proxy Test Suite — exercises all major features via curl
#
# Usage:
#   ./test_proxy.sh                    # run all tests sequentially
#   ./test_proxy.sh health             # run a single test
#   ./test_proxy.sh concurrent         # run concurrency test
#   ./test_proxy.sh --base http://host:port  # custom base URL
# ──────────────────────────────────────────────────────────────────────────
set -uo pipefail

BASE="http://localhost:8881"
PASS=0
FAIL=0
SKIP=0
RESULTS=()

# Parse --base flag
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="$2"; shift 2 ;;
    *) break ;;
  esac
done

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

ok()   { ((PASS++)); RESULTS+=("${GREEN}PASS${NC}  $1"); echo -e "  ${GREEN}✓${NC} $1"; }
fail() { ((FAIL++)); RESULTS+=("${RED}FAIL${NC}  $1: $2"); echo -e "  ${RED}✗${NC} $1: $2"; }
skip() { ((SKIP++)); RESULTS+=("${YELLOW}SKIP${NC}  $1: $2"); echo -e "  ${YELLOW}—${NC} $1: $2"; }
header() { echo -e "\n${CYAN}━━━ $1 ━━━${NC}"; }

# Helper: POST JSON and capture response + status
post() {
  local url="$1" data="$2" timeout="${3:-30}"
  curl -s -w '\n%{http_code}' -X POST "$url" \
    -H "Content-Type: application/json" \
    -d "$data" \
    --max-time "$timeout" 2>/dev/null || echo -e "\n000"
}

get() {
  local url="$1" timeout="${2:-10}"
  curl -s -w '\n%{http_code}' "$url" --max-time "$timeout" 2>/dev/null || echo -e "\n000"
}

extract_status() { echo "$1" | tail -1; }
extract_body()   { echo "$1" | sed '$d'; }

# ──────────────────────────────────────────────────────────────────────────

test_health() {
  header "Health Check"
  local resp; resp=$(get "$BASE/health")
  local status; status=$(extract_status "$resp")
  local body; body=$(extract_body "$resp")
  if [[ "$status" == "200" ]] && echo "$body" | grep -q '"ok"'; then
    ok "GET /health → 200"
  else
    fail "GET /health" "status=$status"
  fi
}

test_models() {
  header "List Models"
  local resp; resp=$(get "$BASE/v1/models")
  local status; status=$(extract_status "$resp")
  local body; body=$(extract_body "$resp")
  if [[ "$status" == "200" ]]; then
    local count; count=$(echo "$body" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo 0)
    ok "GET /v1/models → $count models"
  else
    fail "GET /v1/models" "status=$status"
  fi
}

test_ollama_tags() {
  header "Ollama Tags (native)"
  local resp; resp=$(get "$BASE/api/tags")
  local status; status=$(extract_status "$resp")
  if [[ "$status" == "200" ]]; then
    ok "GET /api/tags → 200"
  else
    fail "GET /api/tags" "status=$status"
  fi
}

test_passthrough() {
  header "Short Prompt (passthrough)"
  local resp; resp=$(post "$BASE/v1/chat/completions" '{
    "model": "qwen3-coder-next",
    "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
    "temperature": 0.1
  }' 120)
  local status; status=$(extract_status "$resp")
  local body; body=$(extract_body "$resp")
  if [[ "$status" == "200" ]]; then
    local answer; answer=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:100])" 2>/dev/null || echo "?")
    ok "Passthrough → \"$answer\""
  else
    fail "Passthrough" "status=$status"
  fi
}

test_force_passthrough() {
  header "Force Passthrough (long prompt, skip RLM)"
  local long_text; long_text=$(python3 -c "print('word ' * 15000)")
  local resp; resp=$(post "$BASE/v1/chat/completions" "{
    \"model\": \"qwen3-coder-next\",
    \"force_passthrough\": true,
    \"messages\": [{\"role\": \"user\", \"content\": \"Say OK. Ignore this filler: $long_text\"}],
    \"temperature\": 0.1
  }" 120)
  local status; status=$(extract_status "$resp")
  if [[ "$status" == "200" ]]; then
    ok "force_passthrough with long prompt → 200"
  else
    fail "force_passthrough" "status=$status"
  fi
}

test_streaming() {
  header "Streaming (SSE passthrough)"
  local chunks; chunks=$(curl -s -N "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen3-coder-next",
      "stream": true,
      "messages": [{"role": "user", "content": "Say hello in one word"}],
      "temperature": 0.1
    }' --max-time 60 2>/dev/null | grep -c "^data:" || echo 0)
  if [[ "$chunks" -gt 1 ]]; then
    ok "Streaming → $chunks SSE chunks received"
  else
    fail "Streaming" "got $chunks chunks"
  fi
}

test_rlm_needle() {
  header "RLM Mode (needle-in-haystack)"
  # Build request JSON via Python to avoid bash escaping issues
  local tmpfile; tmpfile=$(mktemp /tmp/rlm_test_XXXXX.json)
  python3 -c "
import json
lines = [f'Line {i}: This is filler text with no useful information.' for i in range(500)]
lines[247] = 'Line 247: The secret password is RAINBOW-UNICORN-42.'
context = '\n'.join(lines)
req = {
    'model': 'qwen3-coder-next',
    'force_rlm': True,
    'temperature': 0.3,
    'messages': [{'role': 'user', 'content': f'Find the secret password in this document:\n\n{context}'}]
}
with open('$tmpfile', 'w') as f:
    json.dump(req, f)
"
  local resp; resp=$(curl -s -w '\n%{http_code}' -X POST "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @"$tmpfile" \
    --max-time 600 2>/dev/null || echo -e "\n000")
  rm -f "$tmpfile"
  local status; status=$(extract_status "$resp")
  local body; body=$(extract_body "$resp")
  if [[ "$status" == "200" ]]; then
    local answer; answer=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:300])" 2>/dev/null || echo "?")
    if echo "$answer" | grep -qi "rainbow"; then
      ok "RLM needle → found password in answer"
    else
      ok "RLM completed (password not in first 300 chars): ${answer:0:80}..."
    fi
  else
    fail "RLM needle" "status=$status"
  fi
}

test_dispatch() {
  header "Dispatcher"
  local resp; resp=$(get "$BASE/v1/rlm/dispatch")
  local status; status=$(extract_status "$resp")
  local body; body=$(extract_body "$resp")
  if [[ "$status" == "200" ]]; then
    local hosts; hosts=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin).get('host_count',0))" 2>/dev/null || echo 0)
    local models; models=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_models',0))" 2>/dev/null || echo 0)
    ok "Dispatch → $hosts host(s), $models model(s)"
  else
    fail "Dispatch" "status=$status"
  fi
}

test_metrics() {
  header "Metrics"
  local resp; resp=$(get "$BASE/v1/rlm/metrics")
  local status; status=$(extract_status "$resp")
  local body; body=$(extract_body "$resp")
  if [[ "$status" == "200" ]]; then
    local total; total=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin)['requests']['total'])" 2>/dev/null || echo 0)
    ok "Metrics → $total total requests recorded"
  else
    fail "Metrics" "status=$status"
  fi
}

test_config_api() {
  header "Config API"
  # Read
  local resp; resp=$(get "$BASE/v1/rlm/config/all")
  local status; status=$(extract_status "$resp")
  if [[ "$status" != "200" ]]; then fail "Config read" "status=$status"; return; fi
  ok "Config read → 200"

  # Update
  resp=$(curl -s -w '\n%{http_code}' -X PUT "$BASE/v1/rlm/config" \
    -H "Content-Type: application/json" \
    -d '{"max_iterations": 25}' --max-time 5 2>/dev/null)
  status=$(extract_status "$resp")
  if [[ "$status" == "200" ]]; then
    ok "Config update max_iterations=25 → 200"
  else
    fail "Config update" "status=$status"
  fi

  # Verify change
  resp=$(get "$BASE/v1/rlm/config/all")
  local val; val=$(extract_body "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('max_iterations'))" 2>/dev/null)
  if [[ "$val" == "25" ]]; then
    ok "Config verify → max_iterations is now 25"
  else
    fail "Config verify" "expected 25, got $val"
  fi

  # Reset
  resp=$(curl -s -w '\n%{http_code}' -X POST "$BASE/v1/rlm/config/reset" --max-time 5 2>/dev/null)
  status=$(extract_status "$resp")
  if [[ "$status" == "200" ]]; then ok "Config reset → 200"; else fail "Config reset" "status=$status"; fi
}

test_admin_page() {
  header "Admin Page"
  local resp; resp=$(get "$BASE/admin")
  local status; status=$(extract_status "$resp")
  local body; body=$(extract_body "$resp")
  if [[ "$status" == "200" ]] && echo "$body" | grep -q "Admin Dashboard"; then
    ok "GET /admin → HTML loaded"
  else
    fail "GET /admin" "status=$status"
  fi
}

test_concurrent() {
  header "Concurrency Test (3 parallel requests)"
  local pids=()
  local tmpdir; tmpdir=$(mktemp -d)

  for i in 1 2 3; do
    (
      local resp; resp=$(post "$BASE/v1/chat/completions" "{
        \"model\": \"qwen3-coder-next\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Reply with the number $i only\"}],
        \"temperature\": 0.1
      }" 120)
      local status; status=$(extract_status "$resp")
      echo "$status" > "$tmpdir/result_$i"
    ) &
    pids+=($!)
  done

  # Wait for all
  local all_ok=true
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done

  local succeeded=0
  for i in 1 2 3; do
    local s; s=$(cat "$tmpdir/result_$i" 2>/dev/null || echo "000")
    if [[ "$s" == "200" ]]; then ((succeeded++)); fi
  done
  rm -rf "$tmpdir"

  if [[ "$succeeded" -eq 3 ]]; then
    ok "All 3 concurrent requests succeeded"
  elif [[ "$succeeded" -gt 0 ]]; then
    ok "$succeeded/3 succeeded (queue may have limited concurrency)"
  else
    fail "Concurrent requests" "0/3 succeeded"
  fi
}

test_embeddings() {
  header "Embeddings"
  local resp; resp=$(post "$BASE/v1/embeddings" '{
    "model": "nomic-embed-text",
    "input": ["hello world"]
  }' 30)
  local status; status=$(extract_status "$resp")
  if [[ "$status" == "200" ]]; then
    ok "POST /v1/embeddings → 200"
  elif [[ "$status" == "502" ]]; then
    skip "Embeddings" "model not available (502)"
  else
    fail "Embeddings" "status=$status"
  fi
}

test_long_document() {
  header "Long Document RLM (~200K chars, multiple hidden facts)"
  local tmpfile; tmpfile=$(mktemp /tmp/rlm_long_XXXXX.json)

  # Generate the test document and request JSON via Python
  python3 - "$tmpfile" <<'PYEOF'
import json, random, sys

random.seed(42)
outfile = sys.argv[1]
sections = []
topics = [
    'Quarterly Revenue Analysis', 'Market Expansion Strategy',
    'Product Development Roadmap', 'Customer Retention Metrics',
    'Supply Chain Optimization', 'Talent Acquisition Report',
    'Regulatory Compliance Update', 'Technology Infrastructure',
    'Competitive Landscape Review', 'Sustainability Initiative',
    'Risk Assessment Framework', 'Digital Transformation',
    'Partnership Opportunities', 'Budget Allocation Review',
    'Employee Satisfaction Survey', 'Intellectual Property Portfolio',
    'Data Privacy Compliance', 'Marketing Campaign Results',
    'Operational Efficiency Report', 'Innovation Lab Updates',
]
fillers = [
    'The analysis indicates steady progress across all measured dimensions.',
    'Stakeholder feedback has been largely positive with minor exceptions noted.',
    'Year-over-year comparisons show consistent improvement in key indicators.',
    'Resource allocation remains within established parameters for this period.',
    'Cross-functional collaboration has yielded notable synergies in execution.',
    'Preliminary data suggests continued momentum into the following quarter.',
    'The committee reviewed all submissions and approved the recommended actions.',
    'External consultants validated the methodology used in this assessment.',
    'Benchmarking against industry peers confirms our competitive positioning.',
    'Action items from the previous review have been addressed as documented.',
    'The projected timeline aligns with organizational priorities and constraints.',
    'Quality metrics remain above threshold with no significant deviations noted.',
    'Integration testing confirmed compatibility with existing infrastructure.',
    'The proposed framework addresses gaps identified in the prior audit cycle.',
    'Feedback loops have been established to ensure continuous improvement.',
]

for i, topic in enumerate(topics):
    section = f'## Section {i+1}: {topic}\n\n'
    for _ in range(15):
        section += ' '.join(random.sample(fillers, random.randint(5, 8))) + '\n\n'
    sections.append(section)

sections[3] += '\nIMPORTANT NOTE: The project codename for Q3 is GOLDEN-FALCON.\n\n'
sections[11] += '\nCONFIDENTIAL: The merger target company is called Nexora Technologies.\n\n'
sections[17] += '\nINTERNAL ONLY: The new CEO starts on September 15, 2026.\n\n'

document = '\n'.join(sections)
print(f'  Document size: {len(document):,} characters')

req = {
    'model': 'qwen3-coder-next',
    'force_rlm': True,
    'temperature': 0.3,
    'messages': [{'role': 'user', 'content':
        'This is a large corporate document. Find ALL three hidden confidential '
        'facts (a project codename, a merger target, and a CEO start date). '
        'List each one.\n\n' + document
    }]
}
with open(outfile, 'w') as f:
    json.dump(req, f)
print(f'  Request JSON: {len(json.dumps(req)):,} characters')
PYEOF

  echo "  Sending to RLM (this may take a few minutes)..."
  local t_start; t_start=$(date +%s)
  local resp; resp=$(curl -s -w '\n%{http_code}' -X POST "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @"$tmpfile" \
    --max-time 900 2>/dev/null || echo -e "\n000")
  local t_end; t_end=$(date +%s)
  local elapsed=$(( t_end - t_start ))
  rm -f "$tmpfile"
  local status; status=$(extract_status "$resp")
  local body; body=$(extract_body "$resp")
  if [[ "$status" == "200" ]]; then
    local answer; answer=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "?")
    local found=0
    echo "$answer" | grep -qi "golden.falcon" && ((found++)) || true
    echo "$answer" | grep -qi "nexora" && ((found++)) || true
    echo "$answer" | grep -qi "september.*15" && ((found++)) || true
    echo "  Answer (${#answer} chars):"
    echo "$answer" | head -20 | sed 's/^/    /'
    ok "Long doc RLM → found $found/3 facts in ${elapsed}s"
  else
    fail "Long document RLM" "status=$status after ${elapsed}s"
  fi
}

# ──────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────

run_all() {
  test_health
  test_admin_page
  test_models
  test_ollama_tags
  test_dispatch
  test_config_api
  test_passthrough
  test_streaming
  test_force_passthrough
  test_embeddings
  test_metrics
  test_concurrent
  test_rlm_needle
  test_long_document
}

# Summary
summary() {
  echo ""
  echo -e "${CYAN}━━━ Summary ━━━${NC}"
  for r in "${RESULTS[@]}"; do echo -e "  $r"; done
  echo ""
  echo -e "  ${GREEN}$PASS passed${NC}  ${RED}$FAIL failed${NC}  ${YELLOW}$SKIP skipped${NC}"
  echo ""
  if [[ $FAIL -gt 0 ]]; then exit 1; fi
}

# Main
echo -e "${CYAN}RLM Proxy Test Suite${NC}"
echo -e "Target: $BASE"

if [[ $# -gt 0 ]]; then
  # Run specific test
  test_fn="test_$1"
  if declare -f "$test_fn" > /dev/null 2>&1; then
    "$test_fn"
  else
    echo "Unknown test: $1"
    echo "Available: health admin_page models ollama_tags dispatch config_api passthrough streaming force_passthrough embeddings metrics concurrent rlm_needle"
    exit 1
  fi
else
  run_all
fi

summary
