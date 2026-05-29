# Extended API Testing Design: Feature Discovery, Parameter Optimization & Reliability

**Date:** 2026-05-29  
**Repository:** `llm-boundary-tests`  
**Model Under Test:** `taalas-llama3.1-8b` at `https://ai.shenthar.me/v1/chat/completions`

---

## Objective

Exhaustively test the `taalas-llama3.1-8b` API deployment across three dimensions not covered by the initial report:

1. **API Feature Discovery** — probe undocumented API capabilities (`response_format`, `seed`, `stop`, `logprobs`, `tools/function calling`, `n`, `max_tokens` ceiling, message role validation)
2. **Parameter Optimization** — systematic sweeps of temperature, top_p, frequency_penalty, presence_penalty, and combined interaction effects to find optimal settings for quality/consistency
3. **Reliability & Boundaries** — multi-turn conversation degradation, sustained load (1000 req), error catalog, edge input behaviors

---

## Phase 1: API Feature Discovery (`7-api-features/benchmark.py`)

### Test 1.1: `response_format: json_object`
- **Method:** Send request with `response_format: {"type": "json_object"}` and compare JSON parse success rate vs baseline
- **Probes:** 20 prompts (10 structured extraction, 10 free-form)
- **Metrics:** API acceptance (status code), JSON parse success rate vs baseline, response quality

### Test 1.2: `seed` parameter
- **Method:** Send same prompt 5× with `seed=42`, then 5× with `seed=99`, then 5× without seed
- **Probes:** 5 diverse prompts × 3 conditions = 15 total
- **Metrics:** Determinism (identical output with same seed), output difference with different seeds

### Test 1.3: `stop` sequences
- **Method:** Send prompts with single stop token (`"\n"`), multi-stop, and stop token that appears mid-generation
- **Probes:** 10 probes covering: single stop, array of stops, stop in content, stop not in content
- **Metrics:** Correct truncation at stop, no extra content after stop

### Test 1.4: `logprobs`
- **Method:** Request with `logprobs: true, top_logprobs: 5` and inspect response
- **Probes:** 5 prompts with varying output types
- **Metrics:** Does API return logprobs? Structure and completeness of logprob data

### Test 1.5: `tools` / function calling
- **Method:** Send request with `tools` array containing 3 tool definitions (get_weather, search_web, calculate)
- **Probes:** 5 prompts each targeting a different tool
- **Metrics:** Does API accept `tools`? Does it return `tool_calls` in response?

### Test 1.6: `n` (multiple completions)
- **Method:** Send request with `n: 3` and inspect if multiple choices returned
- **Probes:** 5 prompts
- **Metrics:** Choice count, diversity of responses

### Test 1.7: `max_tokens` ceiling
- **Method:** Binary search for maximum output tokens — start at 4096, increase until response truncated or error
- **Probes:** 8–10 probes in binary search
- **Metrics:** Max output tokens before truncation/error

### Test 1.8: Message format validation
- **Method:** Test system/user/assistant role combinations, empty messages, missing fields, invalid roles
- **Probes:** 12 edge-case message formats
- **Metrics:** API error codes, response behavior

---

## Phase 2: Parameter Optimization (`8-parameter-tuning/benchmark.py`)

### Test 2.1: Temperature sweep
- **Values:** [0.0, 0.2, 0.5, 0.7, 1.0]
- **Method:** Each value tested on 10 evaluation probes from the faithfulness/scoring suite. Same prompt sent 5× per temperature for consistency.
- **Metrics:** Mean faithfulness score, consistency (1 - variance), hallucination rate

### Test 2.2: top_p sweep
- **Values:** [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
- **Method:** Fixed temperature=0.7, same 10 probes × 3 iterations per value
- **Metrics:** Output quality, diversity, consistency

### Test 2.3: frequency_penalty sweep
- **Values:** [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]
- **Method:** 10 scope-creep probes (the known failure mode from original report). Each value × 3 iterations.
- **Metrics:** Verbosity reduction (token count), scope creep score, quality preservation

### Test 2.4: presence_penalty sweep
- **Values:** [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]
- **Method:** Same design as frequency_penalty sweep
- **Metrics:** Topic diversity, repetition reduction

### Test 2.5: Best combo discovery
- **Method:** Test top-3 settings from each sweep combined into a single optimal configuration
- **Probes:** Full evaluation suite (20 probes) × 3 iterations
- **Metrics:** Composite quality score

---

## Phase 3: Reliability & Boundaries (`9-reliability/benchmark.py`)

### Test 3.1: Multi-turn conversation (15 turns)
- **Method:** Start with system prompt + one turn, then chain 15 assistant/user alternations. Inject specific facts in early turns and query them later.
- **Probes:** 3 conversation chains × 15 turns each = 45 total calls
- **Metrics:** Fact retention per turn (recall decay curve), context window exhaustion point, instruction drift

### Test 3.2: Sustained load (1000 sequential requests)
- **Method:** 1000 requests sent one-at-a-time (no concurrency) tracking per-request latency and error rate
- **Probes:** 1000 requests across diverse prompt types
- **Metrics:** Error rate by 100-request buckets, latency drift over time, p50/p95/p99 per bucket

### Test 3.3: Error catalog
- **Method:** Send requests designed to trigger each error condition
- **Probes:** Invalid auth, missing fields, oversized body, invalid model, unsupported params, timeout
- **Metrics:** Document all error codes, response bodies, HTTP status codes

### Test 3.4: Edge inputs
- **Method:** 12 edge input types
- **Probes:** Empty string, whitespace-only, very long word (10000 chars), unicode bom, null bytes, emoji-only, 10000-word input, HTML injection, SQL injection, 5000-digit number
- **Metrics:** API response behavior, error rate, crash detection

---

## Execution Plan

1. Write `7-api-features/benchmark.py` — feature discovery tests
2. Write `8-parameter-tuning/benchmark.py` — parameter optimization sweeps
3. Write `9-reliability/benchmark.py` — reliability & stress tests
4. Write `aggregate_results.py` — reads all results, formats report section
5. Run all scripts sequentially
6. Append findings to REPORT.md

All scripts follow the existing repo patterns:
- Self-contained Python + `requests`
- `API_URL`, `API_KEY`, `MODEL` at top
- Structured `results.json` output
- Console progress during execution
