# How to Reproduce This Experiment

Complete step-by-step instructions to reproduce every result in `REPORT.md` from scratch.

---

## Prerequisites

| Requirement | Version used |
|-------------|-------------|
| Python | 3.9.6 or later |
| `requests` | 2.32.5 |
| An OpenAI-compatible LLM API endpoint | see below |
| Git | any recent version |

No other libraries are required. All scripts use only the Python standard library plus `requests`.

---

## 1. Clone the Repository

```bash
git clone https://github.com/KTS-o7/llm-boundary-tests.git
cd llm-boundary-tests
```

---

## 2. Install Dependencies

```bash
pip install requests==2.32.5
```

---

## 3. Configure the API Key

Copy the example env file and add your key:

```bash
cp .env.example .env
```

Edit `.env`:

```
LLM_API_KEY=your_api_key_here
```

Export it into your shell before running any script:

```bash
export LLM_API_KEY=$(grep LLM_API_KEY .env | cut -d= -f2)
```

Every script reads the key from `os.environ.get("LLM_API_KEY", "")`. Nothing is hard-coded. Do **not** commit `.env`.

### Pointing at a Different Endpoint

Each script has a `API_URL` constant near the top. If you are running your own LLM server, change that line:

```python
API_URL = "https://your-server/v1/chat/completions"
```

The endpoint must accept the OpenAI chat completions schema (`model`, `messages`, `temperature`, `max_tokens`).

---

## 4. Run the Benchmarks

Run each benchmark independently. Each script is self-contained — it generates its own dataset internally and writes a `results.json` to its own directory. Results from a previous run are overwritten.

Each benchmark takes **2–10 minutes** depending on concurrency and your network latency to the API.

### Benchmark 1 — High-Throughput Classification Pipeline

**What it measures:** Maximum safe concurrency level; throughput ceiling (items/sec); latency distribution.

```bash
python 1-classification-pipeline/benchmark.py
```

Output: `1-classification-pipeline/results.json`

The script sweeps thread counts (1, 5, 10, 20, 30, 40) and classifies 50 product-review texts per sweep. Expect 5–8 minutes total.

---

### Benchmark 2 — Structured Information Extraction

**What it measures:** NER/extraction quality (F1, precision, recall, consistency) as document word count increases from 200 to 2,500 words.

```bash
python 2-structured-extraction/benchmark.py
```

Output: `2-structured-extraction/results.json`

The script builds synthetic documents at 5 size points (200, 500, 1000, 1500, 2000, 2500 words) and runs extraction 3× per size point to measure consistency.

---

### Benchmark 3 — Query Routing and Cost Reduction

**What it measures:** Routing accuracy (SIMPLE / MEDIUM / COMPLEX); estimated cost savings from pre-filtering.

```bash
python 3-routing-filter/benchmark.py
```

Output: `3-routing-filter/results.json`

60 queries are classified by the LLM and routed. The "slow model" is simulated with `time.sleep(3.0)` — no second API is required.

---

### Benchmark 4 — Parallel Chunked Document Processing

**What it measures:** Chunk-size vs. entity-extraction quality; parallel vs. sequential speedup ratio; optimal overlap.

```bash
python 4-chunked-documents/benchmark.py
```

Output: `4-chunked-documents/results.json`

Tests chunk sizes of 300, 500, 750, and 1000 words at 50-word overlap, then compares sequential vs. parallel execution on the optimal chunk size.

---

### Benchmark 5 — Real-Time Interactive Latency

**What it measures:** p50/p95/p99 latency for four request types (PING, SHORT, MEDIUM, LONG); streaming vs. non-streaming comparison.

```bash
python 5-realtime-interactive/benchmark.py
```

Output: `5-realtime-interactive/results.json`

Sends 20 requests per type (80 total non-streaming + 20 streaming). Takes 2–4 minutes.

---

### Benchmark 6 — Agentic Tool-Call Routing

**What it measures:** Multi-step tool routing accuracy; loop latency per step; maximum viable agent depth.

```bash
python 6-agent-tool-routing/benchmark.py
```

Output: `6-agent-tool-routing/results.json`

Simulates 7 tools (calculator, weather, calendar, search, email, code, database) with a mock execution layer. Runs 15 tasks across depths 1–7.

---

## 4b. Extended Benchmarks (Sections 19–21 of REPORT.md)

These three benchmarks probe undocumented API features, optimize sampling parameters, and stress-test reliability. Run after the 6 core benchmarks.

### Benchmark 7 — API Feature Discovery

**What it measures:** Support for undocumented OpenAI API parameters: `response_format`, `seed`, `stop`, `logprobs`, `tools`/function calling, `n` (multiple choices), `max_tokens` ceiling, and message validation edge cases.

```bash
python 7-api-features/benchmark.py
```

Output: `7-api-features/results.json`

Makes ~80 API calls. Takes 3–5 minutes.

---

### Benchmark 8 — Parameter Optimization

**What it measures:** Temperature sweep (0.0–1.0), top_p sweep, frequency_penalty sweep, presence_penalty sweep, and best-combo comparison vs. defaults.

```bash
python 8-parameter-tuning/benchmark.py
```

Output: `8-parameter-tuning/results.json`

Makes ~610 API calls. Takes 15–25 minutes.

---

### Benchmark 9 — Reliability & Boundaries

**What it measures:** Multi-turn conversation (3 chains × 15 turns), sustained load (1,000 sequential requests), error catalog (10 invalid configurations), and edge inputs (12 edge cases).

```bash
python 9-reliability/benchmark.py
```

Output: `9-reliability/results.json`

Makes ~1,070 API calls. Takes 20–30 minutes (1,000-request sustained load test dominates).

---

## 5. Run the Evaluation Scripts

These scripts read the `results.json` files produced by the benchmarks plus call the API with new probes. Run them after all 6 benchmarks complete.

### Recall Evaluation

**What it measures:** Recall@1 and Recall@3 across classification, routing, extraction, and chunked-document domains.

```bash
python eval_recall.py
```

Output: `eval_recall_results.json`

Reads `results.json` from all 6 benchmark directories. No additional API calls.

---

### Instruction Faithfulness

**What it measures:** How well the model follows explicit schema, type, enum, negation, and numeric-precision constraints.

```bash
python eval_instruction_faithfulness.py
```

Output: `eval_instruction_faithfulness_results.json`

Makes ~40 live API calls. Takes 2–3 minutes.

---

### Intent Faithfulness

**What it measures:** Literal vs. intent interpretation; scope creep; system-prompt vs. user-prompt priority conflicts.

```bash
python eval_intent_faithfulness.py
```

Output: `eval_intent_faithfulness_results.json`

Makes ~50 live API calls across 5 sub-dimensions. Takes 3–5 minutes.

---

### XML vs. Plain-Text System Prompts

**What it measures:** Overall score delta between XML-formatted and plain-text system prompts across 8 task types.

```bash
python eval_xml_vs_plain.py
```

Output: `eval_xml_vs_plain_results.json`

---

### XML Weak-Spots

**What it measures:** Targeted XML variations on the worst-performing task types; identifies the one scenario where XML helps.

```bash
python eval_xml_weakspots.py
```

Output: `eval_xml_weakspots_results.json`

---

## 6. Run the Prompt Engineering Experiments

These are independent A/B experiments. They do not depend on the benchmark or eval results.

### Few-Shot + Output Anchoring

**What it measures:** Zero-shot vs. few-shot on 4 task types; output prefix anchoring effect on JSON validity.

```bash
python prompt-techniques/few-shot/experiment.py
```

Output: `prompt-techniques/few-shot/results.json`

---

### Chain-of-Thought Scratchpad

**What it measures:** CoT vs. no-CoT on 4 tasks (factual recall, arithmetic, classification, extraction).

```bash
python prompt-techniques/cot-scratchpad/experiment.py
```

Output: `prompt-techniques/cot-scratchpad/results.json`

> **Note:** This experiment is expected to show CoT *hurting* performance (−13.3 pp on factual accuracy). This is the correct finding — the model is not a reasoning model.

---

### Instruction Decomposition

**What it measures:** Flat multi-constraint prompts vs. numbered decomposed constraints.

```bash
python prompt-techniques/instruction-decomposition/experiment.py
```

Output: `prompt-techniques/instruction-decomposition/results.json`

---

## 7. Red Team

**What it measures:** Safety and jailbreak resistance across 7 attack categories (20 tests total).

The red-team runner is at the repo root (sibling to this repo, in `~/redteam.py` from the original setup). If you want to run it against this repo's API configuration:

```bash
# From the parent directory where redteam.py lives:
python redteam.py
```

The script prints a colour-coded scorecard to stdout. Results are not written to a file — capture with `tee` if needed:

```bash
python redteam.py | tee redteam_results.txt
```

Expected score: ~40% (8/20 safe). The 12 failures are documented in Table 10.1 of `REPORT.md`.

---

## 8. Recommended Run Order

If you want to reproduce everything end-to-end, this is the safest sequence:

```bash
# 0. Setup
git clone https://github.com/KTS-o7/llm-boundary-tests.git
cd llm-boundary-tests
pip install requests==2.32.5
cp .env.example .env
# edit .env with your API key
export LLM_API_KEY=$(grep LLM_API_KEY .env | cut -d= -f2)

# 1. Core Benchmarks (can run in any order; parallel ok)
python 1-classification-pipeline/benchmark.py
python 2-structured-extraction/benchmark.py
python 3-routing-filter/benchmark.py
python 4-chunked-documents/benchmark.py
python 5-realtime-interactive/benchmark.py
python 6-agent-tool-routing/benchmark.py

# 2. Extended Benchmarks (requires core results)
python 7-api-features/benchmark.py
python 8-parameter-tuning/benchmark.py
python 9-reliability/benchmark.py

# 3. Evaluations (after all benchmarks)
python eval_recall.py
python eval_instruction_faithfulness.py
python eval_intent_faithfulness.py
python eval_xml_vs_plain.py
python eval_xml_weakspots.py

# 4. Prompt technique experiments (independent; run any time)
python prompt-techniques/few-shot/experiment.py
python prompt-techniques/cot-scratchpad/experiment.py
python prompt-techniques/instruction-decomposition/experiment.py
```

Total wall time: approximately **90–180 minutes** depending on API latency (extended benchmarks add 35–60 min).

---

## 9. Verifying Results

After each script completes, the key numbers to check against `REPORT.md` are:

| Script | Key metric to verify | Expected value |
|--------|---------------------|----------------|
| `1-classification-pipeline/benchmark.py` | Throughput ceiling | ~47 items/sec at 30 threads |
| `2-structured-extraction/benchmark.py` | F1 at 1,500 words | ~0.872 |
| `3-routing-filter/benchmark.py` | Routing accuracy | ~81.7% |
| `4-chunked-documents/benchmark.py` | Parallel speedup | ~3.76× |
| `5-realtime-interactive/benchmark.py` | p50 latency (PING) | ~505 ms |
| `6-agent-tool-routing/benchmark.py` | Tool routing accuracy | 100% |
| `eval_instruction_faithfulness.py` | Overall score | ~0.827 |
| `eval_intent_faithfulness.py` | Overall score | ~0.826 |
| `eval_xml_vs_plain.py` | XML delta | −17.9 pp |
| `prompt-techniques/few-shot/experiment.py` | Array sort delta | +10.7 pp |
| `prompt-techniques/cot-scratchpad/experiment.py` | Factual delta | −13.3 pp |
| `7-api-features/benchmark.py` | Supported features (of 8) | 1/8 (only `stop` works) |
| `8-parameter-tuning/benchmark.py` | Optimal temp | 0.0 (default is optimal) |
| `9-reliability/benchmark.py` | Sustained load errors | 0 / 1,000 |

Minor variation (±5%) is expected due to temperature sampling. All benchmarks use `temperature=0` where possible; some tasks require nonzero temperature and will show run-to-run variance.

---

## 10. Expected File Tree After Full Run

```
llm-boundary-tests/
├── 1-classification-pipeline/
│   ├── benchmark.py
│   └── results.json              ← generated
├── 2-structured-extraction/
│   ├── benchmark.py
│   └── results.json              ← generated
├── 3-routing-filter/
│   ├── benchmark.py
│   └── results.json              ← generated
├── 4-chunked-documents/
│   ├── benchmark.py
│   └── results.json              ← generated
├── 5-realtime-interactive/
│   ├── benchmark.py
│   └── results.json              ← generated
├── 6-agent-tool-routing/
│   ├── benchmark.py
│   └── results.json              ← generated
├── 7-api-features/                          ← Section 19
│   ├── benchmark.py
│   └── results.json              ← generated
├── 8-parameter-tuning/                      ← Section 20
│   ├── benchmark.py
│   └── results.json              ← generated
├── 9-reliability/                           ← Section 21
│   ├── benchmark.py
│   └── results.json              ← generated
├── eval_recall.py
├── eval_recall_results.json       ← generated
├── eval_instruction_faithfulness.py
├── eval_instruction_faithfulness_results.json  ← generated
├── eval_intent_faithfulness.py
├── eval_intent_faithfulness_results.json       ← generated
├── eval_xml_vs_plain.py
├── eval_xml_vs_plain_results.json              ← generated
├── eval_xml_weakspots.py
├── eval_xml_weakspots_results.json             ← generated
├── aggregate_results.py
├── aggregate_results.json          ← generated
├── prompt-techniques/
│   ├── few-shot/
│   │   ├── experiment.py
│   │   └── results.json          ← generated
│   ├── cot-scratchpad/
│   │   ├── experiment.py
│   │   └── results.json          ← generated
│   └── instruction-decomposition/
│       ├── experiment.py
│       └── results.json          ← generated
├── .env.example
├── .gitignore
├── README.md
├── EXPERIMENT.md                  ← this file
└── REPORT.md
```

---

## 11. Adapting to a Different Model

To evaluate a different OpenAI-compatible model, change these two constants at the top of every script:

```python
API_URL = "https://your-server/v1/chat/completions"
MODEL   = "your-model-name"
```

A one-liner to do all files at once (review the diff before committing):

```bash
# Replace API URL
find . -name "*.py" -exec sed -i '' \
  's|https://ai.shenthar.me/v1/chat/completions|https://your-server/v1/chat/completions|g' {} +

# Replace model name
find . -name "*.py" -exec sed -i '' \
  's|taalas-llama3.1-8b|your-model-name|g' {} +
```

No other changes are required. The evaluation methodology is model-agnostic.

> **Note on User-Agent blocking:** The specific API tested in this report blocks non-`curl` User-Agent strings. If you encounter `403` errors from a different provider, remove or change the `User-Agent` header in the benchmark scripts.

---

## 12. Troubleshooting

**`401 Unauthorized`** — `LLM_API_KEY` is empty or wrong. Verify with:
```bash
echo $LLM_API_KEY
```

**`413 Request Entity Too Large`** — The context window probe in the context-window section of `eval_recall.py` intentionally triggers this to find the server hard limit. This is expected behaviour, not a bug.

**`ConnectionError` / timeout** — The API endpoint is unreachable. The scripts do not retry by default. Re-run the script; partial results are not saved mid-run.

**Results differ significantly from `REPORT.md`** — Small variance is normal. Large variance (>10 pp) most likely means the model was updated on the server side. Pin the `model` field and check with the server operator.

**`403 Forbidden` from Python but `200` from curl** — The API blocks non-`curl` User-Agent strings. Set `"User-Agent": "curl/8.4.0"` in your request headers. All scripts in this repo handle this automatically.

**`ModuleNotFoundError: requests`** — Run `pip install requests==2.32.5`.

**Hard-coded absolute paths in eval scripts** — Some eval scripts contain absolute paths like `/Users/krishnatejaswis/llm-boundary-tests/`. If you cloned to a different location, update the `BASE` or `RESULTS_F` constants near the top of each eval script, or run them from the repo root after setting `BASE = os.path.dirname(os.path.abspath(__file__))`.
