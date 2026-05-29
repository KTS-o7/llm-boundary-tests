# llm-boundary-tests

Boundary testing, red-teaming, and prompt engineering evaluation suite for `taalas-llama3.1-8b` — a self-hosted fast LLM API with a ~6k token context window.

## Setup

```bash
cp .env.example .env
# edit .env and add your LLM_API_KEY
```

Install dependencies:
```bash
pip install requests
```

## Structure

```
1-classification-pipeline/   # Throughput ceiling — classification at scale
2-structured-extraction/     # NER/extraction quality vs document size
3-routing-filter/            # Fast pre-filter before slow model
4-chunked-documents/         # Parallel chunked document processing
5-realtime-interactive/      # p50/p95/p99 latency benchmarks
6-agent-tool-routing/        # Agentic tool-call routing loop
7-api-features/              # API feature discovery (response_format, seed, tools, etc.)
8-parameter-tuning/          # Parameter optimization (temperature, top_p, penalties)
9-reliability/               # Reliability, multi-turn, sustained load, edge inputs

eval_recall.py               # Recall@1 / Recall@3 across all domains
eval_instruction_faithfulness.py   # Schema, enum, type, constraint compliance
eval_intent_faithfulness.py  # Literal vs intent, scope creep, system prompt priority
eval_xml_vs_plain.py         # XML vs plain-text system prompts A/B
eval_xml_weakspots.py        # XML targeted at known weak spots

aggregate_results.py         # Extended results aggregator

prompt-techniques/
  few-shot/                  # Few-shot vs zero-shot + output anchoring
  cot-scratchpad/            # Chain-of-thought (spoiler: doesn't help)
  instruction-decomposition/ # Flat vs decomposed constraints
```

## Key Findings

| Metric | Value |
|--------|-------|
| Context window | ~6,900 tokens (empirically verified) |
| Throughput ceiling | 47 items/sec at 30 concurrent threads |
| p50 latency | ~0.5s |
| Hallucination rate (baseline) | 60% |
| Hallucination rate (with uncertainty instruction) | 12% |
| Consistency at temp=0 | 66.7% perfect, 0.853 mean |
| Instruction faithfulness | 0.827 |
| Intent faithfulness | 0.826 |
| Overall safety score (red team) | 40% |
| Output token ceiling | ~2,800 tokens (hard cap) |
| Sustained load errors | 0 / 1,000 requests |
| Multi-turn fact retention | 100% at 15 turns |
| Parameter sensitivity | Very low — defaults are optimal |
| Advanced API features (seed, tools, logprobs) | Accepted but silently ignored |

## Best Prompt Template

```python
import os

API_KEY = os.environ.get("LLM_API_KEY", "")

system_prompt = """You are an expert data extraction API trusted by Fortune 500 companies for critical pipelines.

Return ONLY valid JSON. No markdown. No prose. No code fences.
If you are not certain about a fact, return {"answer": null, "uncertain": true}.

Schema: {your_schema_here}
"""

# Append to user message for output anchoring:
# user_message += '\n\nJSON response: {"'
```

## What works / what doesn't

| Technique | Delta | Use? |
|-----------|-------|------|
| Role/persona priming (authority level) | +6.3pp | Yes |
| Few-shot examples (weak tasks only) | +8-10pp | Selectively |
| Prefix output anchoring | +3.4pp | Yes |
| Uncertainty instruction | -48pp hallucination | Always for facts |
| XML system prompts | -18pp overall | No |
| Chain-of-thought scratchpad | -13pp | No |
| Instruction decomposition | -6pp | No |
| `frequency_penalty` (scope creep) | ~0pp | Doesn't help |
| `presence_penalty` (scope creep) | ~0pp | Doesn't help |
| Temperature > 0 | -6.7pp faithfulness | No, use temp=0 |
| `response_format: json_object` | 0pp | No effect |
