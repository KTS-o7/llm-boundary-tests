import os
"""
Two-tier LLM routing benchmark.

Fast model (cheap): taalas-llama3.1-8b via https://ai.shenthar.me
Slow model (expensive): simulated with time.sleep(3.0) + mock response

Strategy:
  1. Send each query to the fast model for CLASSIFICATION.
  2. If SIMPLE  → fast model also ANSWERS it (one extra cheap call).
  3. If MEDIUM/COMPLEX → route to "slow model" (simulated).
  4. Measure accuracy, latency savings, confusion matrix.
"""

import json
import time
import random
import statistics
import requests
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FAST_API_URL  = "https://ai.shenthar.me/v1/chat/completions"
FAST_API_KEY  = os.environ.get("LLM_API_KEY", "")
FAST_MODEL    = "taalas-llama3.1-8b"
SLOW_LATENCY  = 3.0   # seconds to simulate slow model
REQUEST_TIMEOUT = 30  # seconds

# ---------------------------------------------------------------------------
# Dataset  (60 queries: 20 SIMPLE, 20 MEDIUM, 20 COMPLEX)
# ---------------------------------------------------------------------------
DATASET = [
    # ── SIMPLE (20) ─────────────────────────────────────────────────────────
    {"id": "S01", "tier": "SIMPLE",  "query": "What is the capital of France?"},
    {"id": "S02", "tier": "SIMPLE",  "query": "What is 17 × 8?"},
    {"id": "S03", "tier": "SIMPLE",  "query": "Is water a chemical compound? Answer yes or no."},
    {"id": "S04", "tier": "SIMPLE",  "query": "What does HTTP stand for?"},
    {"id": "S05", "tier": "SIMPLE",  "query": "Who wrote Romeo and Juliet?"},
    {"id": "S06", "tier": "SIMPLE",  "query": "What is the boiling point of water in Celsius?"},
    {"id": "S07", "tier": "SIMPLE",  "query": "Convert 5 miles to kilometres."},
    {"id": "S08", "tier": "SIMPLE",  "query": "What year did World War II end?"},
    {"id": "S09", "tier": "SIMPLE",  "query": "What is the largest planet in our solar system?"},
    {"id": "S10", "tier": "SIMPLE",  "query": "What is the speed of light in m/s? (approximate)"},
    {"id": "S11", "tier": "SIMPLE",  "query": "What does DNA stand for?"},
    {"id": "S12", "tier": "SIMPLE",  "query": "Is the Earth closer to the Sun in January or July?"},
    {"id": "S13", "tier": "SIMPLE",  "query": "What color do you get when you mix blue and yellow?"},
    {"id": "S14", "tier": "SIMPLE",  "query": "What is the square root of 144?"},
    {"id": "S15", "tier": "SIMPLE",  "query": "How many sides does a hexagon have?"},
    {"id": "S16", "tier": "SIMPLE",  "query": "What language is spoken in Brazil?"},
    {"id": "S17", "tier": "SIMPLE",  "query": "What is the chemical symbol for gold?"},
    {"id": "S18", "tier": "SIMPLE",  "query": "Who invented the telephone?"},
    {"id": "S19", "tier": "SIMPLE",  "query": "What is 2 to the power of 10?"},
    {"id": "S20", "tier": "SIMPLE",  "query": "Is the sun a star? Yes or no."},

    # ── MEDIUM (20) ─────────────────────────────────────────────────────────
    {"id": "M01", "tier": "MEDIUM",  "query": "Explain why Python lists are mutable but tuples are not, and give a use-case for each."},
    {"id": "M02", "tier": "MEDIUM",  "query": "Debug this code: `def factorial(n): return n * factorial(n)` — what's wrong and fix it."},
    {"id": "M03", "tier": "MEDIUM",  "query": "What is the difference between TCP and UDP? When would you use each?"},
    {"id": "M04", "tier": "MEDIUM",  "query": "Walk me through how HTTPS encryption works at a high level."},
    {"id": "M05", "tier": "MEDIUM",  "query": "Explain the time complexity of merge sort and why it's O(n log n)."},
    {"id": "M06", "tier": "MEDIUM",  "query": "What are the SOLID principles in software engineering? Give a one-line summary of each."},
    {"id": "M07", "tier": "MEDIUM",  "query": "If a train leaves city A at 60 mph and another leaves city B (300 miles away) at 40 mph toward A, when do they meet?"},
    {"id": "M08", "tier": "MEDIUM",  "query": "What causes inflation, and what tools does a central bank use to fight it?"},
    {"id": "M09", "tier": "MEDIUM",  "query": "Describe the difference between supervised and unsupervised machine learning with examples."},
    {"id": "M10", "tier": "MEDIUM",  "query": "Why does JavaScript's `0.1 + 0.2 !== 0.3`? Explain the root cause."},
    {"id": "M11", "tier": "MEDIUM",  "query": "What is recursion? Write a simple recursive function to compute Fibonacci numbers."},
    {"id": "M12", "tier": "MEDIUM",  "query": "Explain how a hash table resolves collisions using chaining."},
    {"id": "M13", "tier": "MEDIUM",  "query": "What is the difference between a process and a thread? When would you use threads?"},
    {"id": "M14", "tier": "MEDIUM",  "query": "Describe how photosynthesis works in 3-4 sentences."},
    {"id": "M15", "tier": "MEDIUM",  "query": "What is the CAP theorem and what does it mean for distributed systems?"},
    {"id": "M16", "tier": "MEDIUM",  "query": "Explain the concept of gradient descent in machine learning."},
    {"id": "M17", "tier": "MEDIUM",  "query": "How does OAuth 2.0 work? Describe the authorization code flow."},
    {"id": "M18", "tier": "MEDIUM",  "query": "Write a SQL query to find the top 3 customers by total spend from an orders table."},
    {"id": "M19", "tier": "MEDIUM",  "query": "What is the difference between REST and GraphQL APIs?"},
    {"id": "M20", "tier": "MEDIUM",  "query": "Explain why quicksort has O(n²) worst-case but O(n log n) average-case complexity."},

    # ── COMPLEX (20) ────────────────────────────────────────────────────────
    {"id": "C01", "tier": "COMPLEX", "query": "Design a distributed rate-limiter that works across 50 microservices with no single point of failure. Discuss trade-offs."},
    {"id": "C02", "tier": "COMPLEX", "query": "Analyze the philosophical tension between free will and determinism from both compatibilist and hard-determinist perspectives."},
    {"id": "C03", "tier": "COMPLEX", "query": "A startup has $2M runway, 18 months, 8 engineers. Evaluate whether to build a monolith or microservices architecture and defend your recommendation."},
    {"id": "C04", "tier": "COMPLEX", "query": "Explain how transformer attention mechanisms work mathematically, including the scaled dot-product attention formula and why scaling is needed."},
    {"id": "C05", "tier": "COMPLEX", "query": "Critique the ethical implications of predictive policing algorithms with respect to systemic bias and due process."},
    {"id": "C06", "tier": "COMPLEX", "query": "Design a globally consistent database schema for a multi-tenant SaaS platform that needs row-level security and audit logging."},
    {"id": "C07", "tier": "COMPLEX", "query": "Compare and contrast Keynesian and Austrian economic theories in the context of the 2008 financial crisis response."},
    {"id": "C08", "tier": "COMPLEX", "query": "Explain RLHF (Reinforcement Learning from Human Feedback) in depth: how it trains LLMs, what reward hacking is, and its limitations."},
    {"id": "C09", "tier": "COMPLEX", "query": "Given a system that processes 10M events/day with P99 latency < 100ms requirements, design the full tech stack and justify each choice."},
    {"id": "C10", "tier": "COMPLEX", "query": "Analyze whether nuclear power should be part of a net-zero 2050 strategy, considering safety, cost, waste, and energy security trade-offs."},
    {"id": "C11", "tier": "COMPLEX", "query": "Write a thorough post-mortem analysis template for a production outage that caused data loss. Include root-cause analysis framework."},
    {"id": "C12", "tier": "COMPLEX", "query": "Explain how zero-knowledge proofs work conceptually and describe a real-world application in privacy-preserving authentication."},
    {"id": "C13", "tier": "COMPLEX", "query": "Evaluate the long-term geopolitical consequences of widespread AI automation on labor markets and social stability across developed vs developing nations."},
    {"id": "C14", "tier": "COMPLEX", "query": "Design an ML pipeline for real-time fraud detection: feature engineering, model choice, online learning, and explainability requirements."},
    {"id": "C15", "tier": "COMPLEX", "query": "Critically analyze the claim 'Agile always outperforms Waterfall'. Under what conditions might Waterfall be superior?"},
    {"id": "C16", "tier": "COMPLEX", "query": "Explain Byzantine fault tolerance: what it is, why it matters in blockchain consensus, and how PBFT achieves it."},
    {"id": "C17", "tier": "COMPLEX", "query": "Design a GDPR-compliant data deletion system for a platform with 100M users where data is replicated across 5 regions."},
    {"id": "C18", "tier": "COMPLEX", "query": "Analyze the trolley problem variants and what each reveals about consequentialist vs deontological ethical frameworks."},
    {"id": "C19", "tier": "COMPLEX", "query": "How would you architect a recommendation engine that handles cold-start, real-time updates, and A/B testing simultaneously?"},
    {"id": "C20", "tier": "COMPLEX", "query": "Explain the P vs NP problem: what it asks, why it matters for cryptography, and what a proof in either direction would mean."},
]

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def call_fast_model(messages: list[dict], temperature: float = 0.1) -> tuple[str, float]:
    """Call the fast model. Returns (content, latency_seconds)."""
    headers = {
        "Authorization": f"Bearer {FAST_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": FAST_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    t0 = time.perf_counter()
    resp = requests.post(FAST_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    latency = time.perf_counter() - t0
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return content, latency


def simulate_slow_model(query: str) -> tuple[str, float]:
    """Simulate a slow expensive model call."""
    t0 = time.perf_counter()
    time.sleep(SLOW_LATENCY)
    latency = time.perf_counter() - t0
    return f"[SLOW MODEL MOCK] Detailed answer for: {query[:60]}...", latency


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
ROUTER_SYSTEM = (
    "You are a query complexity classifier. "
    "Classify the user's query as SIMPLE, MEDIUM, or COMPLEX based on the reasoning depth required.\n\n"
    "SIMPLE: Factual lookups, basic arithmetic, yes/no questions, simple definitions. "
    "A small language model can answer accurately without deep reasoning.\n"
    "MEDIUM: Multi-step reasoning, code debugging, technical explanations, calculations with multiple steps. "
    "Requires moderate reasoning depth.\n"
    "COMPLEX: Deep analysis, long chain-of-thought, nuanced judgment, ethical reasoning, system design, "
    "mathematical proofs, or tasks requiring extensive knowledge synthesis.\n\n"
    "Respond with ONLY valid JSON in this exact format (no markdown, no extra text):\n"
    "{\"tier\": \"SIMPLE|MEDIUM|COMPLEX\", \"reason\": \"one sentence explanation\"}"
)


def route_query(query: str) -> tuple[str, str, float]:
    """
    Ask the fast model to classify the query.
    Returns (predicted_tier, reason, routing_latency).
    Falls back to COMPLEX on parse error.
    """
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user",   "content": query},
    ]
    content, latency = call_fast_model(messages, temperature=0.0)

    # Strip markdown fences if present
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        tier = parsed.get("tier", "COMPLEX").upper().strip()
        if tier not in ("SIMPLE", "MEDIUM", "COMPLEX"):
            tier = "COMPLEX"
        reason = parsed.get("reason", "")
    except json.JSONDecodeError:
        # Try to extract tier keyword from raw text
        upper = content.upper()
        if "SIMPLE" in upper and "MEDIUM" not in upper and "COMPLEX" not in upper:
            tier = "SIMPLE"
        elif "COMPLEX" in upper:
            tier = "COMPLEX"
        elif "MEDIUM" in upper:
            tier = "MEDIUM"
        else:
            tier = "COMPLEX"  # safe fallback
        reason = f"[parse error] raw: {content[:80]}"

    return tier, reason, latency


# ---------------------------------------------------------------------------
# Answer helpers
# ---------------------------------------------------------------------------

def answer_with_fast_model(query: str) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer concisely and accurately."},
        {"role": "user",   "content": query},
    ]
    return call_fast_model(messages, temperature=0.3)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    results = []
    tier_counts     = {"SIMPLE": 0, "MEDIUM": 0, "COMPLEX": 0}
    routing_latencies = []

    # Confusion matrix: actual → predicted
    confusion = {
        "SIMPLE":  {"SIMPLE": 0, "MEDIUM": 0, "COMPLEX": 0},
        "MEDIUM":  {"SIMPLE": 0, "MEDIUM": 0, "COMPLEX": 0},
        "COMPLEX": {"SIMPLE": 0, "MEDIUM": 0, "COMPLEX": 0},
    }

    total_with_routing    = 0.0  # actual wall time
    total_without_routing = len(DATASET) * SLOW_LATENCY  # naive: all slow

    fast_model_answered = 0   # SIMPLE queries handled entirely by fast model
    correct_routings    = 0

    print(f"\n{'='*70}")
    print(f"  TWO-TIER LLM ROUTING BENCHMARK")
    print(f"  Fast model : {FAST_MODEL}")
    print(f"  Slow model : simulated ({SLOW_LATENCY}s)")
    print(f"  Dataset    : {len(DATASET)} queries")
    print(f"{'='*70}\n")

    for i, item in enumerate(DATASET):
        qid        = item["id"]
        true_tier  = item["tier"]
        query      = item["query"]

        print(f"[{i+1:02d}/{len(DATASET)}] {qid} ({true_tier}) — {query[:55]}...")

        # ── Step 1: Route ──────────────────────────────────────────────────
        try:
            pred_tier, reason, route_lat = route_query(query)
        except Exception as e:
            print(f"         ROUTING ERROR: {e} — defaulting to COMPLEX")
            pred_tier, reason, route_lat = "COMPLEX", str(e), 0.0

        routing_latencies.append(route_lat)
        confusion[true_tier][pred_tier] += 1
        tier_counts[pred_tier] += 1
        is_correct = (pred_tier == true_tier)
        if is_correct:
            correct_routings += 1

        # ── Step 2: Answer ─────────────────────────────────────────────────
        if pred_tier == "SIMPLE":
            try:
                answer, answer_lat = answer_with_fast_model(query)
                model_used = "fast"
                fast_model_answered += 1
            except Exception as e:
                answer, answer_lat = f"[ERROR: {e}]", 0.0
                model_used = "fast(error)"
        else:
            answer, answer_lat = simulate_slow_model(query)
            model_used = "slow(simulated)"

        query_total_time = route_lat + answer_lat
        total_with_routing += query_total_time

        print(f"         Routed→{pred_tier:8s} (actual:{true_tier}) {'✓' if is_correct else '✗'}  "
              f"route:{route_lat:.2f}s  answer:{answer_lat:.2f}s  model:{model_used}")

        results.append({
            "id":            qid,
            "true_tier":     true_tier,
            "predicted_tier": pred_tier,
            "correct":       is_correct,
            "query":         query,
            "answer_preview": answer[:120],
            "model_used":    model_used,
            "routing_latency_s": round(route_lat, 3),
            "answer_latency_s":  round(answer_lat, 3),
            "total_latency_s":   round(query_total_time, 3),
            "reason":        reason,
        })

    # ── Metrics ────────────────────────────────────────────────────────────
    routing_accuracy = correct_routings / len(DATASET) * 100
    pct_fast_handled = fast_model_answered / len(DATASET) * 100
    time_saved       = total_without_routing - total_with_routing
    pct_saved        = time_saved / total_without_routing * 100
    avg_route_lat    = statistics.mean(routing_latencies)
    overhead_total   = sum(routing_latencies)  # routing cost

    # Per-tier accuracy
    tier_accuracy = {}
    for t in ("SIMPLE", "MEDIUM", "COMPLEX"):
        total_in_tier = sum(confusion[t].values())
        correct_in_tier = confusion[t][t]
        tier_accuracy[t] = (correct_in_tier / total_in_tier * 100) if total_in_tier else 0

    # Biggest misclassification patterns
    misclass = []
    for true_t in ("SIMPLE", "MEDIUM", "COMPLEX"):
        for pred_t in ("SIMPLE", "MEDIUM", "COMPLEX"):
            if true_t != pred_t and confusion[true_t][pred_t] > 0:
                misclass.append({
                    "actual":    true_t,
                    "predicted": pred_t,
                    "count":     confusion[true_t][pred_t],
                })
    misclass.sort(key=lambda x: -x["count"])

    # Confidence boundary: what fraction does router handle vs escalate
    escalated   = tier_counts["MEDIUM"] + tier_counts["COMPLEX"]
    pct_escalated = escalated / len(DATASET) * 100

    summary = {
        "run_timestamp":          datetime.utcnow().isoformat() + "Z",
        "dataset_size":           len(DATASET),
        "fast_model":             FAST_MODEL,
        "slow_model_latency_s":   SLOW_LATENCY,
        "routing_accuracy_pct":   round(routing_accuracy, 1),
        "tier_accuracy": {
            t: round(v, 1) for t, v in tier_accuracy.items()
        },
        "pct_handled_by_fast_model": round(pct_fast_handled, 1),
        "pct_escalated_to_slow":     round(pct_escalated, 1),
        "total_time_with_routing_s":    round(total_with_routing, 2),
        "total_time_without_routing_s": round(total_without_routing, 2),
        "time_saved_s":            round(time_saved, 2),
        "pct_time_saved":          round(pct_saved, 1),
        "routing_overhead_total_s": round(overhead_total, 2),
        "avg_routing_latency_s":    round(avg_route_lat, 3),
        "confusion_matrix":         confusion,
        "misclassification_patterns": misclass,
        "routing_decisions":       tier_counts,
        "query_results":           results,
    }

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = "/Users/krishnatejaswis/llm-boundary-tests/3-routing-filter/results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"  Routing accuracy      : {routing_accuracy:.1f}%")
    print(f"  Per-tier accuracy     : SIMPLE={tier_accuracy['SIMPLE']:.0f}%  "
          f"MEDIUM={tier_accuracy['MEDIUM']:.0f}%  COMPLEX={tier_accuracy['COMPLEX']:.0f}%")
    print(f"")
    print(f"  Handled by fast model : {fast_model_answered}/{len(DATASET)} ({pct_fast_handled:.1f}%)")
    print(f"  Escalated to slow     : {escalated}/{len(DATASET)} ({pct_escalated:.1f}%)")
    print(f"")
    print(f"  Time WITH routing     : {total_with_routing:.1f}s")
    print(f"  Time WITHOUT routing  : {total_without_routing:.1f}s  (all-slow baseline)")
    print(f"  Time saved            : {time_saved:.1f}s  ({pct_saved:.1f}%)")
    print(f"  Routing overhead      : {overhead_total:.1f}s total  ({avg_route_lat:.2f}s avg/query)")
    print(f"")
    print(f"  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  {'':10s}  {'SIMPLE':>8s}  {'MEDIUM':>8s}  {'COMPLEX':>8s}")
    for true_t in ("SIMPLE", "MEDIUM", "COMPLEX"):
        row = confusion[true_t]
        print(f"  {true_t:10s}  {row['SIMPLE']:>8d}  {row['MEDIUM']:>8d}  {row['COMPLEX']:>8d}")
    print(f"")
    print(f"  Misclassification patterns (ranked):")
    if misclass:
        for m in misclass:
            print(f"    {m['actual']:8s} → {m['predicted']:8s}: {m['count']} queries")
    else:
        print(f"    None — perfect routing!")
    print(f"")
    print(f"  Results saved to: {out_path}")
    print(f"{'='*70}\n")

    return summary


if __name__ == "__main__":
    run_benchmark()
