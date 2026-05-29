import os
#!/usr/bin/env python3
"""
Reliability & Robustness Benchmark for LLM API
Tests multi-turn conversation depth, sustained load capacity, error handling,
and edge-case input robustness.
"""

import json
import time
import statistics
import re
import requests
from datetime import datetime, timezone

API_URL = "https://ai.shenthar.me/v1/chat/completions"
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = "taalas-llama3.1-8b"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "User-Agent": "curl/8.4.0",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def percentile(data, p):
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def compute_stats(latencies):
    if not latencies:
        return {}
    return {
        "min": round(min(latencies), 3),
        "max": round(max(latencies), 3),
        "mean": round(statistics.mean(latencies), 3),
        "p50": round(percentile(latencies, 50), 3),
        "p95": round(percentile(latencies, 95), 3),
        "p99": round(percentile(latencies, 99), 3),
        "count": len(latencies),
    }


def call_api(messages, timeout=60):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=timeout)
        t1 = time.perf_counter()
        status_code = resp.status_code
        if status_code != 200:
            return {
                "ok": False, "latency": t1 - t0, "status_code": status_code,
                "error": f"HTTP {status_code}: {resp.text[:300]}",
            }
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return {
            "ok": True, "latency": t1 - t0, "status_code": status_code,
            "completion_tokens": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "content": content,
        }
    except requests.exceptions.Timeout:
        t1 = time.perf_counter()
        return {"ok": False, "latency": t1 - t0, "status_code": 0, "error": "timeout"}
    except requests.exceptions.ConnectionError as e:
        t1 = time.perf_counter()
        return {"ok": False, "latency": t1 - t0, "status_code": 0, "error": f"connection: {e}"}
    except Exception as e:
        t1 = time.perf_counter()
        return {"ok": False, "latency": t1 - t0, "status_code": 0, "error": str(e)}


def call_api_raw(headers=None, payload=None, data=None, timeout=30):
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            API_URL,
            headers=headers if headers else HEADERS,
            json=payload if data is None else None,
            data=data,
            timeout=timeout,
        )
        t1 = time.perf_counter()
        return {
            "ok": resp.status_code == 200,
            "status": resp.status_code,
            "body": resp.text[:500],
            "latency": t1 - t0,
        }
    except Exception as e:
        t1 = time.perf_counter()
        return {"ok": False, "status": 0, "body": str(e)[:500], "latency": t1 - t0}


# ── Test 1: Multi-turn conversation ──────────────────────────────────────────────

FACTS = [
    "My name is Alice.",
    "I live in Tokyo.",
    "My favorite color is green.",
    "I work as a data scientist.",
    "I have a dog named Max.",
    "I speak Japanese fluently.",
    "My birthday is in December.",
    "I have a younger brother named Tom.",
    "I graduated from Tokyo University.",
    "I love hiking in the mountains.",
]

FACT_QUESTIONS = [
    ("What is my name?", "Alice"),
    ("Where do I live?", "Tokyo"),
    ("What is my favorite color?", "green"),
    ("What is my dog's name?", "Max"),
    ("What do I do for work?", "data scientist"),
]


def run_fact_retention_chain():
    print("  [Chain A] Fact retention...", flush=True)
    messages = [{"role": "system", "content": "You are a helpful assistant. Keep track of facts mentioned during our conversation."}]
    turns = []
    exhausted = False
    exhausted_at = None
    remembered = 0

    for i, fact in enumerate(FACTS):
        turn_num = i + 1
        messages.append({"role": "user", "content": fact})
        result = call_api(messages)
        if result["ok"]:
            messages.append({"role": "assistant", "content": result["content"]})
            status = f"    Turn {turn_num}: OK {result['latency']:.3f}s {result.get('completion_tokens', 0)} tok"
        else:
            exhausted = True
            exhausted_at = turn_num
            status = f"    Turn {turn_num}: ERR {result.get('error', '?')[:80]}"
        turn_record = {"turn": turn_num, "type": "fact_injection", "fact": fact}
        turn_record.update(result)
        turns.append(turn_record)
        print(status, flush=True)
        if exhausted:
            break

    if not exhausted:
        for question, expected in FACT_QUESTIONS:
            turn_num = len(FACTS) + FACT_QUESTIONS.index((question, expected)) + 1
            messages.append({"role": "user", "content": question})
            result = call_api(messages)
            recalled = False
            if result["ok"]:
                recalled = expected.lower() in result["content"].lower()
                if recalled:
                    remembered += 1
                messages.append({"role": "assistant", "content": result["content"]})
            else:
                exhausted = True
                exhausted_at = turn_num
            turn_record = {"turn": turn_num, "type": "fact_query", "question": question, "expected": expected, "recalled": recalled}
            turn_record.update(result)
            turns.append(turn_record)
            if result["ok"]:
                status = f"    Turn {turn_num}: {'CORRECT' if recalled else 'WRONG'} {result['latency']:.3f}s {result.get('completion_tokens', 0)} tok"
            else:
                status = f"    Turn {turn_num}: ERR {result.get('error', '?')[:80]}"
            print(status, flush=True)
            if exhausted:
                break

    return {
        "chain": "A", "name": "Fact retention",
        "turns": turns,
        "facts_remembered": remembered,
        "facts_asked": len(FACT_QUESTIONS) if not exhausted else 0,
        "context_exhausted": exhausted,
        "exhausted_at_turn": exhausted_at,
    }


DRIFT_QUESTIONS = [
    "What is the weather like?",
    "Explain what a computer is.",
    "How does the internet work?",
    "What is the meaning of life?",
    "Explain quantum computing in simple terms.",
    "Describe the theory of relativity.",
    "How does a blockchain work?",
    "Explain the entire history of the Roman Empire.",
    "How do neural networks learn?",
    "What is the best strategy for achieving world peace?",
    "Explain the geopolitical implications of the South China Sea disputes.",
    "How would you design a fully decentralized internet?",
    "Describe how consciousness emerges from neural activity.",
    "Create a comprehensive plan to colonize Mars.",
    "Explain string theory and the holographic principle.",
]


def is_one_sentence(text):
    count = text.count('.') + text.count('!') + text.count('?')
    abbrevs = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Sr.', 'Jr.', 'vs.', 'etc.', 'e.g.', 'i.e.', 'U.S.', 'U.K.']
    for abbr in abbrevs:
        count -= text.count(abbr)
    return count <= 1


def run_instruction_drift_chain():
    print("  [Chain B] Instruction drift...", flush=True)
    messages = [{"role": "system", "content": "Always respond in exactly one sentence. Be concise."}]
    turns = []
    drifted = False
    drifted_at = None

    for i, question in enumerate(DRIFT_QUESTIONS):
        turn_num = i + 1
        messages.append({"role": "user", "content": question})
        result = call_api(messages)
        one_sent = False
        if result["ok"]:
            one_sent = is_one_sentence(result["content"].strip())
            if not one_sent and not drifted:
                drifted = True
                drifted_at = turn_num
            messages.append({"role": "assistant", "content": result["content"]})
        turn_record = {"turn": turn_num, "question": question, "one_sentence": one_sent}
        turn_record.update(result)
        turns.append(turn_record)
        if result["ok"]:
            status = f"    Turn {turn_num}: {'1S' if one_sent else 'DRIFT'} {result['latency']:.3f}s {result.get('completion_tokens', 0)} tok"
        else:
            status = f"    Turn {turn_num}: ERR {result.get('error', '?')[:80]}"
        print(status, flush=True)

    return {
        "chain": "B", "name": "Instruction drift",
        "turns": turns, "drifted": drifted, "drifted_at_turn": drifted_at,
    }


ROLE_QUESTIONS = [
    ("math", "What is 2 + 2?"),
    ("non-math", "What is the capital of France?"),
    ("math", "Solve for x: 3x + 7 = 22"),
    ("non-math", "Who wrote Romeo and Juliet?"),
    ("math", "What is the derivative of x²?"),
    ("non-math", "What is the best movie of all time?"),
    ("math", "Calculate the area of a circle with radius 5."),
    ("non-math", "How do I make pasta?"),
    ("math", "What is the Pythagorean theorem?"),
    ("non-math", "Tell me a joke."),
    ("math", "Solve the quadratic equation x² - 5x + 6 = 0"),
    ("non-math", "What is the meaning of life?"),
    ("math", "What is the integral of e^x dx?"),
    ("non-math", "Who is the president of the United States?"),
    ("math", "What is 2^10?"),
]


def run_role_consistency_chain():
    print("  [Chain C] Role consistency...", flush=True)
    messages = [{"role": "system", "content": "You are a math tutor. Only answer math questions. For non-math questions, say 'I can only help with math.'"}]
    turns = []
    role_violations = 0

    for i, (expected_type, question) in enumerate(ROLE_QUESTIONS):
        turn_num = i + 1
        messages.append({"role": "user", "content": question})
        result = call_api(messages)
        refused = False
        if result["ok"]:
            content_lower = result["content"].lower()
            refusal_phrases = ["i can only help with math", "only help with math", "i'm a math tutor", "i am a math tutor"]
            refused = any(p in content_lower for p in refusal_phrases)
            if expected_type == "math" and refused:
                role_violations += 1
            elif expected_type == "non-math" and not refused:
                role_violations += 1
            messages.append({"role": "assistant", "content": result["content"]})
        turn_record = {"turn": turn_num, "expected": expected_type, "question": question, "refused": refused, "role_correct": (expected_type == "math" and not refused) or (expected_type == "non-math" and refused)}
        turn_record.update(result)
        turns.append(turn_record)
        if result["ok"]:
            tag = "OK" if (expected_type == "math" and not refused) or (expected_type == "non-math" and refused) else "VIOLATION"
            status = f"    Turn {turn_num}: {tag} {result['latency']:.3f}s {result.get('completion_tokens', 0)} tok"
        else:
            status = f"    Turn {turn_num}: ERR {result.get('error', '?')[:80]}"
        print(status, flush=True)

    return {
        "chain": "C", "name": "Role consistency",
        "turns": turns, "role_violations": role_violations, "total_questions": len(ROLE_QUESTIONS),
    }


def run_multi_turn_conversation():
    print("[Test 1] Multi-turn conversation...", flush=True)
    chain_a = run_fact_retention_chain()
    chain_b = run_instruction_drift_chain()
    chain_c = run_role_consistency_chain()

    for c in [chain_a, chain_b, chain_c]:
        for t in c.get("turns", []):
            t.pop("content", None)
    return {"chains": [chain_a, chain_b, chain_c]}


# ── Test 2: Sustained load ───────────────────────────────────────────────────────

LOAD_PROMPTS = [
    ("PING", 'Return {"ok": true}'),
    ("SHORT", "Name 3 colors. Return as JSON array."),
    ("MEDIUM", "Describe HTTP in 3 sentences. Return JSON."),
    ("FACT", "What is the capital of France? Return JSON."),
]


def run_sustained_load():
    print("[Test 2] Sustained load (1000 requests)...", flush=True)
    all_results = []
    t0 = time.perf_counter()
    for i in range(1000):
        ptype, prompt = LOAD_PROMPTS[i % len(LOAD_PROMPTS)]
        result = call_api([{"role": "user", "content": prompt}])
        result["prompt_type"] = ptype
        result.pop("content", None)
        all_results.append(result)
        if (i + 1) % 100 == 0:
            print(f"  Completed {i+1}/1000", flush=True)
    wall = time.perf_counter() - t0

    status_counts = {}
    for r in all_results:
        sc = str(r.get("status_code", 0))
        status_counts[sc] = status_counts.get(sc, 0) + 1

    buckets = []
    for b in range(10):
        start = b * 100
        batch = all_results[start:start + 100]
        errs = sum(1 for r in batch if not r["ok"])
        lats = [r["latency"] for r in batch if r["ok"]]
        toks = [r.get("completion_tokens", 0) for r in batch if r["ok"]]
        buckets.append({
            "bucket": b + 1,
            "requests": f"{start+1}-{start+100}",
            "errors": errs,
            "p50": round(percentile(lats, 50), 3),
            "p95": round(percentile(lats, 95), 3),
            "p99": round(percentile(lats, 99), 3),
            "mean_tokens": round(statistics.mean(toks), 1) if toks else 0,
        })

    total_errs = sum(1 for r in all_results if not r["ok"])
    all_lats = [r["latency"] for r in all_results if r["ok"]]
    return {
        "buckets": buckets,
        "overall": {
            "total_errors": total_errs,
            "error_rate": round(total_errs / len(all_results), 4),
            "mean_latency": round(statistics.mean(all_lats), 3) if all_lats else 0,
            "total_wall_time_s": round(wall, 3),
        },
        "status_codes": {
            "200": status_counts.get("200", 0),
            "others": {k: v for k, v in status_counts.items() if k != "200"},
        },
    }


# ── Test 3: Error catalog ────────────────────────────────────────────────────────

ERROR_TESTS = [
    ("missing_auth", None, None, None),
    ("invalid_api_key", {"Authorization": "Bearer invalid_key_12345", "Content-Type": "application/json", "User-Agent": "curl/8.4.0"}, None, None),
    ("missing_model", None, {"messages": [{"role": "user", "content": "hi"}]}, None),
    ("invalid_model", None, {"model": "nonexistent-model", "messages": [{"role": "user", "content": "hi"}]}, None),
    ("empty_messages", None, {"model": MODEL, "messages": []}, None),
    ("missing_content", None, {"model": MODEL, "messages": [{"role": "user"}]}, None),  # invalid — skip content key
    ("invalid_role", None, {"model": MODEL, "messages": [{"role": "superadmin", "content": "hi"}]}, None),
    ("negative_temp", None, {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "temperature": -5}, None),
    ("large_max_tokens", None, {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100000}, None),
    ("invalid_json", None, None, "not json"),
]


def run_error_catalog():
    print("[Test 3] Error catalog...", flush=True)
    catalog = []
    informative_count = 0
    opaque_count = 0

    for name, headers_override, payload_override, raw_data in ERROR_TESTS:
        h = headers_override if headers_override else (HEADERS if name != "missing_auth" else {"Content-Type": "application/json", "User-Agent": "curl/8.4.0"})
        p = payload_override if payload_override else (None if name == "missing_model" or name == "invalid_model" else None)
        if name == "missing_content":
            p = {"model": MODEL, "messages": [{"role": "user"}]}
        if name == "missing_model":
            p = {"messages": [{"role": "user", "content": "hi"}]}

        result = call_api_raw(headers=h, payload=p, data=raw_data)
        body = result.get("body", "")
        informative = result["status"] != 0 and len(body) > 10 and ("error" in body.lower() or "invalid" in body.lower() or "missing" in body.lower() or "unsupported" in body.lower() or result["status"] >= 400)
        if informative:
            informative_count += 1
        else:
            opaque_count += 1

        catalog.append({
            "test": name,
            "status": result["status"],
            "error_body": body,
            "informative": informative,
        })
        print(f"  {name}: status={result['status']} informative={informative}", flush=True)

    return {
        "catalog": catalog,
        "summary": {"tests": len(ERROR_TESTS), "informative_errors": informative_count, "opaque_errors": opaque_count},
    }


# ── Test 4: Edge inputs ──────────────────────────────────────────────────────────

EDGE_INPUTS = [
    ("empty_string", ""),
    ("whitespace_only", "     "),
    ("very_long_word", "a" * 10000),
    ("unicode_bom", "\ufeffHello"),
    ("null_bytes", "Hello\x00World"),
    ("emoji_only", "🔥🚀🌟💯🎉"),
    ("very_long_input", "word " * 5000),
    ("html_injection", "<script>alert('xss')</script>"),
    ("sql_injection", "'; DROP TABLE users; --"),
    ("long_number", "12345" * 1000),
    ("multi_language", "Hello مرحبا 你好 नमस्ते Bonjour Hola 안녕하세요 Merhaba Ciao こんにちは Привет हेलो สวัสดี Chao Xin chào Hej Hallo Hei Kem cho Hallå Salam Olá Shalom Здраво Ćao Tere हॅलो Sawubona สวัสดี Kaixo Ahoj Hei Salut Здравейте Hyvää päivää Hæ Halo Goddag Bok 你好 నమస్కారம்"),
    ("special_chars_only", "@#$%^&*()_+{}[]|;:',.<>?/~"),
]


def run_edge_inputs():
    print("[Test 4] Edge inputs...", flush=True)
    edge_results = []

    for input_type, prompt_text in EDGE_INPUTS:
        result = call_api([{"role": "user", "content": prompt_text}], timeout=120)
        preview = result.get("content", result.get("error", ""))[:500] if result["ok"] else result.get("error", "")[:500]
        graceful = result["ok"] or result["status_code"] != 0
        edge_results.append({
            "input_type": input_type,
            "status": result.get("status_code", 0),
            "response_preview": preview,
            "error": result.get("error", None) if not result["ok"] else None,
            "handled_gracefully": graceful,
        })
        status_label = "OK" if result["ok"] else f"ERR {result.get('status_code', 0)}"
        print(f"  {input_type}: {status_label}", flush=True)

    return edge_results


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    print("Starting LLM Reliability & Robustness Benchmark", flush=True)
    print(f"Model: {MODEL}", flush=True)
    print(f"API: {API_URL}", flush=True)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}", flush=True)

    multi_turn = run_multi_turn_conversation()
    sustained_load = run_sustained_load()
    error_catalog = run_error_catalog()
    edge = run_edge_inputs()

    results = {
        "meta": {
            "model": MODEL,
            "api_url": API_URL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "test_1_multi_turn": multi_turn,
        "test_2_sustained_load": sustained_load,
        "test_3_error_catalog": error_catalog,
        "test_4_edge_inputs": {"edge_inputs": edge},
    }

    results_path = os.path.join(SCRIPT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}", flush=True)


if __name__ == "__main__":
    main()
