import os
import json
import time
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


def call_api(payload, timeout=120):
    t0 = time.perf_counter()
    try:
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=timeout)
        elapsed = time.perf_counter() - t0
        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text[:1000]}
        return {
            "ok": resp.status_code == 200,
            "status_code": resp.status_code,
            "elapsed": elapsed,
            "data": body,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "ok": False,
            "status_code": None,
            "elapsed": elapsed,
            "error": str(e),
        }


def is_valid_json(text):
    if not text:
        return False
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def truncate(text, max_len=200):
    if text is None:
        return None
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def test_response_format():
    print("\n[response_format] Testing JSON mode...", flush=True)
    prompts = [
        'Extract the name, age, and city from: "Alice is 30 years old and lives in Paris." Return as JSON.',
        'List 3 colors as a JSON array of objects with name and hex fields.',
        'Create a JSON object representing a book with title, author, year, and genres fields.',
        'What is the capital of France?',
        'Write a short poem about programming.',
        'Explain how HTTP works in simple terms.',
        'Return a JSON object with 5 random facts about space.',
        'Convert this to JSON: product name is Widget, price is 19.99, in stock is true.',
        'Generate a JSON array of 3 user profiles each with id, username, and email.',
        'Tell me about machine learning briefly.',
    ]
    with_results = []
    without_results = []
    json_success_with = 0
    json_success_without = 0
    for i, prompt in enumerate(prompts):
        payload_with = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        r = call_api(payload_with)
        content = ""
        if r["ok"]:
            try:
                content = r["data"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                pass
        if content and is_valid_json(content):
            json_success_with += 1
        with_results.append({
            "prompt": prompt[:60],
            "ok": r["ok"],
            "status_code": r["status_code"],
            "content": truncate(content),
        })
        print(f"  [{i+1}/{len(prompts)}] WITH  json_object: {'OK' if r['ok'] else 'ERR'} status={r['status_code']}", flush=True)

        payload_without = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
        }
        r2 = call_api(payload_without)
        content2 = ""
        if r2["ok"]:
            try:
                content2 = r2["data"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                pass
        if content2 and is_valid_json(content2):
            json_success_without += 1
        without_results.append({
            "prompt": prompt[:60],
            "ok": r2["ok"],
            "status_code": r2["status_code"],
            "content": truncate(content2),
        })
        print(f"  [{i+1}/{len(prompts)}] WITHOUT json_object: {'OK' if r2['ok'] else 'ERR'} status={r2['status_code']}", flush=True)

    accepted = all(r["ok"] for r in with_results)

    return {
        "accepted": accepted,
        "json_success_rate_with": json_success_with / len(prompts),
        "json_success_rate_without": json_success_without / len(prompts),
        "sample_responses": {
            "with": with_results[:3],
            "without": without_results[:3],
        },
    }


def test_seed():
    print("\n[seed] Testing determinism with seed parameter...", flush=True)
    prompts = [
        'What is the capital of France? Answer concisely.',
        'Explain what a database is in one sentence.',
        'What is 2 + 2?',
        'Name three primary colors.',
        'Is Python dynamically typed? Answer yes or no.',
    ]
    seed_42_results = []
    seed_99_results = []
    no_seed_results = []

    for pi, prompt in enumerate(prompts):
        print(f"  Prompt {pi+1}: {prompt[:50]}...", flush=True)
        s42 = []
        s99 = []
        ns = []
        for rep in range(5):
            r = call_api({
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "seed": 42,
            })
            content = ""
            if r["ok"]:
                try:
                    content = r["data"]["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    pass
            s42.append({"rep": rep, "ok": r["ok"], "content": content})
            print(f"    seed=42 rep {rep+1}: {'OK' if r['ok'] else 'ERR'}", flush=True)

        for rep in range(5):
            r = call_api({
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "seed": 99,
            })
            content = ""
            if r["ok"]:
                try:
                    content = r["data"]["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    pass
            s99.append({"rep": rep, "ok": r["ok"], "content": content})
            print(f"    seed=99 rep {rep+1}: {'OK' if r['ok'] else 'ERR'}", flush=True)

        for rep in range(5):
            r = call_api({
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
            })
            content = ""
            if r["ok"]:
                try:
                    content = r["data"]["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    pass
            ns.append({"rep": rep, "ok": r["ok"], "content": content})
            print(f"    no-seed rep {rep+1}: {'OK' if r['ok'] else 'ERR'}", flush=True)

        seed_42_results.append(s42)
        seed_99_results.append(s99)
        no_seed_results.append(ns)

    seed_deterministic = True
    for prompt_results in seed_42_results:
        ok_contents = [r["content"] for r in prompt_results if r["ok"]]
        if len(ok_contents) >= 2 and len(set(ok_contents)) != 1:
            seed_deterministic = False
    for prompt_results in seed_99_results:
        ok_contents = [r["content"] for r in prompt_results if r["ok"]]
        if len(ok_contents) >= 2 and len(set(ok_contents)) != 1:
            seed_deterministic = False

    no_seed_deterministic = True
    for prompt_results in no_seed_results:
        ok_contents = [r["content"] for r in prompt_results if r["ok"]]
        if len(ok_contents) >= 2 and len(set(ok_contents)) != 1:
            no_seed_deterministic = False

    seeds_differ = True
    for pi in range(len(prompts)):
        s42_ok = [r["content"] for r in seed_42_results[pi] if r["ok"]]
        s99_ok = [r["content"] for r in seed_99_results[pi] if r["ok"]]
        if s42_ok and s99_ok and s42_ok[0] == s99_ok[0]:
            seeds_differ = False

    return {
        "deterministic_with_seed": seed_deterministic,
        "deterministic_without_seed": no_seed_deterministic,
        "different_seeds_produce_different_outputs": seeds_differ,
    }


def test_stop():
    print("\n[stop] Testing stop sequences...", flush=True)
    probes = [
        {"stop": "\n", "prompt": "Write three short sentences about AI, each on a new line."},
        {"stop": ["\n", "."], "prompt": "Write three short sentences about AI, each on a new line."},
        {"stop": "I", "prompt": "Write a paragraph about artificial intelligence."},
        {"stop": "XXXXXXXX", "prompt": "Write a short haiku about programming."},
        {"stop": ".", "prompt": "Write three sentences about machine learning."},
        {"stop": ["?", "!", "."], "prompt": "Write three sentences about deep learning."},
        {"stop": " intelligence", "prompt": "Define artificial intelligence in one sentence."},
        {"stop": ["\n\n", "."], "prompt": "Write two paragraphs about neural networks."},
        {"stop": "assistant", "prompt": "Write a short greeting to the user."},
        {"stop": [" the ", " The "], "prompt": "Explain what a transformer model is."},
    ]
    results = []
    for i, probe in enumerate(probes):
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": probe["prompt"]}],
            "stop": probe["stop"],
            "max_tokens": 200,
        }
        r = call_api(payload)
        content = ""
        if r["ok"]:
            try:
                content = r["data"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                pass
        stop_str = str(probe["stop"])[:30]
        stop_tokens = [probe["stop"]] if isinstance(probe["stop"], str) else probe["stop"]
        contains_stop = None
        if content:
            contains_stop = any(
                s in content for s in stop_tokens if s.strip()
            )
        results.append({
            "stop": probe["stop"],
            "prompt": probe["prompt"][:60],
            "ok": r["ok"],
            "status_code": r["status_code"],
            "content": truncate(content),
            "contains_stop": contains_stop,
        })
        print(f"  [{i+1}/{len(probes)}] stop={stop_str}: {'OK' if r['ok'] else 'ERR'} status={r['status_code']}", flush=True)

    stop_works = all(r["ok"] for r in results)
    correctly_truncated = sum(1 for r in results if r["ok"] and r.get("contains_stop") is False)
    failed = sum(1 for r in results if not r["ok"])

    return {
        "stop_works": stop_works,
        "correctly_truncated": correctly_truncated,
        "failed": failed,
        "examples": results,
    }


def test_logprobs():
    print("\n[logprobs] Testing logprobs...", flush=True)
    prompts = [
        'The capital of France is',
        'Python is a programming language used for',
        'The square root of 144 is',
        'Water freezes at 0 degrees',
        'The largest planet in our solar system is',
    ]
    results = []
    top_logprobs_count = 0
    sample_data = {}
    for i, prompt in enumerate(prompts):
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "logprobs": True,
            "top_logprobs": 5,
            "max_tokens": 20,
        }
        r = call_api(payload)
        logprob_data = None
        tl_count = 0
        if r["ok"]:
            try:
                choice = r["data"]["choices"][0]
                logprob_data = choice.get("logprobs")
                if logprob_data and "content" in logprob_data:
                    for token_data in logprob_data["content"]:
                        tl = token_data.get("top_logprobs", [])
                        if len(tl) > tl_count:
                            tl_count = len(tl)
            except (KeyError, IndexError):
                pass
        if tl_count > top_logprobs_count:
            top_logprobs_count = tl_count
        if logprob_data and not sample_data:
            sample_data = logprob_data
        results.append({
            "prompt": prompt[:50],
            "ok": r["ok"],
            "status_code": r["status_code"],
            "has_logprobs": logprob_data is not None,
            "max_top_logprobs_found": tl_count,
        })
        lp = "yes" if logprob_data else "no"
        print(f"  [{i+1}/{len(prompts)}] {'OK' if r['ok'] else 'ERR'} status={r['status_code']} logprobs={lp}", flush=True)

    supported = all(r["ok"] for r in results)
    logprobs_returned = any(r.get("has_logprobs") for r in results)

    return {
        "supported": supported,
        "logprobs_returned": logprobs_returned,
        "top_logprobs_returned": top_logprobs_count,
        "sample_data": sample_data if sample_data else None,
    }


def test_tools():
    print("\n[tools] Testing function calling...", flush=True)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and state or country",
                        },
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string",
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to return",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform a mathematical calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate",
                        },
                        "precision": {
                            "type": "integer",
                            "description": "Number of decimal places",
                        },
                    },
                    "required": ["expression"],
                },
            },
        },
    ]

    prompts = [
        "What is the weather in Paris today?",
        "Search the web for latest AI research papers.",
        "Calculate 1234 multiplied by 5678.",
        "What is the weather in Tokyo and calculate 50 times 30?",
        "Search for news about climate change.",
    ]

    results = []
    for i, prompt in enumerate(prompts):
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "tools": tools,
            "tool_choice": "auto",
        }
        r = call_api(payload, timeout=180)
        tool_calls = []
        tool_call_details = []
        if r["ok"]:
            try:
                msg = r["data"]["choices"][0]["message"]
                if "tool_calls" in msg:
                    tool_calls = msg["tool_calls"]
                    tool_call_details = [
                        {
                            "name": tc.get("function", {}).get("name", "?"),
                            "arguments": truncate(tc.get("function", {}).get("arguments", ""), 150),
                        }
                        for tc in tool_calls
                    ]
            except (KeyError, IndexError):
                pass
        results.append({
            "prompt": prompt[:50],
            "ok": r["ok"],
            "status_code": r["status_code"],
            "returned_tool_calls": len(tool_calls) > 0,
            "tool_calls_count": len(tool_calls),
            "tool_calls": tool_call_details,
        })
        tc_status = f"calls={len(tool_calls)}" if tool_calls else "no calls"
        print(f"  [{i+1}/{len(prompts)}] {'OK' if r['ok'] else 'ERR'} status={r['status_code']} {tc_status}", flush=True)

    supported = all(r["ok"] for r in results)
    returned_tool_calls = any(r["returned_tool_calls"] for r in results)

    return {
        "supported": supported,
        "returned_tool_calls": returned_tool_calls,
        "invocation_details": results,
    }


def test_n_choices():
    print("\n[n_choices] Testing multiple completions...", flush=True)
    prompts = [
        'What is the capital of France?',
        'Explain what a database is in one sentence.',
        'What is 2 + 2?',
        'Name three primary colors.',
        'Is Python dynamically typed?',
    ]
    results = []
    for i, prompt in enumerate(prompts):
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "n": 3,
            "max_tokens": 50,
        }
        r = call_api(payload)
        choices = []
        if r["ok"]:
            try:
                choices = [c["message"]["content"] for c in r["data"]["choices"]]
            except (KeyError, IndexError):
                pass
        n_returned = len(choices)
        all_different = len(set(choices)) == n_returned if n_returned > 1 else False
        results.append({
            "prompt": prompt[:50],
            "ok": r["ok"],
            "status_code": r["status_code"],
            "choices_count": n_returned,
            "choices": [truncate(c) for c in choices],
            "all_different": all_different,
        })
        print(f"  [{i+1}/{len(prompts)}] {'OK' if r['ok'] else 'ERR'} status={r['status_code']} n_returned={n_returned} all_diff={all_different}", flush=True)

    supported = all(r["ok"] for r in results)
    max_choices = max(r["choices_count"] for r in results)
    any_different = any(r.get("all_different") for r in results)

    return {
        "supported": supported,
        "choices_count": max_choices,
        "responses_different": any_different,
    }


def test_max_tokens_ceiling():
    print("\n[max_tokens_ceiling] Finding max output tokens...", flush=True)
    prompt = "Write a very long detailed essay about artificial intelligence, keep going until I tell you to stop."
    step_results = []
    low = 4096
    high = None

    current = low
    while True:
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": current,
        }
        print(f"  Testing max_tokens={current}...", flush=True)
        r = call_api(payload, timeout=300)
        completion_tokens = 0
        truncated = False
        if r["ok"]:
            try:
                completion_tokens = r["data"]["usage"]["completion_tokens"]
                truncated = completion_tokens >= current * 0.95
            except (KeyError, TypeError):
                pass

        step_results.append({
            "max_tokens": current,
            "ok": r["ok"],
            "status_code": r["status_code"],
            "completion_tokens": completion_tokens,
            "truncated": truncated,
        })
        print(f"    ok={r['ok']} completion_tokens={completion_tokens} truncated={truncated}", flush=True)

        if not r["ok"]:
            high = current
            break

        if current >= 131072:
            high = current
            break

        current *= 2

    if high is not None and low < high:
        while low < high - 1:
            mid = (low + high) // 2
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": mid,
            }
            print(f"  Binary search: testing max_tokens={mid}...", flush=True)
            r = call_api(payload, timeout=300)
            completion_tokens = 0
            truncated = False
            if r["ok"]:
                try:
                    completion_tokens = r["data"]["usage"]["completion_tokens"]
                    truncated = completion_tokens >= mid * 0.95
                except (KeyError, TypeError):
                    pass

            step_results.append({
                "max_tokens": mid,
                "ok": r["ok"],
                "status_code": r["status_code"],
                "completion_tokens": completion_tokens,
                "truncated": truncated,
            })
            print(f"    ok={r['ok']} completion_tokens={completion_tokens} truncated={truncated}", flush=True)

            if r["ok"]:
                low = mid
            else:
                high = mid

    max_ok_tokens = max((s["max_tokens"] for s in step_results if s["ok"]), default=0)
    max_ct = max((s["completion_tokens"] for s in step_results if s["ok"]), default=0)
    truncation_point = min(
        (s["max_tokens"] for s in step_results if s["ok"] and s["truncated"]),
        default=None,
    )

    return {
        "max_output_tokens": max_ct,
        "max_before_error": max_ok_tokens,
        "truncation_point": truncation_point,
        "step_results": step_results,
    }


def test_message_validation():
    print("\n[message_validation] Testing edge cases...", flush=True)
    cases = [
        {
            "name": "empty_messages_list",
            "payload": {"model": MODEL, "messages": []},
        },
        {
            "name": "empty_content_string",
            "payload": {"model": MODEL, "messages": [{"role": "user", "content": ""}]},
        },
        {
            "name": "missing_role_field",
            "payload": {"model": MODEL, "messages": [{"content": "hello"}]},
        },
        {
            "name": "invalid_role",
            "payload": {"model": MODEL, "messages": [{"role": "superuser", "content": "hello"}]},
        },
        {
            "name": "only_assistant_messages",
            "payload": {"model": MODEL, "messages": [{"role": "assistant", "content": "Hello"}]},
        },
        {
            "name": "system_user_assistant_empty",
            "payload": {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": ""},
                ],
            },
        },
        {
            "name": "very_long_system_prompt",
            "payload": {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "x" * 10000},
                    {"role": "user", "content": "Say hello."},
                ],
            },
        },
        {
            "name": "whitespace_only_content",
            "payload": {"model": MODEL, "messages": [{"role": "user", "content": "   \t   \n   "}]},
        },
        {
            "name": "null_bytes_in_message",
            "payload": {"model": MODEL, "messages": [{"role": "user", "content": "hello\x00world"}]},
        },
        {
            "name": "unicode_only_message",
            "payload": {"model": MODEL, "messages": [{"role": "user", "content": "你好世界"}]},
        },
        {
            "name": "content_as_integer",
            "payload": {"model": MODEL, "messages": [{"role": "user", "content": 42}]},
        },
        {
            "name": "content_as_null",
            "payload": {"model": MODEL, "messages": [{"role": "user", "content": None}]},
        },
    ]

    results = {}
    for case in cases:
        r = call_api(case["payload"], timeout=30)
        error_message = ""
        if r["data"] and isinstance(r["data"], dict):
            if "error" in r["data"]:
                error_message = truncate(str(r["data"]["error"]), 300)
            elif "message" in r["data"]:
                error_message = truncate(str(r["data"]["message"]), 300)
        results[case["name"]] = {
            "status_code": r["status_code"],
            "error_message": error_message,
            "handled_gracefully": r["status_code"] is not None and r["status_code"] != 500,
        }
        print(f"  {case['name']}: status={r['status_code']} graceful={results[case['name']]['handled_gracefully']}", flush=True)

    return results


def main():
    print("Starting LLM API Features Benchmark")
    print(f"Model: {MODEL}")
    print(f"API: {API_URL}")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}\n")

    if not API_KEY:
        print("WARNING: LLM_API_KEY environment variable not set. Requests may fail.", flush=True)

    test_groups = {
        "response_format": test_response_format(),
        "seed": test_seed(),
        "stop": test_stop(),
        "logprobs": test_logprobs(),
        "tools": test_tools(),
        "n_choices": test_n_choices(),
        "max_tokens_ceiling": test_max_tokens_ceiling(),
        "message_validation": test_message_validation(),
    }

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "test_groups": test_groups,
    }

    results_path = os.path.join(SCRIPT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
