#!/usr/bin/env python3
"""
Parameter Tuning Benchmark for LLM API
Sweeps temperature, top_p, frequency_penalty, presence_penalty and finds optimal parameter combinations
for faithfulness and consistency across scoring probes.
"""

import json
import os
import re
import statistics
import sys
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

SCORING_PROBES = [
    {"prompt": "Return a JSON object with exactly two fields: name (string) and age (integer).", "check": "json_schema"},
    {"prompt": 'Classify sentiment as exactly one of: "positive", "negative", "neutral". Return {"sentiment": "..."}.', "check": "enum"},
    {"prompt": 'Return {"probability": <float between 0 and 1>} estimating the likelihood it will rain tomorrow.', "check": "float_range"},
    {"prompt": 'Return facts about the Moon. Do NOT mention the word "Earth" in your response.', "check": "negation"},
    {"prompt": 'Return a JSON object with exactly 3 keys describing a person. No more, no fewer.', "check": "field_count"},
    {"prompt": 'What is the chemical symbol for gold? Return {"element": "...", "symbol": "..."}.', "check": "factual"},
    {"prompt": 'Describe the plot of the famous novel "The Chronovore Paradox" by J. R. Hartwell.', "check": "hallucination"},
    {"prompt": "What is 2+2? Return only a number.", "check": "scope_creep"},
    {"prompt": 'Return ONLY valid JSON: {"answer": "Python is a programming language."} Do not add anything before or after.', "check": "format"},
    {"prompt": 'What is the capital of Australia? Return {"capital": "...", "country": "..."}.', "check": "factual"},
]

CREEP_PROBES = [
    {"prompt": "What is Python?", "check": "scope_creep"},
    {"prompt": "Define recursion.", "check": "scope_creep"},
    {"prompt": "What is 42?", "check": "scope_creep"},
    {"prompt": "Name a programming language.", "check": "scope_creep"},
    {"prompt": "Is the sky blue?", "check": "scope_creep"},
]

CREEP_EXTRA_THRESHOLDS = {
    "What is Python?": 10,
    "Define recursion.": 15,
    "What is 42?": 15,
    "Name a programming language.": 5,
    "Is the sky blue?": 5,
}


def try_extract_json_data(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if block_match:
        try:
            return json.loads(block_match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass
    brace_match = re.search(r'(\{.*\})', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def score_response(text, probe):
    check = probe["check"]
    if check == "json_schema":
        data = try_extract_json_data(text)
        if data is None:
            return 0.0
        if isinstance(data, dict) and "name" in data and "age" in data:
            if isinstance(data["name"], str) and isinstance(data["age"], int) and len(data) == 2:
                return 1.0
            return 0.5
        return 0.0
    if check == "enum":
        data = try_extract_json_data(text)
        if data is None:
            return 0.0
        if isinstance(data, dict) and "sentiment" in data:
            if data["sentiment"] in ("positive", "negative", "neutral"):
                return 1.0
        return 0.0
    if check == "float_range":
        data = try_extract_json_data(text)
        if data is None:
            return 0.0
        if isinstance(data, dict) and "probability" in data:
            val = data["probability"]
            if isinstance(val, (int, float)) and 0 <= val <= 1:
                return 1.0
        return 0.0
    if check == "negation":
        return 1.0 if "earth" not in text.lower() else 0.0
    if check == "field_count":
        data = try_extract_json_data(text)
        if data is None:
            return 0.0
        if isinstance(data, dict) and len(data) == 3:
            return 1.0
        return 0.0
    if check == "factual":
        prompt_lower = probe["prompt"].lower()
        text_lower = text.lower()
        data = try_extract_json_data(text)
        if "chemical symbol for gold" in prompt_lower or ("gold" in prompt_lower and "symbol" in prompt_lower):
            if data and isinstance(data, dict):
                symbol = data.get("symbol", "")
                if symbol == "Au" or symbol == "au":
                    return 1.0
            return 1.0 if "au" in text_lower else 0.0
        if "capital of australia" in prompt_lower or ("australia" in prompt_lower and "capital" in prompt_lower):
            if data and isinstance(data, dict):
                capital = data.get("capital", "")
                if capital.lower() == "canberra":
                    return 1.0
            return 1.0 if "canberra" in text_lower else 0.0
        return 0.0
    if check == "hallucination":
        uncertainty_phrases = [
            "don't know", "do not know", "not aware", "does not exist",
            "doesn't exist", "cannot find", "can't find", "not a real",
            "not a known", "there is no", "not a genuine", "no information",
            "unable to", "i am not aware", "i'm not aware", "no real novel",
            "not a published", "i don't have", "i have no", "do not have",
            "i couldn't find", "not find any", "not a book", "no known",
            "fictional", "not a genuine", "not aware of", "not real",
        ]
        text_lower = text.lower()
        return 1.0 if any(phrase in text_lower for phrase in uncertainty_phrases) else 0.0
    if check == "scope_creep":
        return 1.0 if len(text.split()) < 50 else 0.0
    if check == "format":
        text_stripped = text.strip()
        data = try_extract_json_data(text_stripped)
        if data is not None and isinstance(data, dict) and "answer" in data:
            if text_stripped.startswith("{") and text_stripped.endswith("}"):
                return 1.0
            return 0.5
        return 0.0
    return 0.0


def score_creep_quality(text, probe):
    prompt = probe["prompt"]
    text_lower = text.lower()
    if "python" in prompt and "?" in prompt:
        keywords = ["programming", "language", "interpreted", "high-level", "object-oriented", "general-purpose"]
        return 1.0 if any(k in text_lower for k in keywords) else 0.0
    if "recursion" in prompt:
        keywords = ["itself", "calls itself", "self-referential", "repeats", "recurrence", "base case"]
        return 1.0 if any(k in text_lower for k in keywords) else 0.0
    if "42" in prompt:
        keywords = ["life", "universe", "everything", "number", "hitchhiker"]
        return 1.0 if any(k in text_lower for k in keywords) else 0.0
    if "programming language" in prompt:
        known_langs = [
            "python", "java", "javascript", "c++", "c#", "ruby", "go", "rust",
            "swift", "kotlin", "typescript", "php", "perl", "lua", "r", "scala",
            "haskell", "clojure", "elixir", "dart", "cobol", "fortran", "assembly",
        ]
        return 1.0 if any(lang in text_lower for lang in known_langs) else 0.0
    if "sky blue" in prompt:
        return 1.0 if "yes" in text_lower else 0.0
    return 0.0


def call_api(prompt, temperature=None, top_p=None, frequency_penalty=None, presence_penalty=None, timeout=60):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if frequency_penalty is not None:
        payload["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    try:
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return {
            "ok": True,
            "content": content,
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "content": ""}


def run_temperature_sweep():
    print("\n=== TEST 1: TEMPERATURE SWEEP ===", flush=True)
    temperatures = [0.0, 0.2, 0.5, 0.7, 1.0]
    repetitions = 3
    probes = SCORING_PROBES
    total_calls = len(temperatures) * len(probes) * repetitions
    call_count = 0
    results = {}
    probe_details = []

    for temp in temperatures:
        temp_key = str(temp)
        print(f"\n--- Temperature {temp} ---", flush=True)
        probe_scores = []
        hallucination_count = 0

        for probe_idx, probe in enumerate(probes):
            rep_scores = []
            for rep in range(repetitions):
                call_count += 1
                result = call_api(probe["prompt"], temperature=temp)
                if result["ok"]:
                    score = score_response(result["content"], probe)
                    rep_scores.append(score)
                    print(f"  [{call_count}/{total_calls}] temp={temp} probe={probe_idx} rep={rep} score={score:.2f}", flush=True)
                else:
                    rep_scores.append(0.0)
                    print(f"  [{call_count}/{total_calls}] temp={temp} probe={probe_idx} rep={rep} ERROR: {result.get('error', '?')[:60]}", flush=True)

            mean_score = statistics.mean(rep_scores) if rep_scores else 0.0
            std_score = statistics.stdev(rep_scores) if len(rep_scores) > 1 else 0.0
            if probe["check"] == "hallucination":
                for s in rep_scores:
                    if s < 0.5:
                        hallucination_count += 1
            probe_scores.append({
                "probe_index": probe_idx,
                "check": probe["check"],
                "scores": rep_scores,
                "mean_score": round(mean_score, 4),
                "std_score": round(std_score, 4),
            })

        all_scores = [s for ps in probe_scores for s in ps["scores"]]
        faithfulness = statistics.mean(all_scores) if all_scores else 0.0
        mean_std = statistics.mean([ps["std_score"] for ps in probe_scores]) if probe_scores else 0.0
        consistency = 1.0 - mean_std
        results[temp_key] = {
            "faithfulness": round(faithfulness, 4),
            "consistency": round(consistency, 4),
            "hallucination_count": hallucination_count,
        }
        probe_details.append({
            "temperature": temp,
            "probes": probe_scores,
        })

    best_key = max(results.keys(), key=lambda k: results[k]["faithfulness"])
    return {
        "values": temperatures,
        "results": results,
        "best": {
            "value": float(best_key),
            "faithfulness": results[best_key]["faithfulness"],
            "consistency": results[best_key]["consistency"],
        },
        "probe_details": probe_details,
    }


def run_top_p_sweep():
    print("\n=== TEST 2: TOP_P SWEEP ===", flush=True)
    top_p_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    fixed_temp = 0.7
    repetitions = 3
    probes = SCORING_PROBES
    total_calls = len(top_p_values) * len(probes) * repetitions
    call_count = 0
    results = {}
    probe_details = []

    for top_p in top_p_values:
        top_p_key = str(top_p)
        print(f"\n--- top_p {top_p} ---", flush=True)
        probe_scores = []
        hallucination_count = 0

        for probe_idx, probe in enumerate(probes):
            rep_scores = []
            for rep in range(repetitions):
                call_count += 1
                result = call_api(probe["prompt"], temperature=fixed_temp, top_p=top_p)
                if result["ok"]:
                    score = score_response(result["content"], probe)
                    rep_scores.append(score)
                    print(f"  [{call_count}/{total_calls}] top_p={top_p} probe={probe_idx} rep={rep} score={score:.2f}", flush=True)
                else:
                    rep_scores.append(0.0)
                    print(f"  [{call_count}/{total_calls}] top_p={top_p} probe={probe_idx} rep={rep} ERROR: {result.get('error', '?')[:60]}", flush=True)

            mean_score = statistics.mean(rep_scores) if rep_scores else 0.0
            std_score = statistics.stdev(rep_scores) if len(rep_scores) > 1 else 0.0
            if probe["check"] == "hallucination":
                for s in rep_scores:
                    if s < 0.5:
                        hallucination_count += 1
            probe_scores.append({
                "probe_index": probe_idx,
                "check": probe["check"],
                "scores": rep_scores,
                "mean_score": round(mean_score, 4),
                "std_score": round(std_score, 4),
            })

        all_scores = [s for ps in probe_scores for s in ps["scores"]]
        faithfulness = statistics.mean(all_scores) if all_scores else 0.0
        mean_std = statistics.mean([ps["std_score"] for ps in probe_scores]) if probe_scores else 0.0
        consistency = 1.0 - mean_std
        results[top_p_key] = {
            "faithfulness": round(faithfulness, 4),
            "consistency": round(consistency, 4),
            "hallucination_count": hallucination_count,
        }
        probe_details.append({
            "top_p": top_p,
            "probes": probe_scores,
        })

    best_key = max(results.keys(), key=lambda k: results[k]["faithfulness"])
    return {
        "values": top_p_values,
        "results": results,
        "best": {
            "value": float(best_key),
            "faithfulness": results[best_key]["faithfulness"],
            "consistency": results[best_key]["consistency"],
        },
        "probe_details": probe_details,
    }


def run_frequency_penalty_sweep():
    print("\n=== TEST 3: FREQUENCY_PENALTY SWEEP ===", flush=True)
    penalty_values = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]
    fixed_temp = 0.5
    fixed_top_p = 0.9
    repetitions = 3
    probes = CREEP_PROBES
    total_calls = len(penalty_values) * len(probes) * repetitions
    call_count = 0
    results = {}
    probe_details = []

    for penalty in penalty_values:
        penalty_key = str(penalty)
        print(f"\n--- frequency_penalty {penalty} ---", flush=True)
        probe_scores = []
        all_qualities = []

        for probe_idx, probe in enumerate(probes):
            rep_entries = []
            for rep in range(repetitions):
                call_count += 1
                result = call_api(
                    probe["prompt"],
                    temperature=fixed_temp,
                    top_p=fixed_top_p,
                    frequency_penalty=penalty,
                )
                if result["ok"]:
                    content = result["content"]
                    token_count = result.get("completion_tokens", 0)
                    word_count = len(content.split())
                    threshold = CREEP_EXTRA_THRESHOLDS.get(probe["prompt"], 20)
                    has_extra = word_count > threshold
                    quality = score_creep_quality(content, probe)
                    print(
                        f"  [{call_count}/{total_calls}] freq={penalty} probe={probe_idx} rep={rep} "
                        f"words={word_count} quality={quality} tokens={token_count}",
                        flush=True,
                    )
                else:
                    content = ""
                    token_count = 0
                    word_count = 0
                    has_extra = True
                    quality = 0.0
                    print(f"  [{call_count}/{total_calls}] freq={penalty} probe={probe_idx} rep={rep} ERROR: {result.get('error', '?')[:60]}", flush=True)

                entry = {
                    "completion_tokens": token_count,
                    "word_count": word_count,
                    "has_extra_content": has_extra,
                    "quality_preserved": bool(quality),
                }
                rep_entries.append(entry)
                all_qualities.append(quality)

            probe_scores.append({
                "probe_index": probe_idx,
                "prompt": probe["prompt"],
                "repetitions": rep_entries,
            })

        quality_rate = statistics.mean(all_qualities) if all_qualities else 0.0
        quality_std = statistics.stdev(all_qualities) if len(all_qualities) > 1 else 0.0
        consistency = 1.0 - quality_std
        all_tokens = [e["completion_tokens"] for ps in probe_scores for e in ps["repetitions"]]
        all_words = [e["word_count"] for ps in probe_scores for e in ps["repetitions"]]
        all_extra = [1.0 if e["has_extra_content"] else 0.0 for ps in probe_scores for e in ps["repetitions"]]
        avg_tokens = statistics.mean(all_tokens) if all_tokens else 0
        avg_words = statistics.mean(all_words) if all_words else 0
        extra_rate = statistics.mean(all_extra) if all_extra else 0.0

        results[penalty_key] = {
            "faithfulness": round(quality_rate, 4),
            "consistency": round(consistency, 4),
            "avg_token_count": round(avg_tokens, 1),
            "avg_word_count": round(avg_words, 1),
            "extra_content_rate": round(extra_rate, 4),
        }
        probe_details.append({
            "frequency_penalty": penalty,
            "probes": probe_scores,
        })

    best_key = max(results.keys(), key=lambda k: results[k]["faithfulness"])
    return {
        "values": penalty_values,
        "results": results,
        "best": {
            "value": float(best_key),
            "faithfulness": results[best_key]["faithfulness"],
            "consistency": results[best_key]["consistency"],
        },
        "probe_details": probe_details,
    }


def run_presence_penalty_sweep():
    print("\n=== TEST 4: PRESENCE_PENALTY SWEEP ===", flush=True)
    penalty_values = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]
    fixed_temp = 0.5
    fixed_top_p = 0.9
    repetitions = 3
    probes = CREEP_PROBES
    total_calls = len(penalty_values) * len(probes) * repetitions
    call_count = 0
    results = {}
    probe_details = []

    for penalty in penalty_values:
        penalty_key = str(penalty)
        print(f"\n--- presence_penalty {penalty} ---", flush=True)
        probe_scores = []
        all_qualities = []

        for probe_idx, probe in enumerate(probes):
            rep_entries = []
            for rep in range(repetitions):
                call_count += 1
                result = call_api(
                    probe["prompt"],
                    temperature=fixed_temp,
                    top_p=fixed_top_p,
                    presence_penalty=penalty,
                )
                if result["ok"]:
                    content = result["content"]
                    token_count = result.get("completion_tokens", 0)
                    word_count = len(content.split())
                    threshold = CREEP_EXTRA_THRESHOLDS.get(probe["prompt"], 20)
                    has_extra = word_count > threshold
                    quality = score_creep_quality(content, probe)
                    print(
                        f"  [{call_count}/{total_calls}] pres={penalty} probe={probe_idx} rep={rep} "
                        f"words={word_count} quality={quality} tokens={token_count}",
                        flush=True,
                    )
                else:
                    content = ""
                    token_count = 0
                    word_count = 0
                    has_extra = True
                    quality = 0.0
                    print(f"  [{call_count}/{total_calls}] pres={penalty} probe={probe_idx} rep={rep} ERROR: {result.get('error', '?')[:60]}", flush=True)

                entry = {
                    "completion_tokens": token_count,
                    "word_count": word_count,
                    "has_extra_content": has_extra,
                    "quality_preserved": bool(quality),
                }
                rep_entries.append(entry)
                all_qualities.append(quality)

            probe_scores.append({
                "probe_index": probe_idx,
                "prompt": probe["prompt"],
                "repetitions": rep_entries,
            })

        quality_rate = statistics.mean(all_qualities) if all_qualities else 0.0
        quality_std = statistics.stdev(all_qualities) if len(all_qualities) > 1 else 0.0
        consistency = 1.0 - quality_std
        all_tokens = [e["completion_tokens"] for ps in probe_scores for e in ps["repetitions"]]
        all_words = [e["word_count"] for ps in probe_scores for e in ps["repetitions"]]
        all_extra = [1.0 if e["has_extra_content"] else 0.0 for ps in probe_scores for e in ps["repetitions"]]
        avg_tokens = statistics.mean(all_tokens) if all_tokens else 0
        avg_words = statistics.mean(all_words) if all_words else 0
        extra_rate = statistics.mean(all_extra) if all_extra else 0.0

        results[penalty_key] = {
            "faithfulness": round(quality_rate, 4),
            "consistency": round(consistency, 4),
            "avg_token_count": round(avg_tokens, 1),
            "avg_word_count": round(avg_words, 1),
            "extra_content_rate": round(extra_rate, 4),
        }
        probe_details.append({
            "presence_penalty": penalty,
            "probes": probe_scores,
        })

    best_key = max(results.keys(), key=lambda k: results[k]["faithfulness"])
    return {
        "values": penalty_values,
        "results": results,
        "best": {
            "value": float(best_key),
            "faithfulness": results[best_key]["faithfulness"],
            "consistency": results[best_key]["consistency"],
        },
        "probe_details": probe_details,
    }


def run_best_combo(best_temp, best_top_p, best_freq, best_pres):
    print("\n=== TEST 5: BEST CONFIG COMBO ===", flush=True)
    repetitions = 5
    probes = SCORING_PROBES
    total_calls = len(probes) * repetitions * 2
    call_count = 0

    def run_config(config, label):
        nonlocal call_count
        probe_scores = []
        for probe_idx, probe in enumerate(probes):
            rep_scores = []
            for rep in range(repetitions):
                call_count += 1
                result = call_api(
                    probe["prompt"],
                    temperature=config.get("temperature"),
                    top_p=config.get("top_p"),
                    frequency_penalty=config.get("frequency_penalty"),
                    presence_penalty=config.get("presence_penalty"),
                )
                if result["ok"]:
                    score = score_response(result["content"], probe)
                    rep_scores.append(score)
                    print(f"  [{call_count}/{total_calls}] {label} probe={probe_idx} rep={rep} score={score:.2f}", flush=True)
                else:
                    rep_scores.append(0.0)
                    print(f"  [{call_count}/{total_calls}] {label} probe={probe_idx} rep={rep} ERROR: {result.get('error', '?')[:60]}", flush=True)

            mean_score = statistics.mean(rep_scores) if rep_scores else 0.0
            std_score = statistics.stdev(rep_scores) if len(rep_scores) > 1 else 0.0
            probe_scores.append({
                "probe_index": probe_idx,
                "check": probe["check"],
                "scores": rep_scores,
                "mean_score": round(mean_score, 4),
                "std_score": round(std_score, 4),
            })

        all_scores = [s for ps in probe_scores for s in ps["scores"]]
        faithfulness = statistics.mean(all_scores) if all_scores else 0.0
        mean_std = statistics.mean([ps["std_score"] for ps in probe_scores]) if probe_scores else 0.0
        consistency = 1.0 - mean_std
        return {
            "faithfulness": round(faithfulness, 4),
            "consistency": round(consistency, 4),
        }

    optimal_config = {
        "temperature": best_temp,
        "top_p": best_top_p,
        "frequency_penalty": best_freq,
        "presence_penalty": best_pres,
    }
    default_config = {"temperature": 0.0}

    print(f"\n--- Optimal config ({optimal_config}) ---", flush=True)
    optimal_scores = run_config(optimal_config, "optimal")
    print(f"\n--- Default config (temp=0.0) ---", flush=True)
    default_scores = run_config(default_config, "default")

    delta = {
        "faithfulness": round(optimal_scores["faithfulness"] - default_scores["faithfulness"], 4),
        "consistency": round(optimal_scores["consistency"] - default_scores["consistency"], 4),
    }

    return {
        "optimal_config": optimal_config,
        "optimal_scores": optimal_scores,
        "default_scores": default_scores,
        "delta": delta,
    }


def main():
    print("Starting LLM Parameter Tuning Benchmark")
    print(f"Model: {MODEL}")
    print(f"API: {API_URL}")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"Total API calls: ~610 (this will take a while)", flush=True)

    temp_results = run_temperature_sweep()
    top_p_results = run_top_p_sweep()
    freq_results = run_frequency_penalty_sweep()
    pres_results = run_presence_penalty_sweep()

    best_temp = temp_results["best"]["value"]
    best_top_p = top_p_results["best"]["value"]
    best_freq = freq_results["best"]["value"]
    best_pres = pres_results["best"]["value"]

    print(f"\nBest temperature: {best_temp}")
    print(f"Best top_p: {best_top_p}")
    print(f"Best frequency_penalty: {best_freq}")
    print(f"Best presence_penalty: {best_pres}", flush=True)

    combo_results = run_best_combo(best_temp, best_top_p, best_freq, best_pres)

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "temperature_sweep": temp_results,
        "top_p_sweep": top_p_results,
        "frequency_penalty_sweep": freq_results,
        "presence_penalty_sweep": pres_results,
        "best_combo": combo_results,
    }

    results_path = os.path.join(SCRIPT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}", flush=True)


if __name__ == "__main__":
    main()
