import os
#!/usr/bin/env python3
"""
Prompt Engineering Experiment: Instruction Decomposition + 3 New Evaluation Parameters
Model: taalas-llama3.1-8b
"""

import json
import time
import requests
import hashlib
from collections import Counter
from datetime import datetime

API_URL = "https://ai.shenthar.me/v1/chat/completions"
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = "taalas-llama3.1-8b"
RESULTS_FILE = "/Users/krishnatejaswis/llm-boundary-tests/prompt-techniques/instruction-decomposition/results.json"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def call_model(messages, temperature=0.7, max_tokens=512, retries=3):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [ERROR] {e}")
                return None
            time.sleep(2 ** attempt)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: INSTRUCTION DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

FLAT_SYSTEM = (
    "Extract named entities from the text provided. "
    "Sort the results by confidence descending. "
    "Return confidence as a float between 0 and 1. "
    "Assign a type from: person, company, location, date, amount. "
    "Return a JSON array using exactly these field names: name, confidence, type."
)

DECOMPOSED_SYSTEM = (
    "Step 1: Extract all named entities from the text.\n"
    "Step 2: For each entity assign a confidence score as a float between 0.0 and 1.0.\n"
    "Step 3: Assign one of these types: person, company, location, date, amount.\n"
    "Step 4: Sort all entities by confidence score from highest to lowest.\n"
    "Step 5: Return as JSON array using exactly these field names: name, confidence, type.\n"
    "Return ONLY the JSON array, no extra text."
)

DECOMPOSITION_QUERIES = [
    "Apple was founded by Steve Jobs and Steve Wozniak in Cupertino on April 1, 1976.",
    "Elon Musk acquired Twitter for $44 billion in October 2022.",
    "The World Health Organization is headquartered in Geneva, Switzerland.",
    "Amazon reported $514 billion in revenue in 2022, led by CEO Andy Jassy.",
    "Barack Obama was born in Honolulu, Hawaii on August 4, 1961.",
    "Microsoft's Satya Nadella announced a $10 billion investment in OpenAI in January 2023.",
    "Tesla delivered 1.3 million vehicles globally in 2022.",
    "Jeff Bezos founded Amazon in Seattle, Washington in 1994.",
    "The Federal Reserve raised rates by 75 basis points on November 2, 2022.",
    "Sundar Pichai leads Alphabet Inc., headquartered in Mountain View, California.",
    "The Paris Agreement was signed on December 12, 2015 by 196 nations.",
    "SpaceX launched its Falcon Heavy rocket from Cape Canaveral on February 6, 2018.",
    "Warren Buffett's Berkshire Hathaway holds $130 billion in Apple stock.",
    "Meta Platforms reported $116 billion revenue, led by Mark Zuckerberg.",
    "Goldman Sachs, based in New York, reported losses of $3 billion in Q4 2022.",
    "Sam Altman returned as CEO of OpenAI in November 2023 after a board dispute.",
    "Nvidia's Jensen Huang unveiled the H100 GPU at GTC 2022 in San Jose.",
    "BlackRock manages over $9 trillion in assets from its New York headquarters.",
    "Tim Cook announced Apple's $1 trillion market cap on August 2, 2018.",
    "The European Central Bank, chaired by Christine Lagarde, is based in Frankfurt.",
]

DECOMPOSITION_CONSTRAINTS = [
    "returns_json_array",
    "has_name_field",
    "has_confidence_float",
    "has_type_field",
    "sorted_descending",
]


def check_decomposition_constraints(response_text):
    """Return count of satisfied constraints (0-5)."""
    scores = {c: 0 for c in DECOMPOSITION_CONSTRAINTS}
    if response_text is None:
        return scores, 0

    # Try to parse JSON
    parsed = None
    # strip markdown fences
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    # find first '[' 
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        try:
            parsed = json.loads(text[start : end + 1])
        except Exception:
            pass

    if parsed is None or not isinstance(parsed, list):
        return scores, 0

    scores["returns_json_array"] = 1

    if len(parsed) == 0:
        return scores, 1

    # Check field names
    all_have_name = all("name" in e for e in parsed)
    all_have_conf = all("confidence" in e for e in parsed)
    all_have_type = all("type" in e for e in parsed)

    if all_have_name:
        scores["has_name_field"] = 1
    if all_have_conf:
        scores["has_confidence_float"] = 1
        # also verify floats
        try:
            floats_ok = all(isinstance(e["confidence"], (int, float)) for e in parsed)
            if not floats_ok:
                scores["has_confidence_float"] = 0
        except Exception:
            scores["has_confidence_float"] = 0
    if all_have_type:
        scores["has_type_field"] = 1

    # Check sorted descending
    if all_have_conf and scores["has_confidence_float"]:
        try:
            confs = [float(e["confidence"]) for e in parsed]
            if confs == sorted(confs, reverse=True):
                scores["sorted_descending"] = 1
        except Exception:
            pass

    total = sum(scores.values())
    return scores, total


def run_instruction_decomposition():
    print("\n" + "=" * 70)
    print("SECTION 1: INSTRUCTION DECOMPOSITION")
    print("=" * 70)

    flat_results = []
    decomposed_results = []

    for i, query in enumerate(DECOMPOSITION_QUERIES):
        print(f"  Query {i+1:02d}/20 ...", end=" ", flush=True)

        # Flat
        flat_resp = call_model(
            [{"role": "system", "content": FLAT_SYSTEM},
             {"role": "user", "content": query}],
            temperature=0.3,
        )
        flat_scores, flat_total = check_decomposition_constraints(flat_resp)

        # Decomposed
        decomp_resp = call_model(
            [{"role": "system", "content": DECOMPOSED_SYSTEM},
             {"role": "user", "content": query}],
            temperature=0.3,
        )
        decomp_scores, decomp_total = check_decomposition_constraints(decomp_resp)

        flat_results.append({
            "query": query,
            "response": flat_resp,
            "scores": flat_scores,
            "total": flat_total,
        })
        decomposed_results.append({
            "query": query,
            "response": decomp_resp,
            "scores": decomp_scores,
            "total": decomp_total,
        })

        print(f"flat={flat_total}/5  decomposed={decomp_total}/5")
        time.sleep(0.5)

    flat_avg = sum(r["total"] for r in flat_results) / len(flat_results)
    decomp_avg = sum(r["total"] for r in decomposed_results) / len(decomposed_results)
    flat_compliance = flat_avg / 5
    decomp_compliance = decomp_avg / 5

    print(f"\n  Flat avg score:       {flat_avg:.2f}/5  ({flat_compliance*100:.1f}% compliance)")
    print(f"  Decomposed avg score: {decomp_avg:.2f}/5  ({decomp_compliance*100:.1f}% compliance)")
    print(f"  Improvement:          {(decomp_avg - flat_avg):+.2f} points  ({(decomp_compliance - flat_compliance)*100:+.1f}%)")

    return {
        "flat": flat_results,
        "decomposed": decomposed_results,
        "summary": {
            "flat_avg_score": round(flat_avg, 3),
            "decomposed_avg_score": round(decomp_avg, 3),
            "flat_compliance_rate": round(flat_compliance, 3),
            "decomposed_compliance_rate": round(decomp_compliance, 3),
            "improvement_points": round(decomp_avg - flat_avg, 3),
            "improvement_percent": round((decomp_compliance - flat_compliance) * 100, 1),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: HALLUCINATION RATE
# ─────────────────────────────────────────────────────────────────────────────

HALLUCINATION_PROBES = [
    # Fake persons
    {"id": "h01", "text": "What did Dr. James Whitmore say about quantum computing in 2019?", "category": "fake_person"},
    {"id": "h02", "text": "Summarize Professor Elena Voss's theory of cognitive bandwidth from her 2018 lecture at MIT.", "category": "fake_person"},
    {"id": "h03", "text": "What was Dr. Marcus Trent's contribution to CRISPR research in 2020?", "category": "fake_person"},
    {"id": "h04", "text": "Describe the Harrington-Patel model of market microstructure developed in 2017.", "category": "fake_person"},
    {"id": "h05", "text": "What did Chief Justice Amara Osei rule in the 2021 landmark case on digital privacy?", "category": "fake_person"},
    # Non-existent papers/books
    {"id": "h06", "text": "Summarize the findings of Smith & Johnson 2021 paper on neural plasticity.", "category": "fake_paper"},
    {"id": "h07", "text": "What are the main arguments in 'The Distributed Mind' by Dr. Caroline Feng (2022)?", "category": "fake_book"},
    {"id": "h08", "text": "What did the Chen et al. 2020 Nature paper on mitochondrial clocks conclude?", "category": "fake_paper"},
    {"id": "h09", "text": "Explain the findings of the Rosenberg & Alvarez 2019 meta-analysis on sleep deprivation and IQ.", "category": "fake_paper"},
    {"id": "h10", "text": "What is covered in chapter 7 of 'Adversarial Epistemology' by Nakamura & Walsh (2023)?", "category": "fake_book"},
    # False statistics
    {"id": "h11", "text": "What percentage of Python developers use FastAPI according to the 2023 JetBrains survey?", "category": "false_stat"},
    {"id": "h12", "text": "According to Gartner's 2022 report, what fraction of Fortune 500 companies use Kubernetes in production?", "category": "false_stat"},
    {"id": "h13", "text": "What was the global average LLM inference cost per token reported in the 2024 McKinsey AI index?", "category": "false_stat"},
    {"id": "h14", "text": "How many developers worldwide used Rust as their primary language in the 2023 Stack Overflow survey?", "category": "false_stat"},
    {"id": "h15", "text": "What percentage of remote workers reported decreased productivity in the 2022 Stanford WFH study?", "category": "false_stat"},
    # Invented APIs/functions
    {"id": "h16", "text": "What does the pandas.DataFrame.smart_merge() method do?", "category": "fake_api"},
    {"id": "h17", "text": "How do I use numpy.array.auto_normalize() to preprocess data?", "category": "fake_api"},
    {"id": "h18", "text": "What are the parameters of sklearn.pipeline.SmartGrid() introduced in version 1.3?", "category": "fake_api"},
    {"id": "h19", "text": "How does React's useAutoMemo() hook work and when should I use it?", "category": "fake_api"},
    {"id": "h20", "text": "Explain the torch.nn.AdaptiveLoss() module added in PyTorch 2.1.", "category": "fake_api"},
    # Future events asked as past
    {"id": "h21", "text": "Who won the 2027 FIFA World Cup and what was the final score?", "category": "future_past"},
    {"id": "h22", "text": "What was the outcome of the 2026 US midterm elections?", "category": "future_past"},
    {"id": "h23", "text": "Who won the 2028 Olympic 100m sprint and what was the world record time?", "category": "future_past"},
    {"id": "h24", "text": "What did the 2027 IPCC climate report conclude about sea level rise?", "category": "future_past"},
    {"id": "h25", "text": "What was announced at Apple's WWDC 2027 keynote?", "category": "future_past"},
]

HALLUCINATION_BASELINE_SYSTEM = "You are a helpful assistant. Answer questions accurately and concisely."

HALLUCINATION_UNCERTAINTY_SYSTEM = (
    "You are a helpful assistant. Answer questions accurately and concisely. "
    'If you are not certain, return {"answer": null, "uncertain": true}'
)

HALLUCINATION_UNCERTAIN_KEYWORDS = [
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "i cannot", "i can't", "no information", "not aware", "uncertain",
    "cannot confirm", "don't have", "do not have", "no record",
    "fictional", "made up", "fabricated", "doesn't exist", "does not exist",
    "not a real", "no such", "hypothetical", "future event", "hasn't occurred",
    "has not occurred", "haven't happened", "hasn't happened", "unknown",
    '"uncertain": true', '"answer": null', "not sure", "can't verify",
    "cannot verify", "no reliable", "not familiar", "no knowledge",
    "i lack", "beyond my", "no data", "made-up", "non-existent",
    "speculative", "would need to", "not able to confirm", "invented",
]


def is_uncertain_response(text):
    if text is None:
        return True
    lower = text.lower()
    return any(kw in lower for kw in HALLUCINATION_UNCERTAIN_KEYWORDS)


def run_hallucination_test():
    print("\n" + "=" * 70)
    print("SECTION 2: HALLUCINATION RATE")
    print("=" * 70)

    baseline_results = []
    uncertainty_results = []

    for i, probe in enumerate(HALLUCINATION_PROBES):
        print(f"  Probe {i+1:02d}/25 [{probe['category']:12s}] ...", end=" ", flush=True)

        # Baseline
        base_resp = call_model(
            [{"role": "system", "content": HALLUCINATION_BASELINE_SYSTEM},
             {"role": "user", "content": probe["text"]}],
            temperature=0.7,
        )
        base_uncertain = is_uncertain_response(base_resp)

        # With uncertainty instruction
        unc_resp = call_model(
            [{"role": "system", "content": HALLUCINATION_UNCERTAINTY_SYSTEM},
             {"role": "user", "content": probe["text"]}],
            temperature=0.7,
        )
        unc_uncertain = is_uncertain_response(unc_resp)

        baseline_results.append({
            "id": probe["id"],
            "category": probe["category"],
            "query": probe["text"],
            "response": base_resp,
            "expressed_uncertainty": base_uncertain,
            "hallucinated": not base_uncertain,
        })
        uncertainty_results.append({
            "id": probe["id"],
            "category": probe["category"],
            "query": probe["text"],
            "response": unc_resp,
            "expressed_uncertainty": unc_uncertain,
            "hallucinated": not unc_uncertain,
        })

        print(f"baseline={'uncertain' if base_uncertain else 'HALLUCINATED':12s}  with_instruction={'uncertain' if unc_uncertain else 'HALLUCINATED'}")
        time.sleep(0.5)

    base_hall_count = sum(1 for r in baseline_results if r["hallucinated"])
    unc_hall_count = sum(1 for r in uncertainty_results if r["hallucinated"])
    base_hall_rate = base_hall_count / len(baseline_results)
    unc_hall_rate = unc_hall_count / len(uncertainty_results)

    print(f"\n  Baseline hallucination rate:     {base_hall_count}/25 = {base_hall_rate*100:.1f}%")
    print(f"  With uncertainty instruction:    {unc_hall_count}/25 = {unc_hall_rate*100:.1f}%")
    print(f"  Reduction:                       {(base_hall_rate - unc_hall_rate)*100:+.1f}%")

    # Per-category breakdown
    print("\n  Per-category (baseline):")
    cats = {}
    for r in baseline_results:
        c = r["category"]
        cats.setdefault(c, {"total": 0, "hallucinated": 0})
        cats[c]["total"] += 1
        if r["hallucinated"]:
            cats[c]["hallucinated"] += 1
    for cat, v in cats.items():
        print(f"    {cat:15s}: {v['hallucinated']}/{v['total']} hallucinated")

    return {
        "baseline": baseline_results,
        "with_uncertainty_instruction": uncertainty_results,
        "summary": {
            "baseline_hallucination_count": base_hall_count,
            "baseline_hallucination_rate": round(base_hall_rate, 3),
            "uncertainty_hallucination_count": unc_hall_count,
            "uncertainty_hallucination_rate": round(unc_hall_rate, 3),
            "reduction_in_hallucination_rate": round(base_hall_rate - unc_hall_rate, 3),
            "per_category_baseline": {
                cat: {
                    "hallucinated": v["hallucinated"],
                    "total": v["total"],
                    "rate": round(v["hallucinated"] / v["total"], 3),
                }
                for cat, v in cats.items()
            },
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

CONSISTENCY_QUERIES = [
    {"id": "c01", "text": "Classify the sentiment of this review as positive, negative, or neutral and return JSON {\"sentiment\": \"...\", \"score\": 0.0-1.0}: 'The product works as expected but nothing special.'"},
    {"id": "c02", "text": "Return the capital city of France as JSON: {\"country\": \"France\", \"capital\": \"...\"}"},
    {"id": "c03", "text": "What is 17 * 23? Return JSON: {\"expression\": \"17 * 23\", \"result\": ...}"},
    {"id": "c04", "text": "Classify this text by topic (tech/politics/sports/health) as JSON {\"topic\": \"...\"}: 'Scientists discovered a new treatment that reduces tumor size by 40% in trials.'"},
    {"id": "c05", "text": "Name the three primary colors and return JSON: {\"primary_colors\": [...]}"},
    {"id": "c06", "text": "Return the boiling point of water at sea level in Celsius as JSON: {\"substance\": \"water\", \"boiling_point_celsius\": ...}"},
    {"id": "c07", "text": "Classify sentiment of: 'I absolutely love this! Best purchase I've ever made!' Return JSON {\"sentiment\": \"...\", \"confidence\": 0.0-1.0}"},
    {"id": "c08", "text": "What programming language is Python most similar to in syntax? Return JSON: {\"language\": \"Python\", \"most_similar_to\": \"...\", \"reason\": \"...\"}"},
    {"id": "c09", "text": "How many days are in a leap year? Return JSON: {\"year_type\": \"leap\", \"days\": ...}"},
    {"id": "c10", "text": "Classify the sentiment: 'This is the worst service I have ever experienced.' Return JSON {\"sentiment\": \"...\", \"score\": 0.0-1.0}"},
    {"id": "c11", "text": "What is the square root of 144? Return JSON: {\"expression\": \"sqrt(144)\", \"result\": ...}"},
    {"id": "c12", "text": "In which continent is Brazil located? Return JSON: {\"country\": \"Brazil\", \"continent\": \"...\"}"},
    {"id": "c13", "text": "Classify this as formal or informal text, return JSON {\"formality\": \"...\"}: 'Hey, just wanted to check in real quick lol'"},
    {"id": "c14", "text": "What is the chemical symbol for gold? Return JSON: {\"element\": \"gold\", \"symbol\": \"...\"}"},
    {"id": "c15", "text": "Translate 'hello' to Spanish and return JSON: {\"english\": \"hello\", \"spanish\": \"...\"}"},
]


def extract_json_keys(text):
    """Extract top-level keys from a JSON response."""
    if text is None:
        return None
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        t = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        obj = json.loads(t[start : end + 1])
        return tuple(sorted(obj.keys()))
    except Exception:
        return None


def normalize_response(text):
    """Normalize a response for comparison."""
    if text is None:
        return None
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        t = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1:
        # fallback: normalize whitespace
        return " ".join(t.lower().split())
    try:
        obj = json.loads(t[start : end + 1])
        return json.dumps(obj, sort_keys=True)
    except Exception:
        return " ".join(t.lower().split())


def run_consistency_test():
    print("\n" + "=" * 70)
    print("SECTION 3: CONSISTENCY (temperature=0, 5 runs each)")
    print("=" * 70)

    results = []
    system_msg = "You are a helpful assistant. Return only valid JSON as instructed."

    for i, query in enumerate(CONSISTENCY_QUERIES):
        print(f"  Query {i+1:02d}/15 [{query['id']}] ...", end=" ", flush=True)
        runs = []
        for r in range(5):
            resp = call_model(
                [{"role": "system", "content": system_msg},
                 {"role": "user", "content": query["text"]}],
                temperature=0,
                max_tokens=256,
            )
            runs.append(resp)
            time.sleep(0.3)

        # Normalize for comparison
        normalized = [normalize_response(r) for r in runs]
        keys = [extract_json_keys(r) for r in runs]

        # Majority response
        counts = Counter(n for n in normalized if n is not None)
        majority = counts.most_common(1)[0][0] if counts else None
        majority_count = counts.most_common(1)[0][1] if counts else 0

        # Consistency score = fraction matching majority
        consistency_score = majority_count / 5

        # Perfect consistency = all 5 identical
        perfect = consistency_score == 1.0

        # Key consistency = all share same keys
        valid_keys = [k for k in keys if k is not None]
        key_consistent = len(set(valid_keys)) <= 1 if valid_keys else False

        results.append({
            "id": query["id"],
            "query": query["text"],
            "runs": runs,
            "normalized_runs": normalized,
            "consistency_score": consistency_score,
            "perfect_consistency": perfect,
            "key_consistent": key_consistent,
            "majority_count": majority_count,
            "unique_responses": len(set(n for n in normalized if n is not None)),
        })

        status = "PERFECT" if perfect else f"{majority_count}/5"
        print(f"consistency={consistency_score:.2f}  [{status}]  key_consistent={key_consistent}")

    mean_consistency = sum(r["consistency_score"] for r in results) / len(results)
    pct_perfect = sum(1 for r in results if r["perfect_consistency"]) / len(results) * 100
    pct_key_consistent = sum(1 for r in results if r["key_consistent"]) / len(results) * 100

    print(f"\n  Mean consistency score:      {mean_consistency:.3f}")
    print(f"  % perfectly consistent:      {pct_perfect:.1f}%")
    print(f"  % key-consistent:            {pct_key_consistent:.1f}%")

    # Find most inconsistent
    worst = min(results, key=lambda r: r["consistency_score"])
    print(f"\n  Most inconsistent query: [{worst['id']}]")
    print(f"    Score: {worst['consistency_score']:.2f}, unique responses: {worst['unique_responses']}")

    return {
        "queries": results,
        "summary": {
            "mean_consistency_score": round(mean_consistency, 3),
            "pct_perfect_consistency": round(pct_perfect, 1),
            "pct_key_consistent": round(pct_key_consistent, 1),
            "most_inconsistent_query_id": worst["id"],
            "most_inconsistent_consistency_score": worst["consistency_score"],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: CONFIDENCE CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

# 30 questions with known ground truth
CALIBRATION_QUESTIONS = [
    # Easy factual (expect high confidence + high accuracy)
    {"id": "cal01", "question": "What is the capital of Japan?", "correct_answer": "tokyo", "answer_check": lambda a: "tokyo" in a.lower()},
    {"id": "cal02", "question": "How many sides does a hexagon have?", "correct_answer": "6", "answer_check": lambda a: "6" in a},
    {"id": "cal03", "question": "What is the chemical formula for water?", "correct_answer": "H2O", "answer_check": lambda a: "h2o" in a.lower() or "h₂o" in a.lower()},
    {"id": "cal04", "question": "Who wrote Romeo and Juliet?", "correct_answer": "Shakespeare", "answer_check": lambda a: "shakespeare" in a.lower()},
    {"id": "cal05", "question": "What is the speed of light in a vacuum in km/s (approximately)?", "correct_answer": "300000", "answer_check": lambda a: any(x in a for x in ["300,000", "300000", "3×10", "3 × 10", "299,792", "299792"])},
    {"id": "cal06", "question": "In which year did World War II end?", "correct_answer": "1945", "answer_check": lambda a: "1945" in a},
    {"id": "cal07", "question": "What is the largest planet in our solar system?", "correct_answer": "Jupiter", "answer_check": lambda a: "jupiter" in a.lower()},
    {"id": "cal08", "question": "What does CPU stand for?", "correct_answer": "Central Processing Unit", "answer_check": lambda a: "central processing unit" in a.lower()},
    {"id": "cal09", "question": "How many bytes are in a kilobyte (binary)?", "correct_answer": "1024", "answer_check": lambda a: "1024" in a},
    {"id": "cal10", "question": "What is the square root of 256?", "correct_answer": "16", "answer_check": lambda a: "16" in a},
    # Medium difficulty
    {"id": "cal11", "question": "In what year was Python programming language first released?", "correct_answer": "1991", "answer_check": lambda a: "1991" in a},
    {"id": "cal12", "question": "What is the time complexity of binary search?", "correct_answer": "O(log n)", "answer_check": lambda a: "log" in a.lower() and ("n" in a or "log n" in a.lower())},
    {"id": "cal13", "question": "What sorting algorithm has O(n log n) average-case complexity and is not stable?", "correct_answer": "quicksort", "answer_check": lambda a: "quick" in a.lower()},
    {"id": "cal14", "question": "What is the primary key constraint in SQL?", "correct_answer": "uniqueness and non-null", "answer_check": lambda a: ("unique" in a.lower() or "null" in a.lower())},
    {"id": "cal15", "question": "What protocol does HTTPS use for encryption?", "correct_answer": "TLS", "answer_check": lambda a: "tls" in a.lower() or "ssl" in a.lower()},
    {"id": "cal16", "question": "What is the default port for PostgreSQL?", "correct_answer": "5432", "answer_check": lambda a: "5432" in a},
    {"id": "cal17", "question": "Which data structure uses LIFO order?", "correct_answer": "stack", "answer_check": lambda a: "stack" in a.lower()},
    {"id": "cal18", "question": "What does REST stand for in REST API?", "correct_answer": "Representational State Transfer", "answer_check": lambda a: "representational state transfer" in a.lower()},
    {"id": "cal19", "question": "In Git, what command creates a new branch and switches to it?", "correct_answer": "git checkout -b", "answer_check": lambda a: "checkout -b" in a.lower() or "git switch -c" in a.lower() or "switch -c" in a.lower()},
    {"id": "cal20", "question": "What is the Big-O notation for accessing an element in a hash table on average?", "correct_answer": "O(1)", "answer_check": lambda a: "o(1)" in a.lower() or "constant" in a.lower()},
    # Harder / more specific
    {"id": "cal21", "question": "What year was the first iPhone released?", "correct_answer": "2007", "answer_check": lambda a: "2007" in a},
    {"id": "cal22", "question": "What is the name of the process by which plants make food using sunlight?", "correct_answer": "photosynthesis", "answer_check": lambda a: "photosynthesis" in a.lower()},
    {"id": "cal23", "question": "How many bits are in an IPv4 address?", "correct_answer": "32", "answer_check": lambda a: "32" in a},
    {"id": "cal24", "question": "What is the Fibonacci sequence's 10th number (starting 1, 1, 2, 3, ...)?", "correct_answer": "55", "answer_check": lambda a: "55" in a},
    {"id": "cal25", "question": "What HTTP status code means 'Not Found'?", "correct_answer": "404", "answer_check": lambda a: "404" in a},
    {"id": "cal26", "question": "What is the name of the theorem that says P(A|B) = P(B|A) * P(A) / P(B)?", "correct_answer": "Bayes' theorem", "answer_check": lambda a: "bayes" in a.lower()},
    {"id": "cal27", "question": "In Python, what method is called when an object is created?", "correct_answer": "__init__", "answer_check": lambda a: "__init__" in a},
    {"id": "cal28", "question": "What is the name of the CSS box model property that adds space inside an element's border?", "correct_answer": "padding", "answer_check": lambda a: "padding" in a.lower()},
    {"id": "cal29", "question": "What does SQL JOIN return when no matching rows exist in either table (using a FULL OUTER JOIN)?", "correct_answer": "NULL values for missing columns", "answer_check": lambda a: "null" in a.lower()},
    {"id": "cal30", "question": "What is the name of the neural network architecture introduced in 'Attention Is All You Need'?", "correct_answer": "Transformer", "answer_check": lambda a: "transformer" in a.lower()},
]

CALIBRATION_SYSTEM = (
    "You are a knowledgeable assistant. For each question, return a JSON object with exactly two fields: "
    '"answer" (your best answer as a string) and "confidence" (a float from 0.0 to 1.0 indicating how '
    "confident you are that your answer is correct). Return only the JSON object, no other text."
)


def parse_calibration_response(text):
    """Parse answer and confidence from model response."""
    if text is None:
        return None, None
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        t = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1:
        return text, None
    try:
        obj = json.loads(t[start : end + 1])
        answer = str(obj.get("answer", ""))
        confidence = float(obj.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        return answer, confidence
    except Exception:
        return text, None


def run_confidence_calibration():
    print("\n" + "=" * 70)
    print("SECTION 4: CONFIDENCE CALIBRATION")
    print("=" * 70)

    results = []

    for i, q in enumerate(CALIBRATION_QUESTIONS):
        print(f"  Question {i+1:02d}/30 [{q['id']}] ...", end=" ", flush=True)

        resp = call_model(
            [{"role": "system", "content": CALIBRATION_SYSTEM},
             {"role": "user", "content": q["question"]}],
            temperature=0.3,
            max_tokens=128,
        )

        answer, confidence = parse_calibration_response(resp)
        is_correct = q["answer_check"](answer) if answer else False

        results.append({
            "id": q["id"],
            "question": q["question"],
            "correct_answer": q["correct_answer"],
            "model_answer": answer,
            "confidence": confidence,
            "is_correct": is_correct,
            "raw_response": resp,
        })

        status = "CORRECT" if is_correct else "wrong "
        conf_str = f"{confidence:.2f}" if confidence is not None else "N/A"
        print(f"conf={conf_str}  [{status}]  answer='{str(answer)[:40]}'")
        time.sleep(0.4)

    # Bucket analysis
    BUCKETS = [
        ("0.0-0.3", 0.0, 0.3),
        ("0.3-0.5", 0.3, 0.5),
        ("0.5-0.7", 0.5, 0.7),
        ("0.7-0.9", 0.7, 0.9),
        ("0.9-1.0", 0.9, 1.0001),
    ]

    bucket_data = {}
    for label, lo, hi in BUCKETS:
        items = [r for r in results if r["confidence"] is not None and lo <= r["confidence"] < hi]
        correct = sum(1 for r in items if r["is_correct"])
        total = len(items)
        accuracy = correct / total if total > 0 else None
        mid = (lo + hi) / 2 if hi < 1.001 else 0.95
        bucket_data[label] = {
            "count": total,
            "correct": correct,
            "accuracy": round(accuracy, 3) if accuracy is not None else None,
            "mid_confidence": round(mid, 2),
            "calibration_error": round(abs(mid - accuracy), 3) if accuracy is not None else None,
        }

    # Overall accuracy
    overall_correct = sum(1 for r in results if r["is_correct"])
    overall_accuracy = overall_correct / len(results)

    # Mean calibration error (only over non-empty buckets)
    errors = [b["calibration_error"] for b in bucket_data.values() if b["calibration_error"] is not None]
    mean_cal_error = sum(errors) / len(errors) if errors else None

    # Overconfident/underconfident verdict
    high_conf = [r for r in results if r["confidence"] is not None and r["confidence"] >= 0.7]
    if high_conf:
        high_conf_acc = sum(1 for r in high_conf if r["is_correct"]) / len(high_conf)
        avg_high_conf = sum(r["confidence"] for r in high_conf) / len(high_conf)
        if avg_high_conf - high_conf_acc > 0.15:
            verdict = "OVERCONFIDENT"
        elif high_conf_acc - avg_high_conf > 0.15:
            verdict = "UNDERCONFIDENT"
        else:
            verdict = "WELL_CALIBRATED"
    else:
        high_conf_acc = None
        verdict = "INSUFFICIENT_DATA"

    print(f"\n  Overall accuracy: {overall_correct}/30 = {overall_accuracy*100:.1f}%")
    print(f"\n  Confidence Calibration Table:")
    print(f"  {'Bucket':10s} {'Count':6s} {'Accuracy':10s} {'Mid Conf':10s} {'Cal Error':10s}")
    print(f"  {'-'*50}")
    for label, data in bucket_data.items():
        acc_str = f"{data['accuracy']:.2f}" if data['accuracy'] is not None else "N/A"
        err_str = f"{data['calibration_error']:.3f}" if data['calibration_error'] is not None else "N/A"
        bar = ""
        if data['accuracy'] is not None:
            bar_len = int(data['accuracy'] * 20)
            bar = "[" + "#" * bar_len + "." * (20 - bar_len) + "]"
        print(f"  {label:10s} {data['count']:6d} {acc_str:10s} {data['mid_confidence']:10.2f} {err_str:10s} {bar}")

    print(f"\n  Mean calibration error: {mean_cal_error:.3f}" if mean_cal_error else "\n  Mean calibration error: N/A")
    print(f"  Verdict: {verdict}")
    if high_conf_acc is not None:
        print(f"  (avg high-confidence={avg_high_conf:.2f}, high-conf accuracy={high_conf_acc:.2f})")

    return {
        "questions": results,
        "bucket_analysis": bucket_data,
        "summary": {
            "overall_accuracy": round(overall_accuracy, 3),
            "overall_correct": overall_correct,
            "mean_calibration_error": round(mean_cal_error, 3) if mean_cal_error else None,
            "calibration_verdict": verdict,
            "high_confidence_avg": round(avg_high_conf, 3) if high_conf else None,
            "high_confidence_accuracy": round(high_conf_acc, 3) if high_conf_acc else None,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTERESTING EXAMPLES EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_interesting_examples(decomp, hall, consistency, calibration):
    examples = {}

    # Instruction decomposition: biggest gap
    gaps = []
    for f, d in zip(decomp["flat"], decomp["decomposed"]):
        gaps.append({
            "query": f["query"][:80],
            "flat_score": f["total"],
            "decomposed_score": d["total"],
            "gap": d["total"] - f["total"],
        })
    gaps.sort(key=lambda x: abs(x["gap"]), reverse=True)
    examples["instruction_decomposition"] = {
        "biggest_improvement": next((g for g in gaps if g["gap"] > 0), gaps[0]),
        "biggest_regression": next((g for g in gaps if g["gap"] < 0), None),
        "example_flat_response": decomp["flat"][0]["response"],
        "example_decomposed_response": decomp["decomposed"][0]["response"],
    }

    # Hallucination: most interesting (hallucinated vs didn't)
    hallucinated_base = [r for r in hall["baseline"] if r["hallucinated"]]
    not_hallucinated = [r for r in hall["baseline"] if not r["hallucinated"]]
    if hallucinated_base:
        ex = hallucinated_base[0]
        examples["hallucination"] = {
            "example_hallucinated": {
                "query": ex["query"],
                "response_snippet": (ex["response"] or "")[:200],
                "category": ex["category"],
            }
        }
    if not_hallucinated:
        ex = not_hallucinated[0]
        examples["hallucination"]["example_uncertain"] = {
            "query": ex["query"],
            "response_snippet": (ex["response"] or "")[:200],
            "category": ex["category"],
        }
    # Uncertainty instruction helped:
    helped = [i for i, (b, u) in enumerate(zip(hall["baseline"], hall["with_uncertainty_instruction"]))
              if b["hallucinated"] and not u["hallucinated"]]
    if helped:
        idx = helped[0]
        examples["hallucination"]["instruction_helped_example"] = {
            "query": hall["baseline"][idx]["query"],
            "baseline_response": (hall["baseline"][idx]["response"] or "")[:200],
            "with_instruction_response": (hall["with_uncertainty_instruction"][idx]["response"] or "")[:200],
        }

    # Consistency: most inconsistent
    worst = min(consistency["queries"], key=lambda r: r["consistency_score"])
    examples["consistency"] = {
        "most_inconsistent": {
            "id": worst["id"],
            "query": worst["query"][:80],
            "consistency_score": worst["consistency_score"],
            "sample_runs": worst["runs"][:3],
        },
        "perfect_example": next(
            ({"id": r["id"], "query": r["query"][:80], "response": r["runs"][0]}
             for r in consistency["queries"] if r["perfect_consistency"]),
            None
        ),
    }

    # Calibration: overconfident wrong + underconfident correct
    wrong_high_conf = [r for r in calibration["questions"]
                       if not r["is_correct"] and r["confidence"] is not None and r["confidence"] >= 0.8]
    right_low_conf = [r for r in calibration["questions"]
                      if r["is_correct"] and r["confidence"] is not None and r["confidence"] <= 0.4]
    examples["calibration"] = {}
    if wrong_high_conf:
        ex = max(wrong_high_conf, key=lambda r: r["confidence"])
        examples["calibration"]["overconfident_wrong"] = {
            "question": ex["question"],
            "model_answer": ex["model_answer"],
            "correct_answer": ex["correct_answer"],
            "confidence": ex["confidence"],
        }
    if right_low_conf:
        ex = min(right_low_conf, key=lambda r: r["confidence"])
        examples["calibration"]["underconfident_correct"] = {
            "question": ex["question"],
            "model_answer": ex["model_answer"],
            "correct_answer": ex["correct_answer"],
            "confidence": ex["confidence"],
        }

    return examples


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def print_final_summary(decomp, hall, consistency, calibration):
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    ds = decomp["summary"]
    print(f"\n[1] INSTRUCTION DECOMPOSITION")
    print(f"    Flat compliance:       {ds['flat_compliance_rate']*100:.1f}%  ({ds['flat_avg_score']:.2f}/5 avg)")
    print(f"    Decomposed compliance: {ds['decomposed_compliance_rate']*100:.1f}%  ({ds['decomposed_avg_score']:.2f}/5 avg)")
    print(f"    Improvement:           {ds['improvement_percent']:+.1f}%  ({ds['improvement_points']:+.2f} points)")

    hs = hall["summary"]
    print(f"\n[2] HALLUCINATION RATE")
    print(f"    Baseline:              {hs['baseline_hallucination_rate']*100:.1f}%  ({hs['baseline_hallucination_count']}/25 hallucinated)")
    print(f"    With uncertainty inst: {hs['uncertainty_hallucination_rate']*100:.1f}%  ({hs['uncertainty_hallucination_count']}/25 hallucinated)")
    print(f"    Reduction:             {hs['reduction_in_hallucination_rate']*100:+.1f}%")

    cs = consistency["summary"]
    print(f"\n[3] CONSISTENCY")
    print(f"    Mean consistency:      {cs['mean_consistency_score']:.3f}")
    print(f"    Perfectly consistent:  {cs['pct_perfect_consistency']:.1f}%")
    print(f"    Key-consistent:        {cs['pct_key_consistent']:.1f}%")

    cals = calibration["summary"]
    print(f"\n[4] CONFIDENCE CALIBRATION")
    print(f"    Overall accuracy:      {cals['overall_accuracy']*100:.1f}%  ({cals['overall_correct']}/30 correct)")
    print(f"    Mean calibration err:  {cals['mean_calibration_error']}")
    print(f"    Verdict:               {cals['calibration_verdict']}")
    if cals['high_confidence_avg']:
        print(f"    High-conf avg:         {cals['high_confidence_avg']:.2f}  accuracy: {cals['high_confidence_accuracy']:.2f}")


def main():
    print(f"Prompt Engineering Experiment — taalas-llama3.1-8b")
    print(f"Started: {datetime.now().isoformat()}")

    # Run all sections
    decomp_results = run_instruction_decomposition()
    hall_results = run_hallucination_test()
    consistency_results = run_consistency_test()
    calibration_results = run_confidence_calibration()

    # Extract interesting examples
    interesting = extract_interesting_examples(
        decomp_results, hall_results, consistency_results, calibration_results
    )

    # Print final summary
    print_final_summary(decomp_results, hall_results, consistency_results, calibration_results)

    # Save results
    output = {
        "metadata": {
            "model": MODEL,
            "api_url": API_URL,
            "experiment_date": datetime.now().isoformat(),
        },
        "instruction_decomposition": decomp_results,
        "hallucination_rate": hall_results,
        "consistency": consistency_results,
        "confidence_calibration": calibration_results,
        "interesting_examples": interesting,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
