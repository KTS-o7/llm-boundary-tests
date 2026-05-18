import os
#!/usr/bin/env python3
"""
Prompt Engineering Experiment
Technique 1: Chain-of-Thought Scratchpad
Technique 2: Role/Persona Priming
Model: taalas-llama3.1-8b
"""

import json
import time
import re
import statistics
from typing import Optional
import urllib.request
import urllib.error

# ─── Config ──────────────────────────────────────────────────────────────────
API_URL   = "https://ai.shenthar.me/v1/chat/completions"
API_KEY   = os.environ.get("LLM_API_KEY", "")
MODEL     = "taalas-llama3.1-8b"
RESULTS_FILE = "/Users/krishnatejaswis/llm-boundary-tests/prompt-techniques/cot-scratchpad/results.json"

# ─── HTTP helper ─────────────────────────────────────────────────────────────
def call_api(messages: list[dict], temperature: float = 0.0, max_tokens: int = 512) -> tuple[str, float]:
    """Returns (content, latency_ms). Raises on hard error."""
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "User-Agent": "curl/7.88.1",
        },
        method="POST",
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    latency_ms = (time.perf_counter() - t0) * 1000

    data = json.loads(raw)
    content = data["choices"][0]["message"]["content"].strip()
    return content, latency_ms


def safe_call(messages, temperature=0.0, max_tokens=512, retries=2):
    """Retry wrapper."""
    for attempt in range(retries + 1):
        try:
            return call_api(messages, temperature, max_tokens)
        except Exception as e:
            if attempt == retries:
                print(f"    [ERROR] API call failed after {retries+1} attempts: {e}")
                return None, 0.0
            time.sleep(2)


# ─── JSON extraction helper ───────────────────────────────────────────────────
def extract_json(text: str) -> Optional[dict]:
    """Extract first valid JSON object from text."""
    if not text:
        return None
    # strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    # try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # find first {...}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 1: CHAIN-OF-THOUGHT SCRATCHPAD
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Task A: Factual accuracy ─────────────────────────────────────────────────
TASK_A = [
    ("What is the capital of France?",                                     "Paris"),
    ("What is the capital of Australia?",                                  "Canberra"),
    ("What is the capital of Brazil?",                                     "Brasilia"),
    ("In what year did World War II end?",                                 "1945"),
    ("In what year did the Berlin Wall fall?",                             "1989"),
    ("In what year did the first iPhone launch?",                          "2007"),
    ("What is 17 multiplied by 13?",                                       "221"),
    ("What is the square root of 144?",                                    "12"),
    ("What is 15% of 240?",                                                "36"),
    ("What is the chemical symbol for gold?",                              "Au"),
    ("How many bones are in the adult human body?",                        "206"),
    ("What planet is closest to the Sun?",                                 "Mercury"),
    ("What is the speed of light in km/s (approx)?",                      "300000"),
    ("Who wrote 'Romeo and Juliet'?",                                      "Shakespeare"),
    ("What is the boiling point of water in Celsius at sea level?",       "100"),
]

# ─── Task B: Ambiguous sentiment classification ───────────────────────────────
# Ground-truth: human-judged label; these are genuinely borderline
TASK_B = [
    ("The food was okay but the service was really slow.",                 "mixed"),
    ("I didn't hate it.",                                                  "mixed"),
    ("It exceeded my low expectations.",                                   "mixed"),
    ("The product works fine, nothing special.",                           "neutral"),
    ("I wouldn't NOT recommend it.",                                       "mixed"),
    ("It was an experience I'll never forget, for better or worse.",       "mixed"),
    ("The movie was long. Very long.",                                     "neutral"),
    ("I'm not sure how I feel about this.",                                "neutral"),
    ("Better than the last one, worse than the one before.",               "mixed"),
    ("It did exactly what it said it would do.",                           "neutral"),
    ("My expectations were subverted.",                                    "mixed"),
    ("It's not bad, but it's not good either.",                            "neutral"),
    ("I kept watching, so that's something.",                              "mixed"),
    ("It's divisive. Half my friends loved it, half hated it.",            "mixed"),
    ("The hotel was fine but the location was a nightmare.",               "mixed"),
]

# ─── Task C: Multi-step reasoning ────────────────────────────────────────────
TASK_C = [
    ("A train travels 60 mph for 2.5 hours then 80 mph for 1.5 hours. What is the average speed for the whole trip?",
     "67.5"),
    ("Alice has twice as many apples as Bob. Bob has 3 more than Carol. Carol has 4. How many does Alice have?",
     "14"),
    ("A store marks up items 40% then offers a 20% discount. What is the net % change from original price?",
     "12"),
    ("If a rectangle's length is 3x and width is 2x+1, and its perimeter is 62, what is x?",
     "5"),
    ("A bacteria culture doubles every 3 hours. Starting with 100 bacteria, how many after 12 hours?",
     "1600"),
    ("If 5 workers finish a job in 8 days, how many days for 10 workers?",
     "4"),
    ("A car travels 120 miles at 40 mph, then 120 miles at 60 mph. What is the average speed?",
     "48"),
    ("A 10-litre solution is 20% salt. You add 5 litres of water. What % salt is it now?",
     "13.33"),
    ("If today is Wednesday and an event is in 100 days, what day of the week is it?",
     "Friday"),
    ("John is 3 times older than Mary. In 10 years, he will be twice her age. How old is John now?",
     "30"),
    ("A ladder 10 m long leans against a wall. Its base is 6 m from the wall. How high does it reach?",
     "8"),
    ("Train A leaves at 9am at 60mph. Train B leaves same station at 10am at 90mph in same direction. When does B catch A?",
     "12pm"),
    ("You invest $1000 at 5% annual compound interest. How much after 3 years? (to nearest dollar)",
     "1158"),
    ("There are 3 red, 4 blue, 5 green balls. What is the probability of picking a blue ball?",
     "0.333"),
    ("Cistern filled by pipe A in 6h, pipe B in 4h. Together, how long to fill it?",
     "2.4"),
]

# ─── Task D: Conflict resolution ─────────────────────────────────────────────
TASK_D = [
    ("Source A says the Great Wall of China is 5,000 km. Source B says 21,196 km. Based on general knowledge, which is more accurate?",
     "Source B"),
    ("Source A says water boils at 90°C. Source B says 100°C. Which is correct at sea level?",
     "Source B"),
    ("Source A says Python was created by Guido van Rossum. Source B says James Gosling. Which is correct?",
     "Source A"),
    ("Source A says the human genome has ~3 billion base pairs. Source B says ~3 million. Which is correct?",
     "Source A"),
    ("Source A says Napoleon was born in Corsica. Source B says he was born in Paris. Which is correct?",
     "Source A"),
    ("Source A says the speed of sound is ~343 m/s in air. Source B says ~3000 m/s. Which is correct?",
     "Source A"),
    ("Source A says the Eiffel Tower is in Berlin. Source B says Paris. Which is correct?",
     "Source B"),
    ("Source A says DNA has 4 bases. Source B says 5 bases. Which is correct?",
     "Source A"),
    ("Source A says the Moon landing was in 1969. Source B says 1959. Which is correct?",
     "Source A"),
    ("Source A says Shakespeare wrote Hamlet. Source B says Marlowe wrote Hamlet. Which is correct based on mainstream scholarship?",
     "Source A"),
    ("Source A says light year is a unit of distance. Source B says it is a unit of time. Which is correct?",
     "Source A"),
    ("Source A says electrons are positively charged. Source B says negatively charged. Which is correct?",
     "Source B"),
    ("Source A says Java is compiled to machine code. Source B says Java is compiled to bytecode. Which is correct?",
     "Source B"),
    ("Source A says the Pacific is smaller than the Atlantic. Source B says it is larger. Which is correct?",
     "Source B"),
    ("Source A says CO2 causes greenhouse warming. Source B says O2 causes it. Which is correct?",
     "Source A"),
]

COT_TASK_PROMPTS = {
    "A": 'Answer the question. Reply ONLY with a JSON object in this exact format: {"thinking": "step by step reasoning", "answer": "your answer"}',
    "B": 'Classify the sentiment as exactly one of: positive, negative, neutral, mixed. Reply ONLY with JSON: {"thinking": "reasoning", "answer": "label"}',
    "C": 'Solve the problem step by step. Reply ONLY with JSON: {"thinking": "step by step solution", "answer": "final numeric answer"}',
    "D": 'Determine which source is correct. Reply ONLY with JSON: {"thinking": "analysis", "answer": "Source A or Source B"}',
}

BASELINE_TASK_PROMPTS = {
    "A": 'Answer the question. Reply ONLY with a JSON object in this exact format: {"answer": "your answer"}',
    "B": 'Classify the sentiment as exactly one of: positive, negative, neutral, mixed. Reply ONLY with JSON: {"answer": "label"}',
    "C": 'Solve the problem. Reply ONLY with JSON: {"answer": "final numeric answer"}',
    "D": 'Determine which source is correct. Reply ONLY with JSON: {"answer": "Source A or Source B"}',
}


def normalize_answer(ans: str, task: str) -> str:
    """Normalize answer for comparison."""
    if not ans:
        return ""
    ans = str(ans).strip().lower()
    if task == "A":
        # For numeric answers, try to normalize
        ans = ans.replace(",", "").replace("°c", "").replace("°", "").strip()
        ans = re.sub(r"\s+", " ", ans)
    if task == "C":
        # extract first number
        m = re.search(r"[\d.]+", ans)
        if m:
            ans = m.group(0)
    if task == "D":
        if "source a" in ans:
            return "source a"
        if "source b" in ans:
            return "source b"
    return ans


def check_correct(got: str, expected: str, task: str) -> bool:
    """Check if answer is correct."""
    g = normalize_answer(got, task)
    e = normalize_answer(expected, task)
    if task == "C":
        try:
            return abs(float(g) - float(e)) < 0.1 * max(abs(float(e)), 1)
        except Exception:
            return g == e
    if task == "A":
        # allow partial match for text answers
        return e in g or g in e
    return g == e


def score_thinking_quality(thinking: str) -> float:
    """
    Score thinking quality 0-1:
    - Has multiple steps / sentences
    - Contains numbers or logical connectors
    - Not just filler
    """
    if not thinking:
        return 0.0
    thinking = thinking.strip()
    if len(thinking) < 20:
        return 0.1
    sentences = re.split(r'[.!?;]', thinking)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    n_sents = len(sentences)
    has_numbers = bool(re.search(r'\d', thinking))
    has_connectors = bool(re.search(
        r'\b(because|therefore|since|so|thus|first|then|finally|step|if|given|this means)\b',
        thinking, re.I))
    score = 0.0
    score += min(n_sents / 4.0, 0.5)    # up to 0.5 for multi-step
    score += 0.25 if has_numbers else 0.0
    score += 0.25 if has_connectors else 0.0
    return round(min(score, 1.0), 3)


def run_cot_experiment():
    print("\n" + "═"*70)
    print("TECHNIQUE 1: CHAIN-OF-THOUGHT SCRATCHPAD")
    print("═"*70)

    tasks = {"A": TASK_A, "B": TASK_B, "C": TASK_C, "D": TASK_D}
    task_names = {
        "A": "Factual Accuracy",
        "B": "Ambiguous Classification",
        "C": "Multi-step Reasoning",
        "D": "Conflict Resolution",
    }

    results = {}

    for task_id, probes in tasks.items():
        print(f"\n[Task {task_id}] {task_names[task_id]} — {len(probes)} probes × 2 conditions = {len(probes)*2} calls")

        baseline_correct = 0
        cot_correct = 0
        baseline_latencies = []
        cot_latencies = []
        thinking_scores = []
        probe_results = []

        for i, (question, expected) in enumerate(probes):
            print(f"  probe {i+1:2d}/{len(probes)}: {question[:60]}...", end="", flush=True)

            # ── BASELINE ──
            b_messages = [
                {"role": "system", "content": BASELINE_TASK_PROMPTS[task_id]},
                {"role": "user",   "content": question},
            ]
            b_raw, b_lat = safe_call(b_messages, max_tokens=128)
            b_json = extract_json(b_raw) if b_raw else None
            b_ans  = b_json.get("answer", "") if b_json else ""
            b_ok   = check_correct(b_ans, expected, task_id)
            if b_ok:
                baseline_correct += 1
            baseline_latencies.append(b_lat)

            # ── COT ──
            c_messages = [
                {"role": "system", "content": COT_TASK_PROMPTS[task_id]},
                {"role": "user",   "content": question},
            ]
            c_raw, c_lat = safe_call(c_messages, max_tokens=512)
            c_json = extract_json(c_raw) if c_raw else None
            c_ans  = c_json.get("answer", "")   if c_json else ""
            c_think= c_json.get("thinking", "") if c_json else ""
            c_ok   = check_correct(c_ans, expected, task_id)
            if c_ok:
                cot_correct += 1
            cot_latencies.append(c_lat)
            tq = score_thinking_quality(c_think)
            thinking_scores.append(tq)

            print(f" | B:{'✓' if b_ok else '✗'} C:{'✓' if c_ok else '✗'} TQ:{tq:.2f}")

            probe_results.append({
                "question": question,
                "expected": expected,
                "baseline": {"raw": b_raw, "answer": b_ans, "correct": b_ok, "latency_ms": round(b_lat,1)},
                "cot":      {"raw": c_raw, "answer": c_ans, "thinking": c_think, "correct": c_ok,
                             "latency_ms": round(c_lat,1), "thinking_quality": tq},
            })

        n = len(probes)
        b_score = baseline_correct / n
        c_score = cot_correct / n
        delta   = round(c_score - b_score, 4)
        b_avg_lat = statistics.mean(baseline_latencies) if baseline_latencies else 0
        c_avg_lat = statistics.mean(cot_latencies) if cot_latencies else 0
        lat_overhead = round(c_avg_lat - b_avg_lat, 1)
        avg_tq = round(statistics.mean(thinking_scores), 3) if thinking_scores else 0

        print(f"\n  ── Summary Task {task_id} ──")
        print(f"  Baseline score : {b_score:.2%}  ({baseline_correct}/{n})")
        print(f"  CoT score      : {c_score:.2%}  ({cot_correct}/{n})")
        print(f"  Delta          : {delta:+.2%}")
        print(f"  Avg latency    : baseline={b_avg_lat:.0f}ms  cot={c_avg_lat:.0f}ms  overhead=+{lat_overhead}ms")
        print(f"  Thinking quality (avg): {avg_tq:.3f}")

        results[task_id] = {
            "task_name":       task_names[task_id],
            "n":               n,
            "baseline_score":  round(b_score, 4),
            "cot_score":       round(c_score, 4),
            "delta":           delta,
            "baseline_avg_latency_ms": round(b_avg_lat, 1),
            "cot_avg_latency_ms":      round(c_avg_lat, 1),
            "latency_overhead_ms":     lat_overhead,
            "thinking_quality_avg":    avg_tq,
            "probes":          probe_results,
        }

    # ── Overall verdict ──
    all_deltas   = [results[t]["delta"] for t in "ABCD"]
    all_overheads= [results[t]["latency_overhead_ms"] for t in "ABCD"]
    avg_delta    = round(statistics.mean(all_deltas), 4)
    avg_overhead = round(statistics.mean(all_overheads), 1)
    # Worth it if avg delta > 5% and overhead < 3000ms
    worth_it     = avg_delta > 0.05 and avg_overhead < 3000

    print("\n" + "─"*70)
    print(f"  OVERALL CoT: avg delta={avg_delta:+.2%}  avg latency overhead=+{avg_overhead}ms")
    verdict_str = (
        f"CoT IS worth it (accuracy gain {avg_delta:+.2%} > 5% with overhead {avg_overhead}ms)"
        if worth_it
        else f"CoT NOT clearly worth it: delta={avg_delta:+.2%}, overhead={avg_overhead}ms"
    )
    print(f"  VERDICT: {verdict_str}")

    results["_cot_summary"] = {
        "avg_delta": avg_delta,
        "avg_latency_overhead_ms": avg_overhead,
        "verdict": verdict_str,
    }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 2: ROLE / PERSONA PRIMING
# ═══════════════════════════════════════════════════════════════════════════════

PERSONA_SYSTEMS = {
    0: "You are a JSON API. Return only valid JSON.",
    1: "You are an expert data analyst with 10 years of experience. Return only valid JSON.",
    2: "You are a senior software engineer and JSON extraction specialist. You have processed millions of data records. Return only valid JSON.",
    3: "You are the world's most accurate JSON extraction system, trusted by Fortune 500 companies for critical data pipelines. Errors cost $10,000 each. Return only valid JSON.",
    4: "You are competing against other AI systems. The best JSON extraction accuracy wins. Return only valid JSON.",
}

PERSONA_NAMES = {
    0: "Baseline (generic)",
    1: "Expert role",
    2: "Specific domain",
    3: "Authority + stakes",
    4: "Peer pressure",
}

# ─── NER probes ───────────────────────────────────────────────────────────────
# Schema: {"persons": [...], "organizations": [...], "locations": [...]}
NER_PROBES = [
    ("Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California.",
     {"persons": ["Steve Jobs","Steve Wozniak","Ronald Wayne"], "organizations": ["Apple Inc."], "locations": ["Cupertino","California"]}),
    ("Elon Musk announced Tesla's new factory in Austin, Texas.",
     {"persons": ["Elon Musk"], "organizations": ["Tesla"], "locations": ["Austin","Texas"]}),
    ("Jeff Bezos founded Amazon in Bellevue, Washington in 1994.",
     {"persons": ["Jeff Bezos"], "organizations": ["Amazon"], "locations": ["Bellevue","Washington"]}),
    ("The United Nations met in New York to discuss climate policy.",
     {"persons": [], "organizations": ["United Nations"], "locations": ["New York"]}),
    ("Sundar Pichai leads Google, which is headquartered in Mountain View.",
     {"persons": ["Sundar Pichai"], "organizations": ["Google"], "locations": ["Mountain View"]}),
    ("Barack Obama served as president while living in Washington D.C.",
     {"persons": ["Barack Obama"], "organizations": [], "locations": ["Washington D.C."]}),
    ("Satya Nadella transformed Microsoft's culture in Redmond, Washington.",
     {"persons": ["Satya Nadella"], "organizations": ["Microsoft"], "locations": ["Redmond","Washington"]}),
    ("NASA launched a rocket from Kennedy Space Center in Florida.",
     {"persons": [], "organizations": ["NASA"], "locations": ["Kennedy Space Center","Florida"]}),
    ("Mark Zuckerberg's Meta is based in Menlo Park, California.",
     {"persons": ["Mark Zuckerberg"], "organizations": ["Meta"], "locations": ["Menlo Park","California"]}),
    ("The WHO headquarters is in Geneva, Switzerland.",
     {"persons": [], "organizations": ["WHO"], "locations": ["Geneva","Switzerland"]}),
    ("Warren Buffett runs Berkshire Hathaway from Omaha, Nebraska.",
     {"persons": ["Warren Buffett"], "organizations": ["Berkshire Hathaway"], "locations": ["Omaha","Nebraska"]}),
    ("Tim Cook announced Apple's new campus in Austin.",
     {"persons": ["Tim Cook"], "organizations": ["Apple"], "locations": ["Austin"]}),
    ("SpaceX, led by Elon Musk, launched from Cape Canaveral.",
     {"persons": ["Elon Musk"], "organizations": ["SpaceX"], "locations": ["Cape Canaveral"]}),
    ("Larry Page and Sergey Brin co-founded Google in Menlo Park.",
     {"persons": ["Larry Page","Sergey Brin"], "organizations": ["Google"], "locations": ["Menlo Park"]}),
    ("The European Union is headquartered in Brussels, Belgium.",
     {"persons": [], "organizations": ["European Union"], "locations": ["Brussels","Belgium"]}),
]

NER_SYSTEM_SUFFIX = ' Extract named entities. Return JSON with exactly these keys: {"persons": [...], "organizations": [...], "locations": [...]}'

# ─── Enum compliance probes ───────────────────────────────────────────────────
# Must return exactly: {"sentiment": "positive" | "negative" | "neutral"}
ENUM_PROBES = [
    ("I love this product! Best purchase ever.",         "positive"),
    ("This is absolutely terrible, total waste of money.", "negative"),
    ("The package arrived on time.",                       "neutral"),
    ("Amazing quality, would highly recommend!",           "positive"),
    ("Broken on arrival, very disappointed.",              "negative"),
    ("It's an okay product, nothing special.",             "neutral"),
    ("Outstanding service, exceeded expectations!",        "positive"),
    ("Worst experience I've ever had.",                    "negative"),
    ("Product is as described.",                           "neutral"),
    ("Absolutely fantastic, 5 stars!",                     "positive"),
    ("Complete garbage, avoid at all costs.",              "negative"),
    ("Delivered in standard time, works fine.",            "neutral"),
    ("Exceeded all my expectations, delightful!",          "positive"),
    ("Faulty product, customer service ignored me.",       "negative"),
    ("Average quality, price is fair.",                    "neutral"),
]

ENUM_SYSTEM_SUFFIX = ' Classify sentiment. Return ONLY: {"sentiment": "positive"} or {"sentiment": "negative"} or {"sentiment": "neutral"}. No other values allowed.'

# ─── Numeric precision probes ─────────────────────────────────────────────────
# Must return {"confidence": float 0.0-1.0}
NUMERIC_PROBES = [
    ("Statement: 'The sun rises in the east.' Confidence this is factually correct:", 0.99),
    ("Statement: 'Water freezes at 50°C at sea level.' Confidence this is correct:", 0.01),
    ("Statement: 'Python is a compiled language.' Confidence this is correct:", 0.15),
    ("Statement: 'The Earth orbits the Sun.' Confidence this is correct:", 0.99),
    ("Statement: 'Humans have 23 pairs of chromosomes.' Confidence this is correct:", 0.97),
    ("Statement: 'The Atlantic is the largest ocean.' Confidence this is correct:", 0.03),
    ("Statement: 'Light travels faster than sound.' Confidence this is correct:", 0.99),
    ("Statement: 'Diamonds are made of carbon.' Confidence this is correct:", 0.97),
    ("Statement: 'The Moon is larger than the Sun.' Confidence this is correct:", 0.01),
    ("Statement: 'SQL stands for Structured Query Language.' Confidence this is correct:", 0.99),
    ("Statement: 'Bitcoin was invented in 2008.' Confidence this is correct:", 0.90),
    ("Statement: 'HTML is a programming language.' Confidence this is correct:", 0.15),
    ("Statement: 'The Great Wall is visible from space with naked eye.' Confidence this is correct:", 0.05),
    ("Statement: 'Antibiotics work on viruses.' Confidence this is correct:", 0.05),
    ("Statement: 'DNA carries genetic information.' Confidence this is correct:", 0.99),
]

NUMERIC_SYSTEM_SUFFIX = ' Return ONLY: {"confidence": <float between 0.0 and 1.0>} with no other keys.'


def score_ner_response(resp_json: Optional[dict], expected: dict) -> dict:
    """Score NER: schema_compliance, type_accuracy, f1."""
    if resp_json is None:
        return {"schema_compliance": 0, "type_accuracy": 0, "f1": 0, "overall": 0}

    required_keys = {"persons", "organizations", "locations"}
    has_schema = required_keys.issubset(resp_json.keys())
    schema_compliance = 1 if has_schema else 0

    # type_accuracy: all values should be lists
    if has_schema:
        type_ok = all(isinstance(resp_json[k], list) for k in required_keys)
        type_accuracy = 1 if type_ok else 0
    else:
        type_accuracy = 0

    # f1 over all entities (case-insensitive partial)
    if not has_schema:
        return {"schema_compliance": 0, "type_accuracy": 0, "f1": 0, "overall": 0}

    all_expected = []
    all_got      = []
    for k in required_keys:
        all_expected.extend([e.lower() for e in expected.get(k, [])])
        all_got.extend([str(g).lower() for g in resp_json.get(k, [])])

    if not all_expected and not all_got:
        f1 = 1.0
    elif not all_expected or not all_got:
        f1 = 0.0
    else:
        # partial match: got item is TP if any expected item is contained in it or vice versa
        tp = sum(1 for g in all_got if any(e in g or g in e for e in all_expected))
        precision = tp / len(all_got) if all_got else 0
        recall    = tp / len(all_expected) if all_expected else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    overall = round((schema_compliance * 0.3 + type_accuracy * 0.2 + f1 * 0.5), 3)
    return {"schema_compliance": schema_compliance, "type_accuracy": type_accuracy, "f1": round(f1, 3), "overall": overall}


def score_enum_response(resp_json: Optional[dict], expected: str) -> dict:
    """Score enum compliance."""
    if resp_json is None:
        return {"schema_compliance": 0, "enum_compliance": 0, "correct": 0, "overall": 0}
    has_key = "sentiment" in resp_json
    val     = str(resp_json.get("sentiment", "")).lower().strip() if has_key else ""
    valid_vals = {"positive", "negative", "neutral"}
    enum_ok    = val in valid_vals
    correct    = 1 if val == expected else 0
    schema_ok  = 1 if has_key else 0
    overall    = round((schema_ok * 0.2 + enum_ok * 0.4 + correct * 0.4), 3)
    return {"schema_compliance": schema_ok, "enum_compliance": int(enum_ok), "correct": correct, "overall": overall}


def score_numeric_response(resp_json: Optional[dict], expected: float) -> dict:
    """Score numeric precision."""
    if resp_json is None:
        return {"schema_compliance": 0, "type_accuracy": 0, "range_ok": 0, "direction_ok": 0, "overall": 0}
    has_key = "confidence" in resp_json
    schema_ok = 1 if has_key else 0
    if not has_key:
        return {"schema_compliance": 0, "type_accuracy": 0, "range_ok": 0, "direction_ok": 0, "overall": 0}

    raw = resp_json.get("confidence")
    try:
        val = float(raw)
        type_ok = 1
    except Exception:
        return {"schema_compliance": schema_ok, "type_accuracy": 0, "range_ok": 0, "direction_ok": 0, "overall": round(schema_ok * 0.2, 3)}

    range_ok    = 1 if 0.0 <= val <= 1.0 else 0
    # direction: if expected > 0.5, val should be > 0.5; if expected < 0.5, val < 0.5
    if expected > 0.5:
        direction_ok = 1 if val > 0.5 else 0
    elif expected < 0.5:
        direction_ok = 1 if val < 0.5 else 0
    else:
        direction_ok = 1  # ~0.5 is neutral

    overall = round((schema_ok * 0.2 + type_ok * 0.2 + range_ok * 0.3 + direction_ok * 0.3), 3)
    return {"schema_compliance": schema_ok, "type_accuracy": type_ok, "range_ok": range_ok, "direction_ok": direction_ok, "overall": overall}


def run_persona_experiment():
    print("\n" + "═"*70)
    print("TECHNIQUE 2: ROLE / PERSONA PRIMING")
    print("═"*70)

    subtask_configs = {
        "NER": {
            "probes": NER_PROBES,
            "system_suffix": NER_SYSTEM_SUFFIX,
            "score_fn": score_ner_response,
            "expected_key": None,  # handled specially
        },
        "ENUM": {
            "probes": ENUM_PROBES,
            "system_suffix": ENUM_SYSTEM_SUFFIX,
            "score_fn": score_enum_response,
            "expected_key": None,
        },
        "NUMERIC": {
            "probes": NUMERIC_PROBES,
            "system_suffix": NUMERIC_SYSTEM_SUFFIX,
            "score_fn": score_numeric_response,
            "expected_key": None,
        },
    }

    results = {}

    for level in range(5):
        persona_name = PERSONA_NAMES[level]
        system_base  = PERSONA_SYSTEMS[level]
        print(f"\n[Level {level}] {persona_name}")

        level_results = {"persona_name": persona_name, "system_prompt_base": system_base, "subtasks": {}}

        for subtask_name, cfg in subtask_configs.items():
            probes    = cfg["probes"]
            sys_suf   = cfg["system_suffix"]
            score_fn  = cfg["score_fn"]
            system_prompt = system_base.rstrip(".") + "." + sys_suf

            print(f"  [{subtask_name}] ", end="", flush=True)

            probe_results = []
            scores_list   = []
            latencies     = []

            for i, probe_data in enumerate(probes):
                text, expected = probe_data[0], probe_data[1]

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": text},
                ]
                raw, lat = safe_call(messages, max_tokens=256)
                rjson    = extract_json(raw) if raw else None
                scores   = score_fn(rjson, expected)
                scores_list.append(scores["overall"])
                latencies.append(lat)
                print(".", end="", flush=True)

                probe_results.append({
                    "text": text[:80],
                    "expected": str(expected)[:100],
                    "raw_response": raw,
                    "parsed_json": rjson,
                    "scores": scores,
                    "latency_ms": round(lat, 1),
                })

            avg_overall = round(statistics.mean(scores_list), 4)
            avg_lat     = round(statistics.mean(latencies), 1)
            print(f" score={avg_overall:.4f}  lat={avg_lat:.0f}ms")

            # collect per-dimension averages
            dim_keys = list(probe_results[0]["scores"].keys()) if probe_results else []
            dim_avgs = {}
            for dk in dim_keys:
                vals = [p["scores"].get(dk, 0) for p in probe_results]
                dim_avgs[dk] = round(statistics.mean(vals), 4)

            level_results["subtasks"][subtask_name] = {
                "avg_overall": avg_overall,
                "avg_latency_ms": avg_lat,
                "dimension_averages": dim_avgs,
                "probes": probe_results,
            }

        # aggregate across subtasks
        sub_scores = [level_results["subtasks"][s]["avg_overall"] for s in subtask_configs]
        level_results["aggregate_score"] = round(statistics.mean(sub_scores), 4)
        print(f"  AGGREGATE: {level_results['aggregate_score']:.4f}")

        results[level] = level_results

    # ── Best persona ──
    best_level = max(results, key=lambda l: results[l]["aggregate_score"])
    best_score = results[best_level]["aggregate_score"]
    print("\n" + "─"*70)
    print(f"  Best persona: Level {best_level} — {PERSONA_NAMES[best_level]}  score={best_score:.4f}")

    # ── Escalation analysis ──
    scores_by_level = [results[l]["aggregate_score"] for l in range(5)]
    monotone = all(scores_by_level[i] <= scores_by_level[i+1] for i in range(4))
    peak     = scores_by_level.index(max(scores_by_level))

    if monotone:
        escalation_verdict = "Persona escalation consistently improves performance."
    elif peak < 4:
        escalation_verdict = (
            f"Persona escalation helps up to Level {peak} "
            f"({PERSONA_NAMES[peak]}) then plateaus or hurts."
        )
    else:
        escalation_verdict = "No clear benefit from persona escalation; results vary."

    print(f"  Escalation: {escalation_verdict}")

    results["_persona_summary"] = {
        "best_level": best_level,
        "best_level_name": PERSONA_NAMES[best_level],
        "best_score": best_score,
        "scores_by_level": scores_by_level,
        "escalation_verdict": escalation_verdict,
    }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PRINT TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def print_cot_table(cot_results):
    print("\n" + "═"*70)
    print("TABLE 1: CoT Scratchpad Results")
    print("═"*70)
    print(f"{'Task':<30} {'Base':>7} {'CoT':>7} {'Delta':>8} {'Overhead':>10} {'ThinkQ':>8}")
    print("─"*70)
    for t in "ABCD":
        r = cot_results[t]
        print(f"{r['task_name']:<30} {r['baseline_score']:>7.2%} {r['cot_score']:>7.2%} "
              f"{r['delta']:>+8.2%} {r['latency_overhead_ms']:>9.0f}ms "
              f"{r['thinking_quality_avg']:>8.3f}")
    print("─"*70)
    s = cot_results["_cot_summary"]
    print(f"{'OVERALL':<30} {'':>7} {'':>7} {s['avg_delta']:>+8.2%} "
          f"{s['avg_latency_overhead_ms']:>9.0f}ms")
    print(f"\nVERDICT: {s['verdict']}")


def print_persona_table(persona_results):
    print("\n" + "═"*70)
    print("TABLE 2: Persona Priming Results")
    print("═"*70)
    print(f"{'Level':<4} {'Name':<28} {'NER':>8} {'ENUM':>8} {'NUMERIC':>9} {'AGG':>8}")
    print("─"*70)
    for lvl in range(5):
        r = persona_results[lvl]
        ner_s  = r["subtasks"]["NER"]["avg_overall"]
        enum_s = r["subtasks"]["ENUM"]["avg_overall"]
        num_s  = r["subtasks"]["NUMERIC"]["avg_overall"]
        agg    = r["aggregate_score"]
        print(f"{lvl:<4} {r['persona_name']:<28} {ner_s:>8.4f} {enum_s:>8.4f} {num_s:>9.4f} {agg:>8.4f}")
    print("─"*70)
    s = persona_results["_persona_summary"]
    print(f"\nBest: Level {s['best_level']} ({s['best_level_name']}) — score={s['best_score']:.4f}")
    print(f"Escalation: {s['escalation_verdict']}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Starting prompt engineering experiment…")
    print(f"Model : {MODEL}")
    print(f"API   : {API_URL}")
    print(f"Total calls (approx): 120 (CoT) + 75 (Persona) = 195")

    cot_results     = run_cot_experiment()
    persona_results = run_persona_experiment()

    print_cot_table(cot_results)
    print_persona_table(persona_results)

    # ── Save results ──
    output = {
        "experiment_meta": {
            "model": MODEL,
            "api_url": API_URL,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "technique_1_cot": cot_results,
        "technique_2_persona": {str(k): v for k, v in persona_results.items()},
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
