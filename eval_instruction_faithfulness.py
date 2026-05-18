import os
"""
Instruction Faithfulness Evaluation
Measures how well the model follows explicit formatting, schema, and constraint instructions.
"""

import json
import math
import re
import time
import statistics
from collections import Counter
from typing import Any

import urllib.request
import urllib.error
import ssl

# ── Wilson score 95% CI ───────────────────────────────────────────────────────
def wilson_ci_95(p: float, n: int):
    """Wilson score 95% CI for a proportion p with n observations."""
    if n == 0:
        return (0.0, 1.0)
    z = 1.96
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (round(max(0.0, centre - margin), 3), round(min(1.0, centre + margin), 3))


# ── API config ────────────────────────────────────────────────────────────────
API_URL   = "https://ai.shenthar.me/v1/chat/completions"
API_KEY   = os.environ.get("LLM_API_KEY", "")
MODEL     = "taalas-llama3.1-8b"
RESULTS_F = "/Users/krishnatejaswis/llm-boundary-tests/eval_instruction_faithfulness_results.json"


# ── HTTP helper ───────────────────────────────────────────────────────────────
_SSL_CTX = ssl.create_default_context()

def call_api(system_prompt: str, user_message: str, temperature: float = 0.2) -> str:
    payload = json.dumps({
        "model": MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    }).encode()

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) EvalScript/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def safe_call(system_prompt: str, user_message: str) -> str:
    """Call with retries on transient errors."""
    for attempt in range(3):
        try:
            return call_api(system_prompt, user_message)
        except Exception as exc:
            if attempt == 2:
                return f"__ERROR__: {exc}"
            time.sleep(2 ** attempt)
    return "__ERROR__: max retries"


# ── JSON extraction helper ────────────────────────────────────────────────────
def extract_json(text: str) -> Any:
    """
    Try to parse JSON from a response.
    Strategy:
    1. Extract content of the FIRST code fence block (``` or ```json) and parse it.
    2. If no fence, try parsing the whole text.
    3. Fall back to finding the first {...} or [...] via regex.
    """
    # Strategy 1: extract content of first code fence
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Strategy 2: try parsing stripped full text (after removing all fences)
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: find first {...} or [...] greedily
    for pattern in (r'\{.*\}', r'\[.*\]'):
        m = re.search(pattern, cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass

    # Strategy 4: find outermost JSON object using brace counting
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if not in_string:
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:idx+1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 1 – Schema Strictness
# ─────────────────────────────────────────────────────────────────────────────
def run_schema_strictness() -> list[dict]:
    """
    Give exact JSON schema. Check required fields present, no extra fields,
    correct types.
    """
    probes = [
        {
            "system": (
                'You must respond ONLY with a JSON object that has EXACTLY these fields: '
                '"title" (string), "year" (integer), "rating" (float), "available" (boolean). '
                'No other fields. No explanation.'
            ),
            "user": "Tell me about the movie Inception.",
        },
        {
            "system": (
                'Respond ONLY with a JSON object containing exactly these fields: '
                '"name" (string), "age" (integer), "active" (boolean). '
                'Do not add any other fields. No prose.'
            ),
            "user": "Create a user profile for Alice, age 30, who is active.",
        },
        {
            "system": (
                'Return a JSON object with exactly: "product" (string), "price" (float), '
                '"in_stock" (boolean), "quantity" (integer). Nothing else.'
            ),
            "user": "Describe a laptop product entry.",
        },
        {
            "system": (
                'Output ONLY a JSON object with fields: "country" (string), '
                '"population" (integer), "gdp_usd_billions" (float), "is_g20" (boolean). '
                'No extra fields. No markdown.'
            ),
            "user": "Give me data for Germany.",
        },
        {
            "system": (
                'Respond with a JSON object having exactly these keys: '
                '"event" (string), "date" (string), "attendees" (integer), "virtual" (boolean). '
                'Nothing else.'
            ),
            "user": "Describe a company all-hands meeting.",
        },
        {
            "system": (
                'Return ONLY a JSON object with: "word" (string), "syllables" (integer), '
                '"is_noun" (boolean), "definition" (string). Exactly these 4 fields.'
            ),
            "user": "Analyze the word 'serendipity'.",
        },
        {
            "system": (
                'Output a JSON object with exactly these fields: "city" (string), '
                '"temperature_celsius" (float), "humidity_percent" (integer), "sunny" (boolean). '
                'No additional fields.'
            ),
            "user": "Give weather data for Tokyo.",
        },
        {
            "system": (
                'Respond ONLY with JSON having exactly: "author" (string), "pages" (integer), '
                '"published_year" (integer), "fiction" (boolean). Four fields only.'
            ),
            "user": "Describe a book.",
        },
        {
            "system": (
                'Output a JSON object with exactly: "ticker" (string), "price" (float), '
                '"volume" (integer), "bullish" (boolean). No extra keys.'
            ),
            "user": "Give me stock info for AAPL.",
        },
        {
            "system": (
                'Return ONLY a JSON object with these 4 fields: "recipe" (string), '
                '"calories" (integer), "vegetarian" (boolean), "prep_minutes" (integer). '
                'No other fields.'
            ),
            "user": "Describe a pasta dish.",
        },
    ]

    required_fields_and_types = [
        {"title": str, "year": int, "rating": float, "available": bool},
        {"name": str, "age": int, "active": bool},
        {"product": str, "price": float, "in_stock": bool, "quantity": int},
        {"country": str, "population": int, "gdp_usd_billions": float, "is_g20": bool},
        {"event": str, "date": str, "attendees": int, "virtual": bool},
        {"word": str, "syllables": int, "is_noun": bool, "definition": str},
        {"city": str, "temperature_celsius": float, "humidity_percent": int, "sunny": bool},
        {"author": str, "pages": int, "published_year": int, "fiction": bool},
        {"ticker": str, "price": float, "volume": int, "bullish": bool},
        {"recipe": str, "calories": int, "vegetarian": bool, "prep_minutes": int},
    ]

    results = []
    for i, (probe, schema) in enumerate(zip(probes, required_fields_and_types)):
        response = safe_call(probe["system"], probe["user"])
        obj = extract_json(response)

        constraints_given = [
            "all_required_fields_present",
            "no_extra_fields",
            "correct_types",
            "valid_json",
        ]
        constraints_satisfied = []
        failure_modes = []

        if obj is None or not isinstance(obj, dict):
            failure_modes.append("invalid_json_or_not_object")
        else:
            # valid JSON object
            constraints_satisfied.append("valid_json")

            # required fields
            missing = [f for f in schema if f not in obj]
            if not missing:
                constraints_satisfied.append("all_required_fields_present")
            else:
                failure_modes.append(f"missing_fields:{missing}")

            # no extra fields
            extra = [f for f in obj if f not in schema]
            if not extra:
                constraints_satisfied.append("no_extra_fields")
            else:
                failure_modes.append(f"extra_fields:{extra}")

            # type correctness
            type_errors = []
            for field, expected_type in schema.items():
                if field in obj:
                    val = obj[field]
                    if expected_type is float:
                        if not isinstance(val, (int, float)):
                            type_errors.append(f"{field}:expected_float")
                    elif expected_type is int:
                        if not isinstance(val, int) or isinstance(val, bool):
                            type_errors.append(f"{field}:expected_int")
                    elif expected_type is bool:
                        if not isinstance(val, bool):
                            type_errors.append(f"{field}:expected_bool")
                    elif expected_type is str:
                        if not isinstance(val, str):
                            type_errors.append(f"{field}:expected_str")
            if not type_errors:
                constraints_satisfied.append("correct_types")
            else:
                failure_modes.append(f"type_errors:{type_errors}")

        faithfulness = len(constraints_satisfied) / len(constraints_given)
        results.append({
            "category": "schema_strictness",
            "probe_id": i,
            "instruction_text": probe["system"],
            "response": response[:500],
            "constraints_given": constraints_given,
            "constraints_satisfied": constraints_satisfied,
            "faithfulness_score": faithfulness,
            "failure_mode": failure_modes[0] if failure_modes else "none",
        })
        print(f"  [schema_strictness #{i}] score={faithfulness:.2f} failures={failure_modes}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 2 – Field Count Compliance
# ─────────────────────────────────────────────────────────────────────────────
def run_field_count_compliance() -> list[dict]:
    probes = [
        ("Return a JSON object with EXACTLY 3 fields describing a car.", 3, "Describe a car."),
        ("Return a JSON object with EXACTLY 5 fields about a country.", 5, "Describe France."),
        ("Return a JSON object with EXACTLY 2 fields.", 2, "Describe the sun."),
        ("Return a JSON object with EXACTLY 7 fields about a person.", 7, "Describe a software engineer."),
        ("Return a JSON object with EXACTLY 4 fields about a book.", 4, "Describe a mystery novel."),
        ("Return a JSON object with EXACTLY 6 fields about a recipe.", 6, "Describe a chocolate cake recipe."),
        ("Return a JSON object with EXACTLY 1 field.", 1, "State the capital of Japan."),
        ("Return a JSON object with EXACTLY 8 fields about a city.", 8, "Describe New York City."),
        ("Return a JSON object with EXACTLY 3 fields about a sport.", 3, "Describe basketball."),
        ("Return a JSON object with EXACTLY 5 fields about a company.", 5, "Describe Apple Inc."),
    ]

    results = []
    for i, (system_prompt, expected_count, user_msg) in enumerate(probes):
        response = safe_call(system_prompt, user_msg)
        obj = extract_json(response)

        constraints_given = ["valid_json", f"exactly_{expected_count}_fields"]
        constraints_satisfied = []
        failure_modes = []

        if obj is None or not isinstance(obj, dict):
            failure_modes.append("invalid_json_or_not_object")
        else:
            constraints_satisfied.append("valid_json")
            actual = len(obj)
            if actual == expected_count:
                constraints_satisfied.append(f"exactly_{expected_count}_fields")
            elif actual < expected_count:
                failure_modes.append(f"too_few_fields:{actual}_vs_{expected_count}")
            else:
                failure_modes.append(f"too_many_fields:{actual}_vs_{expected_count}")

        faithfulness = len(constraints_satisfied) / len(constraints_given)
        results.append({
            "category": "field_count_compliance",
            "probe_id": i,
            "instruction_text": system_prompt,
            "response": response[:500],
            "constraints_given": constraints_given,
            "constraints_satisfied": constraints_satisfied,
            "faithfulness_score": faithfulness,
            "failure_mode": failure_modes[0] if failure_modes else "none",
        })
        print(f"  [field_count #{i}] expected={expected_count} score={faithfulness:.2f} failures={failure_modes}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 3 – Value Constraint (Strict Enum)
# ─────────────────────────────────────────────────────────────────────────────
def run_value_constraint() -> list[dict]:
    SENTIMENTS = {"positive", "negative", "neutral"}
    PRIORITIES  = {"low", "medium", "high", "critical"}
    STATUSES    = {"pending", "active", "completed", "cancelled"}

    probes = [
        {
            "system": 'Return JSON with one field "sentiment". The value MUST be exactly one of: positive, negative, neutral. No other values.',
            "user": "This movie was absolutely fantastic!",
            "field": "sentiment", "allowed": SENTIMENTS,
        },
        {
            "system": 'Return JSON with one field "sentiment". The value MUST be exactly one of: positive, negative, neutral.',
            "user": "I hate waiting in long queues.",
            "field": "sentiment", "allowed": SENTIMENTS,
        },
        {
            "system": 'Return JSON with one field "sentiment". The value MUST be exactly one of: positive, negative, neutral.',
            "user": "The weather today is okay.",
            "field": "sentiment", "allowed": SENTIMENTS,
        },
        {
            "system": 'Return JSON with a field "priority". The value MUST be exactly one of: low, medium, high, critical. No variations.',
            "user": "Fix the login button color mismatch.",
            "field": "priority", "allowed": PRIORITIES,
        },
        {
            "system": 'Return JSON with a field "priority". The value MUST be exactly one of: low, medium, high, critical.',
            "user": "The production database is down.",
            "field": "priority", "allowed": PRIORITIES,
        },
        {
            "system": 'Return JSON with a field "status". The value MUST be exactly one of: pending, active, completed, cancelled.',
            "user": "The task was finished yesterday.",
            "field": "status", "allowed": STATUSES,
        },
        {
            "system": 'Return JSON with a field "status". The value MUST be exactly one of: pending, active, completed, cancelled.',
            "user": "The meeting has been called off.",
            "field": "status", "allowed": STATUSES,
        },
        {
            "system": 'Return JSON with a field "sentiment". The value MUST be exactly one of: positive, negative, neutral.',
            "user": "The food was not terrible but not great either.",
            "field": "sentiment", "allowed": SENTIMENTS,
        },
        {
            "system": 'Return JSON with a field "priority". The value MUST be exactly one of: low, medium, high, critical.',
            "user": "Add a dark mode option to the settings page.",
            "field": "priority", "allowed": PRIORITIES,
        },
        {
            "system": 'Return JSON with a field "sentiment". The value MUST be exactly one of: positive, negative, neutral.',
            "user": "It was an average performance from the team.",
            "field": "sentiment", "allowed": SENTIMENTS,
        },
    ]

    results = []
    for i, probe in enumerate(probes):
        response = safe_call(probe["system"], probe["user"])
        obj = extract_json(response)

        constraints_given = ["valid_json", "strict_enum_value"]
        constraints_satisfied = []
        failure_modes = []

        if obj is None or not isinstance(obj, dict):
            failure_modes.append("invalid_json_or_not_object")
        else:
            constraints_satisfied.append("valid_json")
            field = probe["field"]
            allowed = probe["allowed"]
            if field not in obj:
                failure_modes.append(f"missing_field:{field}")
            else:
                val = str(obj[field]).strip().lower()
                if val in allowed:
                    constraints_satisfied.append("strict_enum_value")
                else:
                    failure_modes.append(f"invalid_enum_value:'{obj[field]}'_not_in_{allowed}")

        faithfulness = len(constraints_satisfied) / len(constraints_given)
        results.append({
            "category": "value_constraint",
            "probe_id": i,
            "instruction_text": probe["system"],
            "response": response[:500],
            "constraints_given": constraints_given,
            "constraints_satisfied": constraints_satisfied,
            "faithfulness_score": faithfulness,
            "failure_mode": failure_modes[0] if failure_modes else "none",
        })
        print(f"  [value_constraint #{i}] score={faithfulness:.2f} failures={failure_modes}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 4 – Negation Instructions
# ─────────────────────────────────────────────────────────────────────────────
def run_negation_instructions() -> list[dict]:
    probes = [
        {
            "system": 'Return a JSON object with sentiment analysis. Do NOT include any explanations or prose outside the JSON.',
            "user": "This product is amazing!",
            "forbidden_check": lambda r, obj: bool(re.search(r'[A-Za-z]', re.sub(r'\{.*\}', '', r, flags=re.DOTALL).strip())),
            "forbidden_label": "prose_outside_json",
        },
        {
            "system": 'Return JSON with fields "topic" and "summary". Do NOT add a confidence score.',
            "user": "Summarize the topic of climate change.",
            "forbidden_check": lambda r, obj: obj is not None and isinstance(obj, dict) and "confidence" in {k.lower() for k in obj.keys()},
            "forbidden_label": "confidence_score_present",
        },
        {
            "system": 'Return a flat JSON object. Do NOT use nested objects or arrays.',
            "user": "Describe a person with name, age, and city.",
            "forbidden_check": lambda r, obj: obj is not None and isinstance(obj, dict) and any(isinstance(v, (dict, list)) for v in obj.values()),
            "forbidden_label": "nested_objects_or_arrays_present",
        },
        {
            "system": 'Return a JSON object with "answer". Do NOT include your reasoning or chain-of-thought.',
            "user": "What is 15 * 17?",
            "forbidden_check": lambda r, obj: obj is not None and isinstance(obj, dict) and any(k.lower() in {"reasoning", "steps", "chain_of_thought", "explanation", "thought"} for k in obj.keys()),
            "forbidden_label": "reasoning_fields_present",
        },
        {
            "system": 'Return JSON with "category" and "score". Do NOT include examples.',
            "user": "Classify the quality of this essay.",
            "forbidden_check": lambda r, obj: obj is not None and isinstance(obj, dict) and any(k.lower() in {"examples", "example", "sample", "samples"} for k in obj.keys()),
            "forbidden_label": "examples_field_present",
        },
        {
            "system": 'Return a JSON object. Do NOT wrap the JSON in markdown code blocks.',
            "user": "Give me data about the planet Mars.",
            "forbidden_check": lambda r, obj: "```" in r,
            "forbidden_label": "markdown_code_block_used",
        },
        {
            "system": 'Return JSON with "result". Do NOT add a "source" or "reference" field.',
            "user": "What is the speed of light?",
            "forbidden_check": lambda r, obj: obj is not None and isinstance(obj, dict) and any(k.lower() in {"source", "reference", "ref", "sources"} for k in obj.keys()),
            "forbidden_label": "source_reference_field_present",
        },
        {
            "system": 'Return JSON with "name" and "description". Do NOT include a timestamp or date field.',
            "user": "Describe a new software feature.",
            "forbidden_check": lambda r, obj: obj is not None and isinstance(obj, dict) and any(k.lower() in {"timestamp", "date", "time", "created_at", "updated_at"} for k in obj.keys()),
            "forbidden_label": "timestamp_field_present",
        },
        {
            "system": 'Respond with JSON. Do NOT use boolean values; use strings "yes" or "no" instead.',
            "user": "Is Python a compiled language?",
            "forbidden_check": lambda r, obj: obj is not None and isinstance(obj, dict) and any(isinstance(v, bool) for v in obj.values()),
            "forbidden_label": "boolean_values_present",
        },
        {
            "system": 'Return a JSON object. Do NOT include null or None values.',
            "user": "Give information about a book you know.",
            "forbidden_check": lambda r, obj: obj is not None and isinstance(obj, dict) and any(v is None for v in obj.values()),
            "forbidden_label": "null_values_present",
        },
    ]

    results = []
    for i, probe in enumerate(probes):
        response = safe_call(probe["system"], probe["user"])
        obj = extract_json(response)

        constraints_given = ["valid_json", f"no_{probe['forbidden_label']}"]
        constraints_satisfied = []
        failure_modes = []

        is_valid_json = obj is not None
        if not is_valid_json:
            failure_modes.append("invalid_json")
        else:
            constraints_satisfied.append("valid_json")

        try:
            forbidden_present = probe["forbidden_check"](response, obj)
        except Exception as e:
            forbidden_present = False

        if not forbidden_present:
            constraints_satisfied.append(f"no_{probe['forbidden_label']}")
        else:
            failure_modes.append(f"violated_negation:{probe['forbidden_label']}")

        faithfulness = len(constraints_satisfied) / len(constraints_given)
        results.append({
            "category": "negation_instructions",
            "probe_id": i,
            "instruction_text": probe["system"],
            "response": response[:500],
            "constraints_given": constraints_given,
            "constraints_satisfied": constraints_satisfied,
            "faithfulness_score": faithfulness,
            "failure_mode": failure_modes[0] if failure_modes else "none",
        })
        print(f"  [negation #{i}] score={faithfulness:.2f} failures={failure_modes}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 5 – Format Override Resistance
# ─────────────────────────────────────────────────────────────────────────────
def run_format_override_resistance() -> list[dict]:
    """
    Instruction says JSON but user query implies another format.
    Check if model still returns valid JSON.
    """
    system = 'You must ALWAYS respond with valid JSON. No matter what the user asks, your output must be parseable JSON.'

    probes = [
        "List the top 5 programming languages.",
        "Write a haiku about autumn.",
        "Give me a step-by-step recipe for pancakes.",
        "Explain quantum entanglement in simple terms.",
        "Write a short poem about the ocean.",
        "Tell me a joke.",
        "Give me bullet points on how to stay productive.",
        "Describe the pros and cons of electric cars.",
        "Write a motivational quote.",
        "Enumerate the planets in our solar system.",
    ]

    results = []
    for i, user_msg in enumerate(probes):
        response = safe_call(system, user_msg)
        obj = extract_json(response)

        constraints_given = ["output_is_valid_json"]
        constraints_satisfied = []
        failure_modes = []

        if obj is not None:
            constraints_satisfied.append("output_is_valid_json")
        else:
            failure_modes.append("non_json_output_despite_json_instruction")

        faithfulness = len(constraints_satisfied) / len(constraints_given)
        results.append({
            "category": "format_override_resistance",
            "probe_id": i,
            "instruction_text": system,
            "response": response[:500],
            "constraints_given": constraints_given,
            "constraints_satisfied": constraints_satisfied,
            "faithfulness_score": faithfulness,
            "failure_mode": failure_modes[0] if failure_modes else "none",
        })
        print(f"  [format_override #{i}] score={faithfulness:.2f} failures={failure_modes}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 6 – Numeric Precision
# ─────────────────────────────────────────────────────────────────────────────
def run_numeric_precision() -> list[dict]:
    probes = [
        {
            "system": 'Return JSON with a "confidence" field. The value MUST be a float between 0.0 and 1.0 (not 0–100).',
            "user": "How confident are you that Python is popular?",
            "fields": ["confidence"],
        },
        {
            "system": 'Return JSON with "relevance" and "quality" scores, both floats between 0.0 and 1.0.',
            "user": "Rate this article about climate change.",
            "fields": ["relevance", "quality"],
        },
        {
            "system": 'Return JSON with "probability" as a float between 0.0 and 1.0.',
            "user": "What is the probability it will rain tomorrow in London?",
            "fields": ["probability"],
        },
        {
            "system": 'Return JSON with "toxicity_score" as a float between 0.0 and 1.0.',
            "user": "Rate the toxicity: 'You are completely wrong about everything!'",
            "fields": ["toxicity_score"],
        },
        {
            "system": 'Return JSON with "similarity" score as a float between 0.0 and 1.0.',
            "user": "How similar are 'cat' and 'kitten'?",
            "fields": ["similarity"],
        },
        {
            "system": 'Return JSON with "urgency" as a float between 0.0 and 1.0.',
            "user": "Server is down! Customers cannot checkout.",
            "fields": ["urgency"],
        },
        {
            "system": 'Return JSON with "readability" and "complexity" as floats between 0.0 and 1.0.',
            "user": "Rate this sentence: 'The cat sat on the mat.'",
            "fields": ["readability", "complexity"],
        },
        {
            "system": 'Return JSON with "novelty_score" as a float between 0.0 and 1.0.',
            "user": "Rate the novelty of this idea: using AI to water plants.",
            "fields": ["novelty_score"],
        },
        {
            "system": 'Return JSON with "sentiment_score" as a float between 0.0 and 1.0 where 0.0 is most negative and 1.0 is most positive.',
            "user": "I absolutely love this new feature!",
            "fields": ["sentiment_score"],
        },
        {
            "system": 'Return JSON with "risk_score" as a float between 0.0 and 1.0.',
            "user": "Assess the risk: investing all savings in a single startup.",
            "fields": ["risk_score"],
        },
    ]

    results = []
    for i, probe in enumerate(probes):
        response = safe_call(probe["system"], probe["user"])
        obj = extract_json(response)

        constraints_given = ["valid_json", "fields_are_floats", "values_in_0_to_1_range"]
        constraints_satisfied = []
        failure_modes = []

        if obj is None or not isinstance(obj, dict):
            failure_modes.append("invalid_json_or_not_object")
        else:
            constraints_satisfied.append("valid_json")

            all_floats = True
            all_in_range = True
            for field in probe["fields"]:
                if field not in obj:
                    all_floats = False
                    all_in_range = False
                    failure_modes.append(f"missing_field:{field}")
                    continue
                val = obj[field]
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    all_floats = False
                    failure_modes.append(f"not_numeric:{field}={val}")
                elif not (0.0 <= float(val) <= 1.0):
                    all_in_range = False
                    failure_modes.append(f"out_of_range:{field}={val}")

            if all_floats:
                constraints_satisfied.append("fields_are_floats")
            if all_in_range and all_floats:
                constraints_satisfied.append("values_in_0_to_1_range")

        faithfulness = len(constraints_satisfied) / len(constraints_given)
        results.append({
            "category": "numeric_precision",
            "probe_id": i,
            "instruction_text": probe["system"],
            "response": response[:500],
            "constraints_given": constraints_given,
            "constraints_satisfied": constraints_satisfied,
            "faithfulness_score": faithfulness,
            "failure_mode": failure_modes[0] if failure_modes else "none",
        })
        print(f"  [numeric_precision #{i}] score={faithfulness:.2f} failures={failure_modes}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 7 – Array Ordering
# ─────────────────────────────────────────────────────────────────────────────
def run_array_ordering() -> list[dict]:
    probes = [
        {
            "system": 'Return a JSON object with "items" array. Each item has "name" (string) and "confidence" (float 0-1). Sort items by confidence DESCENDING (highest first).',
            "user": "List 4 possible causes for server slowdown.",
        },
        {
            "system": 'Return a JSON object with "results" array. Each result has "keyword" and "relevance" (float 0-1). Sort by relevance DESCENDING.',
            "user": "Find keywords related to machine learning.",
        },
        {
            "system": 'Return a JSON object with "products" array. Each product has "name" and "price" (number). Sort by price ASCENDING (cheapest first).',
            "user": "List 4 types of smartphones.",
        },
        {
            "system": 'Return a JSON object with "topics" array. Each topic has "title" and "importance" (float 0-1). Sort by importance DESCENDING.',
            "user": "List key topics in data science.",
        },
        {
            "system": 'Return a JSON object with "candidates" array. Each has "name" and "score" (float 0-1). Sort by score DESCENDING.',
            "user": "Rank 4 programming languages by their popularity.",
        },
        {
            "system": 'Return a JSON object with "steps" array. Each step has "order" (integer, starting from 1) and "description". Sort by order ASCENDING.',
            "user": "List steps to brew coffee.",
        },
        {
            "system": 'Return a JSON object with "risks" array. Each risk has "type" and "severity" (float 0-1). Sort by severity DESCENDING (most severe first).',
            "user": "List risks of cloud migration.",
        },
        {
            "system": 'Return a JSON object with "features" array. Each feature has "name" and "priority" (integer 1=highest). Sort by priority ASCENDING.',
            "user": "List features for a todo app.",
        },
        {
            "system": 'Return a JSON object with "entries" array. Each entry has "label" and "weight" (float 0-1). Sort by weight DESCENDING.',
            "user": "List factors affecting software quality.",
        },
        {
            "system": 'Return a JSON object with "events" array. Each event has "name" and "year" (integer). Sort by year ASCENDING (oldest first).',
            "user": "List 4 major events in computing history.",
        },
    ]

    sort_configs = [
        ("confidence", True),
        ("relevance", True),
        ("price", False),
        ("importance", True),
        ("score", True),
        ("order", False),
        ("severity", True),
        ("priority", False),
        ("weight", True),
        ("year", False),
    ]

    results = []
    for i, (probe, (sort_key, descending)) in enumerate(zip(probes, sort_configs)):
        response = safe_call(probe["system"], probe["user"])
        obj = extract_json(response)

        array_keys = ["items", "results", "products", "topics", "candidates", "steps", "risks", "features", "entries", "events"]
        arr_key = array_keys[i]

        constraints_given = ["valid_json", "has_array", f"sorted_by_{sort_key}_{'desc' if descending else 'asc'}"]
        constraints_satisfied = []
        failure_modes = []

        if obj is None or not isinstance(obj, dict):
            failure_modes.append("invalid_json_or_not_object")
        else:
            constraints_satisfied.append("valid_json")

            # find the array (flexible key lookup)
            arr = None
            for k in obj:
                if isinstance(obj[k], list):
                    arr = obj[k]
                    break

            if arr is None or len(arr) < 2:
                failure_modes.append("no_array_or_too_short")
            else:
                constraints_satisfied.append("has_array")
                # check sorting
                vals = []
                for item in arr:
                    if isinstance(item, dict) and sort_key in item:
                        try:
                            vals.append(float(item[sort_key]))
                        except (TypeError, ValueError):
                            vals.append(None)
                    else:
                        vals.append(None)

                if None in vals:
                    failure_modes.append(f"sort_key_missing_or_non_numeric:{sort_key}")
                else:
                    expected = sorted(vals, reverse=descending)
                    if vals == expected:
                        constraints_satisfied.append(f"sorted_by_{sort_key}_{'desc' if descending else 'asc'}")
                    else:
                        failure_modes.append(f"incorrect_sort_order:got={vals}")

        faithfulness = len(constraints_satisfied) / len(constraints_given)
        results.append({
            "category": "array_ordering",
            "probe_id": i,
            "instruction_text": probe["system"],
            "response": response[:500],
            "constraints_given": constraints_given,
            "constraints_satisfied": constraints_satisfied,
            "faithfulness_score": faithfulness,
            "failure_mode": failure_modes[0] if failure_modes else "none",
        })
        print(f"  [array_ordering #{i}] score={faithfulness:.2f} failures={failure_modes}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 8 – Multi-Constraint Compliance
# ─────────────────────────────────────────────────────────────────────────────
def run_multi_constraint() -> list[dict]:
    """
    Each probe has 4 simultaneous constraints. Measures compliance fraction.
    """
    probes = [
        {
            "system": (
                "Respond ONLY with a JSON object. "
                "Constraint 1: The JSON must have exactly 3 fields. "
                "Constraint 2: Include a field 'sentiment' with value exactly one of: positive, negative, neutral. "
                "Constraint 3: Include a field 'score' as a float between 0.0 and 1.0. "
                "Constraint 4: Do NOT include any explanation outside the JSON."
            ),
            "user": "Analyze: 'This product completely exceeded my expectations!'",
            "checks": [
                ("exactly_3_fields", lambda obj: isinstance(obj, dict) and len(obj) == 3),
                ("valid_sentiment", lambda obj: isinstance(obj, dict) and str(obj.get("sentiment", "")).lower() in {"positive", "negative", "neutral"}),
                ("score_float_0_1", lambda obj: isinstance(obj, dict) and isinstance(obj.get("score"), (int, float)) and not isinstance(obj.get("score"), bool) and 0.0 <= float(obj.get("score", -1)) <= 1.0),
                ("no_prose_outside_json", lambda obj: obj is not None),
            ],
        },
        {
            "system": (
                "Respond ONLY with a JSON object. "
                "Constraint 1: Exactly 4 fields. "
                "Constraint 2: Field 'priority' must be exactly: low, medium, high, or critical. "
                "Constraint 3: Field 'tags' must be an array of strings. "
                "Constraint 4: Do NOT use nested objects."
            ),
            "user": "Classify this bug: The checkout button disappears on mobile devices.",
            "checks": [
                ("exactly_4_fields", lambda obj: isinstance(obj, dict) and len(obj) == 4),
                ("valid_priority", lambda obj: isinstance(obj, dict) and str(obj.get("priority", "")).lower() in {"low", "medium", "high", "critical"}),
                ("tags_is_array", lambda obj: isinstance(obj, dict) and isinstance(obj.get("tags"), list)),
                ("no_nested_objects", lambda obj: isinstance(obj, dict) and not any(isinstance(v, dict) for v in obj.values())),
            ],
        },
        {
            "system": (
                "Respond ONLY with a JSON object. "
                "Constraint 1: Exactly 5 fields. "
                "Constraint 2: All numeric values must be floats between 0.0 and 1.0. "
                "Constraint 3: Field 'category' must be a string. "
                "Constraint 4: Do NOT include boolean values."
            ),
            "user": "Evaluate this article about renewable energy.",
            "checks": [
                ("exactly_5_fields", lambda obj: isinstance(obj, dict) and len(obj) == 5),
                ("numerics_in_range", lambda obj: isinstance(obj, dict) and all(
                    not isinstance(v, (int, float)) or isinstance(v, bool) or (0.0 <= float(v) <= 1.0)
                    for v in obj.values()
                )),
                ("category_is_string", lambda obj: isinstance(obj, dict) and isinstance(obj.get("category"), str)),
                ("no_booleans", lambda obj: isinstance(obj, dict) and not any(isinstance(v, bool) for v in obj.values())),
            ],
        },
        {
            "system": (
                "Respond ONLY with a JSON object. "
                "Constraint 1: Field 'items' must be a sorted array (descending by 'score'). "
                "Constraint 2: Exactly 3 items in the array. "
                "Constraint 3: Each item has exactly 'name' (string) and 'score' (float 0-1). "
                "Constraint 4: No extra fields outside 'items'."
            ),
            "user": "List the top 3 benefits of exercise.",
            "checks": [
                ("items_sorted_desc", lambda obj: isinstance(obj, dict) and isinstance(obj.get("items"), list) and len(obj.get("items", [])) >= 2 and all(
                    obj["items"][j]["score"] >= obj["items"][j+1]["score"]
                    for j in range(len(obj["items"])-1)
                    if isinstance(obj["items"][j].get("score"), (int, float)) and isinstance(obj["items"][j+1].get("score"), (int, float))
                )),
                ("exactly_3_items", lambda obj: isinstance(obj, dict) and len(obj.get("items", [])) == 3),
                ("item_schema_correct", lambda obj: isinstance(obj, dict) and all(
                    isinstance(it, dict) and set(it.keys()) == {"name", "score"}
                    for it in obj.get("items", [])
                )),
                ("no_extra_root_fields", lambda obj: isinstance(obj, dict) and set(obj.keys()) == {"items"}),
            ],
        },
        {
            "system": (
                "Respond ONLY with a JSON object. "
                "Constraint 1: Exactly 3 fields: 'label', 'confidence', 'reason'. "
                "Constraint 2: 'confidence' must be a float between 0.0 and 1.0. "
                "Constraint 3: 'label' must be exactly one of: spam, ham. "
                "Constraint 4: 'reason' must be a string under 50 characters."
            ),
            "user": "Classify: 'Congratulations! You've won a $1000 gift card! Click here!'",
            "checks": [
                ("exactly_label_confidence_reason", lambda obj: isinstance(obj, dict) and set(obj.keys()) == {"label", "confidence", "reason"}),
                ("confidence_float_0_1", lambda obj: isinstance(obj, dict) and isinstance(obj.get("confidence"), (int, float)) and not isinstance(obj.get("confidence"), bool) and 0.0 <= float(obj.get("confidence", -1)) <= 1.0),
                ("label_spam_or_ham", lambda obj: isinstance(obj, dict) and str(obj.get("label", "")).lower() in {"spam", "ham"}),
                ("reason_under_50_chars", lambda obj: isinstance(obj, dict) and isinstance(obj.get("reason"), str) and len(obj.get("reason", "")) <= 50),
            ],
        },
        {
            "system": (
                "Respond ONLY with a JSON object. "
                "Constraint 1: Field 'keywords' is an array of exactly 5 strings. "
                "Constraint 2: All strings in 'keywords' must be lowercase. "
                "Constraint 3: Field 'language' must be exactly one of: english, french, spanish, german. "
                "Constraint 4: No other fields besides 'keywords' and 'language'."
            ),
            "user": "Extract keywords from: 'Machine learning models improve through iterative training on data.'",
            "checks": [
                ("keywords_5_strings", lambda obj: isinstance(obj, dict) and isinstance(obj.get("keywords"), list) and len(obj.get("keywords", [])) == 5 and all(isinstance(k, str) for k in obj.get("keywords", []))),
                ("all_lowercase_keywords", lambda obj: isinstance(obj, dict) and all(k == k.lower() for k in obj.get("keywords", []) if isinstance(k, str))),
                ("valid_language", lambda obj: isinstance(obj, dict) and str(obj.get("language", "")).lower() in {"english", "french", "spanish", "german"}),
                ("only_keywords_and_language", lambda obj: isinstance(obj, dict) and set(obj.keys()) == {"keywords", "language"}),
            ],
        },
        {
            "system": (
                "Respond ONLY with a JSON object. "
                "Constraint 1: Exactly 4 fields. "
                "Constraint 2: Field 'status' must be exactly one of: pass, fail. "
                "Constraint 3: Field 'score' must be an integer between 0 and 100. "
                "Constraint 4: No null values."
            ),
            "user": "Grade this code review: the code has minor style issues but works correctly.",
            "checks": [
                ("exactly_4_fields", lambda obj: isinstance(obj, dict) and len(obj) == 4),
                ("status_pass_or_fail", lambda obj: isinstance(obj, dict) and str(obj.get("status", "")).lower() in {"pass", "fail"}),
                ("score_int_0_100", lambda obj: isinstance(obj, dict) and isinstance(obj.get("score"), int) and not isinstance(obj.get("score"), bool) and 0 <= obj.get("score", -1) <= 100),
                ("no_null_values", lambda obj: isinstance(obj, dict) and not any(v is None for v in obj.values())),
            ],
        },
        {
            "system": (
                "Respond ONLY with a JSON object. "
                "Constraint 1: Field 'entities' is a non-empty array. "
                "Constraint 2: Each entity has 'text' (string) and 'type' (string). "
                "Constraint 3: 'type' must be exactly one of: PERSON, ORG, LOCATION, DATE. "
                "Constraint 4: No extra fields in entity objects."
            ),
            "user": "Extract entities: 'Elon Musk founded SpaceX in Hawthorne, California in 2002.'",
            "checks": [
                ("entities_non_empty", lambda obj: isinstance(obj, dict) and isinstance(obj.get("entities"), list) and len(obj.get("entities", [])) > 0),
                ("entities_have_text_and_type", lambda obj: isinstance(obj, dict) and all(isinstance(e, dict) and "text" in e and "type" in e for e in obj.get("entities", []))),
                ("valid_entity_types", lambda obj: isinstance(obj, dict) and all(e.get("type", "").upper() in {"PERSON", "ORG", "LOCATION", "DATE"} for e in obj.get("entities", []) if isinstance(e, dict))),
                ("no_extra_entity_fields", lambda obj: isinstance(obj, dict) and all(set(e.keys()) == {"text", "type"} for e in obj.get("entities", []) if isinstance(e, dict))),
            ],
        },
        {
            "system": (
                "Respond ONLY with a JSON object. "
                "Constraint 1: Exactly 2 top-level fields. "
                "Constraint 2: Field 'summary' is a string of at most 100 characters. "
                "Constraint 3: Field 'tone' must be exactly one of: formal, informal, neutral. "
                "Constraint 4: Do NOT include nested objects."
            ),
            "user": "Analyze the tone of: 'Hey! Wanna grab coffee sometime? It'd be super fun!'",
            "checks": [
                ("exactly_2_top_fields", lambda obj: isinstance(obj, dict) and len(obj) == 2),
                ("summary_under_100_chars", lambda obj: isinstance(obj, dict) and isinstance(obj.get("summary"), str) and len(obj.get("summary", "")) <= 100),
                ("valid_tone", lambda obj: isinstance(obj, dict) and str(obj.get("tone", "")).lower() in {"formal", "informal", "neutral"}),
                ("no_nested_objects", lambda obj: isinstance(obj, dict) and not any(isinstance(v, dict) for v in obj.values())),
            ],
        },
        {
            "system": (
                "Respond ONLY with a JSON object. "
                "Constraint 1: Exactly 4 fields: 'action', 'object', 'confidence', 'reversible'. "
                "Constraint 2: 'confidence' must be a float between 0.0 and 1.0. "
                "Constraint 3: 'reversible' must use string 'yes' or 'no', not boolean. "
                "Constraint 4: 'action' must be a single verb (one word, lowercase)."
            ),
            "user": "Parse the intent: 'Please delete all the log files from last month.'",
            "checks": [
                ("exactly_action_object_confidence_reversible", lambda obj: isinstance(obj, dict) and set(obj.keys()) == {"action", "object", "confidence", "reversible"}),
                ("confidence_float_0_1", lambda obj: isinstance(obj, dict) and isinstance(obj.get("confidence"), (int, float)) and not isinstance(obj.get("confidence"), bool) and 0.0 <= float(obj.get("confidence", -1)) <= 1.0),
                ("reversible_string_yes_no", lambda obj: isinstance(obj, dict) and str(obj.get("reversible", "")).lower() in {"yes", "no"} and not isinstance(obj.get("reversible"), bool)),
                ("action_single_lowercase_word", lambda obj: isinstance(obj, dict) and isinstance(obj.get("action"), str) and len(obj.get("action", "").split()) == 1 and obj.get("action", "") == obj.get("action", "").lower()),
            ],
        },
    ]

    results = []
    for i, probe in enumerate(probes):
        response = safe_call(probe["system"], probe["user"])
        obj = extract_json(response)

        n_constraints = len(probe["checks"])
        constraints_given = [c[0] for c in probe["checks"]]
        constraints_satisfied = []
        failure_modes = []

        if obj is None:
            failure_modes.append("invalid_json")
        else:
            for check_name, check_fn in probe["checks"]:
                try:
                    passed = check_fn(obj)
                except Exception:
                    passed = False
                if passed:
                    constraints_satisfied.append(check_name)
                else:
                    failure_modes.append(f"failed:{check_name}")

        faithfulness = len(constraints_satisfied) / n_constraints
        results.append({
            "category": "multi_constraint",
            "probe_id": i,
            "instruction_text": probe["system"],
            "response": response[:500],
            "constraints_given": constraints_given,
            "constraints_satisfied": constraints_satisfied,
            "faithfulness_score": faithfulness,
            "failure_mode": failure_modes[0] if failure_modes else "none",
        })
        print(f"  [multi_constraint #{i}] score={faithfulness:.2f} ({len(constraints_satisfied)}/{n_constraints}) failures={failure_modes[:2]}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# DEGRADATION CURVE – faithfulness vs constraint count
# ─────────────────────────────────────────────────────────────────────────────
def run_degradation_curve() -> dict:
    """
    Run probes with 1, 2, 3, 4, 5 simultaneous constraints.
    Run 3 probes per constraint count, take mean.
    """
    base_user = "Analyze this text: 'The quick deployment of the new AI system caused unexpected performance regressions in production.'"

    constraint_sets = {
        1: [(
            'Return a JSON object with a "sentiment" field. The value must be exactly one of: positive, negative, neutral.',
            [("valid_sentiment", lambda obj: isinstance(obj, dict) and str(obj.get("sentiment", "")).lower() in {"positive", "negative", "neutral"})],
        )],
        2: [(
            'Return a JSON object with a "sentiment" field (exactly: positive, negative, or neutral) and a "confidence" field (float 0.0 to 1.0).',
            [
                ("valid_sentiment", lambda obj: isinstance(obj, dict) and str(obj.get("sentiment", "")).lower() in {"positive", "negative", "neutral"}),
                ("confidence_0_1", lambda obj: isinstance(obj, dict) and isinstance(obj.get("confidence"), (int, float)) and not isinstance(obj.get("confidence"), bool) and 0.0 <= float(obj.get("confidence", -1)) <= 1.0),
            ],
        )],
        3: [(
            'Return a JSON object with: "sentiment" (exactly: positive, negative, or neutral), "confidence" (float 0.0 to 1.0), and "category" (string). Exactly these 3 fields.',
            [
                ("valid_sentiment", lambda obj: isinstance(obj, dict) and str(obj.get("sentiment", "")).lower() in {"positive", "negative", "neutral"}),
                ("confidence_0_1", lambda obj: isinstance(obj, dict) and isinstance(obj.get("confidence"), (int, float)) and not isinstance(obj.get("confidence"), bool) and 0.0 <= float(obj.get("confidence", -1)) <= 1.0),
                ("exactly_3_fields", lambda obj: isinstance(obj, dict) and len(obj) == 3),
            ],
        )],
        4: [(
            'Return a JSON object with EXACTLY: "sentiment" (positive/negative/neutral), "confidence" (float 0.0-1.0), "category" (string), "urgent" (use string "yes" or "no", not boolean). Exactly 4 fields.',
            [
                ("valid_sentiment", lambda obj: isinstance(obj, dict) and str(obj.get("sentiment", "")).lower() in {"positive", "negative", "neutral"}),
                ("confidence_0_1", lambda obj: isinstance(obj, dict) and isinstance(obj.get("confidence"), (int, float)) and not isinstance(obj.get("confidence"), bool) and 0.0 <= float(obj.get("confidence", -1)) <= 1.0),
                ("exactly_4_fields", lambda obj: isinstance(obj, dict) and len(obj) == 4),
                ("urgent_string_yes_no", lambda obj: isinstance(obj, dict) and str(obj.get("urgent", "")).lower() in {"yes", "no"} and not isinstance(obj.get("urgent"), bool)),
            ],
        )],
        5: [(
            'Return a JSON object with EXACTLY: "sentiment" (positive/negative/neutral), "confidence" (float 0.0-1.0), "category" (string), "urgent" (string "yes"/"no", not boolean), "keywords" (array of exactly 3 strings). Exactly 5 fields.',
            [
                ("valid_sentiment", lambda obj: isinstance(obj, dict) and str(obj.get("sentiment", "")).lower() in {"positive", "negative", "neutral"}),
                ("confidence_0_1", lambda obj: isinstance(obj, dict) and isinstance(obj.get("confidence"), (int, float)) and not isinstance(obj.get("confidence"), bool) and 0.0 <= float(obj.get("confidence", -1)) <= 1.0),
                ("exactly_5_fields", lambda obj: isinstance(obj, dict) and len(obj) == 5),
                ("urgent_string_yes_no", lambda obj: isinstance(obj, dict) and str(obj.get("urgent", "")).lower() in {"yes", "no"} and not isinstance(obj.get("urgent"), bool)),
                ("keywords_3_strings", lambda obj: isinstance(obj, dict) and isinstance(obj.get("keywords"), list) and len(obj.get("keywords", [])) == 3 and all(isinstance(k, str) for k in obj.get("keywords", []))),
            ],
        )],
    }

    # Run each 3 times for better estimate
    curve = {}
    for n_constraints, probes in constraint_sets.items():
        scores = []
        for _ in range(3):
            sys_prompt, checks = probes[0]
            response = safe_call(sys_prompt, base_user)
            obj = extract_json(response)
            if obj is None:
                score = 0.0
            else:
                passed = sum(1 for _, fn in checks if _safe_check(fn, obj))
                score = passed / len(checks)
            scores.append(score)
        mean_score = statistics.mean(scores)
        curve[n_constraints] = mean_score
        print(f"  [degradation_curve] n_constraints={n_constraints} mean_score={mean_score:.2f}")

    return curve


def _safe_check(fn, obj):
    try:
        return bool(fn(obj))
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS & REPORTING
# ─────────────────────────────────────────────────────────────────────────────
def analyze_results(all_results: list[dict]) -> dict:
    categories = {}
    for r in all_results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    category_stats = {}
    for cat, results in categories.items():
        scores = [r["faithfulness_score"] for r in results]
        fully_compliant = sum(1 for r in results if r["faithfulness_score"] >= 1.0)
        failure_modes = [r["failure_mode"] for r in results if r["failure_mode"] != "none"]
        top_failure = Counter(failure_modes).most_common(1)
        top_failure_mode = top_failure[0][0] if top_failure else "none"

        # simplify failure modes for readability
        simplified = []
        for fm in failure_modes:
            if ":" in fm:
                simplified.append(fm.split(":")[0])
            else:
                simplified.append(fm)
        top_simplified = Counter(simplified).most_common(1)
        top_failure_simplified = top_simplified[0][0] if top_simplified else "none"

        mean_f = round(statistics.mean(scores), 3)
        n_p = len(results)
        category_stats[cat] = {
            "mean_faithfulness": mean_f,
            "ci_95": list(wilson_ci_95(mean_f, n_p)),
            "pct_fully_compliant": round(100 * fully_compliant / n_p, 1),
            "top_failure_mode": top_failure_simplified,
            "n_probes": n_p,
        }

    return category_stats


def find_degradation_threshold(curve: dict):
    """Return the constraint count at which mean faithfulness first drops below 0.7."""
    for n in sorted(curve.keys()):
        if curve[n] < 0.7:
            return n
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("INSTRUCTION FAITHFULNESS EVALUATION")
    print("Model:", MODEL)
    print("=" * 70)

    all_results = []

    print("\n[1/8] Schema Strictness...")
    all_results.extend(run_schema_strictness())

    print("\n[2/8] Field Count Compliance...")
    all_results.extend(run_field_count_compliance())

    print("\n[3/8] Value Constraint (Enum)...")
    all_results.extend(run_value_constraint())

    print("\n[4/8] Negation Instructions...")
    all_results.extend(run_negation_instructions())

    print("\n[5/8] Format Override Resistance...")
    all_results.extend(run_format_override_resistance())

    print("\n[6/8] Numeric Precision...")
    all_results.extend(run_numeric_precision())

    print("\n[7/8] Array Ordering...")
    all_results.extend(run_array_ordering())

    print("\n[8/8] Multi-Constraint Compliance...")
    all_results.extend(run_multi_constraint())

    print("\n[Extra] Degradation Curve...")
    degradation_curve = run_degradation_curve()

    # Analysis
    category_stats = analyze_results(all_results)
    overall_score = round(statistics.mean(r["faithfulness_score"] for r in all_results), 3)

    threshold = find_degradation_threshold(degradation_curve)

    all_failure_modes = [r["failure_mode"] for r in all_results if r["failure_mode"] != "none"]
    simplified_failures = []
    for fm in all_failure_modes:
        if ":" in fm:
            simplified_failures.append(fm.split(":")[0])
        else:
            simplified_failures.append(fm)
    global_top_failure = Counter(simplified_failures).most_common(1)
    global_top_failure_mode = global_top_failure[0][0] if global_top_failure else "none"

    # Save results
    output = {
        "model": MODEL,
        "total_probes": len(all_results),
        "overall_faithfulness_score": overall_score,
        "overall_ci_95": list(wilson_ci_95(overall_score, len(all_results))),
        "category_stats": category_stats,
        "degradation_curve": {str(k): v for k, v in degradation_curve.items()},
        "faithfulness_drops_below_0_7_at_constraint_count": threshold,
        "global_top_failure_mode": global_top_failure_mode,
        "all_results": all_results,
    }

    with open(RESULTS_F, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {RESULTS_F}")

    # Print table
    CATEGORY_LABELS = {
        "schema_strictness":          "Schema Strictness",
        "field_count_compliance":     "Field Count Compliance",
        "value_constraint":           "Value Constraint (Enum)",
        "negation_instructions":      "Negation Instructions",
        "format_override_resistance": "Format Override Resistance",
        "numeric_precision":          "Numeric Precision",
        "array_ordering":             "Array Ordering",
        "multi_constraint":           "Multi-Constraint",
    }

    print("\n" + "=" * 90)
    print(f"{'Category':<28} | {'Mean Faithfulness':^18} | {'% Fully Compliant':^18} | {'Top Failure Mode'}")
    print("-" * 90)
    for cat, stats in category_stats.items():
        label = CATEGORY_LABELS.get(cat, cat)
        print(
            f"{label:<28} | {stats['mean_faithfulness']:^18.3f} | "
            f"{stats['pct_fully_compliant']:^17.1f}% | {stats['top_failure_mode']}"
        )
    print("=" * 90)
    print(f"\nOverall Faithfulness Score : {overall_score:.3f}")
    print(f"Faithfulness < 0.7 starts  : {threshold if threshold else 'never (always >= 0.7)'} constraints")
    print(f"Most common failure mode   : {global_top_failure_mode}")

    print("\nDegradation Curve (constraints → mean faithfulness):")
    for n in sorted(degradation_curve):
        bar = "█" * int(degradation_curve[n] * 20)
        print(f"  {n} constraint(s): {degradation_curve[n]:.2f}  {bar}")

    return output


if __name__ == "__main__":
    main()
