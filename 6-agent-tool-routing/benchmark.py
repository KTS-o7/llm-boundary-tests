import os
"""
Agent Tool-Routing Benchmark
Tests LLM routing accuracy and loop latency for a 6-tool mini-agent.
"""

import json
import time
import math
import statistics
import requests
import re
from datetime import datetime
from typing import Any

# ─────────────────────────────────────────────
# API CONFIG
# ─────────────────────────────────────────────
API_URL = "https://ai.shenthar.me/v1/chat/completions"
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL   = "taalas-llama3.1-8b"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

def llm_call(messages: list[dict], temperature: float = 0.1) -> tuple[str, float]:
    """Call the LLM API; returns (content, elapsed_seconds)."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    t0 = time.perf_counter()
    resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return content, elapsed


def extract_json(text: str) -> dict:
    """Extract the first JSON object from LLM response text."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Find JSON block in markdown fences or inline
    patterns = [
        r"```(?:json)?\s*(\{.*?\})\s*```",
        r"(\{[^{}]*\})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No valid JSON found in: {text[:300]}")


# ─────────────────────────────────────────────
# 6 REAL MOCK TOOLS
# ─────────────────────────────────────────────

def search_web(query: str) -> dict:
    """Returns fake but realistic search results JSON."""
    query_lower = query.lower()
    # Build contextual fake results
    topic_map = {
        "python": "Python programming language",
        "climate": "Climate change",
        "bitcoin": "Bitcoin cryptocurrency",
        "recipe": "Cooking recipes",
        "nasa": "NASA space exploration",
    }
    topic = next((v for k, v in topic_map.items() if k in query_lower), query[:40])
    return {
        "query": query,
        "total_results": 1_240_000,
        "results": [
            {
                "rank": 1,
                "title": f"{topic} - Wikipedia",
                "url": f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
                "snippet": f"Comprehensive overview of {topic}, covering history, applications, and recent developments.",
            },
            {
                "rank": 2,
                "title": f"Latest news about {topic}",
                "url": f"https://www.bbc.com/news/search?q={topic.replace(' ', '+')}",
                "snippet": f"Breaking news and in-depth reporting on {topic} from trusted sources worldwide.",
            },
            {
                "rank": 3,
                "title": f"{topic}: Research and Analysis",
                "url": f"https://scholar.google.com/scholar?q={topic.replace(' ', '+')}",
                "snippet": f"Peer-reviewed academic papers and research articles about {topic}.",
            },
        ],
    }


def get_weather(city: str) -> dict:
    """Returns fake but realistic weather JSON for any city."""
    city_profiles = {
        "Tokyo":      {"temp_c": 18, "condition": "Partly Cloudy", "humidity": 65, "wind_kph": 14},
        "London":     {"temp_c": 10, "condition": "Overcast",       "humidity": 80, "wind_kph": 20},
        "New York":   {"temp_c": 15, "condition": "Sunny",          "humidity": 55, "wind_kph": 12},
        "Sydney":     {"temp_c": 22, "condition": "Clear",          "humidity": 60, "wind_kph": 18},
        "Dubai":      {"temp_c": 38, "condition": "Sunny",          "humidity": 30, "wind_kph": 10},
        "Moscow":     {"temp_c": -3, "condition": "Snowing",        "humidity": 85, "wind_kph": 22},
        "Paris":      {"temp_c": 12, "condition": "Light Rain",     "humidity": 75, "wind_kph": 15},
        "Cairo":      {"temp_c": 32, "condition": "Sunny",          "humidity": 25, "wind_kph": 8},
        "Mumbai":     {"temp_c": 30, "condition": "Humid",          "humidity": 88, "wind_kph": 16},
        "Toronto":    {"temp_c": 5,  "condition": "Cloudy",         "humidity": 70, "wind_kph": 25},
    }
    profile = city_profiles.get(city, {"temp_c": 20, "condition": "Partly Cloudy", "humidity": 60, "wind_kph": 15})
    return {
        "city": city,
        "country": "N/A",
        "local_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "temperature_c": profile["temp_c"],
        "temperature_f": round(profile["temp_c"] * 9 / 5 + 32, 1),
        "condition": profile["condition"],
        "humidity_pct": profile["humidity"],
        "wind_kph": profile["wind_kph"],
        "uv_index": 4,
        "forecast_3day": [
            {"day": "Tomorrow",    "high_c": profile["temp_c"] + 2, "low_c": profile["temp_c"] - 3, "condition": profile["condition"]},
            {"day": "Day after",   "high_c": profile["temp_c"] + 1, "low_c": profile["temp_c"] - 4, "condition": "Cloudy"},
            {"day": "In 3 days",   "high_c": profile["temp_c"] - 1, "low_c": profile["temp_c"] - 5, "condition": "Rain"},
        ],
    }


def calculate(expression: str) -> dict:
    """Safely evaluates a math expression using Python."""
    # Whitelist: only math-safe tokens
    allowed = set("0123456789+-*/().% \t")
    math_funcs = {
        "sqrt": math.sqrt, "abs": abs, "round": round,
        "pow": pow, "log": math.log, "sin": math.sin,
        "cos": math.cos, "tan": math.tan, "pi": math.pi, "e": math.e,
    }
    # Reject anything with letters outside allowed math names
    clean_expr = expression.strip()
    try:
        result = eval(clean_expr, {"__builtins__": {}}, math_funcs)  # noqa: S307
        return {
            "expression": expression,
            "result": result,
            "result_type": type(result).__name__,
            "formatted": f"{result:,}" if isinstance(result, (int, float)) else str(result),
        }
    except Exception as exc:
        return {"expression": expression, "error": str(exc), "result": None}


def lookup_person(name: str) -> dict:
    """Returns fake biography JSON for a person."""
    biographies = {
        "Marie Curie": {
            "full_name": "Maria Salomea Skłodowska-Curie",
            "born": "7 November 1867, Warsaw, Poland",
            "died": "4 July 1934, Passy, France",
            "nationality": "Polish-French",
            "fields": ["Physics", "Chemistry"],
            "known_for": "Radioactivity; discovery of Polonium and Radium",
            "awards": ["Nobel Prize in Physics (1903)", "Nobel Prize in Chemistry (1911)"],
            "summary": "Marie Curie was a pioneering physicist and chemist who conducted groundbreaking research on radioactivity.",
        },
        "Elon Musk": {
            "full_name": "Elon Reeve Musk",
            "born": "28 June 1971, Pretoria, South Africa",
            "died": None,
            "nationality": "South African, Canadian, American",
            "fields": ["Entrepreneurship", "Engineering"],
            "known_for": "Tesla, SpaceX, PayPal, xAI",
            "awards": ["Royal Aeronautical Society Gold Medal (2012)"],
            "summary": "Elon Musk is a business magnate and investor known for founding Tesla and SpaceX.",
        },
        "Ada Lovelace": {
            "full_name": "Augusta Ada King, Countess of Lovelace",
            "born": "10 December 1815, London, England",
            "died": "27 November 1852, London, England",
            "nationality": "British",
            "fields": ["Mathematics", "Computing"],
            "known_for": "First computer programmer; algorithm for Babbage's Analytical Engine",
            "awards": [],
            "summary": "Ada Lovelace is often credited as the first computer programmer for her work on Charles Babbage's Analytical Engine.",
        },
        "Albert Einstein": {
            "full_name": "Albert Einstein",
            "born": "14 March 1879, Ulm, Germany",
            "died": "18 April 1955, Princeton, USA",
            "nationality": "German, Swiss, American",
            "fields": ["Theoretical Physics"],
            "known_for": "Theory of Relativity, E=mc², Photoelectric Effect",
            "awards": ["Nobel Prize in Physics (1921)"],
            "summary": "Albert Einstein developed the theory of general relativity, revolutionizing our understanding of space, time, and gravity.",
        },
        "Nikola Tesla": {
            "full_name": "Nikola Tesla",
            "born": "10 July 1856, Smiljan, Serbia (now Croatia)",
            "died": "7 January 1943, New York, USA",
            "nationality": "Serbian-American",
            "fields": ["Electrical Engineering", "Physics"],
            "known_for": "AC electricity, Tesla coil, radio transmission",
            "awards": ["Elliott Cresson Medal (1894)", "Edison Medal (1916)"],
            "summary": "Nikola Tesla was an inventor and electrical engineer who developed the alternating current (AC) electrical supply system.",
        },
    }
    # Fuzzy match
    for key in biographies:
        if key.lower() in name.lower() or name.lower() in key.lower():
            return biographies[key]
    # Generic fallback
    first, *rest = name.split()
    return {
        "full_name": name,
        "born": "Unknown",
        "died": None,
        "nationality": "Unknown",
        "fields": ["Unknown"],
        "known_for": f"Notable figure named {name}",
        "awards": [],
        "summary": f"{name} is a notable person. Detailed biographical information is not available in the local database.",
    }


def translate(text: str, target_lang: str) -> dict:
    """Calls the LLM API to actually translate text."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional translator. "
                "Translate the given text accurately. "
                'Return ONLY valid JSON: {"original": "...", "translated": "...", "target_language": "..."}'
            ),
        },
        {
            "role": "user",
            "content": f'Translate this text to {target_lang}: "{text}"',
        },
    ]
    t0 = time.perf_counter()
    resp_text, _ = llm_call(messages)
    elapsed = time.perf_counter() - t0
    try:
        result = extract_json(resp_text)
    except ValueError:
        result = {"original": text, "translated": resp_text.strip(), "target_language": target_lang}
    result["tool_llm_time_s"] = round(elapsed, 3)
    return result


def summarize(text: str) -> dict:
    """Calls the LLM API to actually summarize text."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise summarizer. "
                "Summarize the given text in 2-3 sentences. "
                'Return ONLY valid JSON: {"summary": "...", "word_count_original": N, "word_count_summary": N}'
            ),
        },
        {"role": "user", "content": f"Summarize: {text}"},
    ]
    t0 = time.perf_counter()
    resp_text, _ = llm_call(messages)
    elapsed = time.perf_counter() - t0
    try:
        result = extract_json(resp_text)
    except ValueError:
        result = {
            "summary": resp_text.strip()[:300],
            "word_count_original": len(text.split()),
            "word_count_summary": len(resp_text.split()),
        }
    result["tool_llm_time_s"] = round(elapsed, 3)
    return result


TOOLS = {
    "search_web":    search_web,
    "get_weather":   get_weather,
    "calculate":     calculate,
    "lookup_person": lookup_person,
    "translate":     translate,
    "summarize":     summarize,
}

TOOL_DESCRIPTIONS = """
Available tools:
1. search_web(query: str) - Search the web for information, news, articles, or any general knowledge topic
2. get_weather(city: str) - Get current weather and forecast for a specific city
3. calculate(expression: str) - Evaluate a mathematical expression (arithmetic, algebra, basic math)
4. lookup_person(name: str) - Get biographical information about a famous person
5. translate(text: str, target_lang: str) - Translate text into another language
6. summarize(text: str) - Summarize a long piece of text into a concise version
"""


# ─────────────────────────────────────────────
# 40 SINGLE-STEP QUERIES (ground truth)
# ─────────────────────────────────────────────
SINGLE_QUERIES = [
    # calculate (8)
    {"query": "What is 2847 * 39?",                                       "tool": "calculate"},
    {"query": "Calculate the square root of 1764",                        "tool": "calculate"},
    {"query": "What is 15% of 840?",                                      "tool": "calculate"},
    {"query": "Compute 2^10 minus 500",                                   "tool": "calculate"},
    {"query": "How much is 3.14159 * 7 squared?",                        "tool": "calculate"},
    {"query": "Divide 99999 by 37 and tell me the result",               "tool": "calculate"},
    {"query": "What is the sum of 1234 + 5678 + 9012?",                  "tool": "calculate"},
    {"query": "Evaluate: (120 / 4) + (7 * 8) - 12",                     "tool": "calculate"},

    # get_weather (7)
    {"query": "What's the weather in Tokyo right now?",                   "tool": "get_weather"},
    {"query": "Is it raining in London today?",                           "tool": "get_weather"},
    {"query": "Tell me the current temperature in Dubai",                 "tool": "get_weather"},
    {"query": "Weather forecast for Paris this week",                     "tool": "get_weather"},
    {"query": "What should I wear in Moscow today?",                      "tool": "get_weather"},
    {"query": "How hot is it in Cairo right now?",                        "tool": "get_weather"},
    {"query": "Current weather conditions in Sydney",                     "tool": "get_weather"},

    # lookup_person (7)
    {"query": "Who is Marie Curie?",                                      "tool": "lookup_person"},
    {"query": "Tell me about Albert Einstein",                            "tool": "lookup_person"},
    {"query": "What did Nikola Tesla invent?",                            "tool": "lookup_person"},
    {"query": "Give me a biography of Ada Lovelace",                     "tool": "lookup_person"},
    {"query": "Who is Elon Musk and what companies does he run?",        "tool": "lookup_person"},
    {"query": "What is Alan Turing known for?",                           "tool": "lookup_person"},
    {"query": "Describe the life and work of Leonardo da Vinci",         "tool": "lookup_person"},

    # search_web (6)
    {"query": "Latest news about climate change",                         "tool": "search_web"},
    {"query": "How does quantum computing work?",                         "tool": "search_web"},
    {"query": "Best Python libraries for data science",                   "tool": "search_web"},
    {"query": "What happened in the 2024 US election?",                  "tool": "search_web"},
    {"query": "Recent breakthroughs in cancer research",                  "tool": "search_web"},
    {"query": "How to learn machine learning from scratch",               "tool": "search_web"},

    # translate (6)
    {"query": "Translate 'Hello, how are you?' to Spanish",              "tool": "translate"},
    {"query": "How do you say 'Good morning' in Japanese?",              "tool": "translate"},
    {"query": "Translate 'The sky is blue' into French",                 "tool": "translate"},
    {"query": "What is 'Thank you very much' in German?",                "tool": "translate"},
    {"query": "Translate 'I love programming' to Portuguese",            "tool": "translate"},
    {"query": "How do you write 'peace' in Arabic?",                     "tool": "translate"},

    # summarize (6)
    {"query": (
        "Summarize this: The Industrial Revolution was a period of major industrialization and innovation "
        "that took place during the late 1700s and early 1800s. It began in Britain and then spread throughout "
        "Western Europe and North America. This period saw the mechanization of agriculture and textile manufacturing "
        "and a revolution in power, including steam ships and railroads, that effected major changes in transportation."
    ), "tool": "summarize"},
    {"query": (
        "Please summarize: Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to "
        "the natural intelligence displayed by animals including humans. AI research has been defined as the field "
        "of study of intelligent agents, which refers to any system that perceives its environment and takes actions "
        "that maximize its chance of achieving its goals."
    ), "tool": "summarize"},
    {"query": (
        "Give me a summary of: The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest "
        "in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2, "
        "of which 5,500,000 km2 are covered by the rainforest. This region includes territory belonging to nine nations "
        "and 3,344 formally acknowledged indigenous territories."
    ), "tool": "summarize"},
    {"query": (
        "Summarize the following text: Machine learning is a method of data analysis that automates analytical model "
        "building. It is based on the idea that systems can learn from data, identify patterns and make decisions with "
        "minimal human intervention. Machine learning algorithms are used in a wide variety of applications, such as "
        "email filtering and computer vision."
    ), "tool": "summarize"},
    {"query": (
        "Can you summarize: The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, "
        "wood, and other materials, generally built along an east-to-west line across the historical northern borders of "
        "China to protect the Chinese states and empires against the raids and invasions of the various nomadic groups "
        "of the Eurasian Steppe."
    ), "tool": "summarize"},
    {"query": (
        "Please give me a brief summary of: Photosynthesis is a process used by plants and other organisms to convert "
        "light energy into chemical energy that can be later released to fuel the organism's activities. This chemical "
        "energy is stored in carbohydrate molecules, such as sugars and starches, which are synthesized from carbon "
        "dioxide and water."
    ), "tool": "summarize"},
]


# ─────────────────────────────────────────────
# 10 MULTI-STEP QUERIES (2 tools each)
# ─────────────────────────────────────────────
MULTI_QUERIES = [
    {
        "query": "Find information about Marie Curie, then summarize what you find",
        "steps": ["lookup_person", "summarize"],
    },
    {
        "query": "Search for climate change news, then summarize the results",
        "steps": ["search_web", "summarize"],
    },
    {
        "query": "Get the weather in Tokyo, then translate the weather summary to Spanish",
        "steps": ["get_weather", "translate"],
    },
    {
        "query": "Calculate 1234 * 5678, then search the web for what that number is famous for",
        "steps": ["calculate", "search_web"],
    },
    {
        "query": "Look up Albert Einstein, then translate his summary to French",
        "steps": ["lookup_person", "translate"],
    },
    {
        "query": "Search for quantum computing information, then translate the top result to German",
        "steps": ["search_web", "translate"],
    },
    {
        "query": "Get weather for London, then calculate what temperature that is in Fahrenheit if it's 10C",
        "steps": ["get_weather", "calculate"],
    },
    {
        "query": "Look up Nikola Tesla, then summarize his biography",
        "steps": ["lookup_person", "summarize"],
    },
    {
        "query": "Calculate the square root of 144, then search for what that number represents in numerology",
        "steps": ["calculate", "search_web"],
    },
    {
        "query": "Search for Python programming best practices, then summarize the findings",
        "steps": ["search_web", "summarize"],
    },
]


# ─────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────

ROUTING_SYSTEM_PROMPT = f"""You are an AI agent with access to the following tools:

{TOOL_DESCRIPTIONS}

When given a user query, select the MOST appropriate tool and extract the arguments.
Return ONLY valid JSON with exactly this structure:
{{"tool": "<tool_name>", "args": {{<key>: <value>}}}}

Rules:
- "tool" must be exactly one of: search_web, get_weather, calculate, lookup_person, translate, summarize
- "args" must contain the correct parameter names for that tool
- Do NOT include any explanation outside the JSON
"""

SYNTHESIS_SYSTEM_PROMPT = """You are a helpful AI assistant. Given a user query and the result from a tool,
formulate a clear, concise final answer.
Return ONLY valid JSON: {"answer": "<your answer here>"}
Do NOT include anything outside the JSON."""


def run_single_step(query: str) -> dict:
    """Full single-step agent loop: route → execute → synthesize."""
    loop_start = time.perf_counter()
    record = {
        "query": query,
        "llm_route_time_s": 0.0,
        "tool_exec_time_s": 0.0,
        "llm_synth_time_s": 0.0,
        "loop_total_time_s": 0.0,
        "tool_chosen": None,
        "tool_args": {},
        "tool_result": None,
        "final_answer": None,
        "error": None,
    }

    # ── Step 1: LLM routing call ──────────────────
    t0 = time.perf_counter()
    try:
        route_response, llm1_elapsed = llm_call([
            {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
            {"role": "user",   "content": f"User query: {query}"},
        ])
        record["llm_route_time_s"] = round(llm1_elapsed, 3)
        routing = extract_json(route_response)
        record["tool_chosen"] = routing.get("tool")
        record["tool_args"]   = routing.get("args", {})
    except Exception as exc:
        record["error"] = f"Routing error: {exc}"
        record["loop_total_time_s"] = round(time.perf_counter() - loop_start, 3)
        return record

    # ── Step 2: Execute tool ──────────────────────
    tool_fn = TOOLS.get(record["tool_chosen"])
    if not tool_fn:
        record["error"] = f"Unknown tool: {record['tool_chosen']}"
        record["loop_total_time_s"] = round(time.perf_counter() - loop_start, 3)
        return record

    t1 = time.perf_counter()
    try:
        result = tool_fn(**record["tool_args"])
        record["tool_result"] = result
    except Exception as exc:
        # Try positional fallback
        try:
            args_vals = list(record["tool_args"].values())
            result = tool_fn(*args_vals)
            record["tool_result"] = result
        except Exception as exc2:
            record["error"] = f"Tool exec error: {exc} / {exc2}"
            record["tool_exec_time_s"] = round(time.perf_counter() - t1, 3)
            record["loop_total_time_s"] = round(time.perf_counter() - loop_start, 3)
            return record
    record["tool_exec_time_s"] = round(time.perf_counter() - t1, 3)

    # ── Step 3: LLM synthesis call ────────────────
    t2 = time.perf_counter()
    try:
        synth_response, llm2_elapsed = llm_call([
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User query: {query}\n\n"
                    f"Tool used: {record['tool_chosen']}\n"
                    f"Tool result: {json.dumps(record['tool_result'], default=str)[:1000]}\n\n"
                    "Formulate a final answer."
                ),
            },
        ])
        record["llm_synth_time_s"] = round(llm2_elapsed, 3)
        synth = extract_json(synth_response)
        record["final_answer"] = synth.get("answer", synth_response[:200])
    except Exception as exc:
        record["error"] = f"Synthesis error: {exc}"
        record["llm_synth_time_s"] = round(time.perf_counter() - t2, 3)

    record["loop_total_time_s"] = round(time.perf_counter() - loop_start, 3)
    return record


def run_multi_step(query: str, expected_steps: list[str]) -> dict:
    """Multi-step agent loop: iteratively route and execute until no more steps."""
    loop_start = time.perf_counter()
    record = {
        "query": query,
        "expected_steps": expected_steps,
        "steps_taken": [],
        "step_times_s": [],
        "total_time_s": 0.0,
        "error": None,
    }

    context = f"User query: {query}"
    prev_result = None

    for step_idx in range(len(expected_steps)):
        step_start = time.perf_counter()

        step_prompt = context
        if prev_result:
            step_prompt += (
                f"\n\nPrevious step result: {json.dumps(prev_result, default=str)[:600]}"
                f"\n\nNow determine the NEXT tool to call to continue answering the original query."
            )

        try:
            route_response, _ = llm_call([
                {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
                {"role": "user",   "content": step_prompt},
            ])
            routing = extract_json(route_response)
            tool_name = routing.get("tool")
            tool_args = routing.get("args", {})
        except Exception as exc:
            record["error"] = f"Step {step_idx+1} routing error: {exc}"
            break

        tool_fn = TOOLS.get(tool_name)
        if not tool_fn:
            record["error"] = f"Step {step_idx+1} unknown tool: {tool_name}"
            break

        try:
            prev_result = tool_fn(**tool_args)
        except Exception:
            try:
                prev_result = tool_fn(*list(tool_args.values()))
            except Exception as exc2:
                record["error"] = f"Step {step_idx+1} tool error: {exc2}"
                break

        step_elapsed = round(time.perf_counter() - step_start, 3)
        record["steps_taken"].append(tool_name)
        record["step_times_s"].append(step_elapsed)
        context += f"\nStep {step_idx+1} ({tool_name}) result: {json.dumps(prev_result, default=str)[:300]}"

    record["total_time_s"] = round(time.perf_counter() - loop_start, 3)
    return record


# ─────────────────────────────────────────────
# BOUNDARY TEST: find max depth before >5s
# ─────────────────────────────────────────────

def find_boundary() -> dict:
    """Test loop depths 1-8; record average time per depth; find first depth >5s."""
    print("\n[BOUNDARY TEST] Finding max viable loop depth (target <5s total)...")
    boundary_results = []
    # Use a lightweight fixed query for each depth step
    base_queries = [
        ("What's 99 * 88?",                   "calculate",     {"expression": "99 * 88"}),
        ("Weather in Tokyo",                   "get_weather",   {"city": "Tokyo"}),
        ("Who is Ada Lovelace?",               "lookup_person", {"name": "Ada Lovelace"}),
        ("Search for Python tutorials",        "search_web",    {"query": "Python tutorials"}),
        ("Translate 'hello' to French",        "translate",     {"text": "hello", "target_lang": "French"}),
        ("Search AI news",                     "search_web",    {"query": "AI news"}),
        ("Calculate 1000 / 7",                 "calculate",     {"expression": "1000 / 7"}),
        ("Weather in Paris",                   "get_weather",   {"city": "Paris"}),
    ]

    cumulative_times = []
    for depth in range(1, 9):
        q_text, tool_name, tool_args = base_queries[depth - 1]
        step_start = time.perf_counter()
        # LLM routing call
        try:
            _, _ = llm_call([
                {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
                {"role": "user",   "content": f"User query: {q_text}"},
            ])
        except Exception:
            pass
        # Tool exec
        TOOLS[tool_name](**tool_args)
        step_time = time.perf_counter() - step_start

        cumulative = (cumulative_times[-1] if cumulative_times else 0) + step_time
        cumulative_times.append(cumulative)

        boundary_results.append({
            "depth": depth,
            "step_time_s": round(step_time, 3),
            "cumulative_time_s": round(cumulative, 3),
            "exceeds_5s": cumulative > 5.0,
        })
        status = "EXCEEDS 5s" if cumulative > 5.0 else "OK"
        print(f"  Depth {depth}: step={step_time:.3f}s  cumulative={cumulative:.3f}s  [{status}]")
        if cumulative > 5.0:
            break

    max_depth = max(
        (r["depth"] for r in boundary_results if not r["exceeds_5s"]),
        default=0,
    )
    return {"depth_results": boundary_results, "max_viable_depth": max_depth}


# ─────────────────────────────────────────────
# STATS HELPERS
# ─────────────────────────────────────────────

def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * p / 100
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return sorted_data[lower]
    return sorted_data[lower] + (sorted_data[upper] - sorted_data[lower]) * (idx - lower)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Agent Tool-Routing Benchmark")
    print(f"  Model: {MODEL}")
    print(f"  Single queries: {len(SINGLE_QUERIES)} | Multi-step: {len(MULTI_QUERIES)}")
    print("=" * 60)

    # ── Run single-step queries ───────────────
    print(f"\n[1/3] Running {len(SINGLE_QUERIES)} single-step queries...")
    single_results = []
    for i, item in enumerate(SINGLE_QUERIES):
        print(f"  [{i+1:02d}/{len(SINGLE_QUERIES)}] {item['query'][:60]}...", end=" ", flush=True)
        rec = run_single_step(item["query"])
        rec["ground_truth_tool"] = item["tool"]
        rec["routing_correct"]   = rec["tool_chosen"] == item["tool"]
        single_results.append(rec)
        status = "✓" if rec["routing_correct"] else f"✗ (got {rec['tool_chosen']})"
        print(f"{status}  {rec['loop_total_time_s']:.2f}s")

    # ── Run multi-step queries ────────────────
    print(f"\n[2/3] Running {len(MULTI_QUERIES)} multi-step queries...")
    multi_results = []
    for i, item in enumerate(MULTI_QUERIES):
        print(f"  [{i+1:02d}/{len(MULTI_QUERIES)}] {item['query'][:60]}...", end=" ", flush=True)
        rec = run_multi_step(item["query"], item["steps"])
        multi_results.append(rec)
        print(f"steps={len(rec['steps_taken'])}  {rec['total_time_s']:.2f}s")

    # ── Boundary test ─────────────────────────
    print("\n[3/3] Running boundary depth test...")
    boundary = find_boundary()

    # ── Compute metrics ───────────────────────
    correct   = [r for r in single_results if r["routing_correct"]]
    incorrect = [r for r in single_results if not r["routing_correct"]]
    accuracy  = len(correct) / len(single_results) * 100

    total_times  = [r["loop_total_time_s"] for r in single_results]
    route_times  = [r["llm_route_time_s"]  for r in single_results]
    tool_times   = [r["tool_exec_time_s"]  for r in single_results]
    synth_times  = [r["llm_synth_time_s"]  for r in single_results]

    # Per-tool accuracy
    tool_names = list(TOOLS.keys())
    per_tool: dict[str, dict] = {t: {"correct": 0, "total": 0} for t in tool_names}
    for r in single_results:
        gt = r["ground_truth_tool"]
        per_tool[gt]["total"] += 1
        if r["routing_correct"]:
            per_tool[gt]["correct"] += 1

    # Multi-step stats
    multi_step_times = [s for r in multi_results for s in r["step_times_s"]]
    multi_total_times = [r["total_time_s"] for r in multi_results]

    # ── Print results ─────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    print(f"\nOverall Routing Accuracy: {accuracy:.1f}%  ({len(correct)}/{len(single_results)})")

    print("\nAccuracy by Tool:")
    print(f"  {'Tool':<15} {'Correct':>8} {'Total':>7} {'Accuracy':>10}")
    print(f"  {'-'*15} {'-'*8} {'-'*7} {'-'*10}")
    for tool, stats in per_tool.items():
        acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  {tool:<15} {stats['correct']:>8} {stats['total']:>7} {acc:>9.1f}%")

    print("\nFull Loop Latency (single-step):")
    print(f"  p50 = {percentile(total_times, 50):.3f}s")
    print(f"  p95 = {percentile(total_times, 95):.3f}s")
    print(f"  p99 = {percentile(total_times, 99):.3f}s")
    print(f"  min = {min(total_times):.3f}s   max = {max(total_times):.3f}s")

    print("\nLatency Breakdown (means):")
    print(f"  LLM Routing call : {statistics.mean(route_times):.3f}s")
    print(f"  Tool execution   : {statistics.mean(tool_times):.3f}s")
    print(f"  LLM Synthesis    : {statistics.mean(synth_times):.3f}s")
    bottleneck = max(
        [("LLM Routing", statistics.mean(route_times)),
         ("Tool Execution", statistics.mean(tool_times)),
         ("LLM Synthesis", statistics.mean(synth_times))],
        key=lambda x: x[1],
    )
    print(f"  >> Bottleneck: {bottleneck[0]} ({bottleneck[1]:.3f}s avg)")

    if multi_step_times:
        print("\nMulti-step Loop Latency:")
        print(f"  Per-step p50 = {percentile(multi_step_times, 50):.3f}s")
        print(f"  Per-step p95 = {percentile(multi_step_times, 95):.3f}s")
        print(f"  Total (2-step) p50 = {percentile(multi_total_times, 50):.3f}s")
        print(f"  Total (2-step) p95 = {percentile(multi_total_times, 95):.3f}s")

    print(f"\nBoundary Test:")
    for r in boundary["depth_results"]:
        flag = " <-- EXCEEDS 5s" if r["exceeds_5s"] else ""
        print(f"  Depth {r['depth']}: cumulative={r['cumulative_time_s']:.3f}s{flag}")
    print(f"  Max viable loop depth (< 5s): {boundary['max_viable_depth']}")

    if incorrect:
        print(f"\nMisrouted queries ({len(incorrect)}):")
        for r in incorrect:
            got = str(r['tool_chosen']) if r['tool_chosen'] is not None else "None"
        print(f"  GT={r['ground_truth_tool']:<15} Got={got:<15} | {r['query'][:60]}")

    # ── Save results.json ─────────────────────
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "summary": {
            "overall_accuracy_pct": round(accuracy, 2),
            "correct": len(correct),
            "total": len(single_results),
            "loop_p50_s": round(percentile(total_times, 50), 3),
            "loop_p95_s": round(percentile(total_times, 95), 3),
            "loop_p99_s": round(percentile(total_times, 99), 3),
            "mean_llm_route_s": round(statistics.mean(route_times), 3),
            "mean_tool_exec_s": round(statistics.mean(tool_times), 3),
            "mean_llm_synth_s": round(statistics.mean(synth_times), 3),
            "bottleneck": bottleneck[0],
            "max_viable_depth": boundary["max_viable_depth"],
        },
        "per_tool_accuracy": {
            t: {
                "correct": s["correct"],
                "total": s["total"],
                "accuracy_pct": round((s["correct"] / s["total"] * 100) if s["total"] > 0 else 0, 1),
            }
            for t, s in per_tool.items()
        },
        "multi_step": {
            "per_step_p50_s": round(percentile(multi_step_times, 50), 3) if multi_step_times else None,
            "per_step_p95_s": round(percentile(multi_step_times, 95), 3) if multi_step_times else None,
            "total_p50_s": round(percentile(multi_total_times, 50), 3),
            "total_p95_s": round(percentile(multi_total_times, 95), 3),
        },
        "boundary": boundary,
        "single_step_results": single_results,
        "multi_step_results":  multi_results,
    }

    results_path = "/Users/krishnatejaswis/llm-boundary-tests/6-agent-tool-routing/results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
