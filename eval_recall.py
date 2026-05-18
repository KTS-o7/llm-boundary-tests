import os
#!/usr/bin/env python3
"""
Recall Evaluation Script — 6 LLM Benchmark Domains
Measures recall at top-1, top-3, recall degradation curves, and recall floors.
"""

import json
import time
import re
import statistics
import urllib.request
import urllib.error
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
API_URL = "https://ai.shenthar.me/v1/chat/completions"
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL   = "taalas-llama3.1-8b"
BASE    = "/Users/krishnatejaswis/llm-boundary-tests"

# ─────────────────────────────────────────────
# API helper
# ─────────────────────────────────────────────
def call_api(messages: List[Dict], temperature: float = 0.0, max_tokens: int = 512) -> Optional[str]:
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
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"    [API ERROR] {e}")
        return None


def load_results(domain_dir: str) -> Dict:
    path = f"{BASE}/{domain_dir}/results.json"
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Domain 1 — Classification Pipeline
# ─────────────────────────────────────────────
CLASSIFICATION_REQUIRED_FIELDS = ["sentiment", "category", "urgency"]

CLASSIFICATION_PROBES = [
    # (text, complexity_level)
    # Level 1 — simple, single-signal
    ("I love this product! Works perfectly.", 1),
    ("The package arrived broken. Very disappointed.", 1),
    ("Need help resetting my password urgently.", 1),
    ("Great customer service, 5 stars!", 1),
    ("My order is 3 weeks late, no response from support.", 1),
    # Level 2 — moderate mixed signals
    ("Camera is excellent but battery life is poor. Overall decent purchase.", 2),
    ("Product looks great but the instruction manual is confusing. Arrived on time.", 2),
    ("Billing error on invoice #INV-2024-5523. Please refund the duplicate charge.", 2),
    ("Love the app UI but crashes every time I close it. Support ticket open.", 2),
    ("Quality is mediocre for the price. Nothing special but functional.", 2),
    # Level 3 — complex multi-signal
    ("URGENT: Production DB down, $50k/hour revenue loss. Also wanted to say the API docs are outdated.", 3),
    ("Long-time customer extremely frustrated. Three failed deliveries, billing errors, no response from team. "
     "Cancelling subscription unless resolved by EOD.", 3),
    ("The firmware update bricked my device. While I appreciate the new features on other models, "
     "this is unacceptable. Need immediate replacement and partial refund for downtime losses.", 3),
    ("Neutral opinion: product does what it claims, pricing is competitive. Some UI quirks. "
     "Support response time could be better. Overall satisfied but not impressed.", 3),
    ("Mixed feelings: exceptional hardware, terrible software. Battery explodes in heat per reviews. "
     "Legal team may be interested in safety issue. Otherwise love the design.", 3),
    # Level 4 — highly complex / ambiguous
    ("Dear support, I'm writing to express BOTH appreciation and frustration. The premium tier "
     "features are fantastic and justify the cost, but last week's outage caused real business harm. "
     "My team lost 4 hours of work. Compensation would be appropriate. Not urgent but important.", 4),
    ("This is simultaneously a product review AND a bug report. As a review: 4/5 stars, "
     "great build quality. As a bug report: WiFi disconnects every 45 minutes on firmware v2.3.1. "
     "The issue is intermittent but reproducible. No urgency but wanted to document it officially.", 4),
    ("Regarding subscription renewal: I want to DOWNGRADE not cancel, but your portal only shows "
     "cancel option. Meanwhile the auto-renew charged full price. Refund the difference. "
     "Also please pass feedback to product team that the UX is confusing for downgrades.", 4),
    # Level 5 — maximum complexity
    ("This message concerns three distinct issues: (1) A CRITICAL billing discrepancy of $2,400 "
     "on account #AC-88123 requiring immediate resolution before month-end close; (2) A general "
     "feedback point that your onboarding flow improved significantly — kudos to the product team; "
     "(3) A moderate bug where exported CSV files lose formatting when opened in Excel versions < 2019. "
     "Please route accordingly and confirm receipt of all three items.", 5),
    ("Following up on ticket #TKT-44892 from last month: the refund was processed but the wrong "
     "amount ($45 vs $145 owed). Meanwhile our team loves the new dashboard features — honestly "
     "the best UX update in years. One more thing: production API latency jumped to 8s P99 this "
     "morning around 09:00 UTC. Not critical yet but trending toward SLA breach. Please advise.", 5),
]

def eval_domain1_classification() -> Dict:
    print("\n[1] Classification Pipeline — Recall Evaluation")
    print("  Probing required fields: sentiment, category, urgency")

    system = (
        "You are a text classifier. Always respond with valid JSON containing exactly these fields: "
        '"sentiment" (positive/negative/neutral/mixed), '
        '"category" (product_review/support_ticket/billing/feedback/bug_report), '
        '"urgency" (low/medium/high/critical). '
        "No extra keys. No prose."
    )

    results_by_complexity = {}
    all_recalls = []

    for text, complexity in CLASSIFICATION_PROBES:
        label = f"L{complexity}"
        if label not in results_by_complexity:
            results_by_complexity[label] = []

        resp = call_api([
            {"role": "system", "content": system},
            {"role": "user", "content": f"Classify this text:\n\n{text}"},
        ], temperature=0.0, max_tokens=80)

        if resp is None:
            fields_found = 0
        else:
            try:
                # extract JSON even if wrapped in markdown
                m = re.search(r'\{.*\}', resp, re.DOTALL)
                obj = json.loads(m.group()) if m else {}
                fields_found = sum(1 for f in CLASSIFICATION_REQUIRED_FIELDS if f in obj)
            except Exception:
                fields_found = 0

        recall = fields_found / len(CLASSIFICATION_REQUIRED_FIELDS)
        results_by_complexity[label].append(recall)
        all_recalls.append((complexity, recall))
        time.sleep(0.05)

    # recall@1: mean over simple (L1)
    recall_at_1 = statistics.mean(results_by_complexity.get("L1", [0]))
    # recall@3: mean over moderate (L1+L2+L3)
    r3_pool = (results_by_complexity.get("L1", []) +
               results_by_complexity.get("L2", []) +
               results_by_complexity.get("L3", []))
    recall_at_3 = statistics.mean(r3_pool) if r3_pool else 0.0

    # degradation curve
    curve = {}
    for lvl in ["L1", "L2", "L3", "L4", "L5"]:
        vals = results_by_complexity.get(lvl, [])
        curve[lvl] = round(statistics.mean(vals), 4) if vals else None

    # recall floor: first complexity where mean recall < 0.5
    recall_floor = "never"
    for lvl in ["L1", "L2", "L3", "L4", "L5"]:
        v = curve.get(lvl)
        if v is not None and v < 0.5:
            recall_floor = lvl
            break

    print(f"  Recall@1 (L1): {recall_at_1:.3f}")
    print(f"  Recall@3 (L1-3): {recall_at_3:.3f}")
    print(f"  Degradation curve: {curve}")
    print(f"  Recall floor: {recall_floor}")

    return {
        "recall_at_1": round(recall_at_1, 4),
        "recall_at_3": round(recall_at_3, 4),
        "recall_floor": recall_floor,
        "degradation_curve": curve,
        "notes": "Recall = fraction of 3 required JSON fields (sentiment/category/urgency) returned",
        "raw": results_by_complexity,
    }


# ─────────────────────────────────────────────
# Domain 2 — Structured Extraction
# ─────────────────────────────────────────────
GT_ENTITIES = {
    "persons":   ["eleanor voss", "marcus chen", "priya nair", "thomas hartwell",
                  "sofia delgado", "james okafor", "linda beaumont", "raj patel"],
    "companies": ["nexagen biotech", "meridian capital partners", "solaris energy group",
                  "fortridge logistics", "quantum dynamics inc", "blue harbor ventures"],
    "locations": ["san francisco", "singapore", "berlin", "nairobi", "toronto",
                  "austin", "amsterdam", "boston"],
    "dates":     ["march 14 2024", "july 2 2024", "september 30 2024",
                  "october 15 2024", "december 1 2024"],
    "amounts":   ["$2.4 billion", "$850 million", "$340 million",
                  "$17.5 million", "$4.2 billion"],
}

ALL_GT = []
for cat, items in GT_ENTITIES.items():
    for item in items:
        ALL_GT.append((cat, item))

TOTAL_GT = len(ALL_GT)  # 32


def _parse_extraction(resp: str) -> Dict[str, List[str]]:
    """Parse JSON extraction from model response."""
    if not resp:
        return {}
    try:
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return {}


def _entities_found(extracted: Dict, gt_entities: Dict) -> int:
    """Count GT entities found in extraction (case-insensitive, partial match allowed)."""
    found = 0
    for cat, items in gt_entities.items():
        extracted_cat = [str(e).lower() for e in extracted.get(cat, [])]
        for item in items:
            item_l = item.lower()
            for e in extracted_cat:
                if item_l in e or e in item_l:
                    found += 1
                    break
    return found


def eval_domain2_extraction() -> Dict:
    print("\n[2] Structured Extraction — Recall Evaluation")
    print(f"  Ground truth: {TOTAL_GT} entities across 5 categories")

    # Reuse existing results.json for the recall trend across word counts
    existing = load_results("2-structured-extraction")
    existing_results = existing.get("results", [])

    existing_recall_by_words = {}
    for r in existing_results:
        words = r.get("target_words", 0)
        recall = r.get("recall", 0.0)
        existing_recall_by_words[words] = recall

    # Fresh live probes: 5 complexity levels x 5 repetitions = 25 probes
    # We test recall with increasingly long passages (simulated by how many entities we ask about)
    PROBE_PASSAGES = [
        # (word_count_approx, passage_snippet)
        (150, "Eleanor Voss, CEO of NexAgen Biotech, announced a $2.4 billion deal on March 14 2024 "
              "with Meridian Capital Partners in San Francisco."),
        (300, "NexAgen Biotech CEO Eleanor Voss met Marcus Chen from Meridian Capital Partners in San Francisco "
              "on March 14 2024 to finalise a $2.4 billion deal. Thomas Hartwell, CFO, confirmed the terms. "
              "The company also has a presence in Berlin and Singapore with Priya Nair leading APAC operations."),
        (500, "Eleanor Voss and Marcus Chen, along with Priya Nair (APAC), Thomas Hartwell (CFO), "
              "Sofia Delgado (Legal), James Okafor (Ops), Linda Beaumont (Research), and Raj Patel (Engineering) "
              "gathered in San Francisco on March 14 2024 for the $2.4 billion Series D close. "
              "Meridian Capital Partners led. Solaris Energy Group and Fortridge Logistics attended. "
              "Quantum Dynamics Inc and Blue Harbor Ventures also participated. Locations: Berlin, Nairobi, Singapore, "
              "Toronto, Austin, Amsterdam, Boston. Amounts: $850 million, $340 million, $17.5 million, $4.2 billion. "
              "Dates: July 2 2024, September 30 2024, October 15 2024, December 1 2024."),
        (800, "This lengthy report covers the activities of NexAgen Biotech led by Eleanor Voss based in San Francisco. "
              "The $2.4 billion Series D was closed on March 14 2024 by Meridian Capital Partners under Marcus Chen. "
              "Priya Nair heads APAC from Singapore. Thomas Hartwell manages finances from Berlin. "
              "Sofia Delgado oversees legal in Austin. James Okafor runs operations in Nairobi. "
              "Linda Beaumont leads research from Boston. Raj Patel heads engineering in Toronto. "
              "Key investors include Solaris Energy Group, Fortridge Logistics, Quantum Dynamics Inc, Blue Harbor Ventures. "
              "Additional cities include Amsterdam. Financial milestones: $850 million Q2 runway, $340 million R&D, "
              "$17.5 million capex, $4.2 billion total market cap. Important dates: July 2 2024 board meeting, "
              "September 30 2024 fiscal close, October 15 2024 product launch, December 1 2024 annual review. " * 2),
        (1200, "Extended narrative — NexAgen Biotech Series D deep-dive. Eleanor Voss CEO San Francisco. "
               "Marcus Chen Meridian Capital Partners. Priya Nair Singapore. Thomas Hartwell Berlin. "
               "Sofia Delgado Austin. James Okafor Nairobi. Linda Beaumont Boston. Raj Patel Toronto Amsterdam. "
               "Solaris Energy Group, Fortridge Logistics, Quantum Dynamics Inc, Blue Harbor Ventures. "
               "$2.4 billion March 14 2024. $850 million July 2 2024. $340 million September 30 2024. "
               "$17.5 million October 15 2024. $4.2 billion December 1 2024. " * 5),
    ]

    system = (
        'Extract named entities. Respond with JSON only: '
        '{"persons": [...], "companies": [...], "locations": [...], "dates": [...], "amounts": [...]}. '
        'All lowercase. No prose.'
    )

    live_results = []
    for words, passage in PROBE_PASSAGES:
        recalls_for_passage = []
        for _ in range(3):  # 3 repeats per passage length
            resp = call_api([
                {"role": "system", "content": system},
                {"role": "user", "content": f"Extract all named entities from this text:\n\n{passage}"},
            ], temperature=0.0, max_tokens=512)
            extracted = _parse_extraction(resp)
            found = _entities_found(extracted, GT_ENTITIES)
            recall = found / TOTAL_GT
            recalls_for_passage.append(recall)
            time.sleep(0.05)

        mean_recall = statistics.mean(recalls_for_passage)
        live_results.append({"words": words, "mean_recall": round(mean_recall, 4),
                              "per_run": [round(r, 4) for r in recalls_for_passage]})

    # recall@1: recall at shortest text (top-1 single-sentence)
    recall_at_1 = live_results[0]["mean_recall"] if live_results else 0.0
    # recall@3: recall averaged over first 3 complexity levels
    recall_at_3 = statistics.mean([r["mean_recall"] for r in live_results[:3]]) if live_results else 0.0

    # degradation curve from existing results (word count vs recall)
    curve = {str(k): round(v, 4) for k, v in sorted(existing_recall_by_words.items())}
    # augment with live
    for r in live_results:
        curve[f"live_{r['words']}w"] = r["mean_recall"]

    # recall floor from existing: first word count where recall < 0.5
    recall_floor = "never"
    for words in sorted(existing_recall_by_words.keys()):
        if existing_recall_by_words[words] < 0.5:
            recall_floor = f"{words}w"
            break

    print(f"  Recall@1 (150w): {recall_at_1:.3f}")
    print(f"  Recall@3 (avg 150-500w): {recall_at_3:.3f}")
    print(f"  Existing recall curve: {existing_recall_by_words}")
    print(f"  Recall floor: {recall_floor}")

    return {
        "recall_at_1": round(recall_at_1, 4),
        "recall_at_3": round(recall_at_3, 4),
        "recall_floor": recall_floor,
        "degradation_curve": curve,
        "notes": f"Recall = fraction of {TOTAL_GT} GT entities found. Low recall at short docs (entities absent), peaks ~3500w",
        "live_probes": live_results,
        "existing_recall_by_words": {str(k): v for k, v in existing_recall_by_words.items()},
    }


# ─────────────────────────────────────────────
# Domain 3 — Routing Filter
# ─────────────────────────────────────────────
SIMPLE_ROUTING_PROBES = [
    # Clearly SIMPLE — should be routed to fast model
    ("What is 7 × 6?", "SIMPLE", 1),
    ("What color is the sky?", "SIMPLE", 1),
    ("How many days are in a week?", "SIMPLE", 1),
    ("What is the capital of Japan?", "SIMPLE", 1),
    ("Is Python a programming language? Yes or no.", "SIMPLE", 1),
    # Borderline SIMPLE/MEDIUM
    ("What is the boiling point of ethanol?", "SIMPLE", 2),
    ("Name the three states of matter.", "SIMPLE", 2),
    ("Who invented the World Wide Web?", "SIMPLE", 2),
    ("What does RAM stand for?", "SIMPLE", 2),
    ("What year was Python first released?", "SIMPLE", 2),
    # Deliberately tricky phrasing of SIMPLE questions
    ("I need to know: what does the acronym 'CPU' expand to?", "SIMPLE", 3),
    ("Could you tell me, in simple terms, how many planets are in our solar system?", "SIMPLE", 3),
    ("Quick question: is the speed of light finite or infinite?", "SIMPLE", 3),
    ("As a curious student, I want to understand: what is the SI unit of temperature?", "SIMPLE", 3),
    ("In a single sentence, what is the Pythagorean theorem?", "SIMPLE", 3),
    # MEDIUM — should NOT be routed to fast model
    ("Explain the time complexity of bubble sort and why it matters.", "MEDIUM", 4),
    ("What are the trade-offs between SQL and NoSQL databases?", "MEDIUM", 4),
    ("Describe how garbage collection works in Python.", "MEDIUM", 4),
    ("Why does tail recursion matter for functional programming?", "MEDIUM", 4),
    ("Explain the difference between authentication and authorisation.", "MEDIUM", 4),
    # COMPLEX — should NOT be routed to fast model
    ("Design a distributed caching layer for a read-heavy e-commerce site.", "COMPLEX", 5),
    ("Analyze the trade-offs of event sourcing vs traditional CRUD in microservices.", "COMPLEX", 5),
    ("How would you architect a system to handle 1M concurrent websocket connections?", "COMPLEX", 5),
    ("Compare Byzantine fault tolerance to crash fault tolerance in consensus algorithms.", "COMPLEX", 5),
    ("Evaluate ethical concerns of LLM deployment in high-stakes medical diagnosis.", "COMPLEX", 5),
]

ROUTING_SYSTEM = (
    "You are a routing classifier. Classify the complexity of the user query as SIMPLE, MEDIUM, or COMPLEX. "
    "SIMPLE: factual lookups, simple math, yes/no. "
    "MEDIUM: explanations requiring some reasoning, multi-step problems. "
    "COMPLEX: design, analysis, deep domain expertise, ethical reasoning. "
    'Respond with JSON only: {"tier": "SIMPLE"|"MEDIUM"|"COMPLEX", "reason": "one sentence"}. No prose.'
)


def eval_domain3_routing() -> Dict:
    print("\n[3] Routing Filter — Recall Evaluation")
    print("  Measuring recall of SIMPLE tier identification")

    existing = load_results("3-routing-filter")
    # From confusion matrix: SIMPLE tier: 19 correct out of 20 → recall = 0.95
    cm = existing.get("confusion_matrix", {})
    simple_row = cm.get("SIMPLE", {})
    simple_correct = simple_row.get("SIMPLE", 0)
    simple_total = sum(simple_row.values())
    existing_simple_recall = simple_correct / simple_total if simple_total else 0.0

    print(f"  Existing SIMPLE recall: {existing_simple_recall:.3f} ({simple_correct}/{simple_total})")

    # Live probes
    results_by_complexity = {}
    for query, true_tier, complexity in SIMPLE_ROUTING_PROBES:
        label = f"L{complexity}"
        if label not in results_by_complexity:
            results_by_complexity[label] = {"total": 0, "simple_correct": 0, "simple_tp": 0, "simple_fn": 0}

        resp = call_api([
            {"role": "system", "content": ROUTING_SYSTEM},
            {"role": "user", "content": query},
        ], temperature=0.0, max_tokens=80)

        predicted_tier = None
        if resp:
            try:
                m = re.search(r'\{.*\}', resp, re.DOTALL)
                obj = json.loads(m.group()) if m else {}
                predicted_tier = obj.get("tier", "").upper()
            except Exception:
                pass

        results_by_complexity[label]["total"] += 1

        if true_tier == "SIMPLE":
            results_by_complexity[label]["simple_tp"] = results_by_complexity[label].get("simple_tp", 0)
            if predicted_tier == "SIMPLE":
                results_by_complexity[label]["simple_tp"] = results_by_complexity[label]["simple_tp"] + 1
            else:
                results_by_complexity[label]["simple_fn"] = results_by_complexity[label].get("simple_fn", 0) + 1

        time.sleep(0.05)

    # For SIMPLE recall: TP / (TP + FN) for queries where true_tier == SIMPLE
    simple_probes = [(q, t, c) for q, t, c in SIMPLE_ROUTING_PROBES if t == "SIMPLE"]

    # Recompute per complexity level for SIMPLE queries
    simple_by_level = {}
    for query, true_tier, complexity in SIMPLE_ROUTING_PROBES:
        if true_tier != "SIMPLE":
            continue
        label = f"L{complexity}"
        if label not in simple_by_level:
            simple_by_level[label] = {"correct": 0, "total": 0}

        resp = call_api([
            {"role": "system", "content": ROUTING_SYSTEM},
            {"role": "user", "content": query},
        ], temperature=0.0, max_tokens=80)

        predicted = None
        if resp:
            try:
                m = re.search(r'\{.*\}', resp, re.DOTALL)
                obj = json.loads(m.group()) if m else {}
                predicted = obj.get("tier", "").upper()
            except Exception:
                pass

        simple_by_level[label]["total"] += 1
        if predicted == "SIMPLE":
            simple_by_level[label]["correct"] += 1

        time.sleep(0.05)

    # recall@1: simple L1 queries
    r1_data = simple_by_level.get("L1", {"correct": 0, "total": 1})
    recall_at_1 = r1_data["correct"] / r1_data["total"] if r1_data["total"] else 0.0

    # recall@3: L1+L2+L3 simple queries
    r3_correct = sum(simple_by_level.get(f"L{i}", {}).get("correct", 0) for i in [1, 2, 3])
    r3_total = sum(simple_by_level.get(f"L{i}", {}).get("total", 0) for i in [1, 2, 3])
    recall_at_3 = r3_correct / r3_total if r3_total else 0.0

    # degradation curve
    curve = {}
    for lvl in ["L1", "L2", "L3"]:
        d = simple_by_level.get(lvl, {})
        t = d.get("total", 0)
        c = d.get("correct", 0)
        curve[lvl] = round(c / t, 4) if t else None

    # recall floor
    recall_floor = "never"
    for lvl in ["L1", "L2", "L3"]:
        v = curve.get(lvl)
        if v is not None and v < 0.5:
            recall_floor = lvl
            break

    print(f"  Live Recall@1 (L1 SIMPLE): {recall_at_1:.3f}")
    print(f"  Live Recall@3 (L1-3 SIMPLE): {recall_at_3:.3f}")
    print(f"  Degradation: {curve}")
    print(f"  Recall floor: {recall_floor}")

    return {
        "recall_at_1": round(recall_at_1, 4),
        "recall_at_3": round(recall_at_3, 4),
        "recall_floor": recall_floor,
        "degradation_curve": curve,
        "existing_simple_recall": round(existing_simple_recall, 4),
        "notes": "Recall = fraction of SIMPLE queries correctly identified as SIMPLE (TP rate for SIMPLE tier)",
        "simple_by_level": {k: v for k, v in simple_by_level.items()},
    }


# ─────────────────────────────────────────────
# Domain 4 — Chunked Documents
# ─────────────────────────────────────────────

# Canonical ground-truth entities from the NexaGen Corp annual report
CHUNK_GT_ENTITIES = [
    "NexaGen Corp", "Margaret Holbrook", "Victor Okafor", "NexaRoute 3.0",
    "NexaSense", "NexaInsight", "NexaLLM", "Daniel Rios", "Priya Nair",
    "Dr. Lin Wei", "Sophie Lefebvre", "Amanda Thornton", "Raymond Chu",
    "James Calloway", "Yolanda Ferreira", "Claudia Martinovic",
    "$4.87 billion", "$3.96 billion", "$612 million", "$489 million",
    "Singapore", "São Paulo", "Frankfurt", "Dubai", "Nairobi",
    "Guadalajara", "Penang", "Goldman Sachs", "Deutsche Bank",
    "RetailGiant Inc", "Frontier Pharmaceuticals", "BlueStar Automotive Group",
    "Pacific Harvest Foods", "Meridian Construction Holdings",
    "SunPeak Energy Partners", "Port of Rotterdam", "Mitsui & Co",
    "PortoBrasil S.A.", "AWS", "Azure", "Google Cloud",
]
TOTAL_CHUNK_GT = len(CHUNK_GT_ENTITIES)


def eval_domain4_chunked() -> Dict:
    print("\n[4] Chunked Documents — Recall Evaluation")
    print(f"  Ground-truth: {TOTAL_CHUNK_GT} key entities from NexaGen Corp report")

    existing = load_results("4-chunked-documents")

    # Extract entities found per strategy from existing results
    strategy_recalls = []
    for strategy in existing.get("all_results", []):
        chunk_words = strategy.get("chunk_words", 0)
        overlap = strategy.get("overlap_words", 0)
        mode = strategy.get("mode", "")
        unique_entities = [str(e).lower() for e in strategy.get("unique_entities", [])]

        found = 0
        for gt in CHUNK_GT_ENTITIES:
            gt_l = gt.lower()
            for ue in unique_entities:
                if gt_l in ue or ue in gt_l:
                    found += 1
                    break

        recall = found / TOTAL_CHUNK_GT
        strategy_recalls.append({
            "chunk_words": chunk_words,
            "overlap_words": overlap,
            "mode": mode,
            "entities_found": found,
            "recall": round(recall, 4),
        })

    # Live probes: ask the model to extract entities from short to long passages
    # Use progressively longer excerpts from the report
    EXCERPTS = [
        # 100w
        ("100w",
         "NexaGen Corp reported annual revenue of $4.87 billion, up 23% from $3.96 billion in FY2023. "
         "CEO Margaret Holbrook announced NexaRoute 3.0 expansion into Southeast Asia and Latin America. "
         "Net income was $612 million. Victor Okafor was appointed Board Chairman."),
        # 250w
        ("250w",
         "NexaGen Corp FY2024 Highlights: Revenue $4.87 billion (+23%), net income $612 million. "
         "Margaret Holbrook led strategic pivot. NexaRoute 3.0 deployed in Singapore, São Paulo, Frankfurt, Dubai, "
         "Guadalajara, Penang. Victor Okafor joined as Board Chairman. "
         "NexaSense IoT platform: 18 billion events/day. NexaInsight Commodities served 340 agribusinesses. "
         "NexaLLM trained on 4.2 trillion tokens. Raymond Chu led R&D ($487M budget). "
         "Dr. Lin Wei: Q3 2026 product launch. Amanda Thornton: $78M US federal contract. "
         "Sophie Lefebvre: EMEA region 4,200 employees. James Calloway: Phoenix, Arizona data center. "
         "Yolanda Ferreira: Project Connect. Claudia Martinovic: VP Risk. "
         "Partners: Goldman Sachs, Deutsche Bank. Clients: RetailGiant Inc, Frontier Pharmaceuticals, "
         "BlueStar Automotive Group, Pacific Harvest Foods, Meridian Construction Holdings. "
         "SunPeak Energy Partners: renewable deal. Port of Rotterdam: smart-grid. "
         "Mitsui & Co: Japan-Southeast Asia. PortoBrasil S.A.: Port of Santos. "
         "Cloud: AWS, Azure, Google Cloud. Net income: $612M. EPS: $0.22. "
         "FY2025 guidance: $5.8–6.1 billion revenue."),
        # 600w
        ("600w",
         "NexaGen Corp Annual Report FY2024 Summary. NexaGen Corp headquartered Austin Texas CEO Margaret Holbrook "
         "Board Chairman Victor Okafor. Revenue $4.87 billion +23% from $3.96 billion FY2023. "
         "Net income $612 million EPS $0.22 vs $0.18 prior year. "
         "NexaRoute 3.0: supply-chain platform deployed at Singapore, São Paulo, Frankfurt, Dubai, Guadalajara Mexico, Penang Malaysia. "
         "RetailGiant Inc: North America contract. Frontier Pharmaceuticals: $340M deal. "
         "BlueStar Automotive Group: Germany France Spain. Pacific Harvest Foods: Australia New Zealand. "
         "Meridian Construction Holdings: Middle East. NexaSense 2.1: 18 billion events/day 11 billion tags. "
         "Dr. Lin Wei: Q3 2026. Port of Rotterdam Authority: 4,000 IoT sensors $220M. "
         "Nairobi Metropolitan Transport Authority. EirGrid Ireland. HeidelbergTech Manufacturing. "
         "NexaInsight: $820M revenue +34% FY2024. Standard Chartered Bank $500M. "
         "NexaInsight Commodities: 340 agribusinesses. India ILMI PM GatiShakti $102M. "
         "Sophie Lefebvre: EMEA 4,200 staff. UK Germany EirGrid. "
         "Amanda Thornton: US federal $78M Phoenix AZ Columbus OH LEED Platinum SunPeak Energy Partners. "
         "James Calloway: US Defense Logistics Agency. "
         "Takeshi Yamamoto APAC $150M Mitsui & Co Southeast Asia Vietnam Indonesia. "
         "PortoBrasil S.A. Brazil AgroPará CerradoGrain Rio Verde Commodities Port of Santos $65M. "
         "Raymond Chu R&D $487M NexaLLM 4.2 trillion tokens AWS Azure Google Cloud August 2024 ISO27001. "
         "Yolanda Ferreira Project Connect Bangalore Austin Frankfurt São Paulo 2027 40%. "
         "Claudia Martinovic VP Risk. Zebra Technologies Blue Yonder Manhattan Associates Oracle SCM Cloud "
         "DeepRoute Analytics Alphabet. Goldman Sachs Deutsche Bank green bond $1.2B BBB+ Baa1. "
         "Forbes Global 2000 #312 Gartner Magic Quadrant Leader Fortune Best Places ISO 50001. "
         "FY2025 guidance $5.8-6.1B revenue 19-25% EBITDA margin capex $550-650M. " * 2),
    ]

    live_recalls = []
    extr_sys = (
        'Extract all named entities (companies, people, products, locations, amounts, dates). '
        'Respond with JSON: {"entities": [...]}. All entity strings, no categories needed. No prose.'
    )

    for label, passage in EXCERPTS:
        run_recalls = []
        for _ in range(3):
            resp = call_api([
                {"role": "system", "content": extr_sys},
                {"role": "user", "content": f"Extract all entities:\n\n{passage}"},
            ], temperature=0.0, max_tokens=1024)

            extracted = []
            if resp:
                try:
                    m = re.search(r'\{.*\}', resp, re.DOTALL)
                    obj = json.loads(m.group()) if m else {}
                    extracted = [str(e).lower() for e in obj.get("entities", [])]
                except Exception:
                    pass

            found = 0
            for gt in CHUNK_GT_ENTITIES:
                gt_l = gt.lower()
                for e in extracted:
                    if gt_l in e or e in gt_l:
                        found += 1
                        break

            run_recalls.append(found / TOTAL_CHUNK_GT)
            time.sleep(0.05)

        live_recalls.append({
            "label": label,
            "mean_recall": round(statistics.mean(run_recalls), 4),
            "runs": [round(r, 4) for r in run_recalls],
        })

    # recall@1: 100-word excerpt
    recall_at_1 = live_recalls[0]["mean_recall"] if live_recalls else 0.0
    # recall@3: avg across all excerpt lengths
    recall_at_3 = statistics.mean([r["mean_recall"] for r in live_recalls]) if live_recalls else 0.0

    # degradation curve from existing chunking strategies (by chunk size)
    curve = {}
    for sr in sorted(strategy_recalls, key=lambda x: x["chunk_words"]):
        key = f"{sr['chunk_words']}w_{sr['overlap_words']}ov_{sr['mode']}"
        curve[key] = sr["recall"]

    # best recall across strategies
    best_recall = max(sr["recall"] for sr in strategy_recalls) if strategy_recalls else 0.0
    # recall floor: smallest chunk_words where recall < 0.5
    recall_floor = "never"
    for sr in sorted(strategy_recalls, key=lambda x: x["chunk_words"], reverse=True):
        if sr["recall"] < 0.5:
            recall_floor = f"{sr['chunk_words']}w"
            break

    print(f"  Recall@1 (100w excerpt): {recall_at_1:.3f}")
    print(f"  Recall@3 (avg excerpts): {recall_at_3:.3f}")
    print(f"  Best chunking recall: {best_recall:.3f}")
    print(f"  Recall floor: {recall_floor}")

    return {
        "recall_at_1": round(recall_at_1, 4),
        "recall_at_3": round(recall_at_3, 4),
        "recall_floor": recall_floor,
        "best_chunking_recall": round(best_recall, 4),
        "degradation_curve": curve,
        "notes": f"Recall = fraction of {TOTAL_CHUNK_GT} GT entities found. 500w/50ov parallel is optimal strategy.",
        "strategy_recalls": strategy_recalls,
        "live_probes": live_recalls,
    }


# ─────────────────────────────────────────────
# Domain 5 — Realtime Interactive
# ─────────────────────────────────────────────

CATEGORY_EXPECTED_FIELDS = {
    "PING":     ["reply"],
    "ONE_LINER": ["answer"],
    "SHORT":    ["answer", "explanation"],
    "MEDIUM":   ["answer", "explanation", "examples"],
    "LONG":     ["answer", "explanation", "examples", "caveats", "summary"],
}

REALTIME_PROBES = {
    "PING": [
        ("Say only 'pong'", {"reply": "pong"}),
        ("Respond with a single word: hello", {"reply": "hello"}),
        ("Reply with just 'ok'", {"reply": "ok"}),
        ("Respond with the number 42", {"reply": "42"}),
        ("Say only 'yes'", {"reply": "yes"}),
    ],
    "ONE_LINER": [
        ("In one sentence, what is the speed of light?", ["answer"]),
        ("In one sentence, define photosynthesis.", ["answer"]),
        ("In one sentence, what is machine learning?", ["answer"]),
        ("In one sentence, what is DNA?", ["answer"]),
        ("In one sentence, who was Albert Einstein?", ["answer"]),
    ],
    "SHORT": [
        ("In 2-3 sentences with an explanation, what is the greenhouse effect?", ["answer", "explanation"]),
        ("In 2-3 sentences with an explanation, what is a hash function?", ["answer", "explanation"]),
        ("Briefly explain TCP handshake (answer + explanation).", ["answer", "explanation"]),
        ("Briefly explain what a CPU does (answer + explanation).", ["answer", "explanation"]),
        ("Briefly explain Moore's law (answer + explanation).", ["answer", "explanation"]),
    ],
    "MEDIUM": [
        ("Explain recursion with answer, explanation, and two examples.", ["answer", "explanation", "examples"]),
        ("Explain REST APIs with answer, explanation, and examples.", ["answer", "explanation", "examples"]),
        ("Explain gradient descent with answer, explanation, and examples.", ["answer", "explanation", "examples"]),
        ("Explain the OSI model with answer, explanation, and examples.", ["answer", "explanation", "examples"]),
        ("Explain Docker containers with answer, explanation, examples.", ["answer", "explanation", "examples"]),
    ],
    "LONG": [
        ("Give a comprehensive explanation of blockchain with answer, explanation, examples, caveats, and summary.",
         ["answer", "explanation", "examples", "caveats", "summary"]),
        ("Comprehensively explain the CAP theorem with all five fields: answer, explanation, examples, caveats, summary.",
         ["answer", "explanation", "examples", "caveats", "summary"]),
        ("Thoroughly explain machine learning bias with answer, explanation, examples, caveats, summary.",
         ["answer", "explanation", "examples", "caveats", "summary"]),
        ("Comprehensively explain microservices architecture: answer, explanation, examples, caveats, summary.",
         ["answer", "explanation", "examples", "caveats", "summary"]),
        ("Thoroughly explain zero-knowledge proofs: answer, explanation, examples, caveats, summary.",
         ["answer", "explanation", "examples", "caveats", "summary"]),
    ],
}

REALTIME_SYSTEM = (
    "You always respond in valid JSON. Include exactly the fields requested in the user's prompt. "
    "No additional fields unless asked. No prose outside JSON."
)


def _field_recall(response: str, required_fields: List[str]) -> float:
    """Returns fraction of required fields present in JSON response."""
    if not response:
        return 0.0
    try:
        m = re.search(r'\{.*\}', response, re.DOTALL)
        obj = json.loads(m.group()) if m else {}
        found = sum(1 for f in required_fields if f in obj and obj[f])
        return found / len(required_fields)
    except Exception:
        # Also try to find field keywords in raw text
        found = sum(1 for f in required_fields if f in response.lower())
        return found / len(required_fields)


def eval_domain5_realtime() -> Dict:
    print("\n[5] Realtime Interactive — Recall Evaluation")
    print("  Measuring field recall per response category")

    existing = load_results("5-realtime-interactive")

    # All existing samples had ok=True → baseline latency fine
    existing_ok_rate = {}
    for cat, data in existing.get("categories", {}).items():
        samples = data.get("samples", [])
        ok_count = sum(1 for s in samples if s.get("ok", False))
        existing_ok_rate[cat] = ok_count / len(samples) if samples else 0.0

    # Live field recall probes
    category_results = {}

    for cat, probes in REALTIME_PROBES.items():
        required = CATEGORY_EXPECTED_FIELDS[cat]
        cat_recalls = []

        for prompt, fields in probes:
            fields_list = fields if isinstance(fields, list) else list(fields.keys())
            field_str = ", ".join(f'"{f}"' for f in fields_list)
            user_msg = f"{prompt}\n\nRespond with JSON containing: {field_str}."

            resp = call_api([
                {"role": "system", "content": REALTIME_SYSTEM},
                {"role": "user", "content": user_msg},
            ], temperature=0.0, max_tokens=1024)

            recall = _field_recall(resp, fields_list)
            cat_recalls.append(recall)
            time.sleep(0.05)

        category_results[cat] = {
            "required_fields": required,
            "mean_recall": round(statistics.mean(cat_recalls), 4),
            "per_probe": [round(r, 4) for r in cat_recalls],
        }

    # recall@1: PING (simplest, single field)
    recall_at_1 = category_results.get("PING", {}).get("mean_recall", 0.0)
    # recall@3: avg over PING, ONE_LINER, SHORT
    r3_vals = [category_results.get(c, {}).get("mean_recall", 0.0)
               for c in ["PING", "ONE_LINER", "SHORT"]]
    recall_at_3 = statistics.mean(r3_vals)

    # degradation curve
    curve = {cat: category_results[cat]["mean_recall"]
             for cat in ["PING", "ONE_LINER", "SHORT", "MEDIUM", "LONG"]}

    # recall floor: first category where recall < 0.5
    recall_floor = "never"
    for cat in ["PING", "ONE_LINER", "SHORT", "MEDIUM", "LONG"]:
        v = curve.get(cat)
        if v is not None and v < 0.5:
            recall_floor = cat
            break

    print(f"  Recall@1 (PING): {recall_at_1:.3f}")
    print(f"  Recall@3 (PING/ONE_LINER/SHORT): {recall_at_3:.3f}")
    print(f"  Curve: {curve}")
    print(f"  Recall floor: {recall_floor}")

    return {
        "recall_at_1": round(recall_at_1, 4),
        "recall_at_3": round(recall_at_3, 4),
        "recall_floor": recall_floor,
        "degradation_curve": curve,
        "notes": "Recall = % expected JSON fields present. PING=1 field, ONE_LINER=1, SHORT=2, MEDIUM=3, LONG=5",
        "category_details": category_results,
    }


# ─────────────────────────────────────────────
# Domain 6 — Agent Tool Routing
# ─────────────────────────────────────────────

TOOLS_AVAILABLE = ["search_web", "get_weather", "calculate", "lookup_person", "translate", "summarize"]

AGENT_PROBES = [
    # (query, correct_tool, complexity_level)
    # Level 1 — direct keyword match
    ("What is 25 + 37?", "calculate", 1),
    ("What is the weather in Paris?", "get_weather", 1),
    ("Who is Isaac Newton?", "lookup_person", 1),
    ("Search for news about AI", "search_web", 1),
    ("Translate 'thank you' to Italian", "translate", 1),
    ("Summarize: The cat sat on the mat.", "summarize", 1),
    # Level 2 — slightly indirect
    ("How much is 1000 divided by 7?", "calculate", 2),
    ("Is it cold in Moscow today?", "get_weather", 2),
    ("Tell me about Marie Curie's Nobel prizes", "lookup_person", 2),
    ("Find recent news about Tesla stock", "search_web", 2),
    ("How do you say 'goodbye' in Mandarin?", "translate", 2),
    ("Give me the key points of: The Amazon is the largest river by discharge in the world.", "summarize", 2),
    # Level 3 — indirect, requires inference
    ("If I have $500 and spend 30%, what's left?", "calculate", 3),
    ("Should I bring an umbrella in London tomorrow?", "get_weather", 3),
    ("What did Elon Musk found?", "lookup_person", 3),
    ("Look up the latest advancements in CRISPR technology", "search_web", 3),
    ("Wie heißt 'I am happy' auf Deutsch?", "translate", 3),
    ("Please condense this: Photosynthesis converts sunlight into glucose using CO2 and water.", "summarize", 3),
    # Level 4 — ambiguous, multi-intent (should still route to one primary tool)
    ("I need to know the temperature in Tokyo to decide if I need a coat", "get_weather", 4),
    ("What's 15% tip on a $84.50 restaurant bill?", "calculate", 4),
    ("Background info on Alan Turing for a school report", "lookup_person", 4),
    ("Find me information about Python programming for beginners", "search_web", 4),
    ("Convert 'The project deadline is tomorrow' to Spanish for my colleague", "translate", 4),
    ("Distill this into 2 sentences: Machine learning enables systems to learn from experience without being explicitly programmed.", "summarize", 4),
    # Level 5 — complex/misleading
    ("Can you compute how many seconds are in a leap year?", "calculate", 5),
    ("My flight lands in Dubai tomorrow, what should I pack?", "get_weather", 5),
    ("The scientist who split the atom — who was that?", "lookup_person", 5),
    ("Latest research on transformer neural networks published this year", "search_web", 5),
    ("My email says 'Merci beaucoup' — what language and what does it mean?", "translate", 5),
    ("Extract the main idea: In 1969, humans first walked on the Moon during NASA's Apollo 11 mission.", "summarize", 5),
]

AGENT_SYSTEM = (
    f"You are a tool router. Available tools: {', '.join(TOOLS_AVAILABLE)}. "
    "Select exactly ONE tool that best handles the user's query. "
    'Respond with JSON only: {"tool": "<tool_name>", "reason": "brief reason"}. No prose.'
)


def eval_domain6_agent_routing() -> Dict:
    print("\n[6] Agent Tool Routing — Recall Evaluation")
    print("  Measuring correct tool selection recall across 40 new probes")

    existing = load_results("6-agent-tool-routing")
    existing_accuracy = existing.get("summary", {}).get("overall_accuracy_pct", 0.0) / 100.0
    print(f"  Existing accuracy (40 calls): {existing_accuracy:.3f}")

    results_by_complexity = {}
    all_correct = 0
    all_total = 0

    for query, correct_tool, complexity in AGENT_PROBES:
        label = f"L{complexity}"
        if label not in results_by_complexity:
            results_by_complexity[label] = {"correct": 0, "total": 0}

        resp = call_api([
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": query},
        ], temperature=0.0, max_tokens=80)

        predicted_tool = None
        if resp:
            try:
                m = re.search(r'\{.*\}', resp, re.DOTALL)
                obj = json.loads(m.group()) if m else {}
                predicted_tool = obj.get("tool", "").lower().strip()
            except Exception:
                pass

        is_correct = (predicted_tool == correct_tool)
        results_by_complexity[label]["total"] += 1
        all_total += 1
        if is_correct:
            results_by_complexity[label]["correct"] += 1
            all_correct += 1

        time.sleep(0.05)

    overall_recall = all_correct / all_total if all_total else 0.0

    # recall@1: L1 (most direct)
    r1 = results_by_complexity.get("L1", {"correct": 0, "total": 1})
    recall_at_1 = r1["correct"] / r1["total"] if r1["total"] else 0.0

    # recall@3: L1+L2+L3
    r3_correct = sum(results_by_complexity.get(f"L{i}", {}).get("correct", 0) for i in [1, 2, 3])
    r3_total = sum(results_by_complexity.get(f"L{i}", {}).get("total", 0) for i in [1, 2, 3])
    recall_at_3 = r3_correct / r3_total if r3_total else 0.0

    # degradation curve
    curve = {}
    for lvl in ["L1", "L2", "L3", "L4", "L5"]:
        d = results_by_complexity.get(lvl, {})
        t = d.get("total", 0)
        c = d.get("correct", 0)
        curve[lvl] = round(c / t, 4) if t else None

    # recall floor
    recall_floor = "never"
    for lvl in ["L1", "L2", "L3", "L4", "L5"]:
        v = curve.get(lvl)
        if v is not None and v < 0.5:
            recall_floor = lvl
            break

    print(f"  Overall live recall: {overall_recall:.3f} ({all_correct}/{all_total})")
    print(f"  Recall@1 (L1): {recall_at_1:.3f}")
    print(f"  Recall@3 (L1-3): {recall_at_3:.3f}")
    print(f"  Curve: {curve}")
    print(f"  Recall floor: {recall_floor}")

    return {
        "recall_at_1": round(recall_at_1, 4),
        "recall_at_3": round(recall_at_3, 4),
        "recall_floor": recall_floor,
        "overall_live_recall": round(overall_recall, 4),
        "existing_accuracy": round(existing_accuracy, 4),
        "degradation_curve": curve,
        "notes": "Recall = fraction of 30 probes where correct tool was selected. Existing run: 40/40=1.0",
        "results_by_complexity": results_by_complexity,
    }


# ─────────────────────────────────────────────
# Print Results Table
# ─────────────────────────────────────────────
def print_table(results: Dict):
    print("\n" + "=" * 110)
    print(f"{'RECALL EVALUATION RESULTS':^110}")
    print("=" * 110)
    print(f"{'Domain':<30} {'Recall@1':>10} {'Recall@3':>10} {'Recall Floor':>15}  Notes")
    print("-" * 110)

    domains = [
        ("1-Classification", results["domain1"]),
        ("2-Extraction",     results["domain2"]),
        ("3-Routing-Filter", results["domain3"]),
        ("4-Chunked-Docs",   results["domain4"]),
        ("5-Realtime",       results["domain5"]),
        ("6-Agent-Routing",  results["domain6"]),
    ]

    for name, d in domains:
        r1   = d.get("recall_at_1", 0.0)
        r3   = d.get("recall_at_3", 0.0)
        flr  = d.get("recall_floor", "N/A")
        note = d.get("notes", "")[:55]
        print(f"{name:<30} {r1:>10.3f} {r3:>10.3f} {str(flr):>15}  {note}")

    print("=" * 110)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print(f"Recall Evaluation — {datetime.now().isoformat()}")
    print(f"Model: {MODEL}  API: {API_URL}")
    print("Running live API probes for all 6 domains...")

    results = {}

    results["domain1"] = eval_domain1_classification()
    results["domain2"] = eval_domain2_extraction()
    results["domain3"] = eval_domain3_routing()
    results["domain4"] = eval_domain4_chunked()
    results["domain5"] = eval_domain5_realtime()
    results["domain6"] = eval_domain6_agent_routing()

    output = {
        "meta": {
            "run_at": datetime.now().isoformat(),
            "model": MODEL,
            "api_url": API_URL,
        },
        "domains": results,
    }

    out_path = f"{BASE}/eval_recall_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    print_table(results)

    # Print degradation curves
    print("\n--- Degradation Curves ---")
    for domain_key, domain_name in [
        ("domain1", "Classification"),
        ("domain2", "Extraction"),
        ("domain3", "Routing-Filter"),
        ("domain4", "Chunked-Docs"),
        ("domain5", "Realtime"),
        ("domain6", "Agent-Routing"),
    ]:
        curve = results[domain_key].get("degradation_curve", {})
        print(f"\n  [{domain_name}]")
        for k, v in curve.items():
            bar_len = int((v or 0) * 30)
            bar = "█" * bar_len
            print(f"    {str(k):>30}: {(v or 0):5.3f}  {bar}")


if __name__ == "__main__":
    main()
