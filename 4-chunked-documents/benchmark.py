"""
benchmark.py - Optimal chunk size and overlap finder for parallel document processing
Tests various chunking strategies against a ~6k token context window LLM API.
"""

import json
import time
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

# ─── API Configuration ───────────────────────────────────────────────────────
API_URL = "https://ai.shenthar.me/v1/chat/completions"
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = "taalas-llama3.1-8b"
MAX_CONTEXT_TOKENS = 6000  # approximate

# ─── Realistic Long Business Document (~5000 words) ───────────────────────────
DOCUMENT = """
NEXAGEN CORP ANNUAL STRATEGIC REVIEW
Q4 2024 – GLOBAL OPERATIONS REPORT

Prepared by: Office of the Chief Strategy Officer
Issued: December 15, 2024

─────────────────────────────────────
EXECUTIVE SUMMARY
─────────────────────────────────────

NexaGen Corp, headquartered in Austin, Texas, concluded fiscal year 2024 with
consolidated revenues of $4.87 billion USD, representing a 23% increase
year-over-year from $3.96 billion in FY2023. Net income reached $612 million,
up from $489 million in the prior year. Chief Executive Officer Margaret Holbrook
attributed growth to the successful global rollout of the company's flagship
AI-powered logistics platform, NexaRoute 3.0, and aggressive market expansion
in Southeast Asia and Latin America.

The company employs 18,400 staff across 34 countries as of December 2024,
having added 2,800 net new employees during the fiscal year. Headquartered
in Austin, Texas, NexaGen maintains regional headquarters in Singapore, São Paulo,
Frankfurt, and Dubai. Its primary manufacturing operations are located in
Guadalajara, Mexico, and Penang, Malaysia.

Board Chairman Victor Okafor praised the management team's execution but noted
that supply-chain volatility and rising interest rates present headwinds entering
2025. The Board approved a $400 million share buyback program and raised the
quarterly dividend from $0.18 to $0.22 per share, effective Q1 2025.

─────────────────────────────────────
SECTION 1: BUSINESS SEGMENT PERFORMANCE
─────────────────────────────────────

1.1  Enterprise Logistics Solutions

The Enterprise Logistics Solutions (ELS) segment generated $2.1 billion in
revenue, accounting for 43% of total company revenue. This represents a 31%
increase versus FY2023's $1.6 billion. Key contracts won during the year include
a five-year, $340 million deal with RetailGiant Inc. to manage last-mile
delivery optimization across 1,200 distribution centers in North America, and
a three-year, $95 million contract with Frontier Pharmaceuticals to provide
cold-chain logistics monitoring.

Senior Vice President of ELS, Daniel Ríos, noted that NexaRoute 3.0's
machine-learning route optimization reduced customer fuel costs by an average
of 17% compared to legacy systems, a key differentiator in competitive bids.
The segment's adjusted EBITDA margin improved to 28.4% from 24.1%.

New customers added in ELS during FY2024 include:
- RetailGiant Inc. (North America)
- Frontier Pharmaceuticals (Global)
- BlueStar Automotive Group (Germany, France, Spain)
- Pacific Harvest Foods (Australia, New Zealand)
- Meridian Construction Holdings (Middle East)

Customer churn in ELS was 4.2%, down from 6.1% in FY2023, reflecting
improved customer success programs launched under VP of Customer Experience,
Priya Nair.

1.2  Smart Infrastructure Division

The Smart Infrastructure Division (SID) posted revenues of $1.45 billion,
up 18% from $1.23 billion in FY2023. SID focuses on IoT-enabled asset tracking,
predictive maintenance software, and industrial automation consulting.

A landmark achievement was the $220 million smart-port contract awarded by
the Port of Rotterdam Authority, enabling NexaGen to deploy over 4,000 IoT
sensors and its NexaSense analytics platform across Europe's largest cargo port.
Expected project completion is Q3 2026. Additionally, the Nairobi Metropolitan
Transport Authority awarded SID a $45 million contract for smart traffic
management systems in the Kenyan capital.

Chief Technology Officer Dr. Lin Wei revealed that SID's NexaSense 2.1 platform
now processes over 18 billion sensor events per day globally, up from 11 billion
in 2023. NexaGen filed 47 new patents in FY2024, bringing its total active patent
portfolio to 312.

1.3  Financial & Data Services

The Financial & Data Services (FDS) segment, NexaGen's newest division launched
in Q2 2022, reached $820 million in revenue in FY2024, exceeding the internal
target of $750 million. FDS offers supply-chain finance, invoice factoring, and
market-intelligence subscription products.

The NexaInsight data platform grew its subscriber base to 9,200 enterprise
clients, generating $310 million in recurring subscription revenue, implying
an average contract value of approximately $33,700 per client. Chief Financial
Officer Amanda Thornton stated that FDS carries the highest gross margin in the
portfolio at 61%, versus the company average of 47%.

Notable FDS milestones:
- Partnership with Standard Chartered Bank to co-finance $500M in supply-chain
  receivables for emerging-market SMEs.
- Launch of NexaInsight Commodities, a real-time agricultural commodities
  price-intelligence tool used by 340 agri-businesses globally.
- Regulatory approval in the EU for NexaGen Financial Services GmbH to operate
  as a licensed payment institution under PSD2.

─────────────────────────────────────
SECTION 2: GEOGRAPHIC PERFORMANCE
─────────────────────────────────────

2.1  North America

North America remained NexaGen's largest market, contributing $1.92 billion
or 39% of global revenue. Year-over-year growth was 19%, led by the RetailGiant
Inc. contract and expansion of SID services to municipal governments in Chicago,
Dallas, and Toronto. The U.S. federal government extended a $78 million contract
to provide logistics analytics for the Defense Logistics Agency.

Regional President for North America, James Calloway, announced the opening of
two new data centers in Phoenix, Arizona, and Columbus, Ohio, with a combined
CapEx of $290 million. The Columbus facility achieved LEED Platinum certification
and operates on 100% renewable energy sourced from a 15-year power purchase
agreement with SunPeak Energy Partners.

2.2  Europe, Middle East & Africa (EMEA)

EMEA revenues reached $1.38 billion, a 27% increase. The Port of Rotterdam
contract and a $60 million smart-grid analytics deal with EirGrid, the Irish
transmission system operator, were major contributors. Germany overtook the UK
as NexaGen's largest single-country market in EMEA for the first time, driven by
automotive supply-chain contracts with BlueStar Automotive Group and HeidelbergTech
Manufacturing.

NexaGen EMEA headquarters were relocated from London to Frankfurt in September
2024 to optimize for post-Brexit EU regulatory access. Regional head, Sophie
Lefebvre, oversees 4,200 employees across 14 European countries, plus teams in
Dubai and Nairobi.

2.3  Asia-Pacific (APAC)

APAC was the fastest-growing region with revenues of $1.09 billion, up 34%
from $813 million. Singapore serves as the APAC hub, with major operations in
Japan, South Korea, Australia, Vietnam, and India. NexaGen signed a memorandum
of understanding with Japan's Mitsui & Co. to explore joint ventures in
smart-port logistics across Southeast Asian ports, with a combined indicative
investment of $150 million over five years.

In India, NexaGen won a ₹850 crore (approximately $102 million) contract from
India Logistics Modernization Initiative (ILMI) under the national PM GatiShakti
infrastructure program to digitize freight corridors. APAC President, Takeshi
Yamamoto, noted that Vietnam and Indonesia are earmarked for accelerated
investment, with new offices opening in Ho Chi Minh City and Jakarta in Q1 2025.

2.4  Latin America

Latin America generated $480 million, up 21%, anchored by NexaGen's São Paulo
hub. Brazil represents 58% of regional revenues. Key new contracts include a
$65 million deal with PortoBrasil S.A. to modernize container tracking at the
Port of Santos, South America's largest port. NexaGen also signed agreements
with three major Brazilian agribusiness exporters — AgroParaná, CerradoGrain,
and Rio Verde Commodities — to use NexaInsight Commodities for real-time
export pricing intelligence.

─────────────────────────────────────
SECTION 3: TECHNOLOGY & INNOVATION
─────────────────────────────────────

3.1  NexaRoute 3.0 Platform

NexaRoute 3.0, released in March 2024, is built on a microservices architecture
deployed across AWS, Azure, and Google Cloud in a multi-cloud configuration.
The platform leverages a proprietary large-language-model fine-tuned on 14 years
of logistics data, branded as NexaLLM. Dr. Lin Wei disclosed that NexaLLM was
trained on 4.2 trillion tokens of supply-chain, freight, and geospatial data.

Key performance benchmarks for NexaRoute 3.0:
- Route optimization latency: 340 milliseconds average (down from 1.2 seconds)
- Concurrent route calculations: 2.4 million per second
- Uptime SLA: 99.97% (26 minutes of unplanned downtime in FY2024)
- Carbon emission reduction for customers: 14.3% average

3.2  Cybersecurity & Compliance

NexaGen achieved ISO 27001 re-certification for all 34 global offices in
August 2024. Chief Information Security Officer, Raymond Chu, confirmed that
the company completed a red-team exercise with external firm CyberForge Solutions
and remediated all 12 critical findings within the 90-day SLA. NexaGen was not
subject to any material data breaches in FY2024.

New compliance milestones include GDPR Article 28 audit completion across EU
processors, SOC 2 Type II certification for FDS platforms, and readiness
assessment for the EU AI Act requirements affecting NexaLLM by Q2 2025.

3.3  R&D Investment

Total R&D expenditure in FY2024 was $487 million, equaling 10% of revenue,
consistent with management's stated target of 9-11% of revenue. Of this amount,
$180 million was directed toward NexaRoute 3.0 and NexaLLM development, $145
million toward NexaSense IoT platform upgrades, and $162 million toward FDS
platform and data analytics capabilities. NexaGen's R&D workforce comprises
3,400 engineers and data scientists, including 420 machine-learning specialists.

─────────────────────────────────────
SECTION 4: HUMAN CAPITAL & ESG
─────────────────────────────────────

4.1  Workforce

NexaGen added 2,800 employees net in FY2024, reaching 18,400 globally. Employee
turnover fell to 11.2% from 14.7%, following the introduction of a revised
equity compensation program and expanded mental health benefits globally.
Chief People Officer, Yolanda Ferreira, credited "Project Connect," an internal
culture initiative, with improving employee engagement scores from 67% to 79%.

Diversity metrics improved: women now represent 41% of the global workforce
(up from 38%) and 33% of senior leadership (up from 28%). NexaGen set a target
of 40% women in senior leadership by 2027. Employees from underrepresented
ethnic minorities represent 29% of the U.S. workforce, up from 25%.

Top 5 hiring locations in FY2024:
1. Bangalore, India (580 hires) — primarily R&D and data science
2. Austin, Texas, USA (420 hires) — corporate and engineering
3. Singapore (310 hires) — APAC operations and sales
4. Frankfurt, Germany (250 hires) — EMEA operations and compliance
5. São Paulo, Brazil (215 hires) — Latin America expansion

4.2  Environmental, Social & Governance

NexaGen achieved carbon neutrality for Scope 1 and Scope 2 emissions in all
owned offices and data centers in 2024, two years ahead of the original 2026
target. Total Scope 1 + 2 emissions were 42,000 tonnes CO2e, fully offset via
verified carbon credits from the Madre de Dios REDD+ project in Peru and the
Rimba Raya Biodiversity Reserve in Indonesia.

Scope 3 emissions (customer and supply-chain emissions enabled by NexaGen's
platform) remain a challenge at an estimated 14 million tonnes CO2e. The company
committed to an independent Scope 3 audit in 2025 and set a 30% Scope 3 reduction
target by 2030 in alignment with Science Based Targets initiative (SBTi) criteria.

Water usage at manufacturing facilities was reduced by 22% through closed-loop
cooling system upgrades in Guadalajara and Penang. NexaGen was included in the
Dow Jones Sustainability Index (DJSI) World for the third consecutive year and
achieved a CDP Climate Score of A-.

Community investment totaled $28 million in FY2024, including:
- $12 million in STEM education programs in Brazil, India, and Nigeria
- $8 million for digital-literacy training targeting 45,000 SMEs in ASEAN
- $5 million for the NexaGen Foundation, supporting refugee workforce
  integration programs in Germany, Kenya, and Canada
- $3 million in disaster-relief logistics support following Typhoon Mawar

─────────────────────────────────────
SECTION 5: RISK FACTORS & OUTLOOK
─────────────────────────────────────

5.1  Macroeconomic & Geopolitical Risks

NexaGen's global footprint exposes it to currency risk, geopolitical disruption,
and regulatory fragmentation. The strengthening U.S. dollar reduced EMEA and
APAC revenue contributions by an estimated $87 million on a constant-currency
basis. Management hedges approximately 60% of non-USD revenues using forward
contracts, primarily through Goldman Sachs and Deutsche Bank as counterparties.

Supply-chain disruptions triggered by the Red Sea shipping crisis in late 2023
continued to affect some ELS customers through Q1 2024. NexaGen's resilience
team, led by VP of Risk, Claudia Martinovic, developed a dynamic re-routing
capability for NexaRoute 3.0 that reduced average customer delivery delays from
6.4 days to 2.1 days during the peak disruption period.

5.2  Competitive Landscape

NexaGen competes primarily against Zebra Technologies, Blue Yonder (acquired by
Panasonic), Manhattan Associates, and Oracle SCM Cloud in its core markets.
The entry of Alphabet subsidiary DeepRoute Analytics into logistics AI in
Q3 2024 is considered a medium-term competitive threat. CEO Margaret Holbrook
emphasized that NexaGen's proprietary training dataset and 14-year customer
relationships provide durable competitive moats.

5.3  Financial Outlook FY2025

Management issued FY2025 guidance of:
- Revenue: $5.8 – $6.1 billion (19-25% growth)
- Adjusted EBITDA margin: 29-31%
- Free Cash Flow: $800 million – $1 billion
- CapEx: $550 – $650 million (primarily data center and R&D)

The guidance assumes no material deterioration in global trade volumes and
excludes any acquisitions. NexaGen's balance sheet carries $1.2 billion net
cash after a $600 million green bond issuance in October 2024, rated BBB+ by
S&P Global and Baa1 by Moody's.

The company has identified four acquisition targets in the $150–400 million range,
primarily in AI-driven warehouse automation and last-mile delivery robotics.
CEO Holbrook confirmed that a definitive agreement on one target is expected
in Q1 2025, though no further details were disclosed.

─────────────────────────────────────
SECTION 6: AWARDS AND RECOGNITION
─────────────────────────────────────

In FY2024, NexaGen Corp received the following recognitions:
- Forbes Global 2000 (Rank: #312, up from #389)
- Gartner Magic Quadrant Leader for Supply Chain Management Platforms
  (third consecutive year)
- Fast Company "World's Most Innovative Companies" (#7 in Enterprise Technology)
- Fortune "Best Places to Work" (included for the first time)
- ISO 50001 Energy Management certification for Guadalajara manufacturing facility
- Singapore Economic Development Board "Anchor Investment of the Year" award

─────────────────────────────────────
CONCLUSION
─────────────────────────────────────

Fiscal year 2024 was a defining year for NexaGen Corp. Driven by the global
deployment of NexaRoute 3.0, strategic geographic expansion, and a disciplined
focus on high-margin recurring revenue, the company delivered record financial
results while advancing its ESG commitments ahead of schedule. Chairman Okafor
and CEO Holbrook expressed confidence that NexaGen is well-positioned to achieve
its medium-term vision of becoming the world's leading intelligent supply-chain
platform company, targeting $10 billion in annual revenue by FY2028.

The Board of Directors extends gratitude to all 18,400 NexaGen employees,
customers, and partners worldwide for their contributions to this milestone year.

[END OF REPORT]
"""

# ─── Chunking Strategies to Test ─────────────────────────────────────────────
STRATEGIES = [
    {"chunk_words": 500,  "overlap_words": 0},
    {"chunk_words": 500,  "overlap_words": 50},
    {"chunk_words": 1000, "overlap_words": 0},
    {"chunk_words": 1000, "overlap_words": 100},
    {"chunk_words": 2000, "overlap_words": 200},
    {"chunk_words": 3000, "overlap_words": 300},
]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def split_into_words(text: str) -> list[str]:
    return text.split()


def chunk_document(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    """Split document into word-based chunks with optional overlap."""
    words = split_into_words(text)
    chunks = []
    step = max(1, chunk_words - overlap_words)
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += step
    return chunks


def _build_prompt(chunk: str) -> str:
    return (
        "You are an expert information extractor. "
        "Extract ALL key facts, named entities, numbers, and important details from the text below. "
        "Return ONLY valid JSON (no markdown, no code fences) with exactly two keys:\n"
        "  \"entities\": a flat list of unique strings (people, companies, places, products, "
        "numbers with context, dates, financial figures, metrics)\n"
        "  \"key_facts\": a list of concise fact strings (max 20 words each)\n"
        "Example output: {\"entities\": [\"NexaGen Corp\", \"$4.87 billion\"], "
        "\"key_facts\": [\"NexaGen revenue grew 23% to $4.87B in FY2024\"]}\n"
        "Do NOT include any text outside the JSON object.\n\n"
        f"TEXT:\n{chunk}"
    )


def _parse_llm_response(content: str) -> tuple[list[str], list[str]]:
    """Try multiple strategies to extract JSON from LLM response."""
    if not content:
        raise ValueError("Empty response content")

    # Strategy 1: find outermost JSON object
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        parsed = json.loads(json_match.group())
        entities = [str(e).strip() for e in parsed.get("entities", []) if str(e).strip()]
        key_facts = [str(f).strip() for f in parsed.get("key_facts", []) if str(f).strip()]
        return entities, key_facts

    # Strategy 2: try the whole content
    parsed = json.loads(content)
    entities = [str(e).strip() for e in parsed.get("entities", []) if str(e).strip()]
    key_facts = [str(f).strip() for f in parsed.get("key_facts", []) if str(f).strip()]
    return entities, key_facts


def call_llm(chunk: str, chunk_index: int, max_retries: int = 2) -> dict[str, Any]:
    """Call LLM API to extract key facts and entities from a chunk as JSON."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    last_error = None
    raw_content = ""
    for attempt in range(max_retries + 1):
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": _build_prompt(chunk)}],
            "temperature": 0,
            "max_tokens": 1200,
        }
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            raw_content = resp.json()["choices"][0]["message"]["content"].strip()
            entities, key_facts = _parse_llm_response(raw_content)
            return {
                "chunk_index": chunk_index,
                "entities": entities,
                "key_facts": key_facts,
                "error": None,
                "raw_response_preview": raw_content[:200],
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc} | raw_preview: {repr(raw_content[:200])}"
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))

    return {
        "chunk_index": chunk_index,
        "entities": [],
        "key_facts": [],
        "error": last_error,
        "raw_response_preview": raw_content[:200],
    }


def normalize_entity(e: str) -> str:
    """Lower-case and strip for dedup comparison."""
    return re.sub(r'\s+', ' ', e.lower().strip())


def merge_results(chunk_results: list[dict]) -> dict[str, Any]:
    """Merge results from all chunks, deduplicating entities."""
    all_entities_raw: list[str] = []
    all_facts: list[str] = []
    errors = []
    for r in chunk_results:
        if r["error"]:
            errors.append(f"chunk {r['chunk_index']}: {r['error']}")
        all_entities_raw.extend(r["entities"])
        all_facts.extend(r["key_facts"])

    # Count total before dedup
    total_before_dedup = len(all_entities_raw)

    # Deduplicate (case-insensitive, whitespace-normalized)
    seen_normalized: set[str] = set()
    deduped: list[str] = []
    for e in all_entities_raw:
        norm = normalize_entity(e)
        if norm not in seen_normalized:
            seen_normalized.add(norm)
            deduped.append(e)

    duplicate_count = total_before_dedup - len(deduped)
    duplicate_rate = (duplicate_count / total_before_dedup * 100) if total_before_dedup > 0 else 0.0

    return {
        "unique_entities": deduped,
        "entities_count": len(deduped),
        "key_facts_count": len(all_facts),
        "total_entity_mentions": total_before_dedup,
        "duplicate_count": duplicate_count,
        "duplicate_rate_pct": round(duplicate_rate, 1),
        "errors": errors,
    }


def run_strategy_parallel(chunk_words: int, overlap_words: int) -> dict[str, Any]:
    """Run a single chunking strategy with full parallelism."""
    chunks = chunk_document(DOCUMENT, chunk_words, overlap_words)
    num_chunks = len(chunks)

    t_start = time.perf_counter()
    chunk_results = []
    with ThreadPoolExecutor(max_workers=min(num_chunks, 20)) as executor:
        futures = {executor.submit(call_llm, chunk, i): i for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            chunk_results.append(future.result())
    t_end = time.perf_counter()

    # Sort results by chunk index for consistency
    chunk_results.sort(key=lambda x: x["chunk_index"])
    merged = merge_results(chunk_results)

    return {
        "chunk_words": chunk_words,
        "overlap_words": overlap_words,
        "mode": "parallel",
        "num_chunks": num_chunks,
        "wall_time_sec": round(t_end - t_start, 3),
        **merged,
    }


def run_strategy_sequential(chunk_words: int, overlap_words: int) -> dict[str, Any]:
    """Run a single chunking strategy sequentially (no parallelism)."""
    chunks = chunk_document(DOCUMENT, chunk_words, overlap_words)
    num_chunks = len(chunks)

    t_start = time.perf_counter()
    chunk_results = [call_llm(chunk, i) for i, chunk in enumerate(chunks)]
    t_end = time.perf_counter()

    merged = merge_results(chunk_results)

    return {
        "chunk_words": chunk_words,
        "overlap_words": overlap_words,
        "mode": "sequential",
        "num_chunks": num_chunks,
        "wall_time_sec": round(t_end - t_start, 3),
        **merged,
    }


def score_strategy(result: dict) -> float:
    """
    Score a strategy for optimality:
    Higher = better.
    Rewards: many unique entities, low duplicate rate, fast wall time.
    Penalizes: errors.
    """
    entities = result["entities_count"]
    dup_penalty = result["duplicate_rate_pct"] / 100.0  # 0–1
    time_penalty = result["wall_time_sec"] / 60.0       # normalize to ~1 at 60s
    error_penalty = len(result["errors"]) * 5
    return entities * (1 - dup_penalty) - time_penalty * 2 - error_penalty


def print_table(results: list[dict]) -> None:
    """Print a formatted summary table."""
    header = (
        f"{'Strategy':<22} {'Mode':<12} {'Chunks':>6} {'WallTime(s)':>12} "
        f"{'Entities':>9} {'DupRate%':>9} {'Errors':>7} {'Score':>8}"
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        label = f"{r['chunk_words']}w/{r['overlap_words']}w-ovlp"
        print(
            f"{label:<22} {r['mode']:<12} {r['num_chunks']:>6} "
            f"{r['wall_time_sec']:>12.2f} {r['entities_count']:>9} "
            f"{r['duplicate_rate_pct']:>9.1f} {len(r['errors']):>7} "
            f"{score_strategy(r):>8.2f}"
        )
    print(sep)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    doc_words = len(split_into_words(DOCUMENT))
    print(f"\nDocument word count: {doc_words:,}")
    print(f"Model: {MODEL}  |  Context: ~{MAX_CONTEXT_TOKENS} tokens\n")

    all_results: list[dict] = []

    # ── Phase 1: Run all strategies in PARALLEL mode ──
    print("=" * 60)
    print("PHASE 1: Testing all chunking strategies (PARALLEL mode)")
    print("=" * 60)
    for strat in STRATEGIES:
        cw, ow = strat["chunk_words"], strat["overlap_words"]
        label = f"{cw}w/{ow}w-overlap"
        print(f"\n  Running {label} ...", flush=True)
        result = run_strategy_parallel(cw, ow)
        result["score"] = score_strategy(result)
        all_results.append(result)
        print(
            f"    → {result['num_chunks']} chunks | "
            f"{result['wall_time_sec']:.2f}s | "
            f"{result['entities_count']} entities | "
            f"dup rate {result['duplicate_rate_pct']}% | "
            f"errors: {len(result['errors'])}"
        )

    # ── Identify best strategy from parallel results ──
    parallel_results = [r for r in all_results if r["mode"] == "parallel"]
    best = max(parallel_results, key=score_strategy)
    best_cw, best_ow = best["chunk_words"], best["overlap_words"]
    print(f"\nBest parallel strategy (by score): {best_cw}w / {best_ow}w-overlap")

    # ── Phase 2: Run best strategy SEQUENTIALLY for comparison ──
    print("\n" + "=" * 60)
    print(f"PHASE 2: Sequential baseline for best strategy ({best_cw}w/{best_ow}w-overlap)")
    print("=" * 60)
    print(f"\n  Running sequential {best_cw}w/{best_ow}w ...", flush=True)
    seq_result = run_strategy_sequential(best_cw, best_ow)
    seq_result["score"] = score_strategy(seq_result)
    all_results.append(seq_result)
    print(
        f"    → {seq_result['num_chunks']} chunks | "
        f"{seq_result['wall_time_sec']:.2f}s | "
        f"{seq_result['entities_count']} entities | "
        f"dup rate {seq_result['duplicate_rate_pct']}% | "
        f"errors: {len(seq_result['errors'])}"
    )

    # ── Summary Table ──
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print_table(all_results)

    # ── Parallel vs Sequential Comparison ──
    par_time = best["wall_time_sec"]
    seq_time = seq_result["wall_time_sec"]
    speedup = seq_time / par_time if par_time > 0 else float("inf")
    print(f"\nParallel vs Sequential ({best_cw}w/{best_ow}w-overlap, {best['num_chunks']} chunks):")
    print(f"  Parallel wall time:   {par_time:.2f}s")
    print(f"  Sequential wall time: {seq_time:.2f}s")
    print(f"  Speedup factor:       {speedup:.2f}x")

    # ── Entity Coverage Across Chunk Sizes ──
    print("\nEntity coverage by chunk size (parallel only):")
    print(f"  {'ChunkSize':>12} {'Overlap':>8} {'Entities':>10} {'DupRate%':>10}")
    for r in parallel_results:
        print(f"  {r['chunk_words']:>12} {r['overlap_words']:>8} {r['entities_count']:>10} {r['duplicate_rate_pct']:>10.1f}")

    # ── Quality degradation near 6k limit ──
    near_limit = [r for r in parallel_results if r["chunk_words"] >= 2000]
    if near_limit:
        print("\nQuality check for large chunk sizes (near 6k token limit):")
        for r in near_limit:
            est_tokens = r["chunk_words"] * 1.35  # rough words-to-tokens ratio
            print(
                f"  {r['chunk_words']}w (≈{int(est_tokens)} tokens): "
                f"{r['entities_count']} entities, dup rate {r['duplicate_rate_pct']}%, "
                f"errors: {len(r['errors'])}"
            )

    # ── Save Results ──
    output = {
        "document_word_count": doc_words,
        "model": MODEL,
        "api_url": API_URL,
        "optimal_strategy": {
            "chunk_words": best_cw,
            "overlap_words": best_ow,
            "score": round(best["score"], 3),
            "entities_found": best["entities_count"],
            "duplicate_rate_pct": best["duplicate_rate_pct"],
            "parallel_wall_time_sec": par_time,
            "sequential_wall_time_sec": seq_time,
            "speedup_factor": round(speedup, 2),
        },
        "all_results": all_results,
    }
    results_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"\nOPTIMAL STRATEGY: chunk_size={best_cw}w, overlap={best_ow}w")
    print(f"  Score: {best['score']:.2f} | Entities: {best['entities_count']} | "
          f"Dup rate: {best['duplicate_rate_pct']}% | "
          f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
