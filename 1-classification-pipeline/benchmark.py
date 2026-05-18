import os
#!/usr/bin/env python3
"""
LLM API Throughput Ceiling Benchmark
Tests high-volume text classification to find the maximum safe concurrency level.
"""

import json
import time
import statistics
import concurrent.futures
from datetime import datetime
import urllib.request
import urllib.error

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
API_URL = "https://ai.shenthar.me/v1/chat/completions"
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL   = "taalas-llama3.1-8b"

# ─────────────────────────────────────────────
# Dataset: 50 real diverse texts
# ─────────────────────────────────────────────
TEXTS = [
    # Product Reviews (1–15)
    "Absolutely love this blender! It crushes ice in seconds and the motor is whisper-quiet. "
    "Cleanup is a breeze since all parts are dishwasher-safe. Five stars without hesitation.",

    "Bought these headphones for my daily commute. The noise cancellation is decent but the "
    "ear pads feel cheap after just two months of use. Battery life is solid at 22 hours though.",

    "This laptop runs hot under any real workload. The fan screams at full blast constantly and "
    "the keyboard flex is unacceptable at this price point. Returning it tomorrow.",

    "Third pair of these running shoes. The cushioning is unmatched for long distances and they "
    "hold up beautifully after 500+ miles. Width sizing runs a bit narrow, order half a size up.",

    "Coffee maker stopped heating water after exactly 90 days—one day outside the return window. "
    "Customer service was dismissive. Never buying this brand again.",

    "The smart thermostat paid for itself in two months on my electricity bill. Setup took "
    "under 15 minutes and the app is intuitive. Scheduling features are outstanding.",

    "Mattress arrived compressed in a box and expanded perfectly. Sleeping cooler than ever "
    "and back pain I've had for years has almost completely disappeared. Worth every penny.",

    "Stand mixer looks gorgeous on the counter but the bowl capacity is deceptively small. "
    "For anything beyond a single batch of cookies you'll need to split the recipe.",

    "Camera quality on this phone is genuinely the best I've ever used. Low-light shots "
    "are stunning. Battery drains a bit fast if you use GPS navigation all day.",

    "Bought the premium subscription for cloud storage and discovered 'unlimited' means "
    "capped at 2TB in the fine print. Misleading advertising at best.",

    "This umbrella survived a brutal storm that snapped my last three umbrellas. The "
    "fiberglass ribs are noticeably more flexible. Compact enough for a backpack.",

    "Wireless charger heats my phone to an uncomfortable level during charging. "
    "Slow charge speed combined with the heat concerns me about long-term battery health.",

    "Gaming chair delivered with a scratched armrest and one missing bolt. "
    "Replacement parts arrived in 48 hours with an apology note. Good recovery.",

    "Air purifier reduced allergy symptoms noticeably within the first week. "
    "Filter replacement costs are steep but the performance justifies it.",

    "The cookbook's recipes are overly complex for the skill level it claims to target. "
    "Ingredient lists assume access to specialty stores most people don't have nearby.",

    # Support Tickets (16–28)
    "URGENT: Production database is completely down. All customer transactions failing. "
    "Revenue impact is approximately $50,000 per hour. Need immediate escalation to on-call DBA.",

    "My account has been locked for 72 hours and password reset emails never arrive. "
    "I have a critical presentation tomorrow that requires access to my files. Please help ASAP.",

    "Invoice #INV-2024-8847 shows a duplicate charge of $299 on my credit card dated March 15. "
    "Please initiate a refund. I've already contacted my bank but prefer to resolve directly.",

    "The mobile app crashes every time I try to upload a photo larger than 2MB. "
    "Running iOS 17.4 on iPhone 14 Pro. This started after the 3.2.1 update last Tuesday.",

    "I'd like to upgrade my subscription from the Basic plan to the Professional plan "
    "but the upgrade button on the billing page returns a 500 error. How do I proceed?",

    "Data export has been stuck at 67% for over 6 hours. The export contains 18 months "
    "of transaction records I need for an audit filing due end of business today.",

    "Our entire team lost access to the shared workspace after the admin accidentally "
    "deleted the primary workspace. Is there a way to restore deleted workspaces?",

    "API rate limit was reduced without any notification or email to developers. "
    "Our production integration is now throttled. Please restore previous limits or explain the change.",

    "Package tracking shows delivered but nothing was left at my address. Neighbors "
    "confirm they did not receive it either. Requesting investigation and replacement shipment.",

    "Subscription was auto-renewed at the old price after we negotiated a lower annual rate. "
    "The difference is $840. Please apply the correct pricing and issue a credit.",

    "Two-factor authentication is no longer accepting valid codes from my authenticator app. "
    "I'm locked out of a production system that requires access every 4 hours.",

    "Need to transfer ownership of 3 enterprise accounts to a new admin email before "
    "the current admin leaves the company on Friday. What's the process?",

    "The bulk import tool rejected our CSV file of 15,000 contacts with a vague 'format error'. "
    "I've attached the file. What specific field is causing the failure?",

    # News Headlines (29–38)
    "Federal Reserve holds interest rates steady amid persistent inflation concerns, "
    "signaling a cautious approach as economic data remains mixed heading into the third quarter.",

    "Wildfire burning through 40,000 acres in northern California forces evacuation of "
    "three communities; firefighters report containment at 15 percent with wind conditions worsening.",

    "Tech giant announces layoffs affecting 12,000 employees globally, citing AI-driven "
    "restructuring and a strategic pivot toward autonomous systems development.",

    "Scientists achieve breakthrough in room-temperature superconductivity using a new "
    "hydrogen-rich compound, a discovery that could transform power transmission within a decade.",

    "Municipal water system in the region tests positive for elevated PFAS levels; "
    "health officials urge residents to use bottled water until further notice.",

    "Coalition of 23 nations signs landmark biodiversity treaty pledging to protect "
    "30 percent of land and ocean by 2030, with binding enforcement mechanisms.",

    "Stock markets close lower for the fourth consecutive session as recession fears mount "
    "and consumer confidence index hits its lowest reading since early 2020.",

    "Hospital chain reports ransomware attack affecting electronic health records across "
    "14 facilities; surgeries postponed and patients diverted to neighboring hospitals.",

    "City council votes 7-2 to approve 10,000-unit affordable housing development "
    "on former industrial land, the largest such project in the city's history.",

    "Electric vehicle startup files for bankruptcy after failing to secure Series D funding, "
    "leaving 2,400 employees jobless and pre-orders from 18,000 customers unfulfilled.",

    # Tweets / Social Media (39–50)
    "just spent 3 hours on hold with my insurance company only to be told to call back "
    "tomorrow. this is genuinely the worst customer service experience of my entire life",

    "okay the new album dropped and it's absolutely unhinged in the best way possible. "
    "track 7 alone justifies the entire runtime. streaming on repeat all day",

    "reminder that tipping culture has completely spiraled out of control when a "
    "self-checkout kiosk asks if you want to leave a gratuity. where does it end",

    "our dog figured out how to open the refrigerator and ate an entire rotisserie chicken "
    "at 2am. she looks absolutely zero percent sorry about it and honestly respect",

    "flying for the first time since 2019 and genuinely shocked by how much prices have "
    "gone up. also the legroom situation is somehow even worse. peak capitalism",

    "six months into learning to code and i just built my first full stack app from scratch. "
    "it's a mess but it works and i genuinely cannot believe what humans are capable of building",

    "the audacity of this grocery store to charge $9 for a dozen eggs when eight months ago "
    "it was $2.49. can someone explain supply chains to me like i'm five",

    "just found out my company quietly removed remote work from the employee handbook "
    "with zero announcement. job searching starts tonight",

    "hot take: if your open-plan office has 'collaboration areas' but no quiet focus rooms "
    "you've optimized entirely for extroverts and managers, not for actual work",

    "bought a plant six months ago fully expecting it to die because that's my track record. "
    "it's now 4 feet tall and i have somehow become a plant person. identity crisis",

    "traffic app routed me through a 'faster' backroad that added 45 minutes to my commute. "
    "arrived at the meeting late covered in dust from an unpaved road. technology is a gift",

    "genuinely emotional watching my kid ride a bike without training wheels for the first time. "
    "parenthood is this constant cycle of teaching them to need you less and somehow that's beautiful",
]

assert len(TEXTS) == 50, f"Expected 50 texts, got {len(TEXTS)}"

# ─────────────────────────────────────────────
# Classification prompt
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a precise text classifier. Respond ONLY with a JSON object, no markdown, "
    "no explanation. The JSON must have exactly these keys: "
    '"sentiment" (positive|negative|neutral), '
    '"category" (product|support|news|social), '
    '"urgency" (high|medium|low).'
)

def build_payload(text: str) -> bytes:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify this text:\n\n{text}"},
        ],
        "max_tokens": 80,
        "temperature": 0,
    }
    return json.dumps(payload).encode("utf-8")


def classify_text(text: str) -> dict:
    """Send one classification request; return result dict with timing."""
    start = time.perf_counter()
    error = None
    result = None
    status_code = None

    try:
        req = urllib.request.Request(
            API_URL,
            data=build_payload(text),
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
        with urllib.request.urlopen(req, timeout=60) as resp:
            status_code = resp.status
            body = json.loads(resp.read().decode("utf-8"))
            raw = body["choices"][0]["message"]["content"].strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw)
    except urllib.error.HTTPError as e:
        status_code = e.code
        error = f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        error = f"URLError: {e.reason}"
    except json.JSONDecodeError as e:
        error = f"JSONDecodeError: {e}"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    elapsed = time.perf_counter() - start
    return {
        "latency": elapsed,
        "result": result,
        "error": error,
        "status_code": status_code,
    }


# ─────────────────────────────────────────────
# Run one concurrency level
# ─────────────────────────────────────────────
def run_level(concurrency: int, texts: list[str]) -> dict:
    print(f"  Running concurrency={concurrency} ({len(texts)} texts)...", end="", flush=True)
    latencies = []
    errors = []
    t_start = time.perf_counter()

    if concurrency == 1:
        for text in texts:
            r = classify_text(text)
            latencies.append(r["latency"])
            if r["error"]:
                errors.append(r["error"])
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(classify_text, t) for t in texts]
            for f in concurrent.futures.as_completed(futures):
                r = f.result()
                latencies.append(r["latency"])
                if r["error"]:
                    errors.append(r["error"])

    total_time = time.perf_counter() - t_start
    n = len(texts)
    error_rate = len(errors) / n

    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies_sorted)
    p95 = latencies_sorted[int(0.95 * n) - 1]
    p99 = latencies_sorted[int(0.99 * n) - 1]
    throughput = n / total_time

    print(f" done in {total_time:.1f}s  |  {throughput:.2f} items/s  |  errors={len(errors)}/{n}")

    return {
        "concurrency": concurrency,
        "n_texts": n,
        "total_time_s": round(total_time, 3),
        "throughput_items_per_sec": round(throughput, 3),
        "p50_latency_s": round(p50, 3),
        "p95_latency_s": round(p95, 3),
        "p99_latency_s": round(p99, 3),
        "error_count": len(errors),
        "error_rate": round(error_rate, 4),
        "errors_sample": errors[:5],
    }


# ─────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  LLM API Throughput Ceiling Benchmark")
    print(f"  Model : {MODEL}")
    print(f"  URL   : {API_URL}")
    print(f"  Texts : {len(TEXTS)}")
    print(f"  Start : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # Phase 1: fixed concurrency levels (1, 5, 10)
    phase1_levels = [1, 5, 10]
    # Phase 2: ceiling search (15, 20, 25, 30, 40, 50)
    phase2_levels = [15, 20, 25, 30, 40, 50]

    all_results = []
    baseline_latency = None
    ceiling_level = None
    ceiling_result = None

    print("\n--- Phase 1: Baseline concurrency sweep ---")
    for c in phase1_levels:
        r = run_level(c, TEXTS)
        all_results.append(r)
        if baseline_latency is None:
            baseline_latency = r["p50_latency_s"]

    print("\n--- Phase 2: Ceiling search ---")
    for c in phase2_levels:
        r = run_level(c, TEXTS)
        all_results.append(r)

        latency_ratio = r["p50_latency_s"] / baseline_latency if baseline_latency else 1
        prev = all_results[-2]
        throughput_gain = (r["throughput_items_per_sec"] - prev["throughput_items_per_sec"]) / prev["throughput_items_per_sec"]

        if r["error_rate"] > 0.05:
            print(f"  *** CEILING HIT at concurrency={c}: error_rate={r['error_rate']:.1%} > 5% ***")
            ceiling_level = prev["concurrency"]
            ceiling_result = prev
            break
        if latency_ratio > 3.0:
            print(f"  *** CEILING HIT at concurrency={c}: p50 latency ratio={latency_ratio:.2f}x > 3x baseline ***")
            ceiling_level = prev["concurrency"]
            ceiling_result = prev
            break
        if throughput_gain < 0.05:
            print(f"  *** THROUGHPUT PLATEAU at concurrency={c}: only {throughput_gain:.1%} gain vs previous level — ceiling is {prev['concurrency']} ***")
            ceiling_level = prev["concurrency"]
            ceiling_result = prev
            break
    else:
        # Never hit ceiling — highest level tested is the ceiling
        ceiling_level = all_results[-1]["concurrency"]
        ceiling_result = all_results[-1]
        print(f"  No ceiling hit — highest tested level ({ceiling_level}) appears stable.")

    # ─────────────────────────────────────────
    # Summary table
    # ─────────────────────────────────────────
    print("\n" + "=" * 79)
    print(f"{'Conc':>6}  {'Total(s)':>9}  {'Items/s':>8}  {'p50(s)':>7}  {'p95(s)':>7}  {'p99(s)':>7}  {'ErrRate':>8}")
    print("-" * 79)
    for r in all_results:
        ceiling_marker = " <-- CEILING" if r["concurrency"] == ceiling_level else ""
        print(
            f"{r['concurrency']:>6}  "
            f"{r['total_time_s']:>9.2f}  "
            f"{r['throughput_items_per_sec']:>8.2f}  "
            f"{r['p50_latency_s']:>7.3f}  "
            f"{r['p95_latency_s']:>7.3f}  "
            f"{r['p99_latency_s']:>7.3f}  "
            f"{r['error_rate']:>7.1%}"
            f"{ceiling_marker}"
        )
    print("=" * 79)

    print(f"\n  THROUGHPUT CEILING: concurrency={ceiling_level}")
    print(f"  Max safe throughput : {ceiling_result['throughput_items_per_sec']:.2f} items/sec")
    print(f"  p50 latency         : {ceiling_result['p50_latency_s']:.3f}s")
    print(f"  p95 latency         : {ceiling_result['p95_latency_s']:.3f}s")
    print(f"  p99 latency         : {ceiling_result['p99_latency_s']:.3f}s")
    print(f"  Error rate          : {ceiling_result['error_rate']:.1%}")

    # ─────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────
    output = {
        "meta": {
            "api_url": API_URL,
            "model": MODEL,
            "n_texts": len(TEXTS),
            "run_at": datetime.now().isoformat(),
        },
        "results": all_results,
        "ceiling": {
            "concurrency": ceiling_level,
            "throughput_items_per_sec": ceiling_result["throughput_items_per_sec"],
            "p50_latency_s": ceiling_result["p50_latency_s"],
            "p95_latency_s": ceiling_result["p95_latency_s"],
            "p99_latency_s": ceiling_result["p99_latency_s"],
            "error_rate": ceiling_result["error_rate"],
        },
    }

    results_path = "/Users/krishnatejaswis/llm-boundary-tests/1-classification-pipeline/results.json"
    with open(results_path, "w") as fh:
        json.dump(output, fh, indent=2)

    print(f"\n  Results saved to: {results_path}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 79)


if __name__ == "__main__":
    main()
