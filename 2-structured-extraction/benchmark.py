import os
"""
NER/Structured Extraction Quality Benchmark
Tests extraction reliability as document size increases toward a ~6k token context window.
"""

import json
import re
import time
import urllib.request
import urllib.error
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
API_URL = "https://ai.shenthar.me/v1/chat/completions"
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL   = "taalas-llama3.1-8b"
REPEATS = 3
TARGET_WORD_COUNTS = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]

# ---------------------------------------------------------------------------
# GROUND TRUTH  (canonical, lowercase for matching)
# ---------------------------------------------------------------------------
GROUND_TRUTH = {
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

# ---------------------------------------------------------------------------
# ARTICLE CORPUS  — one large article (~4 000+ words) that contains every entity.
# Smaller sizes are taken as the first N words of this text.
# ---------------------------------------------------------------------------
BASE_ARTICLE = """
NexAgen Biotech Secures $2.4 Billion in Landmark Funding Round Amid Global Expansion Push

SAN FRANCISCO — In what analysts are calling one of the most significant financing events of the year,
NexAgen Biotech announced on March 14 2024 that it has closed a $2.4 billion Series D funding round
led by Meridian Capital Partners. The deal, which was months in the making, positions the biotechnology
company as a dominant force in the global gene-therapy market and signals renewed investor appetite
for high-risk, high-reward life-sciences ventures.

Eleanor Voss, the chief executive of NexAgen Biotech, described the milestone as transformative.
"This capital will allow us to accelerate clinical trials across three continents and bring our lead
therapy candidate to patients far sooner than previously possible," she told reporters at a packed press
conference held at the company's headquarters in San Francisco. Voss, who joined NexAgen Biotech six
years ago after a decade at a rival pharmaceutical group, has been widely credited with steering the
firm through a difficult post-pandemic period.

Marcus Chen, a managing director at Meridian Capital Partners and the architect of the deal, said the
firm had been tracking NexAgen Biotech for nearly two years before committing to the $2.4 billion
investment. "The science is extraordinary, the management team is exceptional, and the market
opportunity is enormous," Chen said in a statement released from Meridian Capital Partners' offices
in Singapore. He added that the firm expects to maintain its position as lead investor through a
potential initial public offering, which insiders say could come as early as late 2025.

The financing comes at a pivotal moment for the biotechnology sector. Rising interest rates and a
broader pullback in venture capital activity have forced many startups to scale back ambitions or
accept down rounds. NexAgen Biotech's ability to attract $2.4 billion at a valuation that sources
close to the transaction put at roughly $14 billion suggests that investors remain willing to back
companies with differentiated technology and a credible path to commercialisation.

Thomas Hartwell, a senior analyst at Blue Harbor Ventures in Boston, said the round sends an
important signal to the wider market. "When Meridian Capital Partners writes a cheque of this size,
it tells everyone else that there is serious money still circulating in biotech," Hartwell explained.
"The NexAgen Biotech deal will likely unlock further institutional commitments across the sector in
the coming months."

--- Expansion Into Africa and Europe ---

Alongside the funding announcement, NexAgen Biotech revealed plans to open two new research
laboratories — one in Nairobi and one in Berlin — by September 30 2024. The Nairobi facility,
a joint venture with Fortridge Logistics, which will manage the supply-chain and cold-storage
infrastructure required for gene therapies in sub-Saharan Africa, is expected to create more
than 400 local jobs. The Berlin site, meanwhile, will focus on regulatory affairs and clinical
operations for the European Union market.

Priya Nair, NexAgen Biotech's chief scientific officer, flew to Nairobi in February to scout
locations and meet with government officials. Nair, a Cambridge-educated molecular biologist who
previously led research at Quantum Dynamics Inc, said the African continent represents an
"enormous unmet medical need" that the company is uniquely positioned to address. "Our platform
technology is modular. We can adapt it to local disease profiles relatively quickly," she said
in an interview conducted in Nairobi. The decision to anchor the African expansion in Nairobi
rather than, say, Lagos or Johannesburg was driven by the city's strong logistical infrastructure
and its proximity to other East African markets.

The Berlin laboratory is being built on a site formerly occupied by a defunct automotive supplier.
James Okafor, NexAgen Biotech's head of European operations, who is based in Berlin, said the
company had chosen the German capital for its deep pool of scientific talent and its central
position within the European single market. "Berlin gives us access to researchers from across
Europe. The talent density here is remarkable," Okafor said. He noted that the company has
already signed preliminary employment agreements with twenty senior scientists who will relocate
to Berlin from other research centres.

--- Fortridge Logistics Partnership ---

The partnership with Fortridge Logistics is viewed as a strategic masterstroke by industry
observers. Gene therapies require ultra-low-temperature storage and meticulously controlled
distribution networks — capabilities that most pharmaceutical companies lack in-house, particularly
in emerging markets. Fortridge Logistics, which operates one of the largest refrigerated
logistics networks in Africa and Southeast Asia, brings precisely those capabilities to the table.

Linda Beaumont, the chief executive of Fortridge Logistics, said the deal had been structured
to ensure that both parties bear proportional risk. "We are not merely a service provider here.
We have skin in the game," Beaumont said during a joint press conference held in Nairobi on
March 14 2024. Fortridge Logistics received $850 million in preferred equity from NexAgen Biotech
as consideration for an exclusive ten-year supply-chain agreement covering thirty-two countries.
Beaumont added that Fortridge Logistics plans to invest $340 million of that sum into cold-chain
infrastructure across Kenya, Tanzania, Uganda, and Rwanda over the next three years.

--- Solaris Energy Group Investment ---

In a separate but related development, Solaris Energy Group disclosed on July 2 2024 that it had
acquired a 6.5 percent stake in NexAgen Biotech through a secondary transaction, paying
approximately $17.5 million for the holding. Raj Patel, the chief investment officer of Solaris
Energy Group, said the company views the investment as part of a broader diversification strategy
into healthcare and life sciences. "Energy transition is our core business, but we are acutely
aware that the companies transforming healthcare will be among the most valuable enterprises of
the next decade," Patel told analysts on a call from Toronto, where Solaris Energy Group is
headquartered.

Patel, who joined Solaris Energy Group from a private equity firm in Amsterdam three years ago,
said the team spent considerable time evaluating the NexAgen Biotech technology platform and
speaking with clinicians who had participated in early-stage trials. "We came away convinced
that the data is robust and that the management team has the operational capacity to execute at
scale," he said. Patel declined to specify whether Solaris Energy Group would increase its stake
ahead of any eventual public offering, saying only that the company is "comfortable with its
current position and will reassess periodically."

--- Quantum Dynamics Inc Licensing Deal ---

Quantum Dynamics Inc, a computational biology firm headquartered in Austin, Texas, announced on
October 15 2024 that it had signed a landmark licensing agreement with NexAgen Biotech covering
a suite of protein-folding algorithms that NexAgen Biotech will use to accelerate drug discovery.
The deal, valued at $4.2 billion over fifteen years, is the largest licensing transaction in
Quantum Dynamics Inc's history and is expected to contribute meaningfully to the company's
revenue from the first quarter of 2025.

Sofia Delgado, the chief executive of Quantum Dynamics Inc, said the agreement validates years
of investment in foundational research. "We have spent a decade building algorithms that we
believe are the gold standard in computational biology. This partnership with NexAgen Biotech
is proof that fundamental science can translate into enormous commercial value," Delgado said at
a press event held at Quantum Dynamics Inc's campus in Austin. She noted that the $4.2 billion
licensing fee will be paid in structured tranches tied to specific development milestones, with
an initial payment due by December 1 2024.

Priya Nair of NexAgen Biotech said the Quantum Dynamics Inc algorithms will cut months off the
company's drug-discovery timeline. "We were doing this work manually with large teams of
computational scientists. Now we can run the same analyses in a fraction of the time," she said.
Eleanor Voss added that the licensing deal had been under negotiation for fourteen months and
that several competing firms had also sought access to the Quantum Dynamics Inc technology.
Marcus Chen of Meridian Capital Partners confirmed that the licensing deal had been a key
consideration in the $2.4 billion funding round, saying: "We knew the Quantum Dynamics Inc
capability was coming. That de-risked a significant part of our underwriting thesis."

--- Market Reaction and Analyst Commentary ---

Shares in publicly traded peers of NexAgen Biotech rose sharply on the day the funding was
announced. Thomas Hartwell at Blue Harbor Ventures published a research note arguing that
the NexAgen Biotech round signals a "regime change" in biotech financing. "For the better part
of two years, the conversation in San Francisco, Boston, and Berlin has been about capital
scarcity. Today's announcement suggests the pendulum is swinging back," Hartwell wrote.

Linda Beaumont of Fortridge Logistics agreed that the market mood is shifting. Speaking at a
logistics industry conference in Amsterdam in early April, Beaumont said: "The NexAgen Biotech
deal has given the whole sector a shot of confidence. We are already fielding enquiries from
other pharmaceutical companies about cold-chain partnerships in Africa." She noted that Fortridge
Logistics expects to finalise at least two additional partnership agreements by September 30 2024.

James Okafor noted that the Berlin laboratory's construction timeline is on track and that the
facility should be ready to accept its first research cohort by the target date. "We broke ground
last month and the contractors have been working around the clock. Berlin is going to be a
flagship site for us," he said.

Raj Patel of Solaris Energy Group echoed the optimism, saying that Solaris intends to leverage
its energy-management expertise to help NexAgen Biotech reduce the carbon footprint of its
manufacturing operations. "Gene therapy manufacturing is extremely energy-intensive. We see an
opportunity to deploy renewable energy solutions across NexAgen Biotech's facilities in San
Francisco, Nairobi, and Berlin," Patel said.

--- Looking Ahead ---

As NexAgen Biotech looks toward the remainder of 2024 and beyond, the company faces a complex
web of opportunities and challenges. Eleanor Voss acknowledged that scaling a gene-therapy
platform globally is "an enormously complicated undertaking" and that the company will need to
navigate varying regulatory regimes, cultural expectations, and infrastructure realities in each
of its target markets. She expressed confidence that the combination of Meridian Capital Partners'
financial backing, Fortridge Logistics' operational infrastructure, and Quantum Dynamics Inc's
computational tools gives NexAgen Biotech a competitive moat that would be difficult to replicate.

Marcus Chen added that Meridian Capital Partners has assembled a board of advisors with deep
experience in international healthcare markets, including two former health ministers from
sub-Saharan African countries and a retired senior regulator from the European Medicines Agency.
"We are not going into these markets naively. We have done the homework," he said.

Priya Nair, who will be splitting her time between San Francisco, Nairobi, and Berlin for the
foreseeable future, said she is "genuinely excited by the prospect of building something that
matters." She added: "Science that stays in the lab is just an interesting hobby. Taking it to
patients in Nairobi or Berlin or San Francisco — that is the whole point."

Sofia Delgado of Quantum Dynamics Inc said she hopes the Austin-based firm's contribution to
the NexAgen Biotech story will be a catalyst for broader recognition of computational biology's
role in modern medicine. "We have been the quiet engine in the background for too long. This
deal puts Quantum Dynamics Inc on the map in a way that nothing else has," she said. The firm
plans to host a scientific symposium in Austin in the first quarter of 2025 to showcase the
breadth of its algorithmic toolkit.

Thomas Hartwell at Blue Harbor Ventures concluded his research note with a bullish outlook:
"NexAgen Biotech is not just a biotech company. It is a bet on a new model of global medicine —
one that combines cutting-edge science, smart logistics, and computational power in a way that
no single competitor can currently match. The $2.4 billion vote of confidence from Meridian
Capital Partners is the clearest endorsement yet that this model is viable." The note, circulated
to clients across Boston, Singapore, Toronto, and Amsterdam, ended with a twelve-month price
target that implied a potential doubling of valuation from the current implied $14 billion.

Linda Beaumont's final word was characteristically pragmatic: "All of this is wonderful in
theory. The test is in the execution. Fortridge Logistics will be judged by whether those
therapies arrive cold, on time, and intact in Nairobi. That is the job." She said Fortridge
Logistics has committed to zero cold-chain failures in its first year of operation under the
NexAgen Biotech agreement, and that the $340 million infrastructure investment is designed to
make that commitment credible.

James Okafor signed off with a note of personal pride: "I have lived in Berlin for eight years.
I know this city. I know what it can do. NexAgen Biotech is going to love it here." He said the
laboratory's official opening ceremony is being planned for September 30 2024, and that
invitations have been extended to senior officials from Germany's Federal Ministry of Health as
well as representatives from the European Medicines Agency.

Raj Patel noted that Solaris Energy Group's Toronto headquarters will serve as a coordination
hub for the NexAgen Biotech investment, with a dedicated team of two analysts and a relationship
manager assigned to monitor the portfolio company's progress. "We treat our minority stakes as
active relationships, not passive positions," he said. The $17.5 million investment, while
modest relative to the overall funding round, is strategically important to Solaris Energy Group
as it builds out what Patel describes as a "life-sciences adjacency pillar" within the firm's
broader portfolio.

The story of NexAgen Biotech, Meridian Capital Partners, Fortridge Logistics, Quantum Dynamics
Inc, Solaris Energy Group, and Blue Harbor Ventures is, at its core, a story about what happens
when capital, science, logistics, and computing converge at the right moment. Whether that
convergence produces the transformative outcomes that Eleanor Voss, Marcus Chen, Priya Nair,
Thomas Hartwell, Sofia Delgado, James Okafor, Linda Beaumont, and Raj Patel are all betting on
remains to be seen. What is clear is that the bets are large, the ambitions are global, and the
clock — anchored by dates like March 14 2024, July 2 2024, September 30 2024, October 15 2024,
and December 1 2024 — is ticking.

--- Regulatory Landscape and Clinical Trial Strategy ---

One of the central challenges facing NexAgen Biotech as it pursues its global expansion agenda
is the patchwork of regulatory frameworks governing gene therapies in different jurisdictions.
In the United States, the Food and Drug Administration has been gradually developing clearer
guidelines, but the process remains lengthy and unpredictable. In Europe, the European Medicines
Agency — with which James Okafor has been in dialogue from the Berlin office — has signalled a
willingness to expedite review for therapies targeting rare diseases, which could benefit
NexAgen Biotech's lead pipeline programmes.

Eleanor Voss has spoken publicly about the need for greater international harmonisation of
gene-therapy regulation. At a forum convened in Singapore in late January, Voss argued that
the current fragmented regulatory environment imposes unnecessary costs and delays that ultimately
hurt patients. Meridian Capital Partners, whose managing director Marcus Chen co-hosted the
Singapore event, has reportedly been lobbying quietly for regulatory reform through its network
of governmental contacts. The fund has invested in three other gene-therapy companies since 2020,
giving Chen a broad perspective on where regulatory bottlenecks most commonly arise.

Priya Nair's scientific team has identified five candidate indications for which NexAgen Biotech
believes it can achieve regulatory approval within three years, subject to the clinical data
holding up. Two of those indications are rare monogenic disorders where the unmet medical need
is acute and the regulatory pathway is well-defined. The other three are more complex
multifactorial conditions where the scientific case is strong but the regulatory journey will
require more creative engagement. Nair said in an interview in Nairobi that the company
deliberately chose this mixed portfolio to balance near-term revenue certainty against long-term
transformative impact. "We cannot be a company that only chases orphan indications for the
economics," she said. "The science compels us to go after the harder problems."

--- Supply Chain Resilience and Cold-Chain Technology ---

Fortridge Logistics, led by Linda Beaumont, is investing a significant portion of its $340 million
infrastructure budget in next-generation cold-chain technology developed by a consortium of
refrigeration engineers and digital-monitoring specialists based in Amsterdam. The technology,
which uses a combination of phase-change materials and real-time IoT sensors, can maintain
gene-therapy products within their required temperature ranges even during extended power outages
or rough handling in transit. Beaumont said the system was piloted on a smaller logistics route
between Nairobi and Kampala with promising results, and that full rollout across the network is
planned to coincide with the official launch of the NexAgen Biotech partnership.

James Okafor noted that cold-chain reliability is not only a scientific imperative but also a
trust-building exercise with health ministries and clinicians across sub-Saharan Africa and
Southeast Asia. "If a single batch of product is compromised, it sets the whole programme back
by years — not just commercially, but in terms of public confidence in gene therapy as a
modality," he said from Berlin. Fortridge Logistics has accordingly built a redundancy framework
that requires any shipment to pass through at least two independent temperature-monitoring
checkpoints before reaching the point of care. Beaumont is personally overseeing the certification
of each checkpoint facility, beginning with the Nairobi hub that is due to open by September 30 2024.

--- Investor Relations and Capital Markets Outlook ---

Blue Harbor Ventures, represented in this narrative by Thomas Hartwell, has been one of the
most vocal proponents of NexAgen Biotech in the public markets conversation. Hartwell's research
note generated significant attention in Boston and beyond, and the analyst has since been invited
to speak at conferences in Singapore and Amsterdam about the evolving biotech investment landscape.
He told attendees at the Amsterdam event that the NexAgen Biotech deal has "rewritten the valuation
playbook" for the sector, arguing that investors must now price in not just the clinical pipeline
but also the logistical and computational infrastructure that companies like NexAgen Biotech are
building around their science.

Raj Patel of Solaris Energy Group has echoed this view from a somewhat different vantage point.
Speaking to a group of institutional investors in Toronto, Patel argued that the energy and
healthcare sectors are converging in ways that most traditional investors are not yet pricing
correctly. He pointed to NexAgen Biotech's energy-intensive manufacturing processes and the
opportunity for Solaris Energy Group to deploy renewable energy solutions that reduce both costs
and carbon emissions. "The $17.5 million we put into NexAgen Biotech is not just a healthcare
bet. It is a bet on the energy-healthcare nexus," he said. Patel added that Solaris Energy Group
is in early discussions with two other gene-therapy manufacturers about similar partnerships,
though he declined to name them.

Marcus Chen of Meridian Capital Partners, meanwhile, is already fielding questions about a
potential initial public offering for NexAgen Biotech. Speaking from the firm's Singapore office,
Chen said the timing of any IPO will depend primarily on the clinical data readouts expected in
late 2024 and early 2025. "We are not in a hurry. The private markets are well-capitalised and
the company does not need public money to execute its current plan," he said. He noted that the
$2.4 billion raised in the most recent round gives NexAgen Biotech a runway of more than three
years at its current burn rate, providing ample time to generate the data that will ultimately
determine the company's public-market value. Eleanor Voss has said she wants to see at least
two of the five clinical programmes reach Phase 2 before taking the company public, a milestone
she believes is achievable by mid-2026.

--- Quantum Dynamics Inc: The Technology Behind the Science ---

The licensing deal between Quantum Dynamics Inc and NexAgen Biotech, worth $4.2 billion over
fifteen years, represents one of the largest technology transfer agreements in the history of
computational biology. Sofia Delgado, the chief executive of Quantum Dynamics Inc, has been
careful to position the transaction not as a sale of the technology but as a deep partnership
in which both companies share in the outcomes. Under the terms of the agreement, Quantum
Dynamics Inc will embed a team of twelve engineers and data scientists within NexAgen Biotech's
San Francisco research operations for the first two years, after which the capability will be
fully transferred to NexAgen Biotech's in-house team.

Delgado said the Austin-based firm is also exploring the possibility of a dedicated research
collaboration with the Berlin laboratory, where the European regulatory data requirements could
provide a rich training ground for the next generation of the protein-folding algorithms. James
Okafor confirmed that conversations between the Berlin team and Quantum Dynamics Inc are
ongoing, with an initial joint workshop planned for early 2025. If the collaboration proves
productive, Delgado said, Quantum Dynamics Inc would consider opening a small satellite office
in Berlin — a prospect that Okafor described as "enormously exciting for the European
computational biology community."

The December 1 2024 deadline for the initial payment tranche under the licensing agreement
has focused minds at both companies. Priya Nair said NexAgen Biotech's finance team, working
closely with Meridian Capital Partners, has structured the payment schedule to align with the
company's expected cash flows from its first commercial partnership in Southeast Asia. Linda
Beaumont of Fortridge Logistics noted that the supply-chain revenues from the NexAgen Biotech
agreement, including the $850 million preferred equity received at inception, give the logistics
company the financial flexibility to support NexAgen Biotech's payment obligations if needed —
though she was careful to note that this is not a formal commitment. "We are partners, not lenders.
But we are deeply invested in NexAgen Biotech's success, in every sense of the word," Beaumont
said.

Thomas Hartwell at Blue Harbor Ventures described the interlocking financial architecture of the
NexAgen Biotech ecosystem — the $2.4 billion from Meridian Capital Partners, the $850 million
equity to Fortridge Logistics, the $340 million infrastructure investment, the $17.5 million
stake from Solaris Energy Group, and the $4.2 billion licensing deal with Quantum Dynamics Inc —
as "one of the most sophisticated capital structures I have seen in twenty years of covering
the sector." He said the structure is designed to ensure that every major stakeholder has skin
in the game and that the incentives of all parties are closely aligned. "This is not a company
with passive investors and indifferent service providers. Everyone in this ecosystem has a reason
to want NexAgen Biotech to succeed," Hartwell said in a call from his Boston office.

Eleanor Voss, asked to summarise the vision behind NexAgen Biotech's ambitious global expansion,
returned to a theme she has articulated consistently since taking the helm: the moral urgency of
making gene therapy accessible beyond the wealthy markets in which it has historically been confined.
"The patients who need these therapies most are often in Nairobi, not in San Francisco or Boston.
They are in communities that have been overlooked by the pharmaceutical industry for generations.
NexAgen Biotech exists to change that," she said. With $2.4 billion in the bank, partners in
Berlin and Nairobi and Austin and Toronto and Singapore and Amsterdam, and a team she describes
as "the best collection of scientific and commercial talent I have ever worked with," Voss appears
to have the resources and the resolve to make good on that promise. The milestones of March 14 2024,
July 2 2024, September 30 2024, October 15 2024, and December 1 2024 are not just calendar dates —
they are the waypoints of a journey that, if it succeeds, could reshape the geography of medicine.

--- Post-Script: Key Figures and Their Roles ---

For readers new to the NexAgen Biotech story, a brief directory of the principal figures may be
helpful. Eleanor Voss is the chief executive, based in San Francisco. Marcus Chen is the deal
architect and managing director at Meridian Capital Partners in Singapore. Priya Nair is the
chief scientific officer, dividing her time between San Francisco, Nairobi, and Berlin.
Thomas Hartwell is the lead biotech analyst at Blue Harbor Ventures in Boston. Sofia Delgado
is the chief executive of Quantum Dynamics Inc in Austin. James Okafor heads European operations
from Berlin. Linda Beaumont leads Fortridge Logistics and oversees its Nairobi hub. Raj Patel
is the chief investment officer of Solaris Energy Group, based in Toronto. Together, these
eight individuals, across San Francisco, Singapore, Nairobi, Berlin, Boston, Austin, Amsterdam,
and Toronto, are building what may become one of the defining healthcare enterprises of the decade.
The financial scaffolding — $2.4 billion, $850 million, $340 million, $17.5 million, $4.2 billion —
and the temporal anchors — March 14 2024, July 2 2024, September 30 2024, October 15 2024,
December 1 2024 — provide the structure within which this ambition will either be realised or fall short.
"""

# Clean up the article
BASE_ARTICLE = BASE_ARTICLE.strip()
BASE_ARTICLE_WORDS = BASE_ARTICLE.split()


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def get_article_of_n_words(n: int) -> str:
    """Return first n words of the base article, or all if n > len."""
    words = BASE_ARTICLE_WORDS[:n]
    return " ".join(words)


def count_words(text: str) -> int:
    return len(text.split())


def naive_token_estimate(text: str) -> int:
    """Rough estimate: ~1.3 tokens per word for English prose."""
    return int(len(text.split()) * 1.3)


def call_api(messages: List[Dict], max_tokens: int = 1024) -> Dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "User-Agent": "python-benchmark/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_entities(article: str) -> Optional[Dict]:
    """Send extraction request, return parsed JSON or None on failure."""
    system_prompt = (
        "You are a precise Named Entity Recognition system. "
        "Extract entities from the provided text and return ONLY valid JSON — "
        "no markdown fences, no explanation, just raw JSON. "
        "The JSON must have exactly these keys: "
        "\"persons\", \"companies\", \"locations\", \"dates\", \"amounts\". "
        "Each key maps to a list of strings. "
        "Normalise each string to lowercase. "
        "Include every entity you can find, do not hallucinate."
    )
    user_prompt = (
        f"Extract all named entities from the following article.\n\n"
        f"Article:\n{article}\n\n"
        "Return ONLY a JSON object with keys: persons, companies, locations, dates, amounts."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    try:
        resp = call_api(messages, max_tokens=1024)
        raw = resp["choices"][0]["message"]["content"].strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
        # Try to isolate the JSON object
        first_brace = raw.find("{")
        last_brace  = raw.rfind("}")
        if first_brace != -1 and last_brace != -1:
            raw = raw[first_brace : last_brace + 1]
        parsed = json.loads(raw)
        # Normalise all values to lowercase strings
        result = {}
        for key in ("persons", "companies", "locations", "dates", "amounts"):
            vals = parsed.get(key, [])
            if isinstance(vals, list):
                result[key] = [str(v).lower().strip() for v in vals]
            else:
                result[key] = []
        # Grab token counts from response if available
        usage = resp.get("usage", {})
        result["_prompt_tokens"]     = usage.get("prompt_tokens", naive_token_estimate(article))
        result["_completion_tokens"] = usage.get("completion_tokens", 0)
        return result
    except Exception as exc:
        print(f"    [ERROR] extraction failed: {exc}")
        return None


def normalise(entity: str) -> str:
    """Lowercase and strip for comparison."""
    return entity.lower().strip()


def fuzzy_match(extracted: List[str], truth: List[str]) -> Tuple[Set, Set, Set]:
    """
    Returns (true_positives, false_positives, false_negatives).
    A match is counted if the truth string appears as a substring of the
    extracted string OR the extracted string appears as a substring of the
    truth string (handles partial / abbreviated matches).
    """
    norm_truth = [normalise(t) for t in truth]
    norm_extracted = [normalise(e) for e in extracted]

    matched_truth    = set()
    matched_extracted = set()

    for i, ext in enumerate(norm_extracted):
        for j, tru in enumerate(norm_truth):
            if tru in ext or ext in tru:
                matched_truth.add(j)
                matched_extracted.add(i)
                break

    true_positives   = matched_extracted   # extracted items that match a truth item
    false_positives  = set(range(len(norm_extracted))) - matched_extracted
    false_negatives  = set(range(len(norm_truth)))     - matched_truth
    return true_positives, false_positives, false_negatives


def compute_metrics(extracted_all: List[Optional[Dict]]) -> Dict:
    """Compute precision, recall, F1 across all entity types, aggregated over runs."""
    all_tp = 0
    all_fp = 0
    all_fn = 0

    per_run_f1 = []

    for extracted in extracted_all:
        if extracted is None:
            per_run_f1.append(0.0)
            continue

        run_tp = run_fp = run_fn = 0
        for key in ("persons", "companies", "locations", "dates", "amounts"):
            tp, fp, fn = fuzzy_match(extracted.get(key, []), GROUND_TRUTH[key])
            run_tp += len(tp)
            run_fp += len(fp)
            run_fn += len(fn)

        all_tp += run_tp
        all_fp += run_fp
        all_fn += run_fn

        precision_r = run_tp / (run_tp + run_fp) if (run_tp + run_fp) > 0 else 0.0
        recall_r    = run_tp / (run_tp + run_fn) if (run_tp + run_fn) > 0 else 0.0
        f1_r = (2 * precision_r * recall_r / (precision_r + recall_r)
                if (precision_r + recall_r) > 0 else 0.0)
        per_run_f1.append(round(f1_r, 4))

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall    = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    f1_std = (
        (sum((x - (sum(per_run_f1) / len(per_run_f1))) ** 2 for x in per_run_f1) / len(per_run_f1)) ** 0.5
        if per_run_f1 else 0.0
    )

    consistent = f1_std < 0.05  # less than 5% std dev across runs

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "per_run_f1": per_run_f1,
        "f1_std":    round(f1_std, 4),
        "consistent": consistent,
    }


def analyse_failures(extracted_all: List[Optional[Dict]], word_count: int) -> Dict:
    """Collect hallucinations and missed entities for failure analysis."""
    hallucinated = {k: [] for k in ("persons", "companies", "locations", "dates", "amounts")}
    missed       = {k: [] for k in ("persons", "companies", "locations", "dates", "amounts")}

    for extracted in extracted_all:
        if extracted is None:
            continue
        for key in ("persons", "companies", "locations", "dates", "amounts"):
            _, fp_idxs, fn_idxs = fuzzy_match(
                extracted.get(key, []), GROUND_TRUTH[key]
            )
            norm_ext  = [normalise(e) for e in extracted.get(key, [])]
            norm_truth = [normalise(t) for t in GROUND_TRUTH[key]]
            for i in fp_idxs:
                hallucinated[key].append(norm_ext[i])
            for j in fn_idxs:
                missed[key].append(norm_truth[j])

    # Deduplicate
    for key in hallucinated:
        hallucinated[key] = list(set(hallucinated[key]))
    for key in missed:
        missed[key] = list(set(missed[key]))

    return {"hallucinated": hallucinated, "missed": missed}


# ---------------------------------------------------------------------------
# MAIN BENCHMARK
# ---------------------------------------------------------------------------

def run_benchmark():
    print("=" * 70)
    print("NER / STRUCTURED EXTRACTION BENCHMARK")
    print(f"Model: {MODEL}  |  Repeats per size: {REPEATS}")
    print("=" * 70)

    # Verify article covers all entities
    art_lower = BASE_ARTICLE.lower()
    print("\nGround-truth entity coverage check:")
    for category, entities in GROUND_TRUTH.items():
        for ent in entities:
            found = ent in art_lower
            marker = "✓" if found else "✗ MISSING"
            print(f"  [{category}] {ent}: {marker}")
    print()

    results = []
    inflection_point = None

    header = (f"{'words':>6}  {'~tokens':>8}  {'precision':>9}  "
              f"{'recall':>7}  {'f1':>6}  {'consistent':>10}")
    print(header)
    print("-" * len(header))

    for target_words in TARGET_WORD_COUNTS:
        article = get_article_of_n_words(target_words)
        actual_words = count_words(article)
        est_tokens   = naive_token_estimate(article)

        print(f"\n[{actual_words} words / ~{est_tokens} tokens] Running {REPEATS} extractions...", flush=True)

        run_results = []
        prompt_tokens_list = []

        for run_idx in range(REPEATS):
            print(f"  Run {run_idx + 1}/{REPEATS}...", end=" ", flush=True)
            extracted = extract_entities(article)
            run_results.append(extracted)
            if extracted:
                pt = extracted.get("_prompt_tokens", est_tokens)
                prompt_tokens_list.append(pt)
                print(f"OK (prompt_tokens={pt})")
            else:
                print("FAILED")
            time.sleep(0.5)  # be polite to the API

        avg_prompt_tokens = (
            round(sum(prompt_tokens_list) / len(prompt_tokens_list))
            if prompt_tokens_list else est_tokens
        )

        metrics = compute_metrics(run_results)
        failures = analyse_failures(run_results, actual_words)

        row = {
            "target_words":       target_words,
            "actual_words":       actual_words,
            "prompt_tokens_est":  est_tokens,
            "prompt_tokens_api":  avg_prompt_tokens,
            "precision":          metrics["precision"],
            "recall":             metrics["recall"],
            "f1":                 metrics["f1"],
            "f1_std":             metrics["f1_std"],
            "per_run_f1":         metrics["per_run_f1"],
            "consistent":         metrics["consistent"],
            "failures":           failures,
            "raw_extractions":    [
                {k: v for k, v in (r if r else {}).items() if not k.startswith("_")}
                for r in run_results
            ],
        }
        results.append(row)

        consistent_str = "yes" if metrics["consistent"] else "NO"
        print(f"  {'words':>5}: {actual_words:>5}  "
              f"tokens: {avg_prompt_tokens:>5}  "
              f"P={metrics['precision']:.3f}  "
              f"R={metrics['recall']:.3f}  "
              f"F1={metrics['f1']:.3f}  "
              f"std={metrics['f1_std']:.3f}  "
              f"consistent={consistent_str}")

        # Track inflection point
        if inflection_point is None and (metrics["f1"] < 0.8 or not metrics["consistent"]):
            inflection_point = row

    # -----------------------------------------------------------------------
    # SUMMARY TABLE
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'doc_size_words':>15}  {'prompt_tokens':>13}  "
          f"{'precision':>9}  {'recall':>7}  {'f1':>6}  {'consistent':>10}")
    print("-" * 80)
    for r in results:
        consistent_str = "yes" if r["consistent"] else "NO"
        flag = " <-- DEGRADED" if r["f1"] < 0.8 or not r["consistent"] else ""
        print(f"{r['actual_words']:>15}  {r['prompt_tokens_api']:>13}  "
              f"{r['precision']:>9.3f}  {r['recall']:>7.3f}  "
              f"{r['f1']:>6.3f}  {consistent_str:>10}{flag}")

    # -----------------------------------------------------------------------
    # FINDINGS
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINDINGS")
    print("=" * 80)

    reliable = [r for r in results if r["f1"] >= 0.8 and r["consistent"]]
    degraded = [r for r in results if r["f1"] < 0.8 or not r["consistent"]]

    if reliable:
        best = max(reliable, key=lambda x: x["actual_words"])
        print(f"\nMax reliable document size: {best['actual_words']} words "
              f"/ {best['prompt_tokens_api']} tokens  "
              f"(F1={best['f1']:.3f}, consistent={best['consistent']})")
    else:
        print("\nNo document size met the reliability threshold (F1 >= 0.8, consistent).")

    if degraded:
        first_deg = degraded[0]
        print(f"\nFirst degradation at: {first_deg['actual_words']} words "
              f"/ {first_deg['prompt_tokens_api']} tokens")
        print(f"  F1={first_deg['f1']:.3f}  "
              f"std={first_deg['f1_std']:.3f}  "
              f"consistent={first_deg['consistent']}")
        if reliable:
            prev = max(reliable, key=lambda x: x["actual_words"])
            drop = prev["f1"] - first_deg["f1"]
            print(f"  F1 drop from previous size: {drop:.3f} ({drop/prev['f1']*100:.1f}%)")

    # Interesting failure modes
    print("\nNotable failure modes:")
    for r in results:
        f = r["failures"]
        hallu_total = sum(len(v) for v in f["hallucinated"].values())
        missed_total = sum(len(v) for v in f["missed"].values())
        if hallu_total > 0 or missed_total > 0:
            print(f"\n  [{r['actual_words']} words]")
            for cat in ("persons", "companies", "locations", "dates", "amounts"):
                if f["hallucinated"][cat]:
                    print(f"    HALLUCINATED [{cat}]: {f['hallucinated'][cat]}")
                if f["missed"][cat]:
                    print(f"    MISSED       [{cat}]: {f['missed'][cat]}")

    # -----------------------------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------------------------
    output_path = "results.json"
    with open(output_path, "w") as fh:
        json.dump(
            {
                "model":            MODEL,
                "api_url":          API_URL,
                "repeats":          REPEATS,
                "ground_truth":     GROUND_TRUTH,
                "inflection_point": inflection_point,
                "results":          results,
            },
            fh,
            indent=2,
        )
    print(f"\nFull results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
