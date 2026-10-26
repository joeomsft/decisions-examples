import json
import os
from dataclasses import dataclass, asdict
from time import perf_counter
from typing import Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from typesafe_sdk import Noul, Score, Choice, RetryPolicy, TypeSafeClient
from dotenv import load_dotenv

from benchmark import TokenPricing

load_dotenv()

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

ENDPOINT = os.environ["FOUNDRY_DECISIONS_ENDPOINT"]
DEPLOYMENT_NAME = "Decision-1"
TOKEN_SCOPE = "https://ai.azure.com/.default"

# Optional USD-per-million-token pricing: DECISIONS_INPUT_USD_PER_MILLION
# and DECISIONS_OUTPUT_USD_PER_MILLION. No prices are assumed.
# The SDK does not expose cache/reasoning counts, so those remain unknown.
# Cost applies the supplied input rate to all reported input tokens; use
# the effective rate for your deployment if billing includes discounts.
# Latency includes the request and SDK response parsing. Workflow timing
# covers extraction through ranking, excluding printing and the explanation stub.


# -------------------------------------------------------------------
# Stubbed SAM domain objects
# -------------------------------------------------------------------

@dataclass
class Submission:
    submission_id: str
    business_description: str
    annual_revenue: int
    employee_count: int
    requested_limit: int
    prior_claims_count: int
    states_of_operation: list[str]
    extracted_text: str


@dataclass
class MarketCandidate:
    market_id: str
    market_name: str
    appetite_summary: str
    max_revenue: int
    max_requested_limit: int
    allowed_states: list[str]
    preferred_industries: list[str]


@dataclass
class MarketScore:
    market_id: str
    market_name: str
    eligibility: str
    appetite_score: float
    financial_fit_score: float
    risk_quality_score: float
    geography_score: float
    final_score: float
    requires_review: bool
    output: dict[str, Any]


# -------------------------------------------------------------------
# Stub 1: Extract facts from the submission
#
# In the real SAM flow, this might use Content Understanding,
# Document Intelligence, OCR, or an existing extraction pipeline.
# -------------------------------------------------------------------

def extract_submission_facts() -> Submission:
    return Submission(
        submission_id="SAM-2026-1042",
        business_description=(
            "Regional commercial roofing contractor serving Michigan "
            "and Ohio. Approximately 20 percent of work involves buildings "
            "over four stories."
        ),
        annual_revenue=8_500_000,
        employee_count=42,
        requested_limit=2_000_000,
        prior_claims_count=1,
        states_of_operation=["MI", "OH"],
        extracted_text=(
            "Commercial roofing contractor. Annual revenue $8.5 million. "
            "42 employees. Requested liability limit $2 million. "
            "One prior claim in five years. Operations in Michigan and Ohio."
        ),
    )


# -------------------------------------------------------------------
# Stub 2: SAM's existing low-cost matching step
#
# This represents retrieval, filtering, rules, embeddings, or another
# inexpensive process that finds potentially relevant markets.
#
# The point is to avoid running Decision-1 against every market in the
# catalog when basic deterministic exclusions can be applied first.
# -------------------------------------------------------------------

def find_candidate_markets(
    submission: Submission,
) -> list[MarketCandidate]:
    del submission

    return [
        MarketCandidate(
            market_id="MKT-101",
            market_name="Great Lakes Specialty",
            appetite_summary=(
                "Targets regional commercial contractors with revenue "
                "below $15 million. Roofing risks are permitted but "
                "height exposure may require underwriting review."
            ),
            max_revenue=15_000_000,
            max_requested_limit=5_000_000,
            allowed_states=["MI", "OH", "IN"],
            preferred_industries=[
                "commercial construction",
                "specialty contractors",
            ],
        ),
        MarketCandidate(
            market_id="MKT-205",
            market_name="Midwest Main Street",
            appetite_summary=(
                "Prefers small contractors with revenue below $5 million. "
                "Commercial roofing is generally outside appetite."
            ),
            max_revenue=5_000_000,
            max_requested_limit=2_000_000,
            allowed_states=["MI", "OH"],
            preferred_industries=[
                "retail",
                "professional services",
                "light contracting",
            ],
        ),
        MarketCandidate(
            market_id="MKT-314",
            market_name="National Contractor Select",
            appetite_summary=(
                "Accepts commercial contractors with revenue up to "
                "$25 million. Roofing is acceptable when loss history "
                "is favorable and operations are confined to approved states."
            ),
            max_revenue=25_000_000,
            max_requested_limit=10_000_000,
            allowed_states=["MI", "OH", "IN", "IL", "WI"],
            preferred_industries=[
                "commercial construction",
                "roofing",
                "specialty contractors",
            ],
        ),
    ]


# -------------------------------------------------------------------
# Decision-1 scoring step
#
# Decision-1 is used for bounded judgments:
#   - eligibility classification
#   - discrete rule scores
#   - binary review flags
#
# The weighted calculation remains in normal application code so the
# formula is deterministic, testable, versioned, and auditable.
# -------------------------------------------------------------------

def score_market_with_decision_model(
    client: TypeSafeClient,
    submission: Submission,
    market: MarketCandidate,
    pricing: TokenPricing | None = None,
) -> MarketScore:

    state = f"""
INSURANCE SUBMISSION

Submission ID: {submission.submission_id}
Business: {submission.business_description}
Annual revenue: ${submission.annual_revenue:,}
Employees: {submission.employee_count}
Requested limit: ${submission.requested_limit:,}
Prior claims: {submission.prior_claims_count}
States of operation: {", ".join(submission.states_of_operation)}

CANDIDATE MARKET

Market ID: {market.market_id}
Market name: {market.market_name}
Appetite: {market.appetite_summary}
Maximum revenue: ${market.max_revenue:,}
Maximum requested limit: ${market.max_requested_limit:,}
Allowed states: {", ".join(market.allowed_states)}
Preferred industries: {", ".join(market.preferred_industries)}
""".strip()

    started = perf_counter()
    response = client.system_one(
        state=state,
        questions={
            # A categorical gate that SAM can use before ranking.
            "eligibility": Choice(
                instructions=(
                    "Classify whether this insurance submission is eligible "
                    "for the candidate market. Base the classification only "
                    "on the supplied submission and market information."
                ),
                criteria={
                    "eligible": (
                        "The submission clearly satisfies the stated "
                        "market constraints."
                    ),
                    "conditional": (
                        "The submission may fit, but one or more factors "
                        "require underwriting review."
                    ),
                    "ineligible": (
                        "The submission clearly violates the stated "
                        "market constraints or appetite."
                    ),
                },
            ),

            # Scores can map to Kingsly's existing SAM rule groups.
            "appetite_fit": Score(
                instructions=(
                    "Score how closely the insured's business and operations "
                    "fit the candidate market's stated appetite."
                ),
                criteria=[
                    "Outside appetite",
                    "Weak fit",
                    "Possible fit",
                    "Good fit",
                    "Excellent fit",
                ],
            ),
            "financial_fit": Score(
                instructions=(
                    "Score the fit based on revenue and requested policy "
                    "limit relative to the market's stated limits."
                ),
                criteria=[
                    "Clearly exceeds market limits",
                    "Material mismatch",
                    "Borderline fit",
                    "Within limits",
                    "Comfortably within limits",
                ],
            ),
            "risk_quality": Score(
                instructions=(
                    "Score the overall risk quality using the business "
                    "description, operational exposures, and prior claims."
                ),
                criteria=[
                    "Very unfavorable",
                    "Unfavorable",
                    "Mixed or uncertain",
                    "Favorable",
                    "Very favorable",
                ],
            ),
            "geography_fit": Score(
                instructions=(
                    "Score whether all states of operation fit the market's "
                    "allowed geographic footprint."
                ),
                criteria=[
                    "Operations are outside the allowed footprint",
                    "Major geographic mismatch",
                    "Partial geographic fit",
                    "Most operations fit",
                    "All operations fit",
                ],
            ),

            # A Noul is useful for a boolean or probabilistic signal.
            "requires_underwriter_review": Noul(
                instructions=(
                    "Does this submission contain an exposure, ambiguity, "
                    "or borderline condition that should be reviewed by a "
                    "human underwriter before placement?"
                )
            ),
        },
    )

    latency_seconds = perf_counter() - started
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    total_tokens = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None else None
    )
    estimated_cost = (
        pricing.estimate_cost(input_tokens, 0, output_tokens)
        if pricing is not None and input_tokens is not None and output_tokens is not None
        else None
    )

    # Normalize SDK scores to a 0-100 range.
    # This is stub logic. Confirm the actual score range returned by
    # the deployed SDK/model and adjust as necessary.
    appetite_score = normalize_score(
        response.scores["appetite_fit"].score
    )
    financial_score = normalize_score(
        response.scores["financial_fit"].score
    )
    risk_score = normalize_score(
        response.scores["risk_quality"].score
    )
    geography_score = normalize_score(
        response.scores["geography_fit"].score
    )

    # Example SAM rule-group weighting.
    #
    # These weights should come from SAM's configured rule definitions,
    # not be permanently hard-coded.
    final_score = calculate_weighted_score(
        scores={
            "appetite_fit": appetite_score,
            "financial_fit": financial_score,
            "risk_quality": risk_score,
            "geography_fit": geography_score,
        },
        weights={
            "appetite_fit": 0.40,
            "financial_fit": 0.20,
            "risk_quality": 0.25,
            "geography_fit": 0.15,
        },
    )

    review_probability = (
        response.nouls["requires_underwriter_review"].noul
    )

    return MarketScore(
        market_id=market.market_id,
        market_name=market.market_name,
        eligibility=response.choices["eligibility"].choice,
        appetite_score=appetite_score,
        financial_fit_score=financial_score,
        risk_quality_score=risk_score,
        geography_score=geography_score,
        final_score=final_score,
        requires_review=review_probability >= 0.50,
        output={
            "review_probability": review_probability,
            "raw_response": str(response),
            "metrics": {
                "model": response.model,
                "latency_seconds": latency_seconds,
                "input_tokens": input_tokens,
                "cached_input_tokens": None,
                "output_tokens": output_tokens,
                "reasoning_tokens": None,
                "total_tokens": total_tokens,
                "estimated_cost_usd": estimated_cost,
            },
        },
    )


# -------------------------------------------------------------------
# Supporting deterministic logic
# -------------------------------------------------------------------

def normalize_score(score: float) -> float:
    score = max(1.0, min(float(score), 5.0))
    return round(((score - 1.0) / 4.0) * 100.0, 2)


def calculate_weighted_score(
    scores: dict[str, float],
    weights: dict[str, float],
) -> float:
    total_weight = sum(weights.values())

    if total_weight <= 0:
        raise ValueError("Rule weights must total more than zero.")

    weighted_total = sum(
        scores[rule_name] * weight
        for rule_name, weight in weights.items()
    )

    return round(weighted_total / total_weight, 2)


def rank_candidates(
    results: list[MarketScore],
) -> list[MarketScore]:
    eligibility_order = {
        "eligible": 2,
        "conditional": 1,
        "ineligible": 0,
    }

    return sorted(
        results,
        key=lambda result: (
            eligibility_order.get(result.eligibility, -1),
            result.final_score,
        ),
        reverse=True,
    )


# -------------------------------------------------------------------
# Optional final generative step
#
# Only the highest-ranked candidates would be passed to a more
# expensive model for a broker-facing narrative, comparison, or
# recommendation.
# -------------------------------------------------------------------

def generate_broker_explanation(
    submission: Submission,
    ranked_results: list[MarketScore],
) -> str:
    del submission

    top_results = ranked_results[:3]

    return (
        "Stub: use a larger generative model to explain the top "
        f"{len(top_results)} markets, supported by Decision-1 scores "
        "and source submission facts."
    )


# -------------------------------------------------------------------
# Main SAM orchestration
# -------------------------------------------------------------------

def main() -> None:
    workflow_started = perf_counter()
    pricing = TokenPricing.model_validate({
        "input_usd_per_million": os.environ.get("DECISIONS_INPUT_USD_PER_MILLION"),
        "output_usd_per_million": os.environ.get("DECISIONS_OUTPUT_USD_PER_MILLION"),
    })
    submission = extract_submission_facts()

    # Existing inexpensive SAM matching stage.
    candidates = find_candidate_markets(submission)

    with DefaultAzureCredential() as credential:
        token_provider = get_bearer_token_provider(
            credential,
            TOKEN_SCOPE,
        )

        with TypeSafeClient(
            api_key=token_provider(),
            base_url=ENDPOINT,
            model=DEPLOYMENT_NAME,
            timeout=60,
            # Hidden retries would under-report billed tokens in a benchmark.
            retry=RetryPolicy(max_retries=0),
        ) as client:

            scoring_started = perf_counter()
            scored_candidates = [
                score_market_with_decision_model(
                    client=client,
                    submission=submission,
                    market=market,
                    pricing=pricing,
                )
                for market in candidates
            ]
            scoring_seconds = perf_counter() - scoring_started

    ranked_candidates = rank_candidates(scored_candidates)
    workflow_seconds = perf_counter() - workflow_started
    call_metrics = [result.output["metrics"] for result in scored_candidates]
    benchmark = {
        "deployment": DEPLOYMENT_NAME,
        "request_count": len(call_metrics),
        "scoring_seconds": scoring_seconds,
        "workflow_seconds": workflow_seconds,
        "decision_latency_seconds": sum(m["latency_seconds"] for m in call_metrics),
        **{
            key: (
                sum(m[key] for m in call_metrics)
                if all(m[key] is not None for m in call_metrics) else None
            )
            for key in (
                "input_tokens", "cached_input_tokens", "output_tokens",
                "reasoning_tokens", "total_tokens", "estimated_cost_usd",
            )
        },
        "pricing_usd_per_million_tokens": pricing.model_dump(),
        "max_retries": 0,
    }

    print(f"Submission: {submission.submission_id}")
    print("Ranked candidate markets:")

    for position, result in enumerate(ranked_candidates, start=1):
        print(
            f"\n{position}. {result.market_name}"
            f"\n   Eligibility: {result.eligibility}"
            f"\n   Final score: {result.final_score:.2f}"
            f"\n   Appetite fit: {result.appetite_score:.2f}"
            f"\n   Financial fit: {result.financial_fit_score:.2f}"
            f"\n   Risk quality: {result.risk_quality_score:.2f}"
            f"\n   Geography fit: {result.geography_score:.2f}"
            f"\n   Requires review: {result.requires_review}"
            f"\n   Decision metrics: {json.dumps(result.output['metrics'])}"
        )

    print(f"\nBenchmark:\n{json.dumps(benchmark, indent=2)}")
    if any(m["total_tokens"] is None for m in call_metrics):
        print("Token usage unavailable for some requests; affected totals are null.")
    print("Cached-input and reasoning token counts are not exposed by the decisions SDK.")
    if benchmark["estimated_cost_usd"] is None:
        print(
            "Cost unavailable: both input/output token counts and "
            "DECISIONS_INPUT_USD_PER_MILLION / DECISIONS_OUTPUT_USD_PER_MILLION "
            "pricing are required."
        )
    explanation = generate_broker_explanation(
        submission,
        ranked_candidates,
    )

    print(f"\nBroker-facing explanation:\n{explanation}")

    # Example structured result for storage or downstream processing.
    result_payload = {
        "submission_id": submission.submission_id,
        "benchmark": benchmark,
        "ranked_markets": [
            asdict(result) for result in ranked_candidates
        ],
    }

    print(f"\nStructured SAM result:\n{json.dumps(result_payload, indent=2)}")


if __name__ == "__main__":
    main()