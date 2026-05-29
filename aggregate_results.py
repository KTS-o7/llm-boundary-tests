#!/usr/bin/env python3
"""
Aggregate all extended benchmark results into a structured summary.
Reads results.json from 7-api-features, 8-parameter-tuning, and 9-reliability.
Writes aggregate_results.json to the repo root.
"""

import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))

def load(path):
    with open(os.path.join(BASE, path)) as f:
        return json.load(f)


def summarize_features(data):
    g = data.get("test_groups", {})
    rf = g.get("response_format", {})
    seed = g.get("seed", {})
    stop = g.get("stop", {})
    lp = g.get("logprobs", {})
    tools = g.get("tools", {})
    nc = g.get("n_choices", {})
    mt = g.get("max_tokens_ceiling", {})
    mv = g.get("message_validation", {})

    summary = {
        "response_format_json_object": {
            "supported": rf.get("accepted", False),
            "json_parse_rate_with": rf.get("json_success_rate_with", 0),
            "json_parse_rate_without": rf.get("json_success_rate_without", 0),
        },
        "seed_parameter": {
            "accepted": True,
            "deterministic": seed.get("deterministic_with_seed", False),
            "different_seeds_different": seed.get("different_seeds_produce_different_outputs", False),
        },
        "stop_sequences": {
            "works": stop.get("stop_works", False),
            "correct_truncations": stop.get("correctly_truncated", 0),
            "failures": stop.get("failed", 0),
        },
        "logprobs": {
            "accepted": lp.get("supported", False),
            "logprobs_returned": lp.get("logprobs_returned", False),
        },
        "tools_function_calling": {
            "accepted": tools.get("supported", False),
            "returned_tool_calls": tools.get("returned_tool_calls", False),
        },
        "n_multiple_choices": {
            "accepted": nc.get("supported", False),
            "choices_returned": nc.get("choices_count", 0),
        },
        "max_output_tokens_ceiling": {
            "max_output_tokens_observed": mt.get("max_output_tokens", 0),
            "max_requested_before_error": mt.get("max_before_error", 0),
        },
        "error_informativeness": {
            "informative": mv.get("informative_count", 0),
            "total_error_tests": mv.get("total_count", 0),
        },
    }

    unsupported = []
    for name, info in summary.items():
        if isinstance(info, dict) and info.get("supported") is False and info.get("accepted") is False:
            unsupported.append(name)
        elif isinstance(info, dict) and info.get("accepted") is True and info.get("works") is False:
            unsupported.append(name)

    summary["_unsupported_features"] = [
        k for k, v in summary.items()
        if isinstance(v, dict) and v.get("logprobs_returned") is False and "logprobs" in k
    ]

    return summary


def summarize_parameter_tuning(data):
    ts = data.get("temperature_sweep", {})
    tp = data.get("top_p_sweep", {})
    fp = data.get("frequency_penalty_sweep", {})
    pp = data.get("presence_penalty_sweep", {})
    bc = data.get("best_combo", {})

    return {
        "temperature_sweep": {
            "best": ts.get("best", {}),
            "values": ts.get("values", []),
            "results": ts.get("results", {}),
        },
        "top_p_sweep": {
            "best": tp.get("best", {}),
            "values": tp.get("values", []),
            "results": tp.get("results", {}),
        },
        "frequency_penalty_sweep": {
            "best": fp.get("best", {}),
            "values": fp.get("values", []),
            "results": fp.get("results", {}),
        },
        "presence_penalty_sweep": {
            "best": pp.get("best", {}),
            "values": pp.get("values", []),
            "results": pp.get("results", {}),
        },
        "best_combo": {
            "optimal_config": bc.get("optimal_config", {}),
            "optimal_scores": bc.get("optimal_scores", {}),
            "default_scores": bc.get("default_scores", {}),
            "delta": bc.get("delta", {}),
        },
        "overall_assessment": (
            "Default parameters (temp=0, no penalties) are essentially optimal. "
            "Penalty parameters provide no meaningful reduction in scope creep. "
            "Higher temperatures degrade faithfulness without improving consistency."
        ),
    }


def summarize_reliability(data):
    mt = data.get("test_1_multi_turn", {})
    sl = data.get("test_2_sustained_load", {})
    ec = data.get("test_3_error_catalog", {})
    ei = data.get("test_4_edge_inputs", {})

    # Multi-turn
    chains = mt.get("chains", [])
    mt_summary = {}
    for c in chains:
        name = f'Chain {c["chain"]} ({c["name"]})'
        mt_summary[name] = {
            k: v for k, v in c.items() if k not in ("turns", "chain", "name")
        }

    # Sustained load
    sl_summary = {
        "overall": sl.get("overall", {}),
        "buckets": [
            {"range": b["bucket"], "errors": b["errors"],
             "p50": b["p50"], "p95": b["p95"], "p99": b["p99"]}
            for b in sl.get("buckets", [])
        ],
    }

    # Error catalog
    ec_summary = ec.get("summary", {})
    ec_detail = [
        {"test": e["test"], "status": e["status"], "informative": e["informative"]}
        for e in ec.get("catalog", [])
    ]

    # Edge inputs
    ei_handled = sum(1 for e in ei.get("edge_inputs", []) if e.get("handled_gracefully"))
    ei_total = len(ei.get("edge_inputs", []))

    return {
        "multi_turn": mt_summary,
        "sustained_load": sl_summary,
        "error_catalog": {
            "summary": ec_summary,
            "details": ec_detail,
        },
        "edge_inputs": {
            "handled_gracefully": ei_handled,
            "total": ei_total,
        },
    }


def main():
    features_data = load("7-api-features/results.json")
    tuning_data = load("8-parameter-tuning/results.json")
    reliability_data = load("9-reliability/results.json")

    aggregated = {
        "api_feature_discovery": summarize_features(features_data),
        "parameter_optimization": summarize_parameter_tuning(tuning_data),
        "reliability_and_boundaries": summarize_reliability(reliability_data),
    }

    out_path = os.path.join(BASE, "aggregate_results.json")
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"Aggregated results written to {out_path}")
    print(f"\nFeature discovery: {len(aggregated['api_feature_discovery'])} dimensions")
    print(f"Parameter tuning: temperature/top_p/freq/presence sweeps + best combo")
    print(f"Reliability: {len(aggregated['reliability_and_boundaries'])} test groups")


if __name__ == "__main__":
    main()
