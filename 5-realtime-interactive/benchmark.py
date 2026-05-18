import os
#!/usr/bin/env python3
"""
Real-time Interactive Latency Benchmark for LLM API
Tests p50/p95/p99 latency across query categories, streaming vs non-streaming,
sustained burst load, and finds the max safe tokens for sub-1s/sub-2s use.
"""

import json
import time
import statistics
import requests
from datetime import datetime, timezone

API_URL = "https://ai.shenthar.me/v1/chat/completions"
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = "taalas-llama3.1-8b"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ── Category prompts ────────────────────────────────────────────────────────

PING_PROMPTS = [
    'Reply with exactly: {"ok": true}',
    'Return only: {"ok": true}',
    'Output: {"ok": true} — nothing else.',
    'JSON: {"ok": true}',
    'Respond with {"ok": true} and nothing more.',
    'Your entire response must be: {"ok": true}',
    'Single JSON: {"ok": true}',
    '{"ok": true} — that is your complete answer.',
    'Just output {"ok": true}',
    'Return the JSON object {"ok": true}',
]

ONE_LINER_PROMPTS = [
    'Is Python interpreted? Reply as JSON: {"answer": "yes" or "no", "reason": "one sentence"}',
    'Classify "The sky is blue" — return {"category": "...", "sentiment": "..."}',
    'What is 2+2? Return {"result": number}',
    'Is HTTP stateless? Return {"answer": "yes"|"no", "note": "brief explanation"}',
    'Capital of France? Return {"city": "...", "country": "..."}',
    'Is JSON a binary format? Return {"answer": "yes"|"no"}',
    'What language runs in browsers natively? Return {"language": "...", "note": "one line"}',
    'Does TCP guarantee delivery? Return {"answer": "yes"|"no", "mechanism": "brief"}',
    'Is REST an acronym? Return {"answer": "yes"|"no", "full_form": "..."}',
    'What port does HTTPS use? Return {"port": number, "protocol": "..."}',
]

SHORT_PROMPTS = [
    'Return a JSON object describing an HTTP 200 response with fields: status, code, message, cacheable, idempotent.',
    'Return a JSON user profile with fields: id, name, email, role, active.',
    'Return a JSON error object with fields: error, code, message, retryable, timestamp.',
    'Return a JSON API rate-limit descriptor: limit, remaining, reset_at, window_seconds, exceeded.',
    'Return a JSON database connection info: host, port, db_name, pool_size, ssl_enabled.',
    'Return a JSON JWT payload: sub, iat, exp, role, issuer.',
    'Return a JSON for a currency: code, symbol, name, decimal_places, country.',
    'Return a JSON pagination object: page, per_page, total, total_pages, has_next.',
    'Return a JSON color descriptor: name, hex, rgb_r, rgb_g, rgb_b.',
    'Return a JSON for a git commit: hash, author, date, message, files_changed.',
]

MEDIUM_PROMPTS = [
    'Return a JSON object for a REST API endpoint descriptor with at least 12 fields: path, method, auth_required, rate_limit, request_body schema, response schema, status_codes array, version, deprecated, description, example_request, example_response.',
    'Return a JSON for a server health-check with 15 fields covering: uptime, memory used/total, cpu_percent, disk_used/total, active_connections, request_rate, error_rate, latency_p50, latency_p99, db_status, cache_status, queue_depth, version, environment, timestamp.',
    'Return a JSON for a shopping cart with 10+ fields: cart_id, user_id, items (array of 3 products each with id/name/qty/price), subtotal, tax, discount, total, currency, created_at, expires_at.',
    'Return a JSON analytics event batch: event_id, session_id, user_id, timestamp, page, referrer, device, browser, country, city, events array of 5 items each with name/value/ts.',
    'Return a JSON for a Kubernetes pod spec with 12+ fields: name, namespace, image, cpu_request, cpu_limit, memory_request, memory_limit, replicas, env_vars (3 items), ports (2 items), labels, restart_policy.',
    'Return a JSON for a financial transaction: txn_id, from_account, to_account, amount, currency, exchange_rate, fee, net_amount, status, created_at, settled_at, reference, metadata object with 4 fields.',
    'Return a JSON for an OAuth2 token response with all standard fields plus: access_token, token_type, expires_in, refresh_token, scope, id_token, issued_at, client_id, user_info object with 5 fields.',
    'Return a JSON for a CI/CD pipeline run: pipeline_id, repo, branch, commit, triggered_by, status, started_at, finished_at, duration_s, stages array of 4 stages each with name/status/duration, artifacts array of 2 items.',
    'Return a JSON for a product catalog item with 15 fields: id, sku, name, description, category, tags array, price, sale_price, currency, stock, weight, dimensions object, images array of 3, supplier, last_updated.',
    'Return a JSON for a DNS record set with record_type, domain, ttl, and records array of 10 entries each having name, type, value, priority.',
]

LONG_PROMPTS = [
    'Return a comprehensive JSON for a cloud infrastructure deployment manifest. Include: project, environment, region, vpc_config (cidr, subnets array of 5), security_groups array of 3 (each with name, rules array of 4), ec2_instances array of 5 (each with id, type, ami, tags, ebs_volumes array of 2), rds_config (engine, version, instance_class, multi_az, backup_retention, parameter_group, subnet_group), elasticache config, s3_buckets array of 4 (each with name, versioning, lifecycle_rules array of 2, cors), iam_roles array of 3 (each with name, policies array of 3), cloudwatch_alarms array of 5, tags, created_at, updated_at. Make all values realistic.',
    'Return a comprehensive JSON for a microservices application registry. Include: app_name, version, services array of 8 services (each with name, image, replicas, cpu, memory, env_vars array of 4, ports array of 2, health_check object, dependencies array, secrets array of 2), databases array of 3 (each with name, type, host, port, credentials_ref), message_queues array of 2 (each with name, type, topics array of 3), api_gateway config with routes array of 10 (each with path, method, service, auth, rate_limit), monitoring config with metrics array of 6, alerts array of 4, deployment_strategy, rollback_config.',
    'Return a comprehensive JSON for a user permissions and RBAC system. Include: system_name, roles array of 6 roles (each with id, name, description, permissions array of 8, resource_types array of 4, conditions object), users array of 5 users (each with id, name, email, roles array, teams array, mfa_enabled, last_login, api_keys array of 2), teams array of 4 teams (each with id, name, members array of 3, role_bindings array of 2), policies array of 5 policies (each with id, name, effect, actions array of 5, resources array of 3, conditions object with 3 fields), audit_log array of 8 recent entries (each with timestamp, actor, action, resource, result).',
    'Return a comprehensive JSON for an e-commerce order fulfillment record. Include: order_id, customer object (id, name, email, phone, tier), billing_address, shipping_address, items array of 8 line items (each with product_id, sku, name, qty, unit_price, discount, subtotal, weight, hs_code), payment object (method, gateway, txn_id, amount, currency, status, captured_at), shipping object (carrier, service, tracking_number, label_url, weight, dimensions, cost, estimated_delivery, events array of 5), warehouse object (id, name, location, picker_id, packed_at, dispatched_at), invoices array of 2 (each with id, url, amount, issued_at), returns array if any, notes array of 3, status_history array of 6, created_at, updated_at.',
    'Return a comprehensive JSON for a machine learning model registry entry. Include: model_id, name, description, task_type, framework, version, created_by, created_at, training_config (dataset, splits, epochs, batch_size, learning_rate, optimizer, loss_function, callbacks array of 3), architecture (layers array of 10 each with type/units/activation, total_params, input_shape, output_shape), metrics (accuracy, precision, recall, f1, auc_roc, confusion_matrix as 2D array, per_class_metrics array of 4), evaluation_datasets array of 3 (each with name, size, metrics object), deployment_config (endpoint, instance_type, min_replicas, max_replicas, autoscaling_config, monitoring_config), feature_importance array of 10 (name, importance_score), version_history array of 4 (version, changes, deployed_at, deprecated_at), tags, compliance_flags array.',
    'Return a comprehensive JSON for a SaaS subscription billing record. Include: subscription_id, account_id, plan (name, tier, price_monthly, price_yearly, features array of 10, limits object with 6 fields), billing_cycle (start, end, status), payment_method (type, last4, expiry, brand), invoices array of 6 (each with id, period, amount, status, line_items array of 3, pdf_url, paid_at), usage_this_period object with 8 metrics, overage_charges array of 3, credits array of 2 (each with amount, reason, expires_at), discount object (code, pct_off, valid_until), next_billing_date, cancel_at_period_end, metadata object with 5 fields, audit_events array of 5.',
    'Return a comprehensive JSON for a distributed tracing span tree. Include: trace_id, root_span (span_id, service, operation, start_ts, end_ts, duration_ms, status, attributes object with 8 fields, events array of 4 each with name/ts/attrs, tags object), child_spans array of 6 (each with span_id, parent_span_id, service, operation, start_ts, end_ts, duration_ms, status, db_query or http_url, attributes object with 5 fields), error_spans array of 2 (span_id, error_type, message, stack_trace first 3 frames), sampling_info, baggage object with 4 keys, resource_attributes object with 8 fields, instrumentation_library.',
    'Return a comprehensive JSON for a content delivery network configuration. Include: distribution_id, domain, origins array of 3 (each with id, type, domain, port, protocol, health_check object, custom_headers array of 2), cache_behaviors array of 5 (each with path_pattern, allowed_methods, cached_methods, ttl_min, ttl_default, ttl_max, compress, headers_to_forward array of 3, cookies_config, query_string_config), geo_restriction (type, locations array of 8), ssl_config (certificate_arn, minimum_protocol, security_policy), waf_acl_id, logging_config (bucket, prefix, include_cookies), custom_error_responses array of 3 (error_code, response_code, response_page, ttl), tags object with 5 entries, aliases array of 4, price_class, http_version, created_time, last_modified.',
    'Return a comprehensive JSON for a complex API gateway rate limiting and quota system. Include: gateway_id, tiers array of 4 (each with name, requests_per_minute, requests_per_day, burst_limit, quota_period, overage_policy), api_keys array of 5 (each with key_id, masked_key, tier, owner_id, scopes array of 4, created_at, expires_at, last_used, usage_today, usage_this_month), endpoints array of 8 (each with path, method, rate_limit_multiplier, bypass_quota, auth_required, cache_ttl), current_window_stats object (window_start, window_end, total_requests, unique_ips, error_count, throttled_count, avg_latency_ms, p99_latency_ms), blocked_ips array of 3 (ip, reason, expires_at, blocked_at), circuit_breakers array of 3 (service, state, failure_count, last_failure, half_open_at).',
    'Return a comprehensive JSON for a full observability stack configuration. Include: stack_name, environment, metrics_config (prometheus_url, scrape_interval, retention_days, recording_rules array of 5 each with name/expr/labels, targets array of 6 each with job/url/labels), alerting_config (alertmanager_url, routes array of 4 each with matchers/receiver/group_wait/group_interval, receivers array of 4 each with name/type/config_object, inhibit_rules array of 3), logging_config (loki_url, log_sources array of 5 each with name/type/labels/pipeline_stages array of 3, retention_days, index_period_hours), tracing_config (tempo_url, sampling_strategies array of 3 each with service/type/param, exporter_config), dashboards array of 5 (each with name, uid, panels array of 4 each with title/type/query/thresholds array), slo_definitions array of 4 (each with name, objective, error_budget, burn_rate_alerts array of 2), oncall_schedule (rotation_days, escalation_policies array of 3), created_at, updated_at.',
]

STREAM_PROMPTS = [
    'Return a JSON user object with fields: id, name, email, role.',
    'Return JSON: {"status": "healthy", "version": "1.0", "uptime": 99.9}',
    'Return a JSON with: country, capital, population, currency.',
    'Return a JSON HTTP response: status_code, headers object with 3 fields, body.',
    'Return a JSON for a git tag: name, commit, author, date, message.',
    'Return a JSON color: name, hex, r, g, b, luminance.',
    'Return a JSON for a DNS A record: name, type, ttl, value.',
    'Return a JSON pagination: page, per_page, total, has_next, has_prev.',
    'Return a JSON error: code, message, details, retryable.',
    'Return a JSON metric: name, value, unit, timestamp, labels object.',
]

# ── Core call functions ─────────────────────────────────────────────────────

def call_non_streaming(prompt: str, timeout: int = 30) -> dict:
    """Make a non-streaming API call, return timing + usage data."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=timeout)
        t1 = time.perf_counter()
        resp.raise_for_status()
        data = resp.json()
        latency = t1 - t0
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return {
            "ok": True,
            "latency": latency,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "chars": len(content),
            "content": content,
        }
    except Exception as e:
        t1 = time.perf_counter()
        return {"ok": False, "latency": t1 - t0, "error": str(e)}


def call_streaming(prompt: str, timeout: int = 30) -> dict:
    """Make a streaming API call; records time-to-first-chunk and total time."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    t0 = time.perf_counter()
    first_chunk_time = None
    chunks = []
    try:
        with requests.post(API_URL, headers=HEADERS, json=payload,
                           stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if line.startswith("data: "):
                    line = line[6:]
                if line.strip() == "[DONE]":
                    break
                try:
                    chunk_data = json.loads(line)
                    delta = chunk_data["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    if token and first_chunk_time is None:
                        first_chunk_time = time.perf_counter()
                    chunks.append(token)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        t1 = time.perf_counter()
        content = "".join(chunks)
        ttfc = (first_chunk_time - t0) if first_chunk_time else (t1 - t0)
        return {
            "ok": True,
            "ttfc": ttfc,
            "total_latency": t1 - t0,
            "chars": len(content),
            "content": content,
        }
    except Exception as e:
        t1 = time.perf_counter()
        return {"ok": False, "ttfc": t1 - t0, "total_latency": t1 - t0, "error": str(e)}


# ── Stats helpers ────────────────────────────────────────────────────────────

def percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def compute_stats(latencies: list) -> dict:
    if not latencies:
        return {}
    return {
        "min": round(min(latencies), 3),
        "max": round(max(latencies), 3),
        "mean": round(statistics.mean(latencies), 3),
        "p50": round(percentile(latencies, 50), 3),
        "p95": round(percentile(latencies, 95), 3),
        "p99": round(percentile(latencies, 99), 3),
        "count": len(latencies),
    }


# ── Main benchmark ───────────────────────────────────────────────────────────

def run_category(name: str, prompts: list) -> dict:
    print(f"\n[{name}] Running {len(prompts)} samples...", flush=True)
    results = []
    for i, prompt in enumerate(prompts):
        r = call_non_streaming(prompt)
        status = f"  [{i+1}/{len(prompts)}] {'OK' if r['ok'] else 'ERR'} "
        if r["ok"]:
            status += f"{r['latency']:.3f}s | {r['completion_tokens']} tokens | {r['chars']} chars"
        else:
            status += f"{r.get('error', '?')[:60]}"
        print(status, flush=True)
        results.append(r)
    return {"name": name, "samples": results}


def run_streaming_comparison(prompts: list) -> dict:
    print(f"\n[STREAMING vs NON-STREAMING] Running {len(prompts)} pairs...", flush=True)
    streaming_results = []
    non_streaming_results = []

    for i, prompt in enumerate(prompts):
        # Non-streaming first
        ns = call_non_streaming(prompt)
        non_streaming_results.append(ns)

        # Streaming
        s = call_streaming(prompt)
        streaming_results.append(s)

        ok_ns = ns.get("ok", False)
        ok_s = s.get("ok", False)
        print(
            f"  [{i+1}/{len(prompts)}] "
            f"NS={'OK' if ok_ns else 'ERR'} {ns.get('latency', 0):.3f}s | "
            f"S={'OK' if ok_s else 'ERR'} ttfc={s.get('ttfc', 0):.3f}s total={s.get('total_latency', 0):.3f}s",
            flush=True
        )

    ns_latencies = [r["latency"] for r in non_streaming_results if r.get("ok")]
    s_ttfc = [r["ttfc"] for r in streaming_results if r.get("ok")]
    s_total = [r["total_latency"] for r in streaming_results if r.get("ok")]

    return {
        "non_streaming": {
            "stats": compute_stats(ns_latencies),
            "samples": non_streaming_results,
        },
        "streaming": {
            "ttfc_stats": compute_stats(s_ttfc),
            "total_stats": compute_stats(s_total),
            "samples": streaming_results,
        },
    }


def run_burst_test(prompts: list, n: int = 20) -> dict:
    """Fire n requests back-to-back with 0 delay."""
    print(f"\n[BURST] Firing {n} requests with 0 delay...", flush=True)
    # Cycle through prompts if needed
    burst_prompts = [prompts[i % len(prompts)] for i in range(n)]
    results = []
    t_burst_start = time.perf_counter()
    for i, prompt in enumerate(burst_prompts):
        r = call_non_streaming(prompt, timeout=60)
        results.append(r)
        print(
            f"  [{i+1}/{n}] {'OK' if r.get('ok') else 'ERR'} {r.get('latency', 0):.3f}s",
            flush=True
        )
    t_burst_end = time.perf_counter()

    latencies = [r["latency"] for r in results if r.get("ok")]
    return {
        "n_requests": n,
        "total_wall_time": round(t_burst_end - t_burst_start, 3),
        "stats": compute_stats(latencies),
        "samples": results,
    }


def find_latency_thresholds(category_results: list) -> dict:
    """
    Correlate completion_tokens with latency to find:
    - max tokens for sub-1s
    - max tokens for sub-2s (interactive threshold)
    """
    token_latency_pairs = []
    for cat in category_results:
        for s in cat["samples"]:
            if s.get("ok") and s.get("completion_tokens", 0) > 0:
                token_latency_pairs.append({
                    "tokens": s["completion_tokens"],
                    "latency": s["latency"],
                    "category": cat["name"],
                })

    # Sort by tokens
    token_latency_pairs.sort(key=lambda x: x["tokens"])

    sub_1s = [p for p in token_latency_pairs if p["latency"] < 1.0]
    sub_2s = [p for p in token_latency_pairs if p["latency"] < 2.0]
    over_2s = [p for p in token_latency_pairs if p["latency"] >= 2.0]

    max_tokens_sub_1s = max((p["tokens"] for p in sub_1s), default=0)
    max_tokens_sub_2s = max((p["tokens"] for p in sub_2s), default=0)
    min_tokens_over_2s = min((p["tokens"] for p in over_2s), default=None)

    # tokens/sec calculation
    tps_list = []
    for cat in category_results:
        for s in cat["samples"]:
            if s.get("ok") and s.get("completion_tokens", 0) > 0 and s.get("latency", 0) > 0:
                tps_list.append(s["completion_tokens"] / s["latency"])

    return {
        "max_tokens_sub_1s": max_tokens_sub_1s,
        "max_tokens_sub_2s": max_tokens_sub_2s,
        "min_tokens_crossing_2s": min_tokens_over_2s,
        "pct_sub_1s": round(len(sub_1s) / len(token_latency_pairs) * 100, 1) if token_latency_pairs else 0,
        "pct_sub_2s": round(len(sub_2s) / len(token_latency_pairs) * 100, 1) if token_latency_pairs else 0,
        "tokens_per_sec_mean": round(statistics.mean(tps_list), 1) if tps_list else 0,
        "tokens_per_sec_p50": round(percentile(tps_list, 50), 1) if tps_list else 0,
        "all_pairs": token_latency_pairs,
    }


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(category_results, streaming_comparison, burst_results, thresholds):
    div = "=" * 70
    print(f"\n{div}")
    print("  REAL-TIME INTERACTIVE LATENCY BENCHMARK RESULTS")
    print(f"  Model: {MODEL}  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(div)

    # Category latencies
    print("\n── CATEGORY LATENCY (non-streaming, seconds) ──────────────────────")
    print(f"{'Category':<16} {'p50':>7} {'p95':>7} {'p99':>7} {'min':>7} {'max':>7} {'mean':>7} {'n':>4}")
    print("-" * 70)
    for cat in category_results:
        lats = [s["latency"] for s in cat["samples"] if s.get("ok")]
        stats = compute_stats(lats)
        if stats:
            print(
                f"{cat['name']:<16} "
                f"{stats['p50']:>7.3f} {stats['p95']:>7.3f} {stats['p99']:>7.3f} "
                f"{stats['min']:>7.3f} {stats['max']:>7.3f} {stats['mean']:>7.3f} "
                f"{stats['count']:>4}"
            )

    # Tokens/sec per category
    print("\n── TOKENS/SEC PER CATEGORY ────────────────────────────────────────")
    print(f"{'Category':<16} {'mean tps':>10} {'p50 tps':>10} {'mean completion_tok':>20}")
    print("-" * 60)
    for cat in category_results:
        ok_samples = [s for s in cat["samples"] if s.get("ok") and s.get("completion_tokens", 0) > 0]
        if ok_samples:
            tps = [s["completion_tokens"] / s["latency"] for s in ok_samples if s["latency"] > 0]
            ctoks = [s["completion_tokens"] for s in ok_samples]
            if tps:
                print(
                    f"{cat['name']:<16} "
                    f"{statistics.mean(tps):>10.1f} "
                    f"{percentile(tps, 50):>10.1f} "
                    f"{statistics.mean(ctoks):>20.1f}"
                )

    # Streaming comparison
    print("\n── STREAMING vs NON-STREAMING (SHORT prompts) ─────────────────────")
    ns_stats = streaming_comparison["non_streaming"]["stats"]
    ttfc_stats = streaming_comparison["streaming"]["ttfc_stats"]
    total_stats = streaming_comparison["streaming"]["total_stats"]
    print(f"  Non-streaming total:  p50={ns_stats.get('p50', 0):.3f}s  p95={ns_stats.get('p95', 0):.3f}s  p99={ns_stats.get('p99', 0):.3f}s")
    print(f"  Streaming TTFC:       p50={ttfc_stats.get('p50', 0):.3f}s  p95={ttfc_stats.get('p95', 0):.3f}s  p99={ttfc_stats.get('p99', 0):.3f}s")
    print(f"  Streaming total:      p50={total_stats.get('p50', 0):.3f}s  p95={total_stats.get('p95', 0):.3f}s  p99={total_stats.get('p99', 0):.3f}s")
    speedup = ns_stats.get("mean", 1) / ttfc_stats.get("mean", 1) if ttfc_stats.get("mean", 0) > 0 else 0
    print(f"  TTFC speedup vs NS total: {speedup:.1f}x faster perceived latency")

    # Burst test
    print("\n── BURST TEST (20 back-to-back requests, 0 delay) ─────────────────")
    b = burst_results["stats"]
    print(f"  Wall time: {burst_results['total_wall_time']:.2f}s for {burst_results['n_requests']} requests")
    print(f"  p50={b.get('p50', 0):.3f}s  p95={b.get('p95', 0):.3f}s  p99={b.get('p99', 0):.3f}s")
    print(f"  min={b.get('min', 0):.3f}s  max={b.get('max', 0):.3f}s  mean={b.get('mean', 0):.3f}s")

    # Compare burst vs baseline (PING/ONE_LINER)
    baseline_cats = [c for c in category_results if c["name"] in ("PING", "ONE_LINER")]
    if baseline_cats:
        baseline_lats = []
        for cat in baseline_cats:
            baseline_lats += [s["latency"] for s in cat["samples"] if s.get("ok")]
        if baseline_lats:
            baseline_mean = statistics.mean(baseline_lats)
            burst_mean = b.get("mean", 0)
            degradation = (burst_mean - baseline_mean) / baseline_mean * 100 if baseline_mean > 0 else 0
            print(f"  Degradation vs baseline mean: {degradation:+.1f}%")

    # Threshold analysis
    print("\n── LATENCY THRESHOLD ANALYSIS ──────────────────────────────────────")
    print(f"  Max completion tokens for sub-1s response:  {thresholds['max_tokens_sub_1s']} tokens")
    print(f"  Max completion tokens for sub-2s response:  {thresholds['max_tokens_sub_2s']} tokens")
    if thresholds["min_tokens_crossing_2s"]:
        print(f"  Min tokens where >2s latency first seen:    {thresholds['min_tokens_crossing_2s']} tokens")
    print(f"  % of responses sub-1s:  {thresholds['pct_sub_1s']}%")
    print(f"  % of responses sub-2s:  {thresholds['pct_sub_2s']}%")
    print(f"  Mean tokens/sec: {thresholds['tokens_per_sec_mean']:.1f}  |  p50 tokens/sec: {thresholds['tokens_per_sec_p50']:.1f}")

    print(f"\n{div}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("Starting LLM Real-Time Interactive Latency Benchmark")
    print(f"Model: {MODEL}")
    print(f"API: {API_URL}")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}\n")

    categories = [
        ("PING",     PING_PROMPTS),
        ("ONE_LINER", ONE_LINER_PROMPTS),
        ("SHORT",    SHORT_PROMPTS),
        ("MEDIUM",   MEDIUM_PROMPTS),
        ("LONG",     LONG_PROMPTS),
    ]

    category_results = []
    for name, prompts in categories:
        result = run_category(name, prompts)
        category_results.append(result)

    streaming_comparison = run_streaming_comparison(STREAM_PROMPTS)

    # Burst test uses ONE_LINER prompts (lightweight, tests throughput under pressure)
    burst_results = run_burst_test(ONE_LINER_PROMPTS, n=20)

    thresholds = find_latency_thresholds(category_results)

    print_report(category_results, streaming_comparison, burst_results, thresholds)

    # Build results.json
    results = {
        "meta": {
            "model": MODEL,
            "api_url": API_URL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "categories": {
            cat["name"]: {
                "latency_stats": compute_stats([s["latency"] for s in cat["samples"] if s.get("ok")]),
                "samples": [
                    {k: v for k, v in s.items() if k != "content"}  # omit content to keep file small
                    for s in cat["samples"]
                ],
            }
            for cat in category_results
        },
        "streaming_comparison": {
            "non_streaming_stats": streaming_comparison["non_streaming"]["stats"],
            "streaming_ttfc_stats": streaming_comparison["streaming"]["ttfc_stats"],
            "streaming_total_stats": streaming_comparison["streaming"]["total_stats"],
        },
        "burst_test": {
            "n_requests": burst_results["n_requests"],
            "total_wall_time": burst_results["total_wall_time"],
            "stats": burst_results["stats"],
        },
        "thresholds": {
            k: v for k, v in thresholds.items() if k != "all_pairs"
        },
        "token_latency_pairs": thresholds["all_pairs"],
    }

    results_path = "/Users/krishnatejaswis/llm-boundary-tests/5-realtime-interactive/results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
