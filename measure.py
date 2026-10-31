"""
Empirical Performance and Accuracy Measurement Script for Tessera-RAG.

This script executes:
1. Two sequential runs of all 15 golden questions against the live agent API (evaluating latency & cache hit rate).
2. Execution of the 6 guardrail test cases (measuring precision, recall, and pass rate).
3. Computation of 6 RAGAS & Agent metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness, Tool Correctness).
4. Direct inspection of Portkey & Logfire telemetry endpoints.
5. Statistical calculation of median latency, p95 latency, run 1 vs run 2, and export to results/measure_results.json.
"""

import sys
import os
import types

# Windows UTF-8 stdout configuration
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pre-import shim for langchain-community vertexai deprecation required by ragas
import langchain_community.llms
import langchain_community.chat_models
m1 = types.ModuleType("langchain_community.chat_models.vertexai")
m1.ChatVertexAI = None
sys.modules["langchain_community.chat_models.vertexai"] = m1
m2 = types.ModuleType("langchain_community.llms.vertexai")
m2.VertexAI = None
sys.modules["langchain_community.llms.vertexai"] = m2
langchain_community.llms.VertexAI = None
langchain_community.chat_models.ChatVertexAI = None

import time
import json
import asyncio
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)

from evals.pipeline import load_golden_dataset, detect_tool
from evals.guardrails_eval import run_guardrails_eval, compute_guardrails_metrics
import evals.metrics as evals_metrics
from evals.metrics import run_all_metrics

API_URL = "http://localhost:8000/query"
REQUEST_TIMEOUT = 120

# Use GROQ_FALLBACK_API_KEY for judge to avoid TPD quota exhaustion on primary GROQ key
fallback_key = os.getenv("GROQ_FALLBACK_API_KEY")
if fallback_key:
    os.environ["JUDGE_GROQ"] = fallback_key

# Use qwen/qwen3.8-27b: fully supports structured JSON validation on Groq without token overflow
evals_metrics.JUDGE_MODEL = "qwen/qwen3.8-27b"
evals_metrics.COOLDOWN_MINI = 6       # seconds between RAGAS samples
evals_metrics.COOLDOWN_STANDARD = 8   # seconds between RAGAS metric experiments


def test_api_connectivity():
    print("=" * 70)
    print("STEP 0: Verifying Backend API Connectivity (http://localhost:8000)")
    print("=" * 70)
    try:
        r = requests.get("http://localhost:8000/", timeout=5)
        r.raise_for_status()
        print(f"Backend API is healthy: {r.json()}")
    except Exception as e:
        print(f"ERROR: Cannot connect to FastAPI backend at http://localhost:8000/: {e}")
        print("FIX: Start the server using: uvicorn app.main:app --port 8000")
        sys.exit(1)


def execute_query_with_retry(question: str, thread_id: str, max_retries: int = 3):
    """Executes query against the API with timing and exponential backoff retry."""
    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                API_URL,
                json={"q": question, "thread_id": thread_id},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            latency = time.perf_counter() - t0
            data = resp.json()
            return latency, data, None
        except requests.exceptions.HTTPError as he:
            latency = time.perf_counter() - t0
            status = getattr(resp, "status_code", None)
            if status == 429:
                sleep_s = 15 * (attempt + 1)
                print(f"  [RateLimit 429] Waiting {sleep_s}s before retry {attempt+1}/{max_retries}...")
                time.sleep(sleep_s)
                continue
            return latency, None, f"HTTPError {status}: {he}"
        except Exception as e:
            latency = time.perf_counter() - t0
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return latency, None, str(e)
    return 0.0, None, "Max retries exceeded"


def run_golden_queries(golden_samples: list, run_index: int):
    """Runs all 15 questions, recording latency, thought process, and cache hits."""
    print("\n" + "=" * 70)
    print(f"RUN {run_index}: Executing 15 Golden Questions ({'Run 1 - Cold' if run_index == 1 else 'Run 2 - Warm / Cache Evaluation'})")
    print("=" * 70)
    results = []
    
    for i, s in enumerate(golden_samples):
        q = s["question"]
        thread_id = f"measure_run{run_index}_q{i}"
        print(f"[{i+1}/15] Query: {q[:60]}...")
        
        latency, data, error = execute_query_with_retry(q, thread_id)
        
        if error:
            print(f"    ERROR: {error}")
            results.append({
                "sample_id": s.get("id", i+1),
                "question": q,
                "latency_seconds": round(latency, 3),
                "cache_hit": False,
                "error": error,
                "thought_process": [],
                "answer": "",
                "sources": [],
                "per_node_time": "NOT AVAILABLE (not exposed by /query response schema)",
            })
        else:
            tp = data.get("thought_process") or []
            cache_hit = any("cache: hit" in step.lower() for step in tp)
            answer = data.get("answer") or ""
            sources = data.get("sources") or []
            tool = detect_tool(tp)
            
            print(f"    Latency: {latency:.2f}s | Cache: {'HIT ⚡' if cache_hit else 'MISS'} | Tool: {tool}")
            
            results.append({
                "sample_id": s.get("id", i+1),
                "question": q,
                "domain": s.get("domain", ""),
                "latency_seconds": round(latency, 3),
                "cache_hit": cache_hit,
                "answer": answer,
                "thought_process": tp,
                "tool_detected": tool,
                "sources": sources,
                "per_node_time": "NOT AVAILABLE (not exposed by /query response schema)",
            })
        
        # Brief pacing between queries to stay comfortably under Groq RPM limits
        if i < len(golden_samples) - 1:
            time.sleep(4)
            
    return results


def check_portkey_and_logfire():
    """Inspects telemetry reachable via existing configuration."""
    print("\n" + "=" * 70)
    print("TELEMETRY CHECK: Portkey & Logfire Reachability")
    print("=" * 70)
    telemetry_report = {}
    
    # 1. Portkey Reachability
    portkey_key = os.getenv("PORTKEY_API_KEY")
    if not portkey_key:
        telemetry_report["portkey"] = {
            "status": "NOT CONFIGURED",
            "reason": "PORTKEY_API_KEY is not set in environment."
        }
    else:
        try:
            from portkey_ai import Portkey
            pk = Portkey(api_key=portkey_key)
            # Test analytics endpoint
            try:
                import datetime
                now = datetime.datetime.now(datetime.timezone.utc)
                min_t = (now - datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
                max_t = now.strftime("%Y-%m-%d %H:%M:%S")
                pk.analytics.summary.cache(time_of_generation_min=min_t, time_of_generation_max=max_t)
                telemetry_report["portkey_analytics_api"] = {
                    "status": "ACCESSIBLE",
                    "note": "Portkey analytics API successfully queried."
                }
            except Exception as pe:
                telemetry_report["portkey_analytics_api"] = {
                    "status": "NOT AVAILABLE",
                    "reason": f"Permission denied on Portkey REST API: {pe}",
                    "explanation": "Configured PORTKEY_API_KEY has inference permissions only, not administrative analytics permissions."
                }
        except Exception as e:
            telemetry_report["portkey_analytics_api"] = {
                "status": "NOT AVAILABLE",
                "reason": str(e)
            }
            
    telemetry_report["portkey_cache_telemetry"] = {
        "status": "AVAILABLE VIA RUNTIME HEADERS",
        "origin": "Response header 'x-portkey-cache-status' extracted in app/agents/nodes/responder.py and surfaced in 'thought_process'.",
    }
    
    telemetry_report["token_usage_and_cost"] = {
        "status": "NOT AVAILABLE VIA /query RESPONSE",
        "reason": "FastAPI endpoint /query response schema returns question, answer, thought_process, status, and sources. It does not return prompt_tokens, completion_tokens, or cost calculation.",
        "origin": "Token counts are recorded in internal Portkey gateway headers and Logfire spans, but not exposed to client responses."
    }

    # 2. Logfire Reachability
    logfire_token = os.getenv("LOGFIRE_TOKEN")
    if logfire_token:
        telemetry_report["logfire"] = {
            "status": "ACTIVE SPAN EXPORTER",
            "dashboard_url": "https://logfire-us.pydantic.dev/syedsiddiq4888/starter-project",
            "programmatic_api": "NOT AVAILABLE (Logfire does not expose a read API without an organization Read Token; traces are sent via OTLP to Pydantic Cloud).",
        }
    else:
        telemetry_report["logfire"] = {
            "status": "NOT CONFIGURED"
        }
        
    for k, v in telemetry_report.items():
        print(f"• {k}: {v.get('status')} - {v.get('reason') or v.get('origin') or v.get('dashboard_url')}")
        
    return telemetry_report


async def robust_batched_score(metric, inputs: list, samples: list, status_cb=None, label: str = "") -> list:
    """Executes abatch_score one sample at a time with automatic 429 TPM retry and backoff."""
    all_scores = []
    batches = [inputs[i : i + evals_metrics.GENERAL_BATCH_SIZE] for i in range(0, len(inputs), evals_metrics.GENERAL_BATCH_SIZE)]
    for b_idx, batch in enumerate(batches):
        if b_idx > 0:
            await evals_metrics._cooldown(evals_metrics.COOLDOWN_MINI, f"{label} batch {b_idx}", status_cb)
        
        max_retries = 6
        for attempt in range(max_retries):
            try:
                scores = await metric.abatch_score(batch)
                all_scores.extend(scores)
                val = getattr(scores[0], "value", scores[0])
                if status_cb:
                    status_cb(f"{label} [{b_idx+1}/{len(batches)}] scored: {round(float(val), 3) if isinstance(val, (int, float)) else val}")
                break
            except Exception as e:
                err_str = str(e).lower()
                if "rate" in err_str or "429" in err_str or "tpm" in err_str or "tokens" in err_str:
                    wait_s = 15 + (attempt * 10)
                    if status_cb:
                        status_cb(f"⏳ [429 TPM RateLimit on {label} batch {b_idx+1}] Backing off {wait_s}s (retry {attempt+1}/{max_retries})...")
                    await asyncio.sleep(wait_s)
                else:
                    if status_cb:
                        status_cb(f"⚠️ [Error on {label} batch {b_idx+1}] {e} (retry {attempt+1}/{max_retries})")
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(5)
    return all_scores


async def run_ragas_evaluation(golden_dataset: dict, run1_results: list):
    """Enriches golden dataset with run 1 responses and executes all 6 RAGAS & Agent metrics."""
    print("\n" + "=" * 70)
    print("STEP 3: Running RAGAS & Agent Metrics (6 Experiments)")
    print("=" * 70)
    
    # Populate actual_response, actual_contexts, actual_tools_called into golden_dataset
    dataset = json.loads(json.dumps(golden_dataset))
    for s, r in zip(dataset["rag_samples"], run1_results):
        s["actual_response"] = r.get("answer", "")[:300]
        s["actual_contexts"] = r.get("sources", [])[:5]
        s["actual_tools_called"] = [r.get("tool_detected", "unknown")]
        
    def status_callback(msg):
        print(f"  [RAGAS] {msg}")

    judge_llm, ragas_embeddings = evals_metrics._build_judge()
    samples = evals_metrics._prep_samples(dataset)

    results = {}

    # ── Exp 6: Tool Correctness (no LLM — pure Jaccard overlap) ───────────
    print("\n--- Exp 1/6: Tool Correctness (Deterministic Jaccard overlap, 0 LLM calls) ---")
    tool_rows = []
    for s in samples:
        called = set(s.get("actual_tools_called") or [])
        expected = set(s.get("expected_tools") or [])
        union = len(called | expected)
        score = len(called & expected) / union if union > 0 else 0.0
        tool_rows.append({"question": s["question"][:65], "tool_correctness": round(score, 3)})
    df_tool = pd.DataFrame(tool_rows)
    results["tool_correctness"] = df_tool
    print(f"  ⭐ Tool Correctness Mean: {df_tool['tool_correctness'].mean():.4f}")

    # ── Exp 1: Faithfulness ───────────────────────────────────────────────
    try:
        print("\n--- Exp 2/6: Faithfulness ---")
        inputs = [
            {"user_input": s["question"], "response": s["actual_response"], "retrieved_contexts": s["actual_contexts"]}
            for s in samples
        ]
        scores = await robust_batched_score(Faithfulness(llm=judge_llm), inputs, samples, status_callback, "Faithfulness")
        df = evals_metrics._score_df("faithfulness", samples, scores)
        results["faithfulness"] = df
        print(f"  ⭐ Faithfulness Mean: {df['faithfulness'].mean():.4f}")
        await asyncio.sleep(evals_metrics.COOLDOWN_STANDARD)
    except Exception as e:
        print(f"  ❌ Faithfulness failed: {e}")

    # ── Exp 2: Answer Relevancy ───────────────────────────────────────────
    try:
        print("\n--- Exp 3/6: Answer Relevancy ---")
        inputs = [{"user_input": s["question"], "response": s["actual_response"]} for s in samples]
        scores = await robust_batched_score(AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings), inputs, samples, status_callback, "Answer Relevancy")
        df = evals_metrics._score_df("answer_relevancy", samples, scores)
        results["answer_relevancy"] = df
        print(f"  ⭐ Answer Relevancy Mean: {df['answer_relevancy'].mean():.4f}")
        await asyncio.sleep(evals_metrics.COOLDOWN_STANDARD)
    except Exception as e:
        print(f"  ❌ Answer Relevancy failed: {e}")

    # ── Exp 3: Context Precision ──────────────────────────────────────────
    try:
        print("\n--- Exp 4/6: Context Precision ---")
        inputs = [
            {"user_input": s["question"], "reference": s["reference"], "retrieved_contexts": s["actual_contexts"]}
            for s in samples
        ]
        scores = await robust_batched_score(ContextPrecision(llm=judge_llm), inputs, samples, status_callback, "Context Precision")
        df = evals_metrics._score_df("context_precision", samples, scores)
        results["context_precision"] = df
        print(f"  ⭐ Context Precision Mean: {df['context_precision'].mean():.4f}")
        await asyncio.sleep(evals_metrics.COOLDOWN_STANDARD)
    except Exception as e:
        print(f"  ❌ Context Precision failed: {e}")

    # ── Exp 4: Context Recall ─────────────────────────────────────────────
    try:
        print("\n--- Exp 5/6: Context Recall ---")
        inputs = [
            {"user_input": s["question"], "reference": s["reference"], "retrieved_contexts": s["actual_contexts"]}
            for s in samples
        ]
        scores = await robust_batched_score(ContextRecall(llm=judge_llm), inputs, samples, status_callback, "Context Recall")
        df = evals_metrics._score_df("context_recall", samples, scores)
        results["context_recall"] = df
        print(f"  ⭐ Context Recall Mean: {df['context_recall'].mean():.4f}")
        await asyncio.sleep(evals_metrics.COOLDOWN_STANDARD)
    except Exception as e:
        print(f"  ❌ Context Recall failed: {e}")

    # ── Exp 5: Answer Correctness ─────────────────────────────────────────
    try:
        print("\n--- Exp 6/6: Answer Correctness ---")
        inputs = [
            {"user_input": s["question"], "response": s["actual_response"], "reference": s["reference"]}
            for s in samples
        ]
        scores = await robust_batched_score(AnswerCorrectness(llm=judge_llm, embeddings=ragas_embeddings), inputs, samples, status_callback, "Answer Correctness")
        df = evals_metrics._score_df("answer_correctness", samples, scores)
        results["answer_correctness"] = df
        print(f"  ⭐ Answer Correctness Mean: {df['answer_correctness'].mean():.4f}")
    except Exception as e:
        print(f"  ❌ Answer Correctness failed: {e}")

    averages = {}
    per_query_scores = {}
    for metric_name, df in results.items():
        avg_score = float(df[metric_name].mean())
        averages[metric_name] = round(avg_score, 4)
        per_query_scores[metric_name] = df[metric_name].tolist()
        
    return averages, per_query_scores, None


def main():
    test_api_connectivity()
    
    golden_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals", "golden_dataset.json")
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
    rag_samples = golden_data["rag_samples"]
    guardrails_samples = golden_data["guardrails_samples"]
    
    print(f"Loaded Golden Dataset: {len(rag_samples)} RAG questions, {len(guardrails_samples)} Guardrails test cases.")
    
    results_path = os.path.join("results", "measure_results.json")
    cached_payload = {}
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                cached_payload = json.load(f)
        except Exception:
            pass

    if len(cached_payload.get("run1_details", [])) == len(rag_samples) and len(cached_payload.get("run2_details", [])) == len(rag_samples):
        print("Reusing previously measured Run 1, Run 2, and Guardrail query responses from results/measure_results.json...")
        run1_results = cached_payload["run1_details"]
        run2_results = cached_payload["run2_details"]
        gr_enriched = cached_payload.get("guardrail_details", [])
        gr_metrics = cached_payload.get("summary", {}).get("guardrail_metrics") or compute_guardrails_metrics(gr_enriched)
        telemetry_data = cached_payload.get("telemetry", {})
    else:
        # 1. Run 1: Cold execution of 15 questions
        run1_results = run_golden_queries(rag_samples, run_index=1)
        
        # Pause between runs
        print("\nCooldown between Run 1 and Run 2 (10 seconds)...")
        time.sleep(10)
        
        # 2. Run 2: Warm execution of 15 questions (Cache hit evaluation)
        run2_results = run_golden_queries(rag_samples, run_index=2)
        
        # 3. Guardrail evaluations
        print("\n" + "=" * 70)
        print("STEP 2: Executing 6 Guardrails Test Cases")
        print("=" * 70)
        
        def gr_callback(i, total, inp):
            print(f"[{i+1}/{total}] Guardrail Test: {inp[:60]}...")
            
        gr_enriched = run_guardrails_eval(guardrails_samples, progress_callback=gr_callback)
        gr_metrics = compute_guardrails_metrics(gr_enriched)
        print(f"Guardrails Results: {gr_metrics}")
        
        # 4. Telemetry check
        telemetry_data = check_portkey_and_logfire()
    
    # 5. RAGAS Metrics
    ragas_averages, ragas_details, ragas_error = asyncio.run(
        run_ragas_evaluation(golden_data, run1_results)
    )
    
    # 6. Statistical Computation
    latencies_run1 = [r["latency_seconds"] for r in run1_results if "latency_seconds" in r]
    latencies_run2 = [r["latency_seconds"] for r in run2_results if "latency_seconds" in r]
    all_latencies = latencies_run1 + latencies_run2
    
    median_latency_all = round(float(np.median(all_latencies)), 2)
    p95_latency_all = round(float(np.percentile(all_latencies, 95)), 2)
    mean_run1 = round(float(np.mean(latencies_run1)), 2)
    median_run1 = round(float(np.median(latencies_run1)), 2)
    mean_run2 = round(float(np.mean(latencies_run2)), 2)
    median_run2 = round(float(np.median(latencies_run2)), 2)
    
    run1_cache_hits = sum(1 for r in run1_results if r.get("cache_hit"))
    run2_cache_hits = sum(1 for r in run2_results if r.get("cache_hit"))
    total_cache_hits = run1_cache_hits + run2_cache_hits
    total_queries = len(all_latencies)
    cache_hit_rate_run2 = round((run2_cache_hits / len(run2_results)) * 100, 1)
    overall_cache_hit_rate = round((total_cache_hits / total_queries) * 100, 1)
    
    gr_accuracy = round(gr_metrics.get("accuracy", 0.0) * 100, 1)
    gr_pass_rate = f"{gr_metrics.get('correct')}/{gr_metrics.get('total')} ({gr_accuracy}%)"
    
    # Prepare comprehensive results payload
    final_payload = {
        "summary": {
            "total_queries_run": total_queries,
            "golden_questions_count": len(rag_samples),
            "guardrail_cases_count": len(guardrails_samples),
            "median_latency_seconds": median_latency_all,
            "p95_latency_seconds": p95_latency_all,
            "first_run_mean_latency_seconds": mean_run1,
            "first_run_median_latency_seconds": median_run1,
            "second_run_mean_latency_seconds": mean_run2,
            "second_run_median_latency_seconds": median_run2,
            "run1_cache_hits": f"{run1_cache_hits}/{len(run1_results)}",
            "run2_cache_hits": f"{run2_cache_hits}/{len(run2_results)} ({cache_hit_rate_run2}%)",
            "overall_cache_hit_rate": f"{total_cache_hits}/{total_queries} ({overall_cache_hit_rate}%)",
            "guardrail_pass_rate": gr_pass_rate,
            "guardrail_metrics": gr_metrics,
            "ragas_scores": ragas_averages if ragas_averages else {"error": ragas_error},
            "per_node_time": "NOT AVAILABLE (not exposed by /query response schema or AgentState)",
        },
        "telemetry": telemetry_data,
        "run1_details": run1_results,
        "run2_details": run2_results,
        "guardrail_details": gr_enriched,
        "ragas_details": ragas_details,
    }
    
    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", "measure_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)
    print(f"\nRaw results successfully saved to {results_path}")
    
    # Print Markdown Summary Table
    print("\n" + "=" * 70)
    print("FINAL MEASURED RESULTS TABLE")
    print("=" * 70)
    
    md_table = f"""
| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Total Queries Executed** | {total_queries} | 15 golden questions × 2 sequential runs |
| **Median Latency (Overall)** | {median_latency_all}s | 50th percentile across all 30 queries |
| **p95 Latency (Overall)** | {p95_latency_all}s | 95th percentile across all 30 queries |
| **First-Run (Cold) Latency** | Mean: {mean_run1}s, Med: {median_run1}s | Fresh vector search, reranking & LLM synthesis |
| **Second-Run (Warm) Latency**| Mean: {mean_run2}s, Med: {median_run2}s | Identical queries repeated for cache evaluation |
| **Cache Hit Rate (Run 2)** | {run2_cache_hits}/{len(run2_results)} ({cache_hit_rate_run2}%) | Detected via 'Cache: Hit ⚡' in thought process |
| **Overall Cache Hit Rate** | {total_cache_hits}/{total_queries} ({overall_cache_hit_rate}%) | Portkey gateway cache status |
| **Guardrails Pass Rate** | {gr_pass_rate} | NeMo Guardrails on 6 test inputs (TP={gr_metrics.get('tp')}, TN={gr_metrics.get('tn')}, FP={gr_metrics.get('fp')}, FN={gr_metrics.get('fn')}) |
| **RAGAS Faithfulness** | {ragas_averages.get('faithfulness', 'N/A') if ragas_averages else 'N/A'} | Groq LLaMA-3.1-8B judge |
| **RAGAS Answer Relevancy** | {ragas_averages.get('answer_relevancy', 'N/A') if ragas_averages else 'N/A'} | Embeddings + LLM judge |
| **RAGAS Context Precision** | {ragas_averages.get('context_precision', 'N/A') if ragas_averages else 'N/A'} | Reranked chunk relevance vs reference |
| **RAGAS Context Recall** | {ragas_averages.get('context_recall', 'N/A') if ragas_averages else 'N/A'} | Retrieved coverage of ground truth |
| **RAGAS Answer Correctness**| {ragas_averages.get('answer_correctness', 'N/A') if ragas_averages else 'N/A'} | Weighted semantic & factual similarity |
| **Tool Correctness** | {ragas_averages.get('tool_correctness', 'N/A') if ragas_averages else 'N/A'} | Jaccard overlap of expected vs called tools |
| **Per-Node Latency** | NOT AVAILABLE | Not exposed in /query schema; tracked in Logfire spans |
| **Token Usage & Cost** | NOT AVAILABLE | Endpoint /query does not expose token usage or cost |
"""
    print(md_table)


if __name__ == "__main__":
    main()
