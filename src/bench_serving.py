
import argparse
import asyncio
import json
import os
import time

import httpx


def percentile(values, p):
    s = sorted(values)
    if not s:
        return 0.0
    k = min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))
    return s[k]


async def one_request(client, url, model, prompt, max_tokens):
    t0 = time.perf_counter()
    try:
        r = await client.post(url, timeout=180, json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        })
        r.raise_for_status()
        tokens = r.json()["usage"]["completion_tokens"]
        return {"ok": True, "latency": time.perf_counter() - t0, "tokens": tokens}
    except Exception as e:
        return {"ok": False, "latency": time.perf_counter() - t0, "tokens": 0,
                "error": str(e)[:100]}


async def run_level(url, model, prompts, concurrency, n_requests, max_tokens):
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        async def guarded(i):
            async with sem:
                return await one_request(client, url, model,
                                         prompts[i % len(prompts)], max_tokens)

        start = time.perf_counter()
        results = await asyncio.gather(*[guarded(i) for i in range(n_requests)])
        wall = time.perf_counter() - start

    ok = [r for r in results if r["ok"]]
    lat = [r["latency"] for r in ok]
    total_tokens = sum(r["tokens"] for r in ok)
    return {
        "concurrency": concurrency,
        "requests": n_requests,
        "errors": n_requests - len(ok),
        "p50_s": round(percentile(lat, 50), 2),
        "p95_s": round(percentile(lat, 95), 2),
        "tokens_per_s_total": round(total_tokens / wall, 1),
        "tokens_per_s_per_request": round(
            sum(r["tokens"] / r["latency"] for r in ok) / max(len(ok), 1), 1),
    }


async def main(cli):
    prompts = [json.loads(l)["instruction"]
               for l in open("data/benchmark.jsonl", encoding="utf-8") if l.strip()]

    print("Warm-up...")
    await run_level(cli.url, cli.model, prompts, 1, 3, cli.max_tokens)

    rows = []
    print(f"\nModel: {cli.model}")
    print("| Concurrency | Errors | P50 (s) | P95 (s) | Tokens/s total | Tokens/s per request |")
    print("|---|---|---|---|---|---|")
    for c in cli.levels:
        n = max(cli.min_requests, c * 4)
        row = await run_level(cli.url, cli.model, prompts, c, n, cli.max_tokens)
        rows.append(row)
        print(f"| {c} | {row['errors']} | {row['p50_s']} | {row['p95_s']} | "
              f"{row['tokens_per_s_total']} | {row['tokens_per_s_per_request']} |")

    good = [r["concurrency"] for r in rows
            if r["errors"] == 0 and r["p95_s"] <= cli.sla_s]
    max_ok = max(good) if good else 0
    print(f"\nHighest tested concurrency with 0 errors and P95 <= {cli.sla_s}s: {max_ok}")

    os.makedirs("outputs/results", exist_ok=True)
    out = f"outputs/results/serving_benchmark_{cli.model.replace('/', '_')}.json"
    with open(out, "w") as f:
        json.dump({"model": cli.model, "max_tokens": cli.max_tokens,
                   "sla_p95_s": cli.sla_s, "max_ok_concurrency": max_ok,
                   "levels": rows}, f, indent=2)
    print("Saved to", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--model", default="medical")
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--min_requests", type=int, default=16)
    p.add_argument("--sla_s", type=float, default=10.0)
    cli = p.parse_args()
    asyncio.run(main(cli))