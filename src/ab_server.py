
import asyncio
import json
import os
import time

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000/v1/chat/completions")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_MODEL = os.getenv("ADAPTER_MODEL", "medical")
RESULTS = "outputs/results"
DISCLAIMER = "Learning project. Not a source of medical advice."

# DEMO_MODE=1 means: answer from saved examples, no GPU needed
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"


def load_results(name):
    path = f"{RESULTS}/{name}_results.json"
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


base_eval = load_results("base")
ft_eval = load_results("finetuned")

bench_lookup = {}
if base_eval and ft_eval:
    for b, f in zip(base_eval["benchmark"], ft_eval["benchmark"]):
        bench_lookup[b["instruction"].strip().lower()] = {
            "category": b["category"],
            "must_include": b["must_include"],
            "base_score": b["score"],
            "finetuned_score": f["score"],
        }

SAVED = {}
if os.path.exists("demo/examples.json"):
    for ex in json.load(open("demo/examples.json", encoding="utf-8")):
        SAVED[ex["prompt"].strip().lower()] = ex


def keyword_score(answer, must_include):
    a = answer.lower()
    return sum(k.lower() in a for k in must_include) / len(must_include)


app = FastAPI(title="Base vs fine-tuned A/B endpoint")


class CompareRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    must_include: list[str] | None = None


async def ask(client, model, prompt, max_tokens):
    t0 = time.perf_counter()
    r = await client.post(BACKEND, timeout=180, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    })
    r.raise_for_status()
    data = r.json()
    return {
        "answer": data["choices"][0]["message"]["content"].strip(),
        "latency_s": round(time.perf_counter() - t0, 2),
        "completion_tokens": data["usage"]["completion_tokens"],
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return open("demo/index.html", encoding="utf-8").read()


@app.get("/health")
def health():
    return {"status": "ok", "base": BASE_MODEL, "adapter": ADAPTER_MODEL,
            "eval_loaded": bool(base_eval and ft_eval), "demo_mode": DEMO_MODE}


@app.get("/examples")
def examples():
    return list(SAVED.values())


@app.post("/compare")
async def compare(req: CompareRequest):
    if DEMO_MODE:
        saved = SAVED.get(req.prompt.strip().lower())
        if saved:
            return saved
        raise HTTPException(
            status_code=404,
            detail="Replay mode: only the saved questions are available. Live mode needs a GPU.")

    async with httpx.AsyncClient() as client:
        base, ft = await asyncio.gather(
            ask(client, BASE_MODEL, req.prompt, req.max_tokens),
            ask(client, ADAPTER_MODEL, req.prompt, req.max_tokens),
        )

    this_question = None
    known = bench_lookup.get(req.prompt.strip().lower())
    if known:
        this_question = {"source": "benchmark", **known}
    elif req.must_include:
        this_question = {
            "source": "request",
            "must_include": req.must_include,
            "base_score": round(keyword_score(base["answer"], req.must_include), 2),
            "finetuned_score": round(keyword_score(ft["answer"], req.must_include), 2),
        }

    overall = None
    if base_eval and ft_eval:
        bs, fs = base_eval["summary"], ft_eval["summary"]
        overall = {
            "benchmark_keyword_score": {"base": bs["benchmark_keyword_score"],
                                        "finetuned": fs["benchmark_keyword_score"]},
            "test_rougeL": {"base": bs["test_rougeL"], "finetuned": fs["test_rougeL"]},
            "n_benchmark": bs["n_benchmark"],
            "n_test": bs["n_test"],
        }

    return {
        "prompt": req.prompt,
        "base": base,
        "finetuned": ft,
        "scores_this_question": this_question,
        "scores_overall": overall,
        "disclaimer": DISCLAIMER,
    }