#!/usr/bin/env python3
"""Concurrency sweep for any OpenAI-compatible server. Standard library only.

  python3 bench/sweep.py --base-url http://127.0.0.1:8000 --model inkling-2-mini \
      --levels 1,8,32,64,128 --name vllm-best [--workload chat|long] [--rounds 2]

For each concurrency level N it runs N closed-loop users; each user sends --rounds requests back to
back. Every request is streamed, and token counts come from the server's usage block (never from
streamed chunks, which carry several tokens each under speculative decoding).

Reported per level:
  agg_tok_s      completion tokens of all requests / (last finish - first start)
  user_tok_s_p50 per-request decode rate (completion_tokens - 1) / (finish - first token), median
  user_tok_s_p10 the same, 10th percentile (the slowest users)
  ttft_p50_s, ttft_p95_s
  max_gap_p95_s  95th percentile of each request's longest pause between streamed deltas (prefill stalls)
  fail           failed requests
A level "meets the SLO" when user_tok_s_p10 >= --slo-tok-s and ttft_p95_s <= --slo-ttft-s.
Results are appended as JSON lines to --out (default docs/raw/sweep.jsonl).

Workloads (fixed, seeded; acceptance of draft tokens depends on content, so never randomise them):
  chat  a long-form instruction from a fixed panel, default reasoning effort, natural EOS, output capped
        at --max-tokens (default 1024); reasoning tokens count as output
  long  a unique ~--long-tokens document (default 8192) and a question about it; 256 output tokens
"""
import argparse, json, os, random, statistics, string, sys, threading, time, urllib.request

PANEL = [
    "Write a Python implementation of an LRU cache with O(1) get and put, with docstrings and tests.",
    "Explain, step by step, how TCP congestion control works, including slow start and fast recovery.",
    "Write a short story (about 600 words) about a lighthouse keeper who finds a message in a bottle.",
    "Compare B-trees and LSM-trees for a write-heavy database workload. Use a table and give a recommendation.",
    "Write a bash script that backs up a directory to a timestamped tar.gz and keeps only the last 7 backups.",
    "Summarise the causes and consequences of the 1929 stock market crash in five paragraphs.",
    "Write a SQL schema for a library system with books, members and loans, and three example queries.",
    "Explain how a transformer language model generates text, for a curious high-school student.",
    "Write a React component for a searchable, sortable table, with TypeScript types.",
    "Plan a three-day trip to Kyoto for a first-time visitor who likes food and temples.",
    "Explain the difference between processes and threads, and when to use each, with examples in Go.",
    "Draft a polite email asking a landlord to fix a broken heater, and a firmer follow-up a week later.",
    "Write a Rust function that parses a CSV line with quoted fields, plus unit tests.",
    "Describe how vaccines train the immune system, including memory B cells and T cells.",
    "Write a Dockerfile and docker-compose.yaml for a Flask app with Postgres and Redis.",
    "Give a beginner's guide to personal budgeting with a worked monthly example.",
]
TOPICS = ("harbor logistics, river barges, grain prices, winter storms, ferry timetables, lighthouse "
          "maintenance, customs paperwork, fishing quotas, shipyard wages, canal locks").split(", ")


def long_prompt(tokens, seed):
    """A unique prose-like document of roughly `tokens` tokens with one fact to retrieve."""
    rng = random.Random(seed)
    code = "".join(rng.choices(string.ascii_uppercase + string.digits, k=8))
    sents, n = [], 0
    while n < tokens * 0.72:  # ~1.35 tokens per word for this kind of prose
        t = rng.choice(TOPICS)
        s = (f"In report {rng.randint(100, 999)}, the clerk noted that {t} changed by {rng.randint(1, 90)} percent "
             f"after the {rng.choice(['spring', 'autumn', 'summer', 'winter'])} meeting of the harbor board.")
        sents.append(s); n += len(s.split())
    sents.insert(len(sents) // 2, f"The vault access code for this archive is {code}.")
    head = f"[archive {seed}-{time.time_ns()}]\n"  # defeats prefix caching across requests
    return head + " ".join(sents) + "\n\nWhat is the vault access code for this archive? Answer with the code only.", code


def chat(base, model, prompt, max_tokens, extra, timeout=3600):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}}
    body.update(extra)
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic(); tf = None; usage = {}; text = []; last = None; gap = 0.0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if piece:
                    now = time.monotonic()
                    if tf is None:
                        tf = now
                    else:
                        gap = max(gap, now - last)
                    last = now
                    text.append(piece)
    t1 = time.monotonic()
    ct = usage.get("completion_tokens") or 0
    tf = tf or t1
    return {"t0": t0, "t1": t1, "ttft": tf - t0, "ct": ct, "pt": usage.get("prompt_tokens"),
            "decode": (ct - 1) / (t1 - tf) if ct > 1 and t1 > tf else None, "gap": gap, "text": "".join(text)}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))] if xs else None


def warm(a, n, extra):
    """N concurrent short requests, so shapes this batch size needs are compiled before the clock starts."""
    ths = [threading.Thread(target=lambda i=i: chat(a.base_url, a.model, PANEL[i % len(PANEL)], 64, extra))
           for i in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()


def level(a, n, extra):
    results, errors, lock = [], [], threading.Lock()
    if not a.no_warmup:
        warm(a, n, extra)

    def user(i):
        for r in range(a.rounds):
            k = i * a.rounds + r
            try:
                if a.workload == "long":
                    prompt, code = long_prompt(a.long_tokens, seed=1000 * n + k)
                    res = chat(a.base_url, a.model, prompt, 256, extra)
                    res["hit"] = code in res["text"]
                else:
                    res = chat(a.base_url, a.model, PANEL[k % len(PANEL)], a.max_tokens, extra)
                with lock:
                    results.append(res)
            except Exception as e:  # noqa: BLE001 - report every failure, keep the sweep going
                with lock:
                    errors.append(repr(e)[:200])

    ths = [threading.Thread(target=user, args=(i,)) for i in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    out = {"name": a.name, "workload": a.workload, "n": n, "rounds": a.rounds, "requests": len(results),
           "fail": len(errors), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if results:
        wall = max(r["t1"] for r in results) - min(r["t0"] for r in results)
        dec = [r["decode"] for r in results if r["decode"]]
        ttft = [r["ttft"] for r in results]
        out.update(agg_tok_s=round(sum(r["ct"] for r in results) / wall, 1),
                   user_tok_s_p50=round(statistics.median(dec), 1) if dec else None,
                   user_tok_s_p10=round(pct(dec, 10), 1) if dec else None,
                   ttft_p50_s=round(statistics.median(ttft), 2), ttft_p95_s=round(pct(ttft, 95), 2),
                   max_gap_p95_s=round(pct([r["gap"] for r in results], 95), 2),
                   mean_completion_tokens=round(statistics.mean(r["ct"] for r in results)),
                   mean_prompt_tokens=round(statistics.mean((r["pt"] or 0) for r in results)))
        if a.workload == "long":
            out["retrieval"] = f"{sum(r['hit'] for r in results)}/{len(results)}"
        out["slo"] = bool(out["user_tok_s_p10"] and out["user_tok_s_p10"] >= a.slo_tok_s
                          and out["ttft_p95_s"] <= a.slo_ttft_s and not errors)
    if errors:
        out["errors"] = errors[:3]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="inkling-2-mini")
    p.add_argument("--name", required=True)
    p.add_argument("--levels", default="1,4,16,32,64,96,128")
    p.add_argument("--workload", choices=["chat", "long"], default="chat")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--long-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--extra", default="{}", help="JSON merged into every request body")
    p.add_argument("--ignore-eos", action="store_true", help="force fixed-length replies (inflates draft acceptance)")
    p.add_argument("--slo-tok-s", type=float, default=10.0)
    p.add_argument("--slo-ttft-s", type=float, default=5.0)
    p.add_argument("--no-warmup", action="store_true", help="skip the unmeasured warm-up burst before each level")
    p.add_argument("--stop-after-fail", action="store_true", help="stop the sweep after a level misses the SLO")
    p.add_argument("--out", default="docs/raw/sweep.jsonl")
    a = p.parse_args()
    a.base_url = a.base_url.rstrip("/")
    extra = {"temperature": a.temperature, **json.loads(a.extra)}
    if a.ignore_eos:
        extra["ignore_eos"] = True
    chat(a.base_url, a.model, "Say hello.", 16, extra)  # warm-up, not measured
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for n in [int(x) for x in a.levels.split(",")]:
        r = level(a, n, extra)
        print(json.dumps(r), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        if a.stop_after_fail and not r.get("slo"):
            break


if __name__ == "__main__":
    sys.exit(main())
