#!/usr/bin/env python3
"""Dashboard-like TTFT probe. Standard library only.

Each request: a shared ~N-token system prompt (like Ask's system prompt and tools, cacheable) plus a
unique user question. Reports TTFT (first streamed delta of any kind), time to first answer token
(content), aggregate and per-user decode tok/s at each concurrency level.
  python3 probe.py URL MODEL NAME LEVELS [SYS_TOKENS] [MAX_TOKENS] [EXTRA_JSON]
"""
import json, statistics, sys, threading, time, urllib.request

url, model, name, levels = sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split(",")]
sys_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 6000
max_tokens = int(sys.argv[6]) if len(sys.argv) > 6 else 400
extra = json.loads(sys.argv[7]) if len(sys.argv) > 7 else {}

RULES = ("You are Senpiper Ask, an assistant for a manufacturing company. Answer using the company's documents, "
         "cite sources, keep answers short, use tables for comparisons, never invent numbers. ")
TOOL = ("Tool {i}: search_documents_{i}(query: string, folder: string, limit: integer) returns matching passages "
        "from the document store with their file names, page numbers and relevance scores. ")
system = RULES + "".join(TOOL.format(i=i) for i in range(sys_tokens // 40))
QUESTIONS = ["What is our leave policy for new joiners?", "Summarise last quarter's safety incidents.",
             "How do I raise a purchase order above 5 lakh?", "Explain the shift allowance rules.",
             "Which vendors supply bearings and what are the lead times?", "Draft a reminder about the audit.",
             "What does the QA checklist say about weld inspection?", "Compare the two coolant suppliers."]


def one(i, out):
    body = {"model": model, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
            "temperature": 0.7, "messages": [{"role": "system", "content": system},
                                              {"role": "user", "content": f"[{i}-{time.time_ns()}] " + QUESTIONS[i % len(QUESTIONS)]}]}
    body.update(extra)
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    t0 = time.time(); first = answer = None; usage = {}
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                    continue
                d = json.loads(line[6:])
                if d.get("usage"):
                    usage = d["usage"]
                for c in d.get("choices") or []:
                    delta = c.get("delta") or {}
                    if any(delta.get(k) for k in ("content", "reasoning", "reasoning_content", "tool_calls")):
                        first = first or time.time()
                    if delta.get("content"):
                        answer = answer or time.time()
        t1 = time.time()
        n = usage.get("completion_tokens", 0)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        out.append(dict(ttft=first - t0, tfa=(answer or t1) - t0, dec=(n - 1) / max(t1 - first, 1e-6), n=n,
                        start=t0, end=t1, prompt=usage.get("prompt_tokens"), cached=cached))
    except Exception as e:  # noqa: BLE001
        out.append(dict(error=str(e)[:200]))


def q(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]


one(999, [])  # warm the shared prefix
for n in levels:
    out = []
    ts = [threading.Thread(target=one, args=(i, out)) for i in range(n * 2)]
    # n concurrent users, two waves
    for wave in (ts[:n], ts[n:]):
        [t.start() for t in wave]; [t.join() for t in wave]
    ok = [o for o in out if "error" not in o]
    agg = sum(o["n"] for o in ok) / (max(o["end"] for o in ok) - min(o["start"] for o in ok)) if ok else 0
    print(json.dumps(dict(name=name, c=n, ok=len(ok), fail=len(out) - len(ok),
                          prompt=ok[0]["prompt"] if ok else None, cached=ok[0]["cached"] if ok else None,
                          ttft_p50=round(q([o["ttft"] for o in ok], .5), 2), ttft_p95=round(q([o["ttft"] for o in ok], .95), 2),
                          answer_p50=round(q([o["tfa"] for o in ok], .5), 2),
                          user_p50=round(q([o["dec"] for o in ok], .5), 1), user_p10=round(q([o["dec"] for o in ok], .1), 1),
                          agg=round(agg, 1), out_tok_mean=round(statistics.mean(o["n"] for o in ok)))), flush=True)
