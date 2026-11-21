#!/usr/bin/env python3
"""GSM8K quality gate for any OpenAI-compatible server. Standard library only.

  python3 bench/gsm8k.py --base-url http://127.0.0.1:8000 --model inkling-2-mini --n 250 --name vllm-best

Takes the first --n questions of the GSM8K test split (openai/gsm8k, main), asks each at temperature 0
with the default reasoning effort, and scores the last number in the reply against the reference.
--concurrency requests run at once, so the gate also exercises batching. Appends one JSON line to
docs/raw/gsm8k.jsonl. The SGLang cookbook reports 93.6-94.2% on all 1,319 questions for its Spark recipes.
"""
import argparse, json, os, re, socket, threading, time, urllib.request

_orig = socket.getaddrinfo
socket.getaddrinfo = lambda h, p, f=0, *a, **k: _orig(h, p, socket.AF_INET, *a, **k)  # the Hub over IPv4
ROWS_API = "https://datasets-server.huggingface.co/rows?dataset=openai/gsm8k&config=main&split=test&offset={}&length=100"


def questions(n, cache):
    if os.path.exists(cache):
        rows = json.load(open(cache))
    else:
        rows = []
        while len(rows) < n:
            got = json.loads(urllib.request.urlopen(ROWS_API.format(len(rows)), timeout=60).read())["rows"]
            if not got:
                break
            rows += [r["row"] for r in got]
        json.dump(rows, open(cache, "w"))
    return rows[:n]


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="inkling-2-mini")
    p.add_argument("--name", required=True)
    p.add_argument("--n", type=int, default=250)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--cache", default="/tmp/gsm8k-test.json")
    a = p.parse_args()
    qs = questions(a.n, a.cache)
    todo, lock, results = list(enumerate(qs)), threading.Lock(), {}

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                i, q = todo.pop(0)
            body = {"model": a.model, "temperature": 0, "max_tokens": a.max_tokens, "messages": [{"role": "user",
                    "content": q["question"] + "\nGive the final answer as a number on its own last line."}]}
            req = urllib.request.Request(a.base_url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
                msg = r["choices"][0]["message"]
                ans = last_number(msg.get("content") or "")
                ref = q["answer"].split("####")[-1].strip().replace(",", "")
                ok = ans is not None and float(ans) == float(ref)
                results[i] = (ok, r["usage"]["completion_tokens"], r["choices"][0].get("finish_reason"))
            except Exception as e:  # noqa: BLE001
                results[i] = (False, 0, repr(e)[:80])

    t0 = time.time()
    ts = [threading.Thread(target=worker) for _ in range(a.concurrency)]
    [t.start() for t in ts]; [t.join() for t in ts]
    correct = sum(r[0] for r in results.values())
    out = {"name": a.name, "n": len(qs), "correct": correct, "accuracy": round(100 * correct / len(qs), 1),
           "mean_completion_tokens": round(sum(r[1] for r in results.values()) / len(qs)),
           "truncated": sum(r[2] == "length" for r in results.values()),
           "errors": sum(isinstance(r[2], str) and r[2] not in ("stop", "length", None) for r in results.values()),
           "wall_s": round(time.time() - t0), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(json.dumps(out))
    os.makedirs("docs/raw", exist_ok=True)
    with open("docs/raw/gsm8k.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
