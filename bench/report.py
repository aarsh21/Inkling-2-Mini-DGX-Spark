#!/usr/bin/env python3
"""Summarise docs/raw/sweep.jsonl: one table per experiment, plus the capacity at each per-user floor.

  python3 bench/report.py [docs/raw/sweep.jsonl] [--only name1,name2]

Capacity at a floor F = the largest measured concurrency whose P10 per-request decode >= F tok/s with
TTFT p95 <= 5 s and no failed requests. Rows are the latest run of each (name, workload, n).
"""
import json, sys

path = next((a for a in sys.argv[1:] if not a.startswith("--") and a.endswith(".jsonl")), "docs/raw/sweep.jsonl")
only = None
if "--only" in sys.argv:
    only = sys.argv[sys.argv.index("--only") + 1].split(",")
rows = {}
for line in open(path):
    if line.strip():
        r = json.loads(line)
        rows[(r["name"], r["workload"], r["n"])] = r
names = []
for (name, wl, _n) in rows:
    if (name, wl) not in names and (only is None or name in only):
        names.append((name, wl))


def cap(rs, floor):
    ok = [r["n"] for r in rs if r.get("user_tok_s_p10") is not None and r["user_tok_s_p10"] >= floor
          and r.get("ttft_p95_s", 99) <= 5 and not r.get("fail")]
    return max(ok) if ok else "-"


print("| Experiment | Workload | ≥20 tok/s | ≥10 tok/s | ≥5 tok/s | Peak aggregate tok/s |")
print("|---|---|---:|---:|---:|---:|")
for name, wl in names:
    rs = sorted((r for k, r in rows.items() if k[0] == name and k[1] == wl), key=lambda r: r["n"])
    peak = max(r.get("agg_tok_s") or 0 for r in rs)
    print(f"| {name} | {wl} | {cap(rs, 20)} | {cap(rs, 10)} | {cap(rs, 5)} | {peak:.0f} |")
for name, wl in names:
    rs = sorted((r for k, r in rows.items() if k[0] == name and k[1] == wl), key=lambda r: r["n"])
    print(f"\n**{name}** ({wl})\n")
    print("| Users | Aggregate tok/s | Per user p50 | Per user p10 | TTFT p50 / p95 s | Longest pause p95 s | Fail |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for r in rs:
        print(f"| {r['n']} | {r.get('agg_tok_s', '-')} | {r.get('user_tok_s_p50', '-')} | {r.get('user_tok_s_p10', '-')} "
              f"| {r.get('ttft_p50_s', '-')} / {r.get('ttft_p95_s', '-')} | {r.get('max_gap_p95_s', '-')} | {r.get('fail', 0)} |")
