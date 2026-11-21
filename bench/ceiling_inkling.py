#!/usr/bin/env python3
"""Decode ceilings for Inkling 2 Mini on one GPU, per concurrency, with and without DSpark.

  python3 bench/ceiling_inkling.py [bandwidth_GB_s] [context_tokens] [dspark_accept]

Bytes read per decode step (one step = one token per stream, or one verify block with DSpark):
  F      fixed weights read every step: BF16 attention and gate weights on all 48 layer visits (~1.8 GB)
         plus the LM head (201,024 x 2,304 BF16 = 0.93 GB); DSpark adds its 0.64 GB draft
  E      routed experts: each of the 48 visits reads the union of its rows' top-8 of 128 experts.
         44 NVFP4 visits x 0.573 GB + 4 MXFP8 visits x 1.05 GB = 29.4 GB when every expert is touched;
         the touched share is 1 - (120/128)^rows under uniform routing
  KV     per stream: 8 full-attention caches (layers 5 and 23 once, 11 and 17 on three passes) and 40
         sliding-window caches of 1,024 tokens, 4 KiB per token per cache in BF16 (+1 draft cache)
rows = streams without speculation, 8 x streams with DSpark (block of 8 verified per step).
aggregate = streams x accept x bandwidth / bytes_per_step. A ceiling: compute also binds at large row counts.
"""
import sys

B = float(sys.argv[1]) if len(sys.argv) > 1 else 220.0
CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
ACC = float(sys.argv[3]) if len(sys.argv) > 3 else 3.5
F, DRAFT, E_FULL = 2.73, 0.64, 29.4
KV_FULL = (8 + 1) * 4096          # bytes per token of context
KV_SWA = (40 + 2) * 4096 * 1024   # bytes per stream, fixed


def step_gb(n, rows, draft):
    experts = E_FULL * (1 - (120 / 128) ** rows)
    kv = n * (KV_FULL * CTX + KV_SWA) / 1e9
    return F + (DRAFT if draft else 0) + experts + kv


print(f"bandwidth {B} GB/s, context {CTX} tokens, DSpark accepted tokens per step {ACC}")
print(f"{'streams':>8} {'plain agg':>10} {'per user':>9} {'dspark agg':>11} {'per user':>9}")
for n in (1, 2, 4, 8, 16, 32, 48, 64, 96, 128):
    plain = n * B / step_gb(n, n, False)
    spec = n * ACC * B / step_gb(n, 8 * n, True)
    print(f"{n:>8} {plain:>10.0f} {plain / n:>9.1f} {spec:>11.0f} {spec / n:>9.1f}")
