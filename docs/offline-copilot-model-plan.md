# Offline Copilot Model Plan — fine-tuned agentic-coding model for 10K-user scale

## Goal
One open, fine-tuned agentic-coding model family that serves 10K users/day
(100 concurrent × 256K context, fully offline) at ~3 cents/session.

## Substrate decision (locked by physics)
256K × 100 concurrent = 25.6M resident KV tokens. Only hybrid linear-attention MoE
makes this cheap (~10KB/token → ~260GB pool on 2× B200; dense models need 460GB-4.8TB).
**Base: Qwen3.5-35B-A3B** (staircase champion, +9pts proven, 262K on one 24GB card)
for agent sessions; **4B/9B-A3B siblings** for volume tier (routing/summary/SQL).

## Recipe (evidence-backed)
1. Staircase SFT — done, 0.5915 → 0.6815 on frozen 381-bench (E0, our result).
2. DPPO RL on Tmax-15k (open, 14.6K environments, Ai2 recipe: FP32 LM head,
   group 32, async vLLM rollouts) + our harvested session data.
   Tmax paper: this recipe on 9B matches GLM-4.6-class terminal performance.
3. Jev replacement: distill JEV routing decisions into the 4B tier (X4),
   CLA (35B fine-tune) takes over agent escalations (X5), shadow → cutover.

## Experiment ladder (every entry: pre-committed gate, GitHub ledger, weights to buckets)
- X1 bake-off: 4B/9B/35B-A3B + Tmax-27B on 381-bench + TB2.1 subset; measure KV/token.
- X2 serving: vLLM vs SGLang on 2× B200, 100-slot replay from harvest traces.
  Gate: TTFT p95 <5s warm, <15s cold-256K; decode ≥25K tok/s fleet.
- X3 quant: IQ4_XS vs Q8 + KV q8/q4 at 256K needle tests. Gate: ≤1pt loss.
- X4 Jev distillation (4B). Gate: catalog top-k ≥ JEV baseline.
- X5 CLA RL run: staircase champion + DPPO on Tmax-15k (~8 H100-equiv × 2-3 days,
  on-demand only; ~$1-2K per run). Gate: bench ≥0.72, then TB2.1 ≥ GLM-4.6 (24.5).
- X6 soak: 72h × 100 slots, KV eviction + prefix-cache policy. Gate: zero data loss,
  error rate <0.5%.

## Cost (10K users/day, fully offline)
| Phase | Monthly |
|---|---|
| Pilot: 2× B200 rental + X4/X5 runs | ~$8K + $1-2K/run |
| Steady state: 2× B200 (Tier1 + CLA escalations) | ~$8-10K |
| End-state owned: 4-6× B200 | $130-210K capex, payback 6-7 mo |
| Per session | **~$0.03** |
| vs GLM API w/cache | ~$33K/mo (rejected: data must stay offline) |

## Timeline
Wk1 X1 (3090 + rented day) → Wk2-3 X2/X3 (2× B200 days) → Wk3-6 X4/X5 RL runs +
shadow → Wk6-8 X6 soak → Jev→CLA cutover → buy decision.

## Data governance
All training/eval data from our own harvest + open datasets (Tmax-15k, CC-BY);
TB2 tasks NEVER in training. All artifacts: github.com/Limoja/staircase35 ledger +
contabo:qwen35-posttraining/ buckets.
