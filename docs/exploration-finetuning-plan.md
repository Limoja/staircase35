# Model Exploration & Fine-Tuning Program — B200 era

Goal: max cost-efficiency offline agentic-coding fleet (10K users, 100×256K).
Every experiment: pre-committed gate, EXPERIMENTS.md ledger, weights→buckets.

## 1. Model inventory to explore

### Agent tier (100 agent slots, escalations)
| Slot | Model | Status / evidence |
|---|---|---|
| A1 | Qwen3.5-35B-A3B staircase champion | 0.6815 bench; hybrid KV; DONE SFT |
| A2 | **Qwen3.8-27B (the 3.8 we had)** | base anchor **0.7399** — strongest raw bench score ever; its staircase B-phase OOM'd on old hw; **B200 removes that blocker — complete its staircase + RL** |
| A3 | Tmax-27B (allenai) | 44.9 TB2.1 published; dense → KV-expensive at 256K; escalation-only or teacher |
| A4 | GLM open-weight class (355B-A32B) | INT4 on 4-6× B200; only if X-gates fail |
| A5 | placeholder: next-gen ~30B hybrid MoE when released | re-run X1 |

### Volume tier (routing, search, SQL, summary)
| Slot | Model | Notes |
|---|---|---|
| V3 | ~3B hybrid slot | evaluate best current 3B (Qwen3.5-3B? + candidates) |
| V4 | Qwen3.5-4B-A3B | primary volume candidate |
| V5 | ~5B slot | fill at selection (Phi/Gemma/Nemotron class vs Qwen) |
| V6 | ~6B slot (Qwen3.5-6B-A3B?) | latency/quality knee |
| V9 | Qwen3.5-9B-A3B | volume ceiling; Tmax-9B evidence (27% TB2.0) |

### Teachers (all offline; TB2 tasks never trained on)
- T1 Tmax-27B — RL-trained, open, terminal-specialist
- T2 Qwen3.8-27B — our bench king (0.7399)
- T3 staircase-35B champion — harness-aligned
- T4 self-distillation loops (student samples, teacher verifies/scores via verifiers à la Tmax-15k graded verifiers)
- Tmax-15k dataset (open, Gemini-generated, CC-BY) = teacher data without API calls

## 2. Approaches to compare (one ladder each)
1. Staircase SFT (ours, proven +9pts) — port to 3.8-27B, 4B, 9B
2. RSFT/EI (E1: neutral at 29 wins; retry at B200 scale k=32 groups)
3. DPPO RL on Tmax-15k (Tmax recipe: FP32 LM head, group 32, async vLLM)
4. DPO win/loss pairs (E2, queued)
5. Teacher→student distillation (logits/SFT traces from T1/T2 → V-tier)
6. Speculative decoding: V4 draft → A1/A2 target (2-3× cheap-session throughput)
7. Quant: IQ4_XS weights + q8/q4 KV; gate ≤1pt loss

## 3. Experiment ladder (B200 assumed available)
- X1 bake-off: all slots, 381-bench + TB2.1-20-subset, measure KV/token + tok/$.
  Gate: pick V-tier winner (≥0.95× of 9B quality at ≥2× tok/$).
- X2 serving: vLLM vs SGLang, 100-slot harvest-trace replay, 2× B200.
  Gates: TTFT p95 <5s warm/<15s cold; ≥25K tok/s fleet decode.
- X3 quant/KV ladder at 256K needle tests.
- E5 complete-3.8-staircase: A2 phase B (OOM'd before) on 1× B200, then eval vs 0.7399.
  Gate: ≥0.78 (its base + our +9pt pattern). If hit → new champion.
- E6 RL round 1: DPPO on best agent model, Tmax-15k, 2-3 days × 4-6 B200 (~$2-3K).
  Gate: bench ≥ +5pts over its SFT start.
- E7 V-tier distillation: T1/T2 → V4/V6 (200K trace pairs), gate ≥0.9× teacher on routing tasks.
- E8 speculative decode integration + bench (gate ≥1.8× throughput, ≤0.5pt loss).
- E9 DPO pairs from E6 rollouts (+2-4 expected).
- X6 soak 72h × 100 slots → cutover Jev→CLA → buy decision.

## 4. B200 budget envelope
Bench weeks (X1-X3, day-rentals): ~$1-2K. Fine-tunes (E5,E7): ~$1K.
RL runs (E6, E9): ~$2-3K each iteration. Steady serving pilot: 2× B200 ~$8K/mo.
Champion artifacts always: GitHub ledger + contabo buckets + GGUF on 3090 fallback.
