# Exhaustive Exploration Plan — Situation / Complication / Resolution
(B200 era · 12 weeks · 20K GPU-hours · fully offline data)

## 1. SITUATION — where we stand

We operate a copilot platform (data.limoja.ai SPA + nhs-portal warehouse copilot,
Jev heuristic routing, rag/llm services behind ch-proxy). Target load: **10K unique
users/day swapping through 100 concurrent agent slots at up to 256K context each**.
Hard constraint: **all data stays offline** (NHS/UCLP). Assets available: prior
fine-tuning know-how (staircase SFT +9pts on a 35B-A3B hybrid), a 381-scenario
frozen bench + harvested session traces, the open Tmax recipe (DPPO RL on 14.6K
synthetic terminal environments), rented B200s at will (on-demand only), a
**20,000 B200 GPU-hour budget**, and a **3-month deadline**.

## 2. COMPLICATION — why this is hard

1. **API economics invert at our scale**: tokens billed per user → $33K-300K/mo;
   GPUs billed per hour (concurrency-capped) → ~$8K/mo. Offline is mandatory
   (privacy) AND economically right — but only if the right model exists.
2. **KV physics gates the model choice**: 100 × 256K = 25.6M resident KV tokens.
   Classic dense >30B needs 0.5-5TB KV (6-30 B200s); hybrid/MLA MoE needs ~260GB
   (2-4 B200s). Wrong architecture = dead on arrival.
3. **The best model × approach × teacher combination is unknown**: candidates span
   1.5B-1T params across dense/MoE/hybrid, system-1 vs system-2 roles; approaches
   span 7 families with 5-20pt swings; teachers span 4 tiers with egress limits.
4. **Quality gap to close**: escalation-tier work needs GLM-4.6-class terminal
   performance (24.5%+ TB2.0) from an offline, affordable model; Jev must be
   replaced without regression.
5. **Time**: 12 weeks with RL instability risk and no second chance at a full
   re-plan if gates miss late.

## 3. RESOLUTION — the exhaustive 12-week search (compute-saturated, ledger-tracked)

### 3.1 Approach taxonomy (7 families, all screened)
[AP1 SFT: vanilla/curriculum-staircase/RSFT/RSFT+replay | AP2 Distillation:
trace-SFT/KD/GKD | AP3 Preference: DPO/KTO/ORPO/SimPO | AP4 Verifier RL:
GRPO/DPPO/RLOO/GSPO/PPO+PRM/curriculum | AP5 Inference-time: best-of-n+ORM/
self-consistency/scaffold | AP6 System: quant/KV/spec-decode/cascades/schedulers |
AP7 Data-centric: difficulty calibration/taxonomy/verifier diversification]

### 3.2 Model matrix (candidates, refreshed at P0)
- System-1 volume (dense 1.5-10B): Qwen3{1.7,4,8}B, Llama-3.x{1,3,8}B, Gemma-3{1,4},
  Phi-4-mini, SmolLM3-3B, Ministral-8B, Nemotron-Nano{2,8,9}B(hybrid), placeholder-S1D
- System-1 volume (MoE/hybrid): Qwen3.5-A3B{4,6,9}B, OLMoE-7B, DeepSeek-V2-Lite,
  Jamba-mini, GRIN-MoE, gpt-oss-20B-A3.6B, placeholder-S1M
- System-2 agent (dense 27-70B): Qwen3-32B, Devstral-26B, Codestral, Llama-3.3-70B,
  Mistral-Large, Nemotron-H-47B(hybrid), Tmax/Qwen3.6-27B, placeholder-S2D
- System-2 agent (MoE/hybrid 35B-1T): Qwen3.5-35B-A3B, Qwen3-Next-80B-A3B,
  gpt-oss-120B-A5.1B, MiniMax-M2-230B-A10B, GLM-4.6-open-355B-A32B,
  DeepSeek-V3.x-671B, Kimi-K2-1T-A32B, Qwen3-235B-A22B, Llama-4-Scout/Maverick,
  placeholder-S2M
- KV rule: hybrid/MLA/MoE fits 2-4 B200 at 100x256K; dense S2 only as INT4 27-70B
  escalation pod.

### 3.3 Teachers (4 tiers, governance-bound)
T1 frontier APIs (public/synthetic seeds ONLY): GPT-5.x-Codex, Opus 5.x, Gemini-3-Pro,
GLM-5.3 | T2 open heavyweights (offline, any data): GLM-4.6-open, DeepSeek-V3.x,
Kimi-K2, Qwen3-235B, gpt-oss-120B | T3 specialists: Tmax-27B, SWE-Gym/SWE-smith
graduates | T4 self+verifier: student rollouts graded by programmatic verifiers
(outcome/threshold/fuzz/multi-protocol) + ORM on our logs. TB2 tasks NEVER trained on.

### 3.4 Phases, gates, gantt (12 weeks)
| Wk | 1-2 | 3-4 | 4-9 | 7-10 | 10-12 |
|---|---|---|---|---|---|
| Track | P0 screen 29 models (quality/KV/tok-per-$) | P1 ladder ALL survivors (~40 runs, 4 recipes) | P2 RL matrix 8 runs + 3 teacher tracks | P3 cascade/spec-dec/quant (concurrent) | soak 1wk, dual shadow, cutover |
| Gate | frontier chart | +5pt over base | +5pt over P1 (wk7 gate; miss => P1 champion ships) | TTFT p95<5s warm, 25K tok/s, <0.5% err | Jev>= parity, TB2.1 final |

Critical path = RL runs + analysis gates (GPUs unlimited). Fallback ladder:
RL miss -> best P1 SFT+distill champion; P3 miss -> reduce concurrent slots to 60.

### 3.5 GPU-hour allocation (20K envelope)
Base 3,700 + full RL matrix 1,800 + teacher tracks 800 + widened ladders 1,200 +
deep TB2.1 600 + ORM/best-of-n 400 + spec-pair sweep 300 + seed-variance reruns 1,500 +
long soak/dual shadow 800 + 30% reserve 6,000 = **~19,100 of 20,000**.
Reserve drawn only against logged failures; if unused by wk10 -> 2nd champion RL
iteration. Cash: ~$105K at $5.5/GPU-h worst case; serving pilot from wk10 ~$8K/mo;
end-state owned 4-6x B200 ($130-210K, payback 6-7 mo vs renting).

Every experiment: pre-committed gate + EXPERIMENTS.md ledger entry + weights to
buckets. Governance: offline-only for sensitive data; API teachers on public seeds.
