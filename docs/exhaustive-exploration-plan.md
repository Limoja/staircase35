# Exhaustive Model × Approach × Teacher Exploration Plan (B200, offline data)

Purpose: expense planning for finding the best cost-efficiency combination for
10K users/day, 100 concurrent × 256K, fully offline. Clean slate — nothing assumed.

## 1. Approach taxonomy (all considered, each gets a ladder if it survives screening)

| # | Family | Approaches | Cost/iteration | Expected gain |
|---|---|---|---|---|
| AP1 | Supervised SFT | vanilla SFT; curriculum/staircase; RSFT/expert-iteration (STaR); replay-mixed RSFT | $ (LoRA, hrs) | +5-10 |
| AP2 | Distillation | teacher-trace SFT; logits/KD distill; on-policy distill (GKD) | $ | +5-15 (small models) |
| AP3 | Preference | DPO; KTO; ORPO; SimPO | $ | +2-4 |
| AP4 | Verifier RL | GRPO; DPPO; RLOO; GSPO; PPO+PRM; curriculum-difficulty RL | $$$ (2-3d multi-GPU) | +5-20 (largest) |
| AP5 | Inference-time | best-of-n + ORM verifier; self-consistency; scaffold/harness tuning; context engineering | $ serving-only | +5-15, no training |
| AP6 | System optim | weight quant (INT4/8, FP8, MXFP4); KV quant; spec-decode; cascades (small→large routing); batch schedulers (vLLM/SGLang/TRT-LLM) | $ eng | 2-10× tok/$ |
| AP7 | Data-centric | difficulty calibration; taxonomy+persona synthesis; verifier diversification; decontamination | $ | multiplies AP4 |

## 2. Model candidate matrix (not restricted; placeholders refreshed at Phase 0)

### System 1 — volume tier (routing, search, summary, SQL; <300ms first token)
| Class | Candidates (sizes 1.5-10B) |
|---|---|
| Dense | Qwen3 {1.7,4,8}B; Llama-3.x {1,3,8}B; Gemma-3 {1,4}; Phi-4-mini; SmolLM3-3B; Ministral-8B; Nemotron-Nano {2,8,9}B(hybrid); placeholder-S1D |
| MoE / hybrid-attn | Qwen3.5-A3B {4,6,9}B; OLMoE-7B; DeepSeek-V2-Lite-16B-A2.4B; Jamba-mini(hybrid); GRIN-MoE; gpt-oss-20B-A3.6B; placeholder-S1M |

### System 2 — agent/escalation tier (256K sessions, terminal work)
| Class | Candidates (10B-700B) |
|---|---|
| Dense | Qwen3-32B; Devstral-26B-dense; Codestral; Llama-3.3-70B; Mistral-Large; Nemotron-H-47B(hybrid); Tmax/Qwen3.6-27B; placeholder-S2D |
| MoE / hybrid-attn | Qwen3.5-35B-A3B; Qwen3-Next-80B-A3B(hybrid!); gpt-oss-120B-A5.1B(MXFP4, 128K); MiniMax-M2-230B-A10B; GLM-4.6-open-355B-A32B; DeepSeek-V3.x-671B-A37B; Kimi-K2-1T-A32B; Qwen3-235B-A22B; Llama-4-{Scout,Maverick}; placeholder-S2M |

KV economics note: at 100×256K, hybrid/MLA/MoE (Qwen3.5-A3B, Qwen3-Next, gpt-oss,
Nemotron-H, MiniMax-M2) fit 2-4× B200; classic dense >30B needs 6-30× B200 —
dense S2 only viable as INT4 27-70B escalation pod.

## 3. Teacher inventory
| Tier | Teachers | Use | Governance |
|---|---|---|---|
| T1 frontier API | GPT-5.x-Codex; Claude Opus 5.x; Gemini-3-Pro; GLM-5.3 | synthetic env/trace generation (Tmax-15k style), hard-task grading | PUBLIC/SYNTHETIC SEEDS ONLY — no NHS/UCLP data egress |
| T2 open heavyweights (offline) | GLM-4.6-open; DeepSeek-V3.x; Kimi-K2; Qwen3-235B; gpt-oss-120B | offline trace generation, KD logits | any data |
| T3 specialists | Tmax-27B; SWE-Gym/SWE-smith-trained; our fine-tunes | domain traces | any data |
| T4 self/verifier | student self-generation + programmatic verifiers (outcome, graded thresholds, fuzz, multi-protocol); ORM trained on our logs | RL rewards, best-of-n reranking | any data, zero egress |

## 4. Exhaustive search plan (matrix: model × approach × teacher)

**P0 Capability screen (wk 1-2, ~$3K):** all S1 (~15) zero-shot on routing/SQL/summary
mini-bench + KV/token@256K + tok/s/$ on 1× B200; all S2 (~14) on 381-bench + TB2.1-20.
Output: quality/$ frontier; cut to top-3 per tier per class. Gate: full frontier chart.

**P1 Recipe ladder (wk 3-6, ~$8K):** each top model × {AP1 SFT, AP2 distill-SFT (T2/T3
teacher), AP3 DPO, AP1-RSFT} standardized LoRA recipe (same data budget: 20K traces).
Output: approach × model gain table → pick 2 models/tier. Gate: quality ≥ baseline+5.

**P2 RL deep-dive (wk 7-12, ~$15K):** top-2 per tier × {GRPO vs DPPO} on Tmax-15k +
self-generated envs (AP7); teacher ablation T1-API vs T2-offline vs T4-self;
difficulty curriculum on/off. 2-3 B200-days per run. Gate: +5 over P1 champion.

**P3 System integration (wk 13-16, ~$6K + $8K/mo serving):** cascade router S1→S2
(threshold tuning), spec-decode pairs (best S1 draft → S2 target), final quant/KV
sweep (AP6), 100-slot 72h soak on 2× B200. Gate: TTFT p95 <5s warm, ≥25K tok/s,
<0.5% err, quality retention ≥98%.

**P4 Champion lock + cutover (wk 17-20, ~$3K):** TB2.1 full run, Jev→model shadow →
cutover, buy-vs-rent decision (owned 4-6× B200 ≈ $130-210K, payback 6-7 mo vs rent).

## 5. Budget envelope (R&D)
P0 $3K + P1 $8K + P2 $15K + P3 $14K + P4 $3K ≈ **$43K over 5 months** + serving
$8K/mo from P3. Contingency +30% (RL instability, re-runs) → **~$55K total**.
Every cell of the matrix lands in EXPERIMENTS.md; weights → buckets (standing rule).
