# EXPERIMENTS.md — append-only ledger

Every experiment: hypothesis, config, gate, result. Gates are pre-committed before
the run. Evaluation is always the frozen 381-scenario internal bench, k=4, temp 0.7.

Decision rule template: an experiment's model replaces the serving model iff its
best-of-4 >= the current champion's best-of-4, measured with the same harness.

## E0 — Staircase A1→A2→B (completed 2026-10-14) — CHAMPION

- **Hypothesis**: the 3-phase mid-training→post-training recipe only pays off when
  completed by the full-context direct-emission phase (B); partial staircases hurt.
- **Data**: A1 1,322 rows (5k-char windows), A2 1,006 rows (20k-char windows),
  B 448-450 rows (57k-char windows ≈ 8k tokens), all from harvested real terminal
  sessions, 89.3% fresh decision points in B, 0 bench leaks (sha-verified).
- **Training**: QLoRA 4-bit, rank 64, lr 1e-4→5e-5, 2-3 epochs per phase, MoE
  17.6M trainable params. Single 3090 24GB. ~101 min for phase B.
- **Gate**: completed-staircase best-of-4 >= 0.5915 (base anchor).
- **Result**: **0.6815** (+0.0900 vs base) — ADOPTED as serving model.
  - by_bucket: exec 0.6976, read 0.7599, other 0.6238, plan 0.5342, edit 0.4334
  - pass@1 0.577, name 0.664, args_f1 0.436, valid 0.872, api_errors 0
  - Dose-response: A1 0.5619 < A2 0.5762 < base 0.5915 < staircase 0.6815
- **Conclusion**: partial doses damage, completion repairs + doubles tool-name
  accuracy. `results/20261014-163425-summary.json`.

## Reference — failed replications (negative results, kept)

- Base+LoRA overlay (same adapter, no staircase): 0.5738
- v1 (identical recipe on 3.5): abandoned at A2
- Qwen3.8-27B 3.8 base anchor: 0.7399; its A2 adapter: 0.6063 (dose damage replicated)
- 3.8 phase B attempt: killed by host OOM (pinned-page RAM leak at step 0-2)

## E1 — Expert iteration round 1 (RUNNING)

- **Hypothesis**: sampling k=8 with the champion on FRESH post-harvest scenarios
  (sessions created after the 2026-10-08 harvest, bench sessions excluded,
  0 bench-context hash hits), keeping only scenarios solved >=1x as
  RSFT-certified contexts, and continuing training (phase C, lr 2e-5) pushes
  best-of-4 past 0.6815. STaR/SWE-Gym-style rejection sampling reports +5-19
  absolute on agentic benchmarks.
- **Pool**: 109 scenarios from 21 fresh opencode sessions + 12 fresh claude-local
  files (bench sessions + 2 noise rows excluded). TB2 tasks are NEVER used for
  training (benchmark contamination ban); this is self-generated data only.
- **Rollout result**: 29/109 solved at least once (26.6%) at temp 0.95, k=8.
  The fresh pool is the hard tail (cluster-debugging sessions, web-tool args
  that are near-unmatchable exactly). `results/e1-rollouts.jsonl`.
- **Config amendment (pre-committed before training)**: 29 E1 rows alone risk
  catastrophic forgetting of the B junction — phase C trains on
  E1-wins (29) + B-replay (150 rows sampled from sft_mid_b.jsonl, seed 42).
  sft_e1.jsonl = 179 rows, sha256 40fb7e09…; Bash share 63.7%.
- **Training**: init staircase-b adapter, max_len 8192, lr 2e-5, 2 epochs,
  warmup 20, same QLoRA rank/gates as B.
- **Gate**: best-of-4 >= 0.6815 on the frozen bench, 381x4.
- **Status**: chain running (train -> merge -> quant -> serve -> eval).

## E2 — (queued) DPO on win/loss pairs from E1 rollouts

- **Hypothesis**: preference pairs (win vs fail trajectory on same scenario) add
  +2-4 over E1 (SWE-Gym finding).
- **Status**: QUEUED.

## E3 — (queued) TB2.1 external evaluation via Harbor

- **Hypothesis**: champion transfers to Terminal Bench 2.1 (89→fixed tasks) at a
  competitive resolution rate for an open 30B-class MoE (GLM-4.6 reference: 24.5%
  TB2.0; GLM-5.3: 41.8% TB2.1; frontier: 60-65%).
- **Config**: harbor CLI + terminus-2 agent against llama.cpp endpoint, 20-task
  subset first. Evaluation ONLY (never training data).
- **Status**: QUEUED.

## Deferred

- GRPO/RL with verifiable rewards: Unsloth docs — 4-bit MoE training unsupported on
  24GB (QLoRA SFT alone ~17.5GB). Requires >=80GB rental.
- Learned best-of-n verifier (outcome reward model on trajectories).
