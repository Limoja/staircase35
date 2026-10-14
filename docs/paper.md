# The Staircase: Completing Mid-Training Unlocks Direct Tool Emission at 262k Context

*Working paper. Code in `code/`, ledger in `EXPERIMENTS.md`, raw eval JSON in `results/`.*

## Abstract

We post-train Qwen3.5-35B-A3B, a 35B-parameter MoE language model (3B active), to
emit the next terminal tool action directly — JSON tool call, verbatim — from
agent-session context up to 262k tokens, using QLoRA on a single RTX 3090 (24GB).
We document a reproducible failure mode and its repair: intermediate "mid-training"
doses of next-action prediction at short context (5k, 20k chars) monotonically
*degrade* a healthy base model's tool-calling (best-of-4 accuracy 0.5915 → 0.5619),
a result replicated across two model versions. Completing the staircase with a
final phase that re-learns the same decision function at pretraining-length context
(57k-char windows) not only repairs the damage but roughly doubles tool-name
selection accuracy (0.344 → 0.664) and beats the base by +9.0 points (0.5915 →
0.6815 best-of-4 on a frozen 381-scenario internal bench). We release the full
training pipeline, evaluation harness, and experiment ledger; model weights are
distributed as GGUF quantizations.

## 1. Introduction

Agentic coding assistants spend most of their life in tool-call loops: read a
file, run a command, edit, verify. Frontier models handle this well; open
sub-40B MoE models lose accuracy as context grows, or drift into narration
instead of acting. Our goal is a model that, given a long real session, emits
the correct next action directly with no scaffold.

We report three contributions:

1. **A dose-response study of mid-training.** Partial staircases (phase A1 at
   5k chars, A2 at 20k chars) hurt every base they touch — monotonic,
   replicated on two model versions (Section 4).
2. **Completion as the active ingredient.** Phase B (57k-char windows, direct
   emission) is what unlocks the gain; it doubles tool-name accuracy.
3. **A full pipeline reproducible on one 24GB GPU** — data harvest, dataset
   builders with leak gates, QLoRA trainer with per-phase gates, merge +
   quantization + serving + evaluation harness, and an append-only experiment
   ledger with pre-committed decision rules.

## 2. Data

All training data is harvested from real terminal-agent sessions (our own
infrastructure logs), never from public benchmarks. Sessions are decomposed
into fixed-length character windows (phase-specific: 5k / 20k / 57k) with
stride = length/4, supervised spans restricted to `{"tool": ..., "input": ...}`
decision points plus their immediate boundary tokens, and deduplicated by
decision-point content hash across phases. Two mechanical gates run before any
phase trains: (a) row-count asserts, (b) n-gram overlap against the frozen
evaluation bench (leak check must return 0 hits).

The evaluation bench is 381 scenarios harvested from *held-out* sessions,
frozen with a content hash, each scenario = context prefix + gold next action,
scored on tool-name match, argument F1, JSON validity, and a safety check
(destructive-command refusal).

## 3. Training

QLoRA (4-bit NF4 base, rank 64, MoE — 17.6M trainable params), unsloth +
TRL SFTTrainer, bf16, one RTX 3090. Phase hyperparameters:

| Phase | Rows | max_len | LR | Epochs | Init from |
|---|---|---|---|---|---|
| A1 | 1,322 | 2048 | 1e-4 | 3 | base |
| A2 | 1,006 | 4096 | 5e-5 | 3 | A1 adapter |
| B | 448 | 8192 | 5e-5 | 2 | A2 adapter |

Each phase gates on supervised-span integrity (trainable-token count) before
starting. Merging applies the cumulative adapter to base, verifies per-module
deltas (230/230 modules, max |Δ| 0.0044), converts to GGUF, quantizes IQ4_XS
(18.9GB), and serves with llama.cpp at 262,144 context, flash-attention on.

## 4. Results

Dose-response on the frozen bench (381 scenarios × 4 samples, temp 0.7):

| Model | Best-of-4 | pass@1 | name | args F1 | valid |
|---|---|---|---|---|---|
| A1-only | 0.5619 | 0.547 | 0.291 | 0.294 | 0.931 |
| A2-only | 0.5762 | — | 0.320 | 0.314 | 0.932 |
| Base | 0.5915 | 0.570 | 0.344 | 0.331 | 0.951 |
| Base + A-adapter overlay | 0.5738 | — | — | — | — |
| **A1→A2→B (complete)** | **0.6815** | 0.577 | **0.664** | **0.436** | 0.872 |

Replication on a second model version (Qwen3.8-27B, different harness instance):
base 0.7399, its A2 adapter 0.6063 — dose damage reproduces; that campaign's
phase B never ran (host OOM), which is what motivated completing it here.

Per-bucket (complete staircase vs A1-only): exec 0.698, read 0.760, plan 0.534,
edit 0.433, other 0.624. The largest repair is in `exec` (+0.16) — direct
command emission at long context.

**Validity trade.** The complete staircase is *less* conservative: it attempts
more tool calls, and 12.8% are malformed (vs 4.9% base). Net effect strongly
positive; a decoding-level or verifier-level repair of malformed attempts is
future work (E2).

## 5. Discussion

The pattern — intermediate doses hurt, completion pays — mirrors curriculum
results in continual learning: each phase overfits its context-length prior and
*erases* the junction behaviour the base had at longer context; the final phase
re-learns the junction at full length and locks in the accumulated signal. The
practical rule for practitioners: **never deploy a partial staircase.**

## 6. Ongoing work

Expert iteration (E1: rejection-sample wins on unsolved scenarios → SFT),
DPO on win/loss pairs (E2), external evaluation on Terminal Bench 2.1 via
Harbor (E3; GLM-4.6 references 24.5% on TB2.0), and GRPO on >=80GB rentals.
Ledger: `EXPERIMENTS.md`.

## Reproducibility

`code/build_mid*.py` (dataset builders), `code/train35*.py` (per-phase QLoRA
trainer with gates), `code/merge35.py` (merge + GGUF), `code/run_eval35.py`
(frozen-bench harness + scorer), `code/extract_scenarios.py` (bench harvest).
Weights: IQ4_XS GGUF on public object storage (link in repo releases).
