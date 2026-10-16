# Fable Recipe — everything recoverable about DavidAU's TURBO-Fable-735 pipeline
Status: assembled 2026-10-16 from public artifacts. EXACT = verbatim from source. INFERRED = reconstructed.

## Stage 0 — base (EXACT)
Qwen/Qwen3.8-27B. 27.78B params BF16, 12 shards + model-mtp-restored.safetensors.

## Stage 1 — Heretic decensor (EXACT, fully reproducible)
Artifact: trohrbaugh/Qwen3.8-27B-heretic-ara (2.5K dl, tag: reproducible).
```bash
pip install "heretic-llm @ git+https://github.com/timrohrbaugh/heretic"
heretic --reproduce reproduce/reproduce.json   # skips search, rebuilds exact model
```
Recorded: seed 705085018, 60 trials (20 startup), KL 0.0535, refusals 0/100.
Targets: attn.o_proj + mlp.down_proj, ARA on, kl_target 0.01, batch 128.
Probe data (public): mlabonne/harmless_alpaca + mlabonne/harmful_behaviors (400 train / 100 eval).
Guardrail: 12 benchmarks (AGIEval, BBH, CommonsenseQA, EQ-Bench, GSM8K, HellaSwag, IFEval, MMLU, MMLU-Pro, PIQA, Winogrande).
Full config.toml archived at: trohrbaugh/Qwen3.8-27B-heretic-ara/resolve/main/reproduce/config.toml

## Stage 2 — COLD FUSION SFT (partial: method known, hyperparams NOT published)
GAIN (per-sample dynamic training adjustment, in-house, no code) + Unsloth trainers.
Output: DavidAU/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1 (base_model: finetune Qwen3.8-27B).
Training data repos (GATED — access via author/Discord David_AU [drawless111]):
DavidAU/Polar-STRICT-Datasets, DavidAU/Reasoning-STRICT-Datasets (+ F451-STRICT for final).
INFERRED starting point: Unsloth Qwen3 SFT defaults (LoRA r16-32, LR ~2e-4, 1-3 epochs, 8-32K seq).

## Stage 3 — Fable Fusion merge (partial: ingredients known, ratios NOT published)
DNA (per card): Fable-Fusion-711-dark-roast (Qwen3.6 line) + Cold-Fusion-GAIN-V1.1.
Tool: mergekit (per author stars + nightmedia tags). Pattern from sibling merges
(e.g. nightmedia/Qwen3.5-9B-Seven = 8 named ingredients in base_model:merge: tags):
multi-model dare_ties/slerp ladder, bench-gated per step. Exact YAML unpublished.

## Stage 4 — re-Heretic + final tune (EXACT per card)
Heretic v1.2+ARA again post-merge, then final fine-tune. Stage-2 KLD 0.0025, refusals 11/100
(deliberately balanced for quality first, refusal-rate second).

## Stage 5 — quant (EXACT)
Dual/imatrix NEO GGUF ("NEO Imatrix dataset" calibration, unpublished), output tensor kept
at 16-bit for ALL quants (+2-4% over stock GGUF), MTP tensors at Q8_0.
Tool-calling floor: >=Q4_K_M (author), Q6 recommended.

## Serving recipe (EXACT, verified by Q38 workbench + Jakaryun harness)
llama-server -c 262144 (1M via --override-kv qwen35.context_length=int:1048576 + yarn scale 4) \
  -np 1 -fa on --jinja --reasoning-format deepseek --temp 0.7 --top-k 64 --min-p 0.05 \
  --top-p 0.95 --repeat-penalty 1.0 [--spec-type draft-mtp --spec-draft-n-max 2 for MTP builds]
reasoning_effort xhigh/medium/low via chat_template_kwargs; enable_thinking false = instruct mode.
Sampling validated A/B on 276-case tool harness: 0.7/64/0.95 >= card 1.0/20/0.95.

## Independent validation (EXACT, third party)
Jakaryun, 276 tasks / 320 live tools, Q8_0, RTX PRO 6000 96GB:
TURBO 254-262 (record 262 MTP) vs stock Qwen3.8 244/244; tune premium +14 cases;
near-miss 4-5 (Fable line only; others 7-19). Discussion #36 on the GGUF repo.
Our Quick12: Fable 0.791 / Tmax 0.774 / base 0.690 / champion-B 0.618 — same ranking.

## What is UNRECOVERABLE without the author
Unsloth LR/schedule/LoRA config; merge YAML ratios/order; GAIN code; NEO imatrix
calibration set; Polar/F451/Reasoning-STRICT contents (gated). Route: Discord or
HF discussion + "Buy Me a Coffee" commission channel (per support page).
