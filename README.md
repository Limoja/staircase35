# staircase35: Post-training a 35B MoE for terminal agentic work on one 24GB GPU

Qwen3.5-35B-A3B (MoE, 3B active) post-trained in a 3-phase SFT "staircase" to emit
the next tool action directly at long context (262k). All phases trained QLoRA on a
single RTX 3090 24GB with unsloth. Served via llama.cpp IQ4_XS.

## Headline result (internal 381-scenario terminal next-action bench, best-of-4)

| Model | Best-of-4 | Tool-name acc | Args F1 | Valid |
|---|---|---|---|---|
| A1-only (mid-training dose 1) | 0.5619 | 0.291 | 0.294 | 0.931 |
| A2-only (mid-training doses 1+2) | 0.5762 | 0.320 | 0.314 | 0.932 |
| Base (no post-training) | 0.5915 | 0.344 | 0.331 | 0.951 |
| **Full staircase (A1→A2→B)** | **0.6815** | **0.664** | **0.436** | 0.872 |

**Key finding:** mid-training doses alone damage every healthy base they touch
(monotonic regression, replicated on two model versions). The recipe's gain arrives
only when the full-context direct-emission phase (B) *completes* the staircase:
tool-name selection doubles (+0.32) at the cost of a small validity dip, netting
+9.0 points over base.

## Why this works (mechanism)

- Phases A1/A2 are low-context next-action prediction. They *overfit short-context
  priors* and corrupt the base's tool-calling behaviour (name accuracy drops 0.34→0.29).
- Phase B re-trains the same decision function at full pretraining-length context
  (windows up to 57k chars / 8k tokens), restoring the long-context junction the
  A-phases destroyed. The result is a model that commits to direct tool emission
  at 262k context — something neither the base nor any A-dose does.

See `docs/paper.md` for the full write-up and `EXPERIMENTS.md` for the running
experiment ledger.

## Layout

- `code/` — dataset builders, QLoRA trainer, merge + GGUF quant pipeline, eval harness
- `results/` — evaluation summaries + decomposition metrics
- `docs/paper.md` — method + results paper
- `EXPERIMENTS.md` — append-only experiment ledger (every experiment fully described)

## Bench

The internal bench is 381 scenarios harvested from real terminal-agent sessions
(shell + file-edit + planning decisions), frozen and leak-checked against all
training corpora. Scenario data itself is private; the builders + scorer are here.

## Serving

llama.cpp, IQ4_XS quant (18.9GB), ctx 262144, `-np 2 -fa on`, single 3090.
