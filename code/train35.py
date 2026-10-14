#!/usr/bin/env python3
"""Staircase trainer: A1(1024) -> A2(2048) -> B(16384) on Qwen3.8-27B LoRA.

Same gates and row-builder as the manifest train.py (§8.2, with the four applied
review fixes). New: PHASES loop; A2/B continue from the previous phase's adapter.
Run on the L40S only (48GB). Effective batch = 8 in every phase.
"""
import os, json
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"          # correct PyTorch spelling (verified vs torch 2.13.0 source)
from collections import defaultdict
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

MODEL_NAME = os.environ.get("BASE_MODEL", "unsloth/Qwen3.8-27B-unsloth-bnb-4bit")
OUTDIR     = os.environ.get("OUT_DIR", "/root/staircase")
START      = os.environ.get("START_PHASE", "a1")   # skip completed phases; init = prev phase output
MAX_SEQ_LEN_DATA = 16384
BASH_CAP   = 800
HEADER     = "assistant\n"
END        = "<|im_end|>"
CANON = {"bash": "Bash", "read": "Read", "write": "Write", "edit": "Edit",
         "glob": "Glob", "grep": "Grep", "webfetch": "WebFetch",
         "websearch": "WebSearch", "todowrite": "TodoWrite", "task": "Task"}

# per-phase constants — A1/A2 locked by token measurement (zero truncation),
# B subject to the P6 memory/throughput probes on the box
PHASES = [
    dict(name="a1", raw="/root/sft_mid_a1.jsonl",      max_len=1024,  bs=4, ga=2, lr=1.5e-4, epochs=2, warmup=25, init=None),
    dict(name="a2", raw="/root/sft_mid_a2.jsonl",      max_len=2048,  bs=2, ga=4, lr=1.0e-4, epochs=2, warmup=25, init=f"{OUTDIR}/a1"),
]


def clean(t: str) -> str:
    return t.replace("\u200b", "").replace("\ufeff", "")


def build_rows(tok, raw, max_len, im_end):
    counts, rows, n_trunc, n_bad = defaultdict(int), [], 0, 0
    for s in (json.loads(l) for l in open(raw)):
        raw_tool = clean(s["target"]["tool"].strip())
        tool = CANON.get(raw_tool.lower(), raw_tool)
        if tool == "Bash" and counts["Bash"] >= BASH_CAP:
            continue
        ctx = clean(s["context"])
        assert "\u200b" not in ctx, "Fatal: ZWS survived cleaning"

        raw_tmpl = tok.apply_chat_template(
            [{"role": "user", "content": ctx}],
            tokenize=False, add_generation_prompt=True)
        j = raw_tmpl.rfind(HEADER)
        assert j != -1, "Fatal: assistant header not found"
        prompt_str = raw_tmpl[:j + len(HEADER)]
        assert prompt_str.endswith("<|im_start|>" + HEADER), \
            f"Fatal: bad junction, tail={prompt_str[-80:]!r}"

        tj   = json.dumps({"tool": tool, "input": s["target"]["input"]},
                          ensure_ascii=False)
        comp = tok.encode(f"<tool_call>\n{tj}\n</tool_call>", add_special_tokens=False) + [im_end]

        pid = tok.encode(prompt_str, add_special_tokens=False)
        if not tok.decode(pid + comp).startswith(prompt_str):
            n_bad += 1   # pathological tokenizer junction: skip the row, keep the run
            continue

        hdr = len(tok.encode(HEADER, add_special_tokens=False))
        budget = max_len - len(comp)
        if budget <= hdr:
            continue
        if len(pid) > budget:
            pid = pid[-budget:]
            n_trunc += 1

        rows.append({"input_ids": pid + comp,
                     "labels": [-100] * len(pid) + comp,
                     "attention_mask": [1] * (len(pid) + len(comp))})
        counts[tool] += 1
    return rows, counts, n_trunc, n_bad


def main():
    model = tok = None
    started = False
    prev_dir = None
    for ph in PHASES:
        if ph["name"] == START:
            started = True
        if not started:
            prev_dir = f"{OUTDIR}/{ph['name']}"
            continue
        if prev_dir and ph["init"] is None:
            ph = dict(ph, init=prev_dir)   # resuming mid-staircase: continue from prior phase
        prev_dir = f"{OUTDIR}/{ph['name']}"
        print(f"\n########## PHASE {ph['name']} ##########", flush=True)
        if ph["init"] is None:
            model, tok = FastLanguageModel.from_pretrained(
                model_name=MODEL_NAME, max_seq_length=ph["max_len"],
                dtype=torch.bfloat16, load_in_4bit=True)
            # this repo ships a VL processor wrapper — unwrap to the raw tokenizer
            if not hasattr(tok, "encode") and hasattr(tok, "tokenizer"):
                tok = tok.tokenizer
            assert hasattr(tok, "encode"), f"Fatal: no tokenizer (got {type(tok).__name__})"
            ids = tok.encode(END, add_special_tokens=False)
            assert len(ids) == 1, f"Fatal: {END} encoded to {ids}"
            im_end = ids[0]
            assert tok.decode([im_end]).strip() == END, "Fatal: terminator round-trip failed"
            if tok.pad_token_id is None:
                tok.pad_token = "<|endoftext|>"
            assert tok.pad_token_id is not None, "Fatal: no pad token"
            print(f"[gate] terminator id={im_end}")
            EXCLUDE = ("visual", "vision", "merger", "embed", "lm_head", "gate_proj", "up_proj", "down_proj")  # DD9: expert-FFN LoRA = 945M trainable OOM; attn+DeltaNet kept
            targets = sorted({n.split(".")[-1] for n, m in model.named_modules()
                              if isinstance(m, torch.nn.Linear)
                              and not any(x in n for x in EXCLUDE)})
            print(f"[gate] {len(targets)} modules: {targets}")
            assert len(targets) >= 10, "Fatal: module set too small"
            assert {"in_proj_qkv", "out_proj", "q_proj"} <= set(targets), f"Fatal: DeltaNet/attn core missing: {targets}"  # DD9 gate
            model = FastLanguageModel.get_peft_model(
                model, r=16, lora_alpha=32, lora_dropout=0.0, target_modules=targets,
                bias="none", use_gradient_checkpointing="unsloth", random_state=42)
        else:
            # P7 SMOKE GATE: the first time this branch runs, run 5 steps and
            # confirm finite loss before the phase is left unattended.
            model, tok = FastLanguageModel.from_pretrained(
                model_name=ph["init"], max_seq_length=ph["max_len"],
                dtype=torch.bfloat16, load_in_4bit=True)
            if not hasattr(tok, "encode") and hasattr(tok, "tokenizer"):
                tok = tok.tokenizer
            assert hasattr(tok, "encode"), f"Fatal: no tokenizer (got {type(tok).__name__})"
            ids = tok.encode(END, add_special_tokens=False)
            assert len(ids) == 1, f"Fatal: {END} encoded to {ids}"
            im_end = ids[0]

        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[gate] trainable {n_tr/1e6:.1f}M")
        assert n_tr > 8e6, "Fatal: adapter not trainable in this phase"  # DD9: ~20M expected

        rows, counts, n_trunc, n_bad = build_rows(tok, ph["raw"], ph["max_len"], im_end)
        print(f"[data] {ph['name']}: {len(rows)} rows, {n_trunc} left-truncated, {n_bad} bad-boundary skipped")
        assert len(rows) > 800, f"Fatal: phase {ph['name']} collapsed to {len(rows)}"
        for k, v in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"   {k}: {v}")
        r0 = rows[0]
        print("--- SUPERVISED SPAN (must be exactly the tool call + terminator) ---")
        print(tok.decode([i for i, l in zip(r0["input_ids"], r0["labels"]) if l != -100]))

        out = f"{OUTDIR}/{ph['name']}"
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=out, per_device_train_batch_size=ph["bs"],
                gradient_accumulation_steps=ph["ga"],
                learning_rate=ph["lr"], lr_scheduler_type="cosine",
                warmup_steps=ph["warmup"], weight_decay=0.01,
                num_train_epochs=ph["epochs"], bf16=True, logging_steps=10,
                save_strategy="steps", save_steps=25, save_total_limit=3,
                report_to="none", seed=42, dataloader_num_workers=int(os.environ.get('DL_WORKERS', '0'))),
            train_dataset=Dataset.from_list(rows),
            data_collator=DataCollatorForSeq2Seq(tok, padding=True))
        trainer.train()
        model.save_pretrained(out)
        tok.save_pretrained(out)
        print(f"PHASE_{ph['name']}_DONE", flush=True)
    print("STAIRCASE_DONE")


if __name__ == "__main__":
    main()
