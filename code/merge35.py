#!/usr/bin/env python3
"""Streaming shard-wise PEFT merge for Qwen3.5-35B-A3B (31GB RAM box).

Merges adapter deltas into /root/q35-hf safetensors IN PLACE (shard at a time,
temp file per shard). Emits delta-magnitude report for the R2 gate:
if deltas are below the IQ4_XS quant step the re-quant can swallow the fine-tune.
"""
import json, os, sys, glob, shutil, tempfile
import torch
from safetensors.torch import load_file, save_file

ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "/root/staircase35/a2"
BASEDIR = sys.argv[2] if len(sys.argv) > 2 else "/root/q35-hf"
REPORT  = sys.argv[3] if len(sys.argv) > 3 else "/root/merge_report.json"

cfg = json.load(open(f"{ADAPTER}/adapter_config.json"))
scale = cfg.get("lora_alpha", 32) / cfg.get("r", 16)
if cfg.get("use_rslora"):
    scale = cfg.get("lora_alpha", 32) / (cfg.get("r", 16) ** 0.5)
print(f"[merge] scale={scale} rslora={cfg.get('use_rslora')}")

adt = load_file(f"{ADAPTER}/adapter_model.safetensors")
pairs = {}
for k in adt:
    if ".lora_A.weight" in k:
        base_key = k.replace(".lora_A.weight", ".weight")
        if base_key.startswith("base_model.model."):
            base_key = base_key[len("base_model.model."):]  # FIX: strip PEFT prefix from the VALUE, not the dict key
        b_key = k.replace(".lora_A.weight", ".lora_B.weight")
        mod = base_key.rsplit(".", 1)[0]           # ...q_proj
        pairs[mod] = (base_key, adt[k], adt[b_key])
print(f"[merge] {len(pairs)} adapted modules")
assert len(pairs) > 50, "Fatal: adapter has too few modules"

# module-name containment map: HF shard keys are like model.layers.N.self_attn.q_proj.weight
# PEFT keys are base_model.model.model.layers... -> strip prefix
def norm(mod):
    return mod.replace("base_model.model.", "") if mod.startswith("base_model.model.") else mod

pairs = {norm(m): v for m, v in pairs.items()}

rep = {"tensors": 0, "merged": 0, "max_abs": 0.0, "sum_abs": 0.0, "n_elem": 0, "per_mod": {}}
os.makedirs(f"{BASEDIR}/merge_tmp", exist_ok=True)

for shard in sorted(glob.glob(f"{BASEDIR}/model*-*.safetensors")):
    tensors = load_file(shard)
    changed = set()
    for mod, (bkey, A, B) in pairs.items():
        if bkey in tensors:
            W = tensors[bkey].to(torch.float32)
            delta = (B.to(torch.float32) @ A.to(torch.float32)) * scale
            dmax = delta.abs().max().item()
            dmean = delta.abs().mean().item()
            rep["tensors"] += 1
            rep["max_abs"] = max(rep["max_abs"], dmax)
            rep["sum_abs"] += delta.abs().sum().item()
            rep["n_elem"] += delta.numel()
            rep["per_mod"][mod.split(".layers.")[-1] if ".layers." in mod else mod] = {
                "max": dmax, "mean": dmean}
            tensors[bkey] = (W + delta).to(tensors[bkey].dtype)
            changed.add(bkey)
            del W, delta
            rep["merged"] += 1
    tmp = f"{BASEDIR}/merge_tmp/{os.path.basename(shard)}"
    save_file(tensors, tmp, metadata={"format": "pt"})
    del tensors
    os.replace(tmp, shard)
    print(f"[merge] {os.path.basename(shard)}: +{len(changed)} merged", flush=True)

# copy any remaining tensors shards not matched by glob (e.g., model.safetensors)
for extra in glob.glob(f"{BASEDIR}/*.safetensors"):
    if "/merge_tmp/" in extra:
        continue
    if os.path.basename(extra) in {os.path.basename(s) for s in glob.glob(f"{BASEDIR}/model*-*.safetensors")}:
        continue
    pass  # nothing to do; all shards match the model*-*.safetensors glob in this repo

os.rmdir(f"{BASEDIR}/merge_tmp")
rep["mean_abs"] = rep["sum_abs"] / max(1, rep["n_elem"])
json.dump(rep, open(REPORT, "w"), indent=1)
print(f"[gate] merged {rep['merged']}/{len(pairs)} modules | max|d|={rep['max_abs']:.5f} mean|d|={rep['mean_abs']:.6f}")
print(f"[gate] IQ4_XS step ~0.017-0.03 rel; delta/max ratios above ~0.25x step survive re-quant")
print("MERGE_DONE")
