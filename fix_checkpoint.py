#!/usr/bin/env python3
"""
fix_checkpoint.py — one-time migration.

Models trained on older `transformers` save DeBERTa LayerNorm parameters as
`LayerNorm.gamma` / `LayerNorm.beta`. transformers >= 5.0 removed the legacy
auto-rename, so those 28 layers silently load as random values and the model
outputs ~0.5 for every ticket. This script renames the keys in place so the
checkpoint loads correctly in ANY transformers version.

Usage: python fix_checkpoint.py --model models/deberta_final
"""

import argparse
import os
import shutil
from safetensors.torch import load_file, save_file


def fix(model_dir):
    path = os.path.join(model_dir, "model.safetensors")
    backup = path + ".bak"
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # Back up first, then read from the backup so the mmap is NOT on the file
    # we are about to overwrite (Windows forbids overwriting a mapped file).
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"Backup written -> {backup}")

    state = load_file(backup)
    renamed = {}
    n = 0
    for k, v in state.items():
        nk = k
        if k.endswith("LayerNorm.gamma"):
            nk = k[: -len("gamma")] + "weight"
        elif k.endswith("LayerNorm.beta"):
            nk = k[: -len("beta")] + "bias"
        if nk != k:
            n += 1
        renamed[nk] = v

    if n == 0:
        print("No .gamma/.beta LayerNorm keys found — checkpoint already clean.")
        return

    save_file(renamed, path, metadata={"format": "pt"})
    print(f"Renamed {n} LayerNorm keys (.gamma->.weight, .beta->.bias) -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/deberta_final")
    fix(parser.parse_args().model)
