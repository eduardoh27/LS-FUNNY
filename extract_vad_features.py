import os
import re
import json
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

BOGOTA = ZoneInfo("America/Bogota")
def log(msg: str):
    ts = datetime.now(BOGOTA).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

def load_vad_lexicon(lex_path: str):
    """Loads the Spanish VAD lexicon → (valence, arousal, dominance)."""
    vad = {}
    with open(lex_path, encoding="utf-8") as f:
        header = next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            # English Word = parts[0], Valence=parts[1], Arousal=parts[2], Dominance=parts[3], Spanish=parts[4]
            try:
                v, a, d = map(float, parts[1:4])
            except ValueError:
                continue
            spanish = parts[4].lower()
            vad[spanish] = (v, a, d)
    return vad

def topk_stats(values, k=3):
    """Returns top k max, top k min, mean and std of a list of floats."""
    if not values:
        return [0.0]*k, [0.0]*k, 0.0, 0.0
    arr = np.array(values)
    asc = np.sort(arr)
    desc = asc[::-1]
    maxk = desc[:k].tolist() if len(arr) >= k else desc.tolist() + [desc[-1]]*(k-len(arr))
    mink = asc[:k].tolist() if len(arr) >= k else asc.tolist() + [asc[0]]*(k-len(arr))
    return maxk, mink, float(arr.mean()), float(arr.std())

def extract_vad_features(text: str, vad_lex):
    # simple tokenization
    tokens = re.findall(r"\b\w+\b", text.lower())
    vals, aras, doms = [], [], []
    for t in tokens:
        if t in vad_lex:
            v, a, d = vad_lex[t]
            vals.append(v); aras.append(a); doms.append(d)
    # compute stats per dimension
    v_max, v_min, v_mean, v_std = topk_stats(vals)
    a_max, a_min, a_mean, a_std = topk_stats(aras)
    d_max, d_min, d_mean, d_std = topk_stats(doms)
    # concatenate in order: max3, min3, mean, std for V, A, D
    features = (
        v_max + v_min + [v_mean, v_std] +
        a_max + a_min + [a_mean, a_std] +
        d_max + d_min + [d_mean, d_std]
    )
    return np.array(features, dtype=np.float32)

def main():
    lex_path     = "Spanish-NRC-VAD-Lexicon.txt"
    meta_path    = "dataset/dataset.jsonl"
    output_dir   = "features/VAD"
    os.makedirs(output_dir, exist_ok=True)

    log("Loading VAD lexicon...")
    vad_lex = load_vad_lexicon(lex_path)
    log(f"  → {len(vad_lex)} entries loaded.")

    total = sum(1 for _ in open(meta_path, 'r', encoding='utf-8'))
    log(f"Reading {total} instances from '{meta_path}'")

    with open(meta_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, start=1):
            entry = json.loads(line)
            vid       = entry["video_id"]
            inst      = entry["instance_id"]
            context   = entry.get("context", "")
            punchline = entry.get("punchline", "")
            text = f"{context} {punchline}"

            log(f"[{idx}/{total}] Processing video={vid} inst={inst}")
            feats = extract_vad_features(text, vad_lex)
            log(f"    → VAD feature shape: {feats.shape}")

            fn = f"{vid}_{inst}.npy"
            out_path = os.path.join(output_dir, fn)
            np.save(out_path, feats)
            log(f"    → Saved to: {out_path}")

    log("VAD feature extraction completed.")

if __name__ == "__main__":
    main()
