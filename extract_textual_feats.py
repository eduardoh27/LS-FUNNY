import os
import json
import torch
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
from zoneinfo import ZoneInfo

BOGOTA = ZoneInfo("America/Bogota")
def log(msg: str):
    ts = datetime.now(BOGOTA).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

def init_model(model_name: str = "xlm-roberta-base"):
    log(f"Loading tokenizer and model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    log("Model ready.")
    return tokenizer, model

def extract_embedding(tokenizer, model, context: str, punchline: str, max_length: int = 128):
    encoded = tokenizer(
        context,
        punchline,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**encoded)
    # Representation of the <s> token at position 0
    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

def main():
    meta_path = "dataset/dataset.jsonl"
    output_dir = "features/text"
    os.makedirs(output_dir, exist_ok=True)

    tokenizer, model = init_model()
    total = sum(1 for _ in open(meta_path, 'r', encoding='utf-8'))
    log(f">>> Reading {total} lines from '{meta_path}'")

    with open(meta_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, start=1):
            entry = json.loads(line)
            vid       = entry.get("video_id")
            inst      = entry.get("instance_id")
            context   = entry.get("context", "")
            punchline = entry.get("punchline", "")

            log(f"[{idx}/{total}] Processing video={vid} inst={inst}")
            vec = extract_embedding(tokenizer, model, context, punchline)
            log(f"    → Embedding shape: {vec.shape}")

            fn = f"{vid}_{inst}.npy"
            out_path = os.path.join(output_dir, fn)
            # Save array as binary float32
            vec.astype('float32').tofile(out_path)
            log(f"    → Saved to: {out_path}")

    log(">>> Text extraction completed.")

if __name__ == "__main__":
    main()
