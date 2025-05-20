import os
import re
import subprocess
import random
import json
from typing import List, Dict

VIDEO_IDS_PATH = os.path.join("raw_data", "video_ids.txt")
TRANSCRIPTS_DIR = os.path.join("raw_data", "transcripts")
VIDEOS_DIR       = os.path.join("raw_data", "videos")
AUDIOS_DIR       = os.path.join("raw_data", "audio")
OUTPUT_DIR       = "dataset"
METADATA_PATH    = os.path.join(OUTPUT_DIR, "dataset.jsonl")

for sub_dir in ("videos", "audios"):
    os.makedirs(os.path.join(OUTPUT_DIR, sub_dir), exist_ok=True)

# Simple VTT parser
def parse_vtt(path: str) -> List[Dict]:
    cues = []
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    # Skip the WEBVTT header
    assert lines[0].startswith("WEBVTT")
    i = 1
    # Read cue blocks
    while i < len(lines):
        # <timestamp> --> <timestamp>
        match = re.match(r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})", lines[i])
        if match:
            start, end = match.groups()
            i += 1
            text_lines = []
            # Read until next blank line or timestamp
            while i < len(lines) and "-->" not in lines[i]:
                text_lines.append(lines[i])
                i += 1
            cues.append({
                "start": start,
                "end": end,
                "text": " ".join(text_lines)
            })
        else:
            i += 1
    return cues

# Remove initial credits
def drop_credits(cues: List[Dict]) -> List[Dict]:
    pattern = re.compile(r"(Traductor|Revisor|Transcriptor):", re.IGNORECASE)
    # Find first cue without credits
    for idx, cue in enumerate(cues):
        if not pattern.search(cue["text"]):
            return cues[idx:]
    return []

# Extract positive instances (context + punchline)
def extract_positives(cues: List[Dict]) -> List[Dict]:
    positives = []
    laugh_indices = [i for i, c in enumerate(cues) if "(Risas)" in c["text"]]
    prev_laugh = None
    for laugh_idx in laugh_indices:
        punch_idx = laugh_idx - 1
        if punch_idx < 0:
            continue
        start_ctx = prev_laugh + 1 if prev_laugh is not None else 0
        context_cues = cues[start_ctx:punch_idx]
        # Join texts
        context_text = " ".join(c["text"] for c in context_cues).strip()
        punchline_text = cues[punch_idx]["text"].replace("(Risas)", "").strip()
        start_time = context_cues[0]["start"] if context_cues else cues[punch_idx]["start"]
        end_time = cues[punch_idx]["end"]
        positives.append({
            "type": "pos",
            "context": context_text,
            "punchline": punchline_text,
            "start": start_time,
            "end": end_time
        })
        prev_laugh = laugh_idx
    return positives

# Extract balanced negative instances
def extract_negatives(cues: List[Dict], n: int) -> List[Dict]:
    negatives = []
    laugh_set = {i for i, c in enumerate(cues) if "(Risas)" in c["text"]}
    attempts = 0
    while len(negatives) < n and attempts < n * 10:
        i = random.randint(1, len(cues) - 3)
        # Skip if this or next two cues contain laughter
        if any(idx in laugh_set for idx in (i, i + 1, i + 2)):
            attempts += 1
            continue
        # Define context since last laughter
        prev_laugh = max([k for k in laugh_set if k < i], default=-1)
        context_cues = cues[prev_laugh + 1:i]
        # Discard if context contains laughter
        if any("(Risas)" in c["text"] for c in context_cues):
            attempts += 1
            continue
        context_text = " ".join(c["text"] for c in context_cues).strip()
        punchline_text = cues[i]["text"].strip()
        start_time = context_cues[0]["start"] if context_cues else cues[i]["start"]
        end_time = cues[i]["end"]
        negatives.append({
            "type": "neg",
            "context": context_text,
            "punchline": punchline_text,
            "start": start_time,
            "end": end_time
        })
        attempts += 1
    return negatives

# FFmpeg function to cut segments
def cut_segment(input_path: str, output_path: str, start: str, end: str):
    cmd = [
        "ffmpeg", "-y",
        "-ss", start,
        "-to", end,
        "-i", input_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-strict", "experimental",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


with open(METADATA_PATH, "w", encoding="utf-8") as meta_file:
    for line in open(VIDEO_IDS_PATH):
        video_id = line.strip()
        cues = parse_vtt(os.path.join(TRANSCRIPTS_DIR, f"{video_id}.vtt"))
        cues = drop_credits(cues)
        positives = extract_positives(cues)
        negatives = extract_negatives(cues, len(positives))
        all_instances = positives + negatives

        for idx, inst in enumerate(all_instances):
            label = inst["type"]
            base_name = f"{video_id}_{label}_{idx}"
            video_output = os.path.join(OUTPUT_DIR, "videos", base_name + ".mp4")
            audio_output = os.path.join(OUTPUT_DIR, "audios", base_name + ".wav")

            # Cut video and audio
            cut_segment(
                os.path.join(VIDEOS_DIR, f"{video_id}.mp4"),
                video_output,
                inst["start"],
                inst["end"]
            )
            cut_segment(
                os.path.join(AUDIOS_DIR, f"{video_id}.wav"),
                audio_output,
                inst["start"],
                inst["end"]
            )

            # Write metadata record
            record = {
                "video_id":    video_id,
                "instance_id": idx,
                "label":       "humor" if label == "pos" else "no_humor",
                "context":     inst["context"],
                "punchline":   inst["punchline"],
                "video_path":  video_output,
                "audio_path":  audio_output,
                "start":       inst["start"],
                "end":         inst["end"]
            }
            meta_file.write(json.dumps(record, ensure_ascii=False) + "\n")
