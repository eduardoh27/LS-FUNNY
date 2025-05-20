import os
import librosa
import numpy as np
from datetime import datetime

def get_librosa_features(path: str, n_bins=10) -> np.ndarray:
    y, sr = librosa.load(path, sr=None)  # Load the audio file with its original sample rate : 441000 samples / 44100 Hz    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Compute MFCC features from the raw signal
    mfcc_delta = librosa.feature.delta(mfcc)  # First-order differences (delta features)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_delta = librosa.feature.delta(S)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    audio_feature = np.vstack((mfcc, mfcc_delta, S, S_delta, spectral_centroid))  # Combine all features
    # Binning data
    idx = np.linspace(0, audio_feature.shape[1] - 1, num=n_bins, dtype=int)
    binned_feature = audio_feature[:, idx]
    return binned_feature

def main():
    input_dir = 'dataset/audios'
    output_dir = 'features/audios'
    os.makedirs(output_dir, exist_ok=True)

    audio_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith('.wav'))
    print(f">>> Found {len(audio_files)} files in {input_dir}")

    for idx, fname in enumerate(audio_files, start=1):
        audio_path = os.path.join(input_dir, fname)
        print(f"[{idx}/{len(audio_files)}] Processing: {fname}")
        feats = get_librosa_features(audio_path)
        out_name = os.path.splitext(fname)[0] + '.npy'
        out_path = os.path.join(output_dir, out_name)
        np.save(out_path, feats)
        print(f"    → Saved to: {out_path}")

    print(">>> Extraction completed.")

if __name__ == '__main__':
    main()
