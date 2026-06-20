"""
Turn the raw CREMA-D wavs into mel-spectrogram features and split them into
five IID client partitions (one per device).

This follows the paper's data setup (Section IV, "Dataset"):
  - keep the four most frequent emotions: Neutral, Happy, Angry, Sad
  - ~5882 clips after dropping Fear and Disgust
  - five IID partitions, each with an 80/20 train/test split

Unlike an earlier version of this script we do NOT add augmentation or client
overlap here - the paper uses clean IID partitions so that the only thing
varying between clients is the hardware, not the data.

Usage:
    python prepare_data.py --data_path ./CREMA-D --out_dir ./client_data
"""

import os
import argparse
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
DURATION = 2.5
AUDIO_LENGTH = int(SAMPLE_RATE * DURATION)

# CREMA-D filename code -> label index. Sorted so the mapping is deterministic.
EMOTION_TO_LABEL = {"ANG": 0, "HAP": 1, "NEU": 2, "SAD": 3}


def load_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    if len(audio) > AUDIO_LENGTH:
        audio = audio[:AUDIO_LENGTH]
    else:
        audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)), "constant")
    return audio


def mel_spectrogram(audio):
    """STFT -> mel filter bank -> log scale. Implements Eq. (3) in the paper."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    return librosa.power_to_db(mel, ref=np.max)


def load_dataset(data_path):
    features, labels = [], []
    for subdir, _, files in os.walk(data_path):
        for fname in files:
            if not fname.lower().endswith(".wav"):
                continue
            # CREMA-D names look like 1001_DFA_ANG_XX.wav -> emotion is field [2]
            emotion = fname.split("_")[2]
            if emotion not in EMOTION_TO_LABEL:
                continue
            audio = load_audio(os.path.join(subdir, fname))
            features.append(mel_spectrogram(audio))
            labels.append(EMOTION_TO_LABEL[emotion])

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # standardize per mel-bin across the whole set
    scaler = StandardScaler()
    flat = features.reshape(-1, features.shape[-1])
    features = scaler.fit_transform(flat).reshape(features.shape)
    return features, labels


def split_iid(features, labels, num_clients=5, seed=42):
    """Shuffle once, then hand each client an equal, class-balanced-ish slice."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(features))
    shards = np.array_split(idx, num_clients)
    return [(features[s], labels[s]) for s in shards]


def save_partitions(partitions, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, (x, y) in enumerate(partitions, start=1):
        # keep an explicit per-client train/test split (80/20) like the paper
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )
        np.save(os.path.join(out_dir, f"client_{i}_x_train.npy"), x_tr)
        np.save(os.path.join(out_dir, f"client_{i}_y_train.npy"), y_tr)
        np.save(os.path.join(out_dir, f"client_{i}_x_test.npy"), x_te)
        np.save(os.path.join(out_dir, f"client_{i}_y_test.npy"), y_te)
        print(f"client {i}: {len(x_tr)} train / {len(x_te)} test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="CREMA-D")
    parser.add_argument("--out_dir", default="client_data")
    parser.add_argument("--num_clients", type=int, default=5)
    args = parser.parse_args()

    feats, labs = load_dataset(args.data_path)
    print(f"loaded {len(feats)} clips, feature shape {feats.shape[1:]}")
    parts = split_iid(feats, labs, num_clients=args.num_clients)
    save_partitions(parts, args.out_dir)
