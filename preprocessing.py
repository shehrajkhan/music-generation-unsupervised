"""
preprocessing.py
Utilities for loading MIDI files and converting them to piano-roll tensors.
"""

import os
import numpy as np
import pretty_midi
import pandas as pd
import torch
from config import CSV_PATH, BASE_FOLDER, SAMPLE_RATE, SEQ_LENGTH


def load_midi(midi_path: str, fs: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load a MIDI file and return a binary piano-roll of shape (128, T).
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    pr = midi_data.get_piano_roll(fs=fs)
    return (pr > 0).astype(np.float32)


def create_sequences(piano_roll: np.ndarray,
                     seq_length: int = SEQ_LENGTH):
    """
    Slide a window over a piano-roll to produce (input, target) pairs.

    Returns
    -------
    X : np.ndarray  shape (N, 128, seq_length)
    y : np.ndarray  shape (N, 128)
    """
    inputs, targets = [], []
    for i in range(piano_roll.shape[1] - seq_length):
        inputs.append(piano_roll[:, i: i + seq_length])
        targets.append(piano_roll[:, i + seq_length])
    return np.array(inputs), np.array(targets)


def to_tensors(X: np.ndarray, y: np.ndarray, device: torch.device):
    """
    Convert numpy arrays to PyTorch tensors and move to device.
    X is transposed to (N, seq_length, 128) for LSTM/Transformer input.
    """
    X_t = torch.FloatTensor(X).transpose(1, 2).to(device)
    y_t = torch.FloatTensor(y).to(device)
    return X_t, y_t


def load_first_midi(n_steps: int = 500):
    """
    Convenience function used by all notebooks:
    loads the first MIDI in MAESTRO and returns a truncated piano-roll.
    """
    metadata   = pd.read_csv(CSV_PATH)
    midi_path  = os.path.join(BASE_FOLDER, metadata['midi_filename'].iloc[0])
    piano_roll = load_midi(midi_path)
    return piano_roll[:, :n_steps]