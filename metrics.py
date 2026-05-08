"""
metrics.py
Quantitative evaluation metrics for symbolic music generation.

All functions accept binary piano-rolls of shape (128, T).
"""

import numpy as np


# ── Pitch Histogram Similarity ────────────────────────────────────────────

def pitch_histogram(pr: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """
    Compute a normalized 128-bin pitch histogram from a piano-roll.
    Each bin counts how many time steps that pitch is active.
    """
    hist = (pr > thresh).sum(axis=1).astype(float)
    total = hist.sum()
    return hist / total if total > 0 else hist


def pitch_histogram_similarity(orig_pr: np.ndarray,
                                gen_pr:  np.ndarray,
                                thresh:  float = 0.5) -> float:
    """
    H(p, q) = sum_i |p_i - q_i|
    L1 distance between pitch histograms.  Lower is more similar.
    Range: [0, 2].
    """
    p = pitch_histogram(orig_pr, thresh)
    q = pitch_histogram(gen_pr,  thresh)
    return float(np.sum(np.abs(p - q)))


# ── Rhythm Diversity ──────────────────────────────────────────────────────

def rhythm_diversity(pr: np.ndarray, thresh: float = 0.5) -> float:
    """
    D_rhythm = #unique_durations / #total_notes
    Measures how varied the note lengths are.  Higher = more diverse.
    """
    durations = []
    for pitch in range(pr.shape[0]):
        active  = pr[pitch, :] > thresh
        diff    = np.diff(active.astype(int), prepend=0, append=0)
        starts  = np.where(diff ==  1)[0]
        ends    = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            durations.append(e - s)
    if not durations:
        return 0.0
    return len(set(durations)) / len(durations)


# ── Repetition Ratio ──────────────────────────────────────────────────────

def repetition_ratio(pr: np.ndarray,
                     thresh: float = 0.5,
                     n: int = 4) -> float:
    """
    R = #repeated_patterns / #total_patterns
    Extracts length-n pitch n-grams and counts how many appear more than once.
    Lower = less repetitive.
    """
    pitches = []
    for t in range(pr.shape[1]):
        active = np.where(pr[:, t] > thresh)[0]
        if len(active):
            pitches.append(int(active[0]))

    if len(pitches) < n + 1:
        return 0.0

    ngrams   = [tuple(pitches[i: i + n]) for i in range(len(pitches) - n)]
    repeated = sum(1 for g in ngrams if ngrams.count(g) > 1)
    return repeated / len(ngrams) if ngrams else 0.0


# ── Perplexity (Task 3) ───────────────────────────────────────────────────

def perplexity_from_loss(avg_cross_entropy_loss: float) -> float:
    """
    Perplexity = exp(L_TR / T)
    Pass in the average BCE loss per token.
    """
    import math
    return math.exp(avg_cross_entropy_loss)