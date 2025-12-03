import numpy as np
from typing import Sequence, Optional, Tuple

class RewardNormalizer:
    """
    Tracks running statistics (mean, std) for reward normalization.
    Uses Welford's online algorithm for numerical stability.
    """
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squared differences from mean

    def update(self, values: np.ndarray):
        """Update running statistics with new batch of values."""
        for value in values.flatten():
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            delta2 = value - self.mean
            self.m2 += delta * delta2

    @property
    def std(self) -> float:
        """Return standard deviation."""
        if self.count < 2:
            return 1.0
        variance = self.m2 / self.count
        return np.sqrt(variance) + self.epsilon

    def normalize(self, values: np.ndarray, clip: Optional[float] = None) -> np.ndarray:
        """
        Z-score normalization: (value - mean) / std
        Optionally clip to [-clip, +clip] to handle outliers.
        """
        normalized = (values - self.mean) / self.std
        if clip is not None:
            normalized = np.clip(normalized, -clip, clip)
        return normalized


def method2_loss(
    de_new: Sequence[float],
    de_old: Sequence[float],
    constraint: Sequence[float],
    verify: Sequence[float],
    x: float = 1.0,
    y: float = 0.3,
    z: float = 0.3,
    f: str = "relu",
    normalizers: Optional[Tuple[RewardNormalizer, RewardNormalizer, RewardNormalizer]] = None,
    normalize_rewards: bool = False,
    clip_normalized: Optional[float] = None,
    constraint_threshold: float = 0.5,
    constraint_sharpness: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute combined reward from difficulty delta, verifier, and constraint scores.

    Args:
        de_new: New difficulty scores
        de_old: Original difficulty scores
        constraint: Constraint scores (lexical diversity + length)
        verify: Verifier scores (LM perplexity proxy)
        x: Weight for difficulty delta
        y: Weight for verifier
        z: Weight for constraint
        f: Function to apply to delta ("relu", "sigmoid", or "none")
        normalizers: Tuple of (delta_normalizer, verifier_normalizer, constraint_normalizer)
        normalize_rewards: Whether to normalize rewards before combining
        clip_normalized: Optional clipping value for normalized rewards (e.g., 3.0 for ±3 std)
        constraint_threshold: Threshold for soft gating (default 0.5)
        constraint_sharpness: Sharpness of the sigmoid gate (higher = sharper transition, default 10.0)

    Returns:
        Tuple of (combined_rewards, raw_delta_values)
    """
    de_new = np.array(de_new, dtype=float)
    de_old = np.array(de_old, dtype=float)
    cons = np.array(constraint, dtype=float)
    ver = np.array(verify, dtype=float)

    delta = de_new - de_old
    if f == "relu":
        delta = np.maximum(0.0, delta)
    elif f == "sigmoid":
        delta = 1/(1 + np.exp(-delta))

    # Soft gating: difficulty reward is modulated by constraint satisfaction
    # constraint_weight = sigmoid((cons - threshold) * sharpness)
    # When cons >> threshold: weight → 1.0 (full difficulty reward)
    # When cons << threshold: weight → 0.0 (no difficulty reward)
    # When cons ≈ threshold: smooth transition
    constraint_weight = 1.0 / (1.0 + np.exp(-constraint_sharpness * (cons - constraint_threshold)))

    # Normalize rewards if requested
    if normalize_rewards and normalizers is not None:
        delta_norm, ver_norm, cons_norm = normalizers

        # Update statistics
        delta_norm.update(delta)
        ver_norm.update(ver)
        cons_norm.update(cons)

        # Normalize each component
        delta_normalized = delta_norm.normalize(delta, clip=clip_normalized)
        ver_normalized = ver_norm.normalize(ver, clip=clip_normalized)
        cons_normalized = cons_norm.normalize(cons, clip=clip_normalized)

        # Combine normalized rewards with soft gating
        sum_reward = x*delta_normalized*constraint_weight + y*ver_normalized + z*cons_normalized
    else:
        # Combine raw rewards with soft gating
        sum_reward = x*delta*constraint_weight + y*ver + z*cons

    return sum_reward, delta
