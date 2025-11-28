#!/usr/bin/env python3
"""
Test script to verify reward normalization works correctly.
"""
import numpy as np
from src.losses import RewardNormalizer, method2_loss

def test_normalizer():
    """Test the RewardNormalizer class."""
    print("Testing RewardNormalizer...")

    normalizer = RewardNormalizer()

    # Simulate some reward values with different scales
    batch1 = np.array([10.0, 15.0, 12.0, 18.0])
    batch2 = np.array([11.0, 14.0, 16.0, 13.0])
    batch3 = np.array([9.0, 17.0, 12.5, 14.5])

    # Update statistics
    normalizer.update(batch1)
    print(f"After batch 1: mean={normalizer.mean:.4f}, std={normalizer.std:.4f}")

    normalizer.update(batch2)
    print(f"After batch 2: mean={normalizer.mean:.4f}, std={normalizer.std:.4f}")

    normalizer.update(batch3)
    print(f"After batch 3: mean={normalizer.mean:.4f}, std={normalizer.std:.4f}")

    # Test normalization
    test_values = np.array([10.0, 13.5, 17.0])
    normalized = normalizer.normalize(test_values)
    print(f"\nTest values: {test_values}")
    print(f"Normalized: {normalized}")

    # Test clipping
    normalized_clipped = normalizer.normalize(test_values, clip=2.0)
    print(f"Normalized (clipped ±2σ): {normalized_clipped}")


def test_method2_loss():
    """Test method2_loss with and without normalization."""
    print("\n\nTesting method2_loss...")

    # Sample data with very different scales
    de_new = [0.7, 0.8, 0.75]  # Difficulty scores (0-1 range)
    de_old = [0.5, 0.6, 0.55]
    constraint = [0.8, 0.7, 0.9]  # Constraint scores (0-1 range)
    verify = [50.0, 60.0, 55.0]  # Verifier scores (much larger scale!)

    # Without normalization
    print("\nWithout normalization:")
    rewards_raw, delta_raw = method2_loss(
        de_new, de_old, constraint, verify,
        x=5.0, y=1.0, z=0.3, f="none",
        normalize_rewards=False
    )
    print(f"Delta: {delta_raw}")
    print(f"Rewards: {rewards_raw}")
    print(f"Reward range: [{rewards_raw.min():.2f}, {rewards_raw.max():.2f}]")

    # With normalization
    print("\nWith normalization:")
    delta_norm = RewardNormalizer()
    ver_norm = RewardNormalizer()
    cons_norm = RewardNormalizer()
    normalizers = (delta_norm, ver_norm, cons_norm)

    rewards_normalized, delta_norm_vals = method2_loss(
        de_new, de_old, constraint, verify,
        x=5.0, y=1.0, z=0.3, f="none",
        normalizers=normalizers,
        normalize_rewards=True,
        clip_normalized=3.0
    )
    print(f"Delta: {delta_norm_vals}")
    print(f"Rewards (normalized): {rewards_normalized}")
    print(f"Reward range: [{rewards_normalized.min():.2f}, {rewards_normalized.max():.2f}]")

    # Show normalizer statistics
    print(f"\nNormalizer statistics:")
    print(f"  Delta: mean={delta_norm.mean:.4f}, std={delta_norm.std:.4f}")
    print(f"  Verifier: mean={ver_norm.mean:.4f}, std={ver_norm.std:.4f}")
    print(f"  Constraint: mean={cons_norm.mean:.4f}, std={cons_norm.std:.4f}")

    # Process another batch to show statistics update
    print("\n\nProcessing second batch...")
    de_new2 = [0.6, 0.85, 0.72]
    de_old2 = [0.4, 0.65, 0.52]
    constraint2 = [0.75, 0.85, 0.8]
    verify2 = [45.0, 70.0, 52.0]

    rewards_normalized2, _ = method2_loss(
        de_new2, de_old2, constraint2, verify2,
        x=5.0, y=1.0, z=0.3, f="none",
        normalizers=normalizers,
        normalize_rewards=True,
        clip_normalized=3.0
    )
    print(f"Rewards (batch 2): {rewards_normalized2}")
    print(f"\nUpdated normalizer statistics:")
    print(f"  Delta: mean={delta_norm.mean:.4f}, std={delta_norm.std:.4f}")
    print(f"  Verifier: mean={ver_norm.mean:.4f}, std={ver_norm.std:.4f}")
    print(f"  Constraint: mean={cons_norm.mean:.4f}, std={cons_norm.std:.4f}")


if __name__ == "__main__":
    test_normalizer()
    test_method2_loss()
    print("\n✓ All tests passed!")
