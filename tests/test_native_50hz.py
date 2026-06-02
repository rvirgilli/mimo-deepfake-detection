"""
Test native 50Hz feature extraction for MiMo frontend.

Verifies that native 50Hz produces 2x the temporal frames compared to 25Hz.
"""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


MODEL_PATH = Path("./models/MiMo-Audio-Tokenizer")


def require_mimo_integration():
    """Skip heavyweight MiMo integration tests unless explicitly enabled."""
    if os.environ.get("RUN_MIMO_INTEGRATION") != "1":
        pytest.skip("Set RUN_MIMO_INTEGRATION=1 to run MiMo model integration tests")
    if not MODEL_PATH.exists():
        pytest.skip(f"MiMo model weights not found: {MODEL_PATH}")


def test_native_50hz_output_shape():
    """Test that native 50Hz produces 2x temporal frames vs 25Hz."""
    require_mimo_integration()
    try:
        from src.frontends import get_frontend
    except ImportError:
        pytest.skip("MiMo frontend not available")

    # Create two frontends: default 25Hz and native 50Hz
    frontend_25hz = get_frontend(
        "mimo",
        model_path="./models/MiMo-Audio-Tokenizer",
        use_bfloat16=True,
        native_50hz=False,
    )

    frontend_50hz = get_frontend(
        "mimo",
        model_path="./models/MiMo-Audio-Tokenizer",
        use_bfloat16=True,
        native_50hz=True,
    )

    # Generate dummy audio: 4 seconds at 24kHz
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio = torch.randn(2, 96000).to(device)  # batch of 2

    # Move frontends to device
    frontend_25hz = frontend_25hz.to(device)
    frontend_50hz = frontend_50hz.to(device)

    # Extract features
    with torch.no_grad():
        features_25hz = frontend_25hz.extract_feat(audio)
        features_50hz = frontend_50hz.extract_feat(audio)

    print(f"25Hz shape: {features_25hz.shape}")
    print(f"50Hz shape: {features_50hz.shape}")

    # Verify dimensions
    batch_25, seq_25, dim_25 = features_25hz.shape
    batch_50, seq_50, dim_50 = features_50hz.shape

    assert batch_25 == batch_50 == 2, "Batch size should match"
    assert dim_25 == dim_50 == 1280, "Feature dimension should be 1280"

    # 50Hz should have approximately 2x the frames (may differ by 1 due to rounding)
    ratio = seq_50 / seq_25
    assert 1.9 < ratio < 2.1, f"Frame ratio should be ~2x, got {ratio:.2f}"

    print(f"Frame ratio (50Hz/25Hz): {ratio:.2f}x")
    print("Test passed!")


def test_upsample_vs_native():
    """Compare upsampled 25Hz vs native 50Hz features."""
    require_mimo_integration()
    try:
        from src.frontends import get_frontend
    except ImportError:
        pytest.skip("MiMo frontend not available")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Upsampled 25Hz → 50Hz
    frontend_upsampled = get_frontend(
        "mimo",
        model_path="./models/MiMo-Audio-Tokenizer",
        use_bfloat16=True,
        upsample_to_50hz=True,
        native_50hz=False,
    )

    # Native 50Hz
    frontend_native = get_frontend(
        "mimo",
        model_path="./models/MiMo-Audio-Tokenizer",
        use_bfloat16=True,
        native_50hz=True,
    )

    audio = torch.randn(1, 96000).to(device)
    frontend_upsampled = frontend_upsampled.to(device)
    frontend_native = frontend_native.to(device)

    with torch.no_grad():
        features_upsampled = frontend_upsampled.extract_feat(audio)
        features_native = frontend_native.extract_feat(audio)

    print(f"Upsampled shape: {features_upsampled.shape}")
    print(f"Native shape: {features_native.shape}")

    # Shapes may differ slightly due to rounding:
    # - Upsampled: doubles 25Hz frames exactly (101 * 2 = 202)
    # - Native: true 50Hz frame count (201)
    assert features_upsampled.shape[0] == features_native.shape[0], "Batch size should match"
    assert features_upsampled.shape[2] == features_native.shape[2], "Feature dim should match"

    # Sequence lengths should be within 1-2 frames
    seq_diff = abs(features_upsampled.shape[1] - features_native.shape[1])
    assert seq_diff <= 2, f"Sequence length should be similar, diff={seq_diff}"
    print(f"Sequence length difference: {seq_diff} frames")

    # Compare overlapping frames to verify they produce different features
    min_len = min(features_upsampled.shape[1], features_native.shape[1])
    diff = (features_upsampled[:, :min_len] - features_native[:, :min_len]).abs().mean()
    print(f"Mean absolute difference: {diff:.4f}")
    assert diff > 0.01, "Features should differ significantly"

    print("Upsampled vs native comparison passed!")


if __name__ == "__main__":
    print("Testing native 50Hz feature extraction\n")
    print("=" * 60)

    print("\n1. Testing output shape ratio...")
    test_native_50hz_output_shape()

    print("\n" + "=" * 60)
    print("\n2. Comparing upsampled vs native 50Hz...")
    test_upsample_vs_native()

    print("\n" + "=" * 60)
    print("\nAll tests passed!")
