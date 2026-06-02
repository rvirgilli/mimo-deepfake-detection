"""
Tests for upsampling modes (linear, nearest, learnable).

Tests the LearnableUpsample class and verifies upsampling behavior
without requiring the MiMo model.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from src.frontends.mimo import LearnableUpsample


class TestLearnableUpsample:
    """Tests for LearnableUpsample module."""

    def test_output_shape_2x(self):
        """Test that output is exactly 2x the input length."""
        upsampler = LearnableUpsample(in_channels=1280)
        x = torch.randn(4, 100, 1280)  # (batch, seq_len, dim)
        out = upsampler(x)
        assert out.shape == (4, 200, 1280)

    def test_various_sequence_lengths(self):
        """Test with various input sequence lengths."""
        upsampler = LearnableUpsample(in_channels=1280)
        for seq_len in [50, 101, 200, 251]:
            x = torch.randn(2, seq_len, 1280)
            out = upsampler(x)
            assert out.shape == (2, seq_len * 2, 1280), f"Failed for seq_len={seq_len}"

    def test_various_channel_dims(self):
        """Test with various channel dimensions."""
        for channels in [128, 512, 1024, 1280, 2560]:
            upsampler = LearnableUpsample(in_channels=channels)
            x = torch.randn(2, 100, channels)
            out = upsampler(x)
            assert out.shape == (2, 200, channels), f"Failed for channels={channels}"

    def test_kernel_sizes(self):
        """Test with different kernel sizes."""
        for kernel_size in [2, 4, 6, 8]:
            upsampler = LearnableUpsample(in_channels=1280, kernel_size=kernel_size)
            x = torch.randn(2, 100, 1280)
            out = upsampler(x)
            assert out.shape == (2, 200, 1280), f"Failed for kernel_size={kernel_size}"

    def test_parameter_count(self):
        """Test parameter count is correct."""
        upsampler = LearnableUpsample(in_channels=1280, kernel_size=4)
        # ConvTranspose1d: weight (in_channels * out_channels * kernel_size) + bias (out_channels)
        # With groups=1: weight (1280 * 1280 * 4) + bias (1280)
        expected = 1280 * 1280 * 4 + 1280
        actual = sum(p.numel() for p in upsampler.parameters())
        assert actual == expected

    def test_gradient_flow(self):
        """Test that gradients flow through the upsampler."""
        upsampler = LearnableUpsample(in_channels=1280)
        x = torch.randn(2, 50, 1280, requires_grad=True)
        out = upsampler(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert upsampler.upsample.weight.grad is not None

    def test_trainable_by_default(self):
        """Test that upsampler parameters are trainable by default."""
        upsampler = LearnableUpsample(in_channels=1280)
        for param in upsampler.parameters():
            assert param.requires_grad

    def test_initialization_stability(self):
        """Test that initial output is reasonable (not exploding)."""
        upsampler = LearnableUpsample(in_channels=1280)
        x = torch.randn(2, 100, 1280)
        out = upsampler(x)
        # Output should be in a reasonable range
        assert out.abs().max() < 100, "Output values too large"
        assert out.std() > 0.001, "Output too collapsed"


class TestUpsamplingModes:
    """Compare different upsampling methods."""

    def test_linear_interpolation(self):
        """Test linear interpolation upsampling."""
        x = torch.randn(2, 100, 1280)
        # Transpose for F.interpolate: (B, C, T)
        x_t = x.transpose(1, 2)
        out_t = F.interpolate(x_t, scale_factor=2, mode="linear", align_corners=False)
        out = out_t.transpose(1, 2)
        assert out.shape == (2, 200, 1280)

    def test_nearest_interpolation(self):
        """Test nearest neighbor upsampling."""
        x = torch.randn(2, 100, 1280)
        x_t = x.transpose(1, 2)
        out_t = F.interpolate(x_t, scale_factor=2, mode="nearest")
        out = out_t.transpose(1, 2)
        assert out.shape == (2, 200, 1280)

    def test_nearest_duplicates_frames(self):
        """Test that nearest neighbor duplicates frames."""
        x = torch.randn(2, 10, 128)
        x_t = x.transpose(1, 2)
        out_t = F.interpolate(x_t, scale_factor=2, mode="nearest")
        out = out_t.transpose(1, 2)
        # Check that consecutive frames are identical
        for i in range(10):
            assert torch.allclose(out[:, 2 * i, :], out[:, 2 * i + 1, :])

    def test_linear_smooths_frames(self):
        """Test that linear interpolation creates intermediate values."""
        x = torch.zeros(1, 2, 1)
        x[0, 0, 0] = 0.0
        x[0, 1, 0] = 1.0
        x_t = x.transpose(1, 2)
        out_t = F.interpolate(x_t, scale_factor=2, mode="linear", align_corners=False)
        out = out_t.transpose(1, 2)
        # Interpolated values should be between 0 and 1
        assert out[0, 1, 0] > 0 and out[0, 1, 0] < 1
        assert out[0, 2, 0] > 0 and out[0, 2, 0] < 1

    def test_learnable_differs_from_linear(self):
        """Test that learnable upsampling produces different output than linear."""
        upsampler = LearnableUpsample(in_channels=128)
        x = torch.randn(2, 50, 128)

        # Learnable
        learnable_out = upsampler(x)

        # Linear
        x_t = x.transpose(1, 2)
        linear_out_t = F.interpolate(x_t, scale_factor=2, mode="linear", align_corners=False)
        linear_out = linear_out_t.transpose(1, 2)

        # Should be different (learnable has trainable params)
        assert not torch.allclose(learnable_out, linear_out)

    def test_all_modes_same_output_shape(self):
        """Test that all modes produce the same output shape."""
        x = torch.randn(2, 100, 1280)
        upsampler = LearnableUpsample(in_channels=1280)

        # Learnable
        learnable_out = upsampler(x)

        # Linear
        x_t = x.transpose(1, 2)
        linear_out = F.interpolate(
            x_t, scale_factor=2, mode="linear", align_corners=False
        ).transpose(1, 2)

        # Nearest
        nearest_out = F.interpolate(x_t, scale_factor=2, mode="nearest").transpose(1, 2)

        assert learnable_out.shape == linear_out.shape == nearest_out.shape == (2, 200, 1280)


class TestLearnableUpsampleTraining:
    """Test learnable upsampler in training scenarios."""

    def test_parameter_update(self):
        """Test that parameters update during training."""
        upsampler = LearnableUpsample(in_channels=128)
        optimizer = torch.optim.SGD(upsampler.parameters(), lr=0.01)

        # Save initial weights
        initial_weight = upsampler.upsample.weight.clone()

        # Training step
        x = torch.randn(4, 50, 128)
        target = torch.randn(4, 100, 128)

        optimizer.zero_grad()
        out = upsampler(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()

        # Weights should have changed
        assert not torch.allclose(upsampler.upsample.weight, initial_weight)

    def test_loss_decreases(self):
        """Test that loss decreases over training iterations."""
        upsampler = LearnableUpsample(in_channels=128)
        optimizer = torch.optim.Adam(upsampler.parameters(), lr=0.001)

        x = torch.randn(4, 50, 128)
        # Target: simply double each frame (nearest neighbor pattern)
        target = x.repeat_interleave(2, dim=1)

        losses = []
        for _ in range(100):
            optimizer.zero_grad()
            out = upsampler(x)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0], "Loss did not decrease during training"

    def test_can_learn_identity_like_pattern(self):
        """Test that upsampler can learn a simple pattern."""
        upsampler = LearnableUpsample(in_channels=32)
        optimizer = torch.optim.Adam(upsampler.parameters(), lr=0.01)

        # Create simple pattern: input -> doubled output
        x = torch.randn(8, 20, 32)
        target = x.repeat_interleave(2, dim=1)

        for _ in range(200):
            optimizer.zero_grad()
            out = upsampler(x)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer.step()

        # Final output should be close to target
        with torch.no_grad():
            final_out = upsampler(x)
            final_loss = F.mse_loss(final_out, target)
            assert final_loss < 0.1, f"Could not learn pattern, loss={final_loss:.4f}"


class TestUpsampleEdgeCases:
    """Test edge cases for upsampling."""

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        upsampler = LearnableUpsample(in_channels=1280)
        x = torch.randn(1, 100, 1280)
        out = upsampler(x)
        assert out.shape == (1, 200, 1280)

    def test_sequence_length_one(self):
        """Test with sequence length of 1."""
        upsampler = LearnableUpsample(in_channels=1280)
        x = torch.randn(2, 1, 1280)
        out = upsampler(x)
        assert out.shape == (2, 2, 1280)

    def test_very_long_sequence(self):
        """Test with very long sequence."""
        upsampler = LearnableUpsample(in_channels=128)
        x = torch.randn(1, 1000, 128)
        out = upsampler(x)
        assert out.shape == (1, 2000, 128)

    def test_cuda_if_available(self):
        """Test on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        upsampler = LearnableUpsample(in_channels=1280).cuda()
        x = torch.randn(2, 100, 1280).cuda()
        out = upsampler(x)
        assert out.shape == (2, 200, 1280)
        assert out.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
