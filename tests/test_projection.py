"""
Tests for projection modules (LinearProjection and MLPProjection).

These tests verify shape transformations and trainable parameter counts
without requiring the MiMo model.
"""

import pytest

torch = pytest.importorskip("torch")

from src.frontends.projection import (
    LinearProjection,
    MLPProjection,
    get_projection,
)


class TestLinearProjection:
    """Tests for LinearProjection module."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        proj = LinearProjection(in_dim=1280, out_dim=128)
        x = torch.randn(4, 100, 1280)  # (batch, seq_len, in_dim)
        out = proj(x)
        assert out.shape == (4, 100, 128)

    def test_different_dimensions(self):
        """Test with various input/output dimensions."""
        test_cases = [
            (1024, 128),  # wav2vec2 -> AASIST
            (1280, 128),  # MiMo -> AASIST
            (2560, 256),  # dual_stream -> larger
            (25600, 512),  # rvq_concat -> compressed
        ]
        for in_dim, out_dim in test_cases:
            proj = LinearProjection(in_dim=in_dim, out_dim=out_dim)
            x = torch.randn(2, 50, in_dim)
            out = proj(x)
            assert out.shape == (2, 50, out_dim), f"Failed for {in_dim} -> {out_dim}"

    def test_parameter_count(self):
        """Test parameter count is correct."""
        proj = LinearProjection(in_dim=1280, out_dim=128)
        # Linear: weight (1280 * 128) + bias (128)
        expected = 1280 * 128 + 128
        actual = sum(p.numel() for p in proj.parameters())
        assert actual == expected

    def test_gradient_flow(self):
        """Test that gradients flow through the projection."""
        proj = LinearProjection(in_dim=1280, out_dim=128)
        x = torch.randn(2, 50, 1280, requires_grad=True)
        out = proj(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert proj.linear.weight.grad is not None


class TestMLPProjection:
    """Tests for MLPProjection module."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        proj = MLPProjection(in_dim=1280, out_dim=128, hidden_dims=[512, 256])
        x = torch.randn(4, 100, 1280)
        out = proj(x)
        assert out.shape == (4, 100, 128)

    def test_single_hidden_layer(self):
        """Test with single hidden layer."""
        proj = MLPProjection(in_dim=1280, out_dim=128, hidden_dims=[512])
        x = torch.randn(2, 50, 1280)
        out = proj(x)
        assert out.shape == (2, 50, 128)

    def test_multiple_hidden_layers(self):
        """Test with multiple hidden layers."""
        proj = MLPProjection(in_dim=1280, out_dim=128, hidden_dims=[640, 320, 160])
        x = torch.randn(2, 50, 1280)
        out = proj(x)
        assert out.shape == (2, 50, 128)

    def test_activations(self):
        """Test different activation functions."""
        activations = ["gelu", "relu", "selu", "swish", "tanh"]
        for act in activations:
            proj = MLPProjection(in_dim=1280, out_dim=128, activation=act)
            x = torch.randn(2, 50, 1280)
            out = proj(x)
            assert out.shape == (2, 50, 128), f"Failed for activation={act}"

    def test_invalid_activation(self):
        """Test that invalid activation raises error."""
        with pytest.raises(ValueError):
            MLPProjection(in_dim=1280, out_dim=128, activation="invalid")

    def test_dropout(self):
        """Test dropout is applied during training."""
        proj = MLPProjection(in_dim=1280, out_dim=128, dropout=0.5)
        proj.train()
        x = torch.randn(2, 50, 1280)

        # Run multiple times - outputs should differ due to dropout
        out1 = proj(x)
        out2 = proj(x)
        assert not torch.allclose(out1, out2)

    def test_no_dropout_in_eval(self):
        """Test dropout is disabled during eval."""
        proj = MLPProjection(in_dim=1280, out_dim=128, dropout=0.5)
        proj.eval()
        x = torch.randn(2, 50, 1280)

        # Run multiple times - outputs should be identical
        out1 = proj(x)
        out2 = proj(x)
        assert torch.allclose(out1, out2)

    def test_batchnorm(self):
        """Test with and without batch normalization."""
        proj_bn = MLPProjection(in_dim=1280, out_dim=128, use_batchnorm=True)
        proj_no_bn = MLPProjection(in_dim=1280, out_dim=128, use_batchnorm=False)

        x = torch.randn(4, 50, 1280)
        out_bn = proj_bn(x)
        out_no_bn = proj_no_bn(x)

        assert out_bn.shape == out_no_bn.shape == (4, 50, 128)

    def test_gradient_flow(self):
        """Test that gradients flow through all layers."""
        proj = MLPProjection(in_dim=1280, out_dim=128, hidden_dims=[512, 256])
        x = torch.randn(2, 50, 1280, requires_grad=True)
        out = proj(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        # Check all linear layers have gradients
        for name, param in proj.named_parameters():
            if "weight" in name:
                assert param.grad is not None, f"No gradient for {name}"


class TestGetProjection:
    """Tests for the get_projection factory function."""

    def test_linear_projection(self):
        """Test creating linear projection."""
        proj = get_projection("linear", in_dim=1280, out_dim=128)
        assert isinstance(proj, LinearProjection)

    def test_mlp_projection(self):
        """Test creating MLP projection."""
        proj = get_projection("mlp", in_dim=1280, out_dim=128)
        assert isinstance(proj, MLPProjection)

    def test_mlp_with_kwargs(self):
        """Test creating MLP projection with custom kwargs."""
        proj = get_projection(
            "mlp",
            in_dim=1280,
            out_dim=128,
            hidden_dims=[640, 320],
            activation="relu",
            dropout=0.2,
        )
        assert isinstance(proj, MLPProjection)
        assert proj.hidden_dims == [640, 320]

    def test_invalid_type(self):
        """Test that invalid projection type raises error."""
        with pytest.raises(ValueError):
            get_projection("invalid", in_dim=1280)


class TestProjectionIntegration:
    """Integration tests for projection modules."""

    def test_mimo_to_aasist_linear(self):
        """Test MiMo feature dimension to AASIST input."""
        proj = get_projection("linear", in_dim=1280, out_dim=128)
        # Simulate MiMo output: batch=8, seq_len=100 (4s @ 25Hz), dim=1280
        mimo_features = torch.randn(8, 100, 1280)
        aasist_input = proj(mimo_features)
        assert aasist_input.shape == (8, 100, 128)

    def test_mimo_to_aasist_mlp(self):
        """Test MiMo feature dimension to AASIST input with MLP."""
        proj = get_projection(
            "mlp",
            in_dim=1280,
            out_dim=128,
            hidden_dims=[512, 256],
        )
        mimo_features = torch.randn(8, 100, 1280)
        aasist_input = proj(mimo_features)
        assert aasist_input.shape == (8, 100, 128)

    def test_dual_stream_projection(self):
        """Test dual_stream (2560-dim) to AASIST."""
        proj = get_projection(
            "mlp",
            in_dim=2560,  # continuous + rvq_sum
            out_dim=128,
            hidden_dims=[1024, 512],
        )
        dual_features = torch.randn(4, 100, 2560)
        aasist_input = proj(dual_features)
        assert aasist_input.shape == (4, 100, 128)

    def test_rvq_concat_projection(self):
        """Test rvq_concat (25600-dim) to AASIST."""
        proj = get_projection(
            "mlp",
            in_dim=25600,  # 20 layers * 1280
            out_dim=128,
            hidden_dims=[2048, 512],
        )
        concat_features = torch.randn(2, 50, 25600)
        aasist_input = proj(concat_features)
        assert aasist_input.shape == (2, 50, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
