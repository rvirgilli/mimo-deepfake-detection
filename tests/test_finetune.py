"""
Tests for fine-tuning strategies (frozen, adapter, lora, partial, gradual).

Uses mock tokenizer to test fine-tuning logic without requiring the MiMo model.
"""

import pytest

torch = pytest.importorskip("torch")
from unittest.mock import MagicMock

import torch.nn as nn


class MockTransformerLayer(nn.Module):
    """Mock transformer layer for testing."""

    def __init__(self, hidden_size: int = 1280):
        super().__init__()
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.mlp = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return x


class MockEncoder(nn.Module):
    """Mock encoder with transformer layers."""

    def __init__(self, num_layers: int = 32, hidden_size: int = 1280):
        super().__init__()
        self.layers = nn.ModuleList([MockTransformerLayer(hidden_size) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MockConfig:
    """Mock config for MiMo tokenizer."""

    d_model = 1280


class MockTokenizer(nn.Module):
    """Mock MiMo tokenizer for testing."""

    def __init__(self, num_layers: int = 32):
        super().__init__()
        self.encoder = MockEncoder(num_layers)
        self.config = MockConfig()

    def forward(self, x):
        return self.encoder(x)


class TestMiMoGradualUnfreeze:
    """Tests for MiMoGradualUnfreeze class."""

    def test_import(self):
        """Test that MiMoGradualUnfreeze can be imported."""
        from src.frontends.mimo_finetune import MiMoGradualUnfreeze

        assert MiMoGradualUnfreeze is not None

    def test_initialization_freezes_all(self):
        """Test that initialization freezes all parameters."""
        from src.frontends.mimo_finetune import MiMoGradualUnfreeze

        tokenizer = MockTokenizer(num_layers=8)
        MiMoGradualUnfreeze(tokenizer, {0: 0, 5: 4})

        # All params should be frozen initially
        for param in tokenizer.parameters():
            assert not param.requires_grad

    def test_default_schedule(self):
        """Test default unfreeze schedule."""
        from src.frontends.mimo_finetune import MiMoGradualUnfreeze

        tokenizer = MockTokenizer(num_layers=32)
        wrapper = MiMoGradualUnfreeze(tokenizer)

        # Default schedule should be present
        assert 0 in wrapper.unfreeze_schedule
        assert wrapper.unfreeze_schedule[0] == 0

    def test_custom_schedule(self):
        """Test custom unfreeze schedule."""
        from src.frontends.mimo_finetune import MiMoGradualUnfreeze

        schedule = {0: 0, 10: 2, 20: 4, 30: 8}
        tokenizer = MockTokenizer(num_layers=8)
        wrapper = MiMoGradualUnfreeze(tokenizer, schedule)

        assert wrapper.unfreeze_schedule == schedule

    def test_on_epoch_start_unfreezes_layers(self):
        """Test that on_epoch_start unfreezes correct number of layers."""
        from src.frontends.mimo_finetune import MiMoGradualUnfreeze

        tokenizer = MockTokenizer(num_layers=8)
        schedule = {0: 0, 5: 2, 10: 4, 15: 8}
        wrapper = MiMoGradualUnfreeze(tokenizer, schedule)

        # Epoch 0: all frozen
        n = wrapper.on_epoch_start(0)
        assert n == 0
        assert wrapper.num_trainable_params() == 0

        # Epoch 5: last 2 unfrozen
        n = wrapper.on_epoch_start(5)
        assert n == 2
        assert wrapper.num_trainable_params() > 0

        # Epoch 10: last 4 unfrozen
        n = wrapper.on_epoch_start(10)
        assert n == 4

        # Epoch 15: all 8 unfrozen
        n = wrapper.on_epoch_start(15)
        assert n == 8

    def test_intermediate_epochs(self):
        """Test epochs between schedule points."""
        from src.frontends.mimo_finetune import MiMoGradualUnfreeze

        tokenizer = MockTokenizer(num_layers=8)
        schedule = {0: 0, 10: 4}
        wrapper = MiMoGradualUnfreeze(tokenizer, schedule)

        # Epoch 5: should still be at 0 layers
        n = wrapper.on_epoch_start(5)
        assert n == 0

        # Epoch 15: should be at 4 layers (from epoch 10)
        n = wrapper.on_epoch_start(15)
        assert n == 4

    def test_get_trainable_params(self):
        """Test get_trainable_params iterator."""
        from src.frontends.mimo_finetune import MiMoGradualUnfreeze

        tokenizer = MockTokenizer(num_layers=8)
        wrapper = MiMoGradualUnfreeze(tokenizer, {0: 0, 5: 2})

        # Initially no trainable params
        params = list(wrapper.get_trainable_params())
        assert len(params) == 0

        # After unfreezing
        wrapper.on_epoch_start(5)
        params = list(wrapper.get_trainable_params())
        assert len(params) > 0

    def test_forward_passthrough(self):
        """Test that forward passes through to tokenizer."""
        from src.frontends.mimo_finetune import MiMoGradualUnfreeze

        tokenizer = MockTokenizer(num_layers=4)
        wrapper = MiMoGradualUnfreeze(tokenizer)

        x = torch.randn(2, 100, 1280)
        out = wrapper(x)
        assert out.shape == x.shape

    def test_get_frozen_status(self):
        """Test get_frozen_status method."""
        from src.frontends.mimo_finetune import MiMoGradualUnfreeze

        tokenizer = MockTokenizer(num_layers=4)
        wrapper = MiMoGradualUnfreeze(tokenizer, {0: 0, 5: 2})

        # Initially all frozen
        status = wrapper.get_frozen_status()
        assert all(s == "frozen" for s in status.values())

        # After unfreezing last 2
        wrapper.on_epoch_start(5)
        status = wrapper.get_frozen_status()
        unfrozen_count = sum(1 for s in status.values() if s == "unfrozen")
        assert unfrozen_count == 2


class TestMiMoPartialFinetune:
    """Tests for MiMoPartialFinetune class."""

    def test_import(self):
        """Test that MiMoPartialFinetune can be imported."""
        from src.frontends.mimo_finetune import MiMoPartialFinetune

        assert MiMoPartialFinetune is not None

    def test_initialization(self):
        """Test initialization unfreezes last N layers."""
        from src.frontends.mimo_finetune import MiMoPartialFinetune

        tokenizer = MockTokenizer(num_layers=8)
        wrapper = MiMoPartialFinetune(tokenizer, n_trainable_layers=2)

        # Check that some params are trainable
        trainable = wrapper.num_trainable_params()
        assert trainable > 0

    def test_n_trainable_layers(self):
        """Test different values of n_trainable_layers."""
        from src.frontends.mimo_finetune import MiMoPartialFinetune

        tokenizer = MockTokenizer(num_layers=8)

        wrapper2 = MiMoPartialFinetune(tokenizer, n_trainable_layers=2)
        params2 = wrapper2.num_trainable_params()

        tokenizer = MockTokenizer(num_layers=8)
        wrapper4 = MiMoPartialFinetune(tokenizer, n_trainable_layers=4)
        params4 = wrapper4.num_trainable_params()

        # More layers = more params
        assert params4 > params2


class TestMiMoWithAdapters:
    """Tests for MiMoWithAdapters class."""

    def test_import(self):
        """Test that MiMoWithAdapters can be imported."""
        from src.frontends.mimo_finetune import MiMoWithAdapters

        assert MiMoWithAdapters is not None

    def test_adapter_dim(self):
        """Test adapter dimension configuration."""
        from src.frontends.mimo_finetune import MiMoWithAdapters

        tokenizer = MockTokenizer(num_layers=8)
        wrapper = MiMoWithAdapters(tokenizer, adapter_dim=64)

        # Should have trainable adapter params
        trainable = wrapper.num_trainable_params()
        assert trainable > 0

    def test_different_adapter_dims(self):
        """Test different adapter bottleneck dimensions."""
        from src.frontends.mimo_finetune import MiMoWithAdapters

        for dim in [32, 64, 128]:
            tokenizer = MockTokenizer(num_layers=4)
            wrapper = MiMoWithAdapters(tokenizer, adapter_dim=dim, n_adapter_layers=2)
            params = wrapper.num_trainable_params()
            assert params > 0, f"No trainable params for adapter_dim={dim}"


class TestMiMoWithLoRA:
    """Tests for MiMoWithLoRA class."""

    def test_import(self):
        """Test that MiMoWithLoRA can be imported."""
        from src.frontends.mimo_finetune import MiMoWithLoRA

        assert MiMoWithLoRA is not None

    def test_lora_rank(self):
        """Test LoRA rank configuration."""
        from src.frontends.mimo_finetune import MiMoWithLoRA

        tokenizer = MockTokenizer(num_layers=8)
        wrapper = MiMoWithLoRA(tokenizer, lora_rank=8)

        trainable = wrapper.num_trainable_params()
        assert trainable > 0

    def test_different_ranks(self):
        """Test different LoRA ranks."""
        from src.frontends.mimo_finetune import MiMoWithLoRA

        results = {}
        for rank in [4, 8, 16]:
            tokenizer = MockTokenizer(num_layers=4)
            wrapper = MiMoWithLoRA(tokenizer, lora_rank=rank)
            results[rank] = wrapper.num_trainable_params()

        # Higher rank = more params
        assert results[16] > results[8] > results[4]


class TestApplyFinetuneStrategy:
    """Tests for apply_finetune_strategy factory function."""

    def test_import(self):
        """Test that apply_finetune_strategy can be imported."""
        from src.frontends.mimo_finetune import apply_finetune_strategy

        assert apply_finetune_strategy is not None


class TestFinetuneIntegration:
    """Integration tests for fine-tuning with MiMoFrontend."""

    @pytest.fixture
    def mock_mimo_frontend(self):
        """Create a mock MiMoFrontend for testing."""
        # Create mock frontend with tokenizer
        frontend = MagicMock()
        frontend.tokenizer = MockTokenizer(num_layers=8)
        frontend._finetune_strategy = "frozen"
        frontend.finetune_wrapper = None
        return frontend

    def test_gradual_unfreeze_with_training_loop(self):
        """Simulate gradual unfreezing in a training loop."""
        from src.frontends.mimo_finetune import MiMoGradualUnfreeze

        tokenizer = MockTokenizer(num_layers=8)
        schedule = {0: 0, 3: 2, 6: 4, 9: 8}
        wrapper = MiMoGradualUnfreeze(tokenizer, schedule)

        trainable_history = []
        for epoch in range(12):
            n_unfrozen = wrapper.on_epoch_start(epoch)
            trainable = wrapper.num_trainable_params()
            trainable_history.append((epoch, n_unfrozen, trainable))

        # Verify progression
        assert trainable_history[0][1] == 0  # Epoch 0: 0 layers
        assert trainable_history[3][1] == 2  # Epoch 3: 2 layers
        assert trainable_history[6][1] == 4  # Epoch 6: 4 layers
        assert trainable_history[9][1] == 8  # Epoch 9: 8 layers

    def test_schedule_from_config(self):
        """Test creating schedule from config dict."""
        from src.frontends.mimo_finetune import MiMoGradualUnfreeze

        # Simulate config structure
        config_schedule = {
            0: 0,
            5: 4,
            10: 8,
            15: 16,
            20: 32,
        }

        tokenizer = MockTokenizer(num_layers=32)
        wrapper = MiMoGradualUnfreeze(tokenizer, config_schedule)

        # Verify schedule was applied
        assert wrapper.on_epoch_start(0) == 0
        assert wrapper.on_epoch_start(7) == 4
        assert wrapper.on_epoch_start(12) == 8


class TestParameterCounting:
    """Tests for parameter counting across strategies."""

    def test_frozen_has_zero_trainable(self):
        """Test that frozen strategy has no trainable encoder params."""
        tokenizer = MockTokenizer(num_layers=8)

        # Freeze all
        for param in tokenizer.parameters():
            param.requires_grad = False

        trainable = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
        assert trainable == 0

    def test_full_has_all_trainable(self):
        """Test that full strategy has all params trainable."""
        tokenizer = MockTokenizer(num_layers=8)

        # Unfreeze all
        for param in tokenizer.parameters():
            param.requires_grad = True

        total = sum(p.numel() for p in tokenizer.parameters())
        trainable = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
        assert trainable == total

    def test_partial_has_subset_trainable(self):
        """Test that partial strategy has subset trainable."""
        from src.frontends.mimo_finetune import MiMoPartialFinetune

        tokenizer = MockTokenizer(num_layers=8)
        wrapper = MiMoPartialFinetune(tokenizer, n_trainable_layers=2)

        total = sum(p.numel() for p in tokenizer.parameters())
        trainable = wrapper.num_trainable_params()

        assert 0 < trainable < total


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
