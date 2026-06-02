"""
Tests for feature extraction strategies.

Tests the FeatureStrategy classes including output dimensions and
get_feature_strategy factory function.
"""

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn


class TestFeatureStrategyImports:
    """Test that feature strategies can be imported."""

    def test_import_base_class(self):
        """Test importing FeatureStrategy base class."""
        from src.frontends.mimo_features import FeatureStrategy

        assert FeatureStrategy is not None

    def test_import_continuous_strategy(self):
        """Test importing ContinuousStrategy."""
        from src.frontends.mimo_features import ContinuousStrategy

        assert ContinuousStrategy is not None

    def test_import_rvq_sum_strategy(self):
        """Test importing RVQSumStrategy."""
        from src.frontends.mimo_features import RVQSumStrategy

        assert RVQSumStrategy is not None

    def test_import_rvq_concat_strategy(self):
        """Test importing RVQConcatStrategy."""
        from src.frontends.mimo_features import RVQConcatStrategy

        assert RVQConcatStrategy is not None

    def test_import_rvq_fine_strategy(self):
        """Test importing RVQFineStrategy."""
        from src.frontends.mimo_features import RVQFineStrategy

        assert RVQFineStrategy is not None

    def test_import_dual_stream_strategy(self):
        """Test importing DualStreamStrategy."""
        from src.frontends.mimo_features import DualStreamStrategy

        assert DualStreamStrategy is not None

    def test_import_weighted_strategy(self):
        """Test importing WeightedRVQStrategy."""
        from src.frontends.mimo_features import WeightedRVQStrategy

        assert WeightedRVQStrategy is not None

    def test_import_factory_function(self):
        """Test importing get_feature_strategy factory."""
        from src.frontends.mimo_features import get_feature_strategy

        assert get_feature_strategy is not None

    def test_import_list_strategies(self):
        """Test importing list_strategies function."""
        from src.frontends.mimo_features import list_strategies

        assert list_strategies is not None


class TestContinuousStrategy:
    """Tests for ContinuousStrategy."""

    def test_output_dimension(self):
        """Test output dimension is 1280."""
        from src.frontends.mimo_features import ContinuousStrategy

        strategy = ContinuousStrategy()
        assert strategy.out_dim == 1280

    def test_is_nn_module(self):
        """Test that strategy is an nn.Module."""
        from src.frontends.mimo_features import ContinuousStrategy

        strategy = ContinuousStrategy()
        assert isinstance(strategy, nn.Module)

    def test_no_trainable_params(self):
        """Test that ContinuousStrategy has no trainable params."""
        from src.frontends.mimo_features import ContinuousStrategy

        strategy = ContinuousStrategy()
        params = list(strategy.parameters())
        assert len(params) == 0


class TestRVQSumStrategy:
    """Tests for RVQSumStrategy."""

    def test_output_dimension(self):
        """Test output dimension is 1280."""
        from src.frontends.mimo_features import RVQSumStrategy

        strategy = RVQSumStrategy()
        assert strategy.out_dim == 1280

    def test_default_n_q(self):
        """Test default number of quantizer layers."""
        from src.frontends.mimo_features import RVQSumStrategy

        strategy = RVQSumStrategy()
        assert strategy.n_q == 20

    def test_custom_n_q(self):
        """Test custom number of quantizer layers."""
        from src.frontends.mimo_features import RVQSumStrategy

        strategy = RVQSumStrategy(n_q=10)
        assert strategy.n_q == 10


class TestRVQConcatStrategy:
    """Tests for RVQConcatStrategy."""

    def test_default_output_dimension(self):
        """Test default output dimension (20 layers * 1280)."""
        from src.frontends.mimo_features import RVQConcatStrategy

        strategy = RVQConcatStrategy()
        assert strategy.out_dim == 20 * 1280  # 25600

    def test_custom_n_q(self):
        """Test custom n_q affects output dimension."""
        from src.frontends.mimo_features import RVQConcatStrategy

        strategy = RVQConcatStrategy(n_q=10)
        assert strategy.out_dim == 10 * 1280  # 12800


class TestRVQFineStrategy:
    """Tests for RVQFineStrategy."""

    def test_default_layers(self):
        """Test default layer selection (layers 2-8)."""
        from src.frontends.mimo_features import RVQFineStrategy

        strategy = RVQFineStrategy()
        assert strategy.start_layer == 2
        assert strategy.end_layer == 8  # Default from config is 8

    def test_custom_layers(self):
        """Test custom layer selection."""
        from src.frontends.mimo_features import RVQFineStrategy

        strategy = RVQFineStrategy(start_layer=4, end_layer=10)
        assert strategy.start_layer == 4
        assert strategy.end_layer == 10

    def test_output_dimension(self):
        """Test output dimension based on selected layers."""
        from src.frontends.mimo_features import RVQFineStrategy

        # Default: layers 2-7 = 6 layers
        strategy = RVQFineStrategy()
        n_layers = strategy.end_layer - strategy.start_layer + 1
        assert strategy.out_dim == n_layers * 1280


class TestDualStreamStrategy:
    """Tests for DualStreamStrategy."""

    def test_output_dimension(self):
        """Test output dimension is 2560 (1280 + 1280)."""
        from src.frontends.mimo_features import DualStreamStrategy

        strategy = DualStreamStrategy()
        assert strategy.out_dim == 2560

    def test_default_n_q(self):
        """Test default number of quantizer layers."""
        from src.frontends.mimo_features import DualStreamStrategy

        strategy = DualStreamStrategy()
        assert strategy.n_q == 20


class TestWeightedRVQStrategy:
    """Tests for WeightedRVQStrategy."""

    def test_output_dimension(self):
        """Test output dimension is 1280."""
        from src.frontends.mimo_features import WeightedRVQStrategy

        strategy = WeightedRVQStrategy()
        assert strategy.out_dim == 1280

    def test_has_trainable_params(self):
        """Test that WeightedRVQStrategy has trainable layer weights."""
        from src.frontends.mimo_features import WeightedRVQStrategy

        strategy = WeightedRVQStrategy()
        params = list(strategy.parameters())
        assert len(params) > 0

    def test_layer_weights_shape(self):
        """Test layer weights have correct shape."""
        from src.frontends.mimo_features import WeightedRVQStrategy

        strategy = WeightedRVQStrategy(n_q=20)
        assert strategy.layer_weights.shape == (20,)

    def test_custom_n_q(self):
        """Test custom number of quantizer layers."""
        from src.frontends.mimo_features import WeightedRVQStrategy

        strategy = WeightedRVQStrategy(n_q=10)
        assert strategy.n_q == 10
        assert strategy.layer_weights.shape == (10,)

    def test_weights_are_learnable(self):
        """Test that layer weights require gradients."""
        from src.frontends.mimo_features import WeightedRVQStrategy

        strategy = WeightedRVQStrategy()
        assert strategy.layer_weights.requires_grad


class TestGetFeatureStrategy:
    """Tests for get_feature_strategy factory function."""

    def test_continuous(self):
        """Test creating continuous strategy."""
        from src.frontends.mimo_features import ContinuousStrategy, get_feature_strategy

        strategy = get_feature_strategy("continuous")
        assert isinstance(strategy, ContinuousStrategy)

    def test_rvq_sum(self):
        """Test creating rvq_sum strategy."""
        from src.frontends.mimo_features import RVQSumStrategy, get_feature_strategy

        strategy = get_feature_strategy("rvq_sum")
        assert isinstance(strategy, RVQSumStrategy)

    def test_rvq_concat(self):
        """Test creating rvq_concat strategy."""
        from src.frontends.mimo_features import RVQConcatStrategy, get_feature_strategy

        strategy = get_feature_strategy("rvq_concat")
        assert isinstance(strategy, RVQConcatStrategy)

    def test_rvq_fine(self):
        """Test creating rvq_fine strategy."""
        from src.frontends.mimo_features import RVQFineStrategy, get_feature_strategy

        strategy = get_feature_strategy("rvq_fine")
        assert isinstance(strategy, RVQFineStrategy)

    def test_dual_stream(self):
        """Test creating dual_stream strategy."""
        from src.frontends.mimo_features import DualStreamStrategy, get_feature_strategy

        strategy = get_feature_strategy("dual_stream")
        assert isinstance(strategy, DualStreamStrategy)

    def test_weighted(self):
        """Test creating weighted strategy."""
        from src.frontends.mimo_features import WeightedRVQStrategy, get_feature_strategy

        strategy = get_feature_strategy("weighted")
        assert isinstance(strategy, WeightedRVQStrategy)

    def test_invalid_strategy(self):
        """Test that invalid strategy name raises error."""
        from src.frontends.mimo_features import get_feature_strategy

        with pytest.raises(ValueError):
            get_feature_strategy("invalid_strategy")

    def test_kwargs_passed_through(self):
        """Test that kwargs are passed to strategy constructor."""
        from src.frontends.mimo_features import get_feature_strategy

        strategy = get_feature_strategy("rvq_sum", n_q=15)
        assert strategy.n_q == 15

    def test_rvq_fine_with_kwargs(self):
        """Test rvq_fine with custom layer range."""
        from src.frontends.mimo_features import get_feature_strategy

        strategy = get_feature_strategy("rvq_fine", start_layer=5, end_layer=15)
        assert strategy.start_layer == 5
        assert strategy.end_layer == 15


class TestListStrategies:
    """Tests for list_strategies function."""

    def test_returns_dict(self):
        """Test that list_strategies returns a dict."""
        from src.frontends.mimo_features import list_strategies

        strategies = list_strategies()
        assert isinstance(strategies, dict)

    def test_all_strategies_listed(self):
        """Test that all strategies are listed."""
        from src.frontends.mimo_features import list_strategies

        strategies = list_strategies()
        expected = ["continuous", "rvq_sum", "rvq_concat", "rvq_fine", "dual_stream", "weighted"]
        for name in expected:
            assert name in strategies

    def test_descriptions_are_strings(self):
        """Test that all descriptions are strings."""
        from src.frontends.mimo_features import list_strategies

        strategies = list_strategies()
        for name, desc in strategies.items():
            assert isinstance(desc, str), f"Description for {name} is not a string"


class TestOutputDimensions:
    """Test output dimensions for all strategies."""

    @pytest.mark.parametrize(
        "strategy_name,expected_dim",
        [
            ("continuous", 1280),
            ("rvq_sum", 1280),
            ("rvq_concat", 25600),
            ("dual_stream", 2560),
            ("weighted", 1280),
        ],
    )
    def test_output_dimensions(self, strategy_name, expected_dim):
        """Test output dimensions for each strategy."""
        from src.frontends.mimo_features import get_feature_strategy

        strategy = get_feature_strategy(strategy_name)
        assert strategy.out_dim == expected_dim, f"{strategy_name} has wrong out_dim"

    def test_rvq_fine_dimension(self):
        """Test rvq_fine dimension (depends on layer range)."""
        from src.frontends.mimo_features import get_feature_strategy

        # Default: layers 2-7 = 6 layers
        strategy = get_feature_strategy("rvq_fine")
        n_layers = strategy.end_layer - strategy.start_layer + 1
        assert strategy.out_dim == n_layers * 1280


class TestUpsampleMethod:
    """Test the _upsample_to_length helper method."""

    def test_upsample_doubles_length(self):
        """Test that upsampling to 2x length works."""
        from src.frontends.mimo_features import ContinuousStrategy

        strategy = ContinuousStrategy()
        features = torch.randn(2, 100, 1280)
        upsampled = strategy._upsample_to_length(features, 200)
        assert upsampled.shape == (2, 200, 1280)

    def test_upsample_arbitrary_length(self):
        """Test upsampling to arbitrary length."""
        from src.frontends.mimo_features import ContinuousStrategy

        strategy = ContinuousStrategy()
        features = torch.randn(2, 100, 1280)
        upsampled = strategy._upsample_to_length(features, 150)
        assert upsampled.shape == (2, 150, 1280)

    def test_upsample_gradient_flow(self):
        """Test that gradients flow through upsampling."""
        from src.frontends.mimo_features import ContinuousStrategy

        strategy = ContinuousStrategy()
        features = torch.randn(2, 100, 1280, requires_grad=True)
        upsampled = strategy._upsample_to_length(features, 200)
        loss = upsampled.sum()
        loss.backward()
        assert features.grad is not None


class TestStrategyNNModule:
    """Test that strategies behave as proper nn.Modules."""

    @pytest.mark.parametrize(
        "strategy_name",
        ["continuous", "rvq_sum", "rvq_concat", "rvq_fine", "dual_stream", "weighted"],
    )
    def test_is_nn_module(self, strategy_name):
        """Test each strategy is an nn.Module."""
        from src.frontends.mimo_features import get_feature_strategy

        strategy = get_feature_strategy(strategy_name)
        assert isinstance(strategy, nn.Module)

    @pytest.mark.parametrize(
        "strategy_name",
        ["continuous", "rvq_sum", "rvq_concat", "rvq_fine", "dual_stream", "weighted"],
    )
    def test_can_move_to_device(self, strategy_name):
        """Test strategies can be moved to devices."""
        from src.frontends.mimo_features import get_feature_strategy

        strategy = get_feature_strategy(strategy_name)
        strategy = strategy.cpu()  # Should not raise

    def test_weighted_cuda_if_available(self):
        """Test WeightedRVQStrategy on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from src.frontends.mimo_features import WeightedRVQStrategy

        strategy = WeightedRVQStrategy().cuda()
        assert strategy.layer_weights.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
