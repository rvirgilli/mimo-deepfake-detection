"""
Tests for RawBoost sample-rate aware scaling and Optuna integration.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rawboost import (
    REFERENCE_SAMPLE_RATE,
    ISD_additive_noise,
    LnL_convolutive_noise,
    ScaledRawBoostParams,
    SSI_additive_noise,
    scale_rawboost_params,
)


class TestScaleRawBoostParams:
    """Tests for scale_rawboost_params function."""

    @pytest.fixture
    def baseline_args(self):
        """Baseline 16kHz RawBoost parameters."""
        return SimpleNamespace(
            N_f=5,
            nBands=5,
            minF=20,
            maxF=8000,
            minBW=100,
            maxBW=1000,
            minCoeff=10,
            maxCoeff=100,
            minG=0,
            maxG=0,
            minBiasLinNonLin=5,
            maxBiasLinNonLin=20,
            P=10,
            g_sd=2,
            SNRmin=10,
            SNRmax=40,
        )

    def test_reference_sample_rate_is_16khz(self):
        """Verify reference sample rate constant."""
        assert REFERENCE_SAMPLE_RATE == 16000

    def test_scaling_at_16khz_no_change(self, baseline_args):
        """At 16kHz, params should be unchanged (except maxF capped at Nyquist-100)."""
        scaled = scale_rawboost_params(baseline_args, 16000, auto_scale=True)

        assert isinstance(scaled, ScaledRawBoostParams)
        assert scaled.scale_factor == 1.0
        assert scaled.sample_rate == 16000
        # maxF capped at Nyquist - 100 = 8000 - 100 = 7900
        assert scaled.maxF == 7900
        assert scaled.minBW == 100
        assert scaled.maxBW == 1000
        assert scaled.minCoeff == 10
        assert scaled.maxCoeff == 100

    def test_scaling_at_24khz(self, baseline_args):
        """At 24kHz, frequency-dependent params should be scaled by 1.5."""
        scaled = scale_rawboost_params(baseline_args, 24000, auto_scale=True)

        assert scaled.scale_factor == 1.5
        assert scaled.sample_rate == 24000
        # maxF: min(8000 * 1.5, 12000 - 100) = min(12000, 11900) = 11900
        assert scaled.maxF == 11900
        # Bandwidth scaled by 1.5
        assert scaled.minBW == 150
        assert scaled.maxBW == 1500
        # Coefficients scaled by 1.5
        assert scaled.minCoeff == 15
        assert scaled.maxCoeff == 150

    def test_scaling_at_48khz(self, baseline_args):
        """At 48kHz, params should be scaled by 3.0."""
        scaled = scale_rawboost_params(baseline_args, 48000, auto_scale=True)

        assert scaled.scale_factor == 3.0
        assert scaled.sample_rate == 48000
        # maxF: min(8000 * 3, 24000 - 100) = min(24000, 23900) = 23900
        assert scaled.maxF == 23900
        assert scaled.minBW == 300
        assert scaled.maxBW == 3000
        assert scaled.minCoeff == 30
        assert scaled.maxCoeff == 300

    def test_unchanged_params_not_scaled(self, baseline_args):
        """Verify amplitude/percentage/dB params are not scaled."""
        scaled = scale_rawboost_params(baseline_args, 24000, auto_scale=True)

        # These should be unchanged regardless of sample rate
        assert scaled.N_f == 5
        assert scaled.nBands == 5
        assert scaled.minF == 20
        assert scaled.minG == 0
        assert scaled.maxG == 0
        assert scaled.minBiasLinNonLin == 5
        assert scaled.maxBiasLinNonLin == 20
        assert scaled.P == 10
        assert scaled.g_sd == 2
        assert scaled.SNRmin == 10
        assert scaled.SNRmax == 40

    def test_auto_scale_false(self, baseline_args):
        """With auto_scale=False, params should not be scaled."""
        scaled = scale_rawboost_params(baseline_args, 24000, auto_scale=False)

        assert scaled.scale_factor == 1.0
        # maxF still capped at Nyquist - 100 for safety
        assert scaled.maxF == 8000  # Not scaled, but within 24kHz Nyquist
        assert scaled.minBW == 100
        assert scaled.maxBW == 1000
        assert scaled.minCoeff == 10
        assert scaled.maxCoeff == 100

    def test_custom_bw_scale(self, baseline_args):
        """Test custom bandwidth scaling (for Optuna tuning)."""
        baseline_args.bw_scale = 1.8  # Custom bandwidth scale
        scaled = scale_rawboost_params(baseline_args, 24000, auto_scale=True)

        # maxF and coefficients still use standard scale_factor (1.5)
        assert scaled.maxF == 11900
        assert scaled.minCoeff == 15
        assert scaled.maxCoeff == 150

        # But bandwidth uses custom bw_scale (1.8)
        assert scaled.minBW == 180  # 100 * 1.8
        assert scaled.maxBW == 1800  # 1000 * 1.8

    def test_custom_bw_scale_without_auto_scale(self, baseline_args):
        """Custom bw_scale should work even without auto_scale."""
        baseline_args.bw_scale = 1.5
        scaled = scale_rawboost_params(baseline_args, 24000, auto_scale=False)

        # No scaling for maxF and coefficients
        assert scaled.maxF == 8000
        assert scaled.minCoeff == 10
        assert scaled.maxCoeff == 100

        # But bw_scale still applies
        assert scaled.minBW == 150
        assert scaled.maxBW == 1500

    def test_maxf_capped_at_nyquist(self, baseline_args):
        """maxF should never exceed Nyquist - 100."""
        # Set maxF very high
        baseline_args.maxF = 20000
        scaled = scale_rawboost_params(baseline_args, 24000, auto_scale=False)

        # Should be capped at Nyquist - 100 = 12000 - 100 = 11900
        assert scaled.maxF == 11900

    def test_default_values_used_when_missing(self):
        """Test that defaults are used for missing attributes."""
        empty_args = SimpleNamespace()
        scaled = scale_rawboost_params(empty_args, 24000, auto_scale=True)

        # Should use defaults and scale appropriately
        assert scaled.N_f == 5
        assert scaled.nBands == 5
        assert scaled.minF == 20
        assert scaled.maxF == 11900  # min(8000 * 1.5, 11900)
        assert scaled.minBW == 150
        assert scaled.maxBW == 1500


class TestRawBoostAugmentation:
    """Test actual RawBoost augmentation with scaled params."""

    @pytest.fixture
    def audio_16khz(self):
        """Generate 4 seconds of random audio at 16kHz."""
        np.random.seed(42)
        return np.random.randn(16000 * 4).astype(np.float32)

    @pytest.fixture
    def audio_24khz(self):
        """Generate 4 seconds of random audio at 24kHz."""
        np.random.seed(42)
        return np.random.randn(24000 * 4).astype(np.float32)

    def test_lnl_convolutive_noise_16khz(self, audio_16khz):
        """Test LnL convolutive noise at 16kHz."""
        output = LnL_convolutive_noise(
            audio_16khz,
            N_f=5,
            nBands=5,
            minF=20,
            maxF=7900,
            minBW=100,
            maxBW=1000,
            minCoeff=10,
            maxCoeff=100,
            minG=0,
            maxG=0,
            minBiasLinNonLin=5,
            maxBiasLinNonLin=20,
            fs=16000,
        )
        assert output.shape == audio_16khz.shape
        assert not np.allclose(output, audio_16khz)  # Should be modified

    def test_lnl_convolutive_noise_24khz_scaled(self, audio_24khz):
        """Test LnL convolutive noise at 24kHz with scaled params."""
        output = LnL_convolutive_noise(
            audio_24khz,
            N_f=5,
            nBands=5,
            minF=20,
            maxF=11900,  # Scaled maxF
            minBW=150,
            maxBW=1500,  # Scaled bandwidth
            minCoeff=15,
            maxCoeff=150,  # Scaled coefficients
            minG=0,
            maxG=0,
            minBiasLinNonLin=5,
            maxBiasLinNonLin=20,
            fs=24000,
        )
        assert output.shape == audio_24khz.shape
        assert not np.allclose(output, audio_24khz)

    def test_ssi_additive_noise_24khz_scaled(self, audio_24khz):
        """Test SSI additive noise at 24kHz with scaled params."""
        output = SSI_additive_noise(
            audio_24khz,
            SNRmin=10,
            SNRmax=40,
            nBands=5,
            minF=20,
            maxF=11900,
            minBW=150,
            maxBW=1500,
            minCoeff=15,
            maxCoeff=150,
            minG=0,
            maxG=0,
            fs=24000,
        )
        assert output.shape == audio_24khz.shape
        assert not np.allclose(output, audio_24khz)

    def test_isd_noise_sample_rate_independent(self, audio_24khz):
        """ISD noise should work identically regardless of sample rate."""
        output = ISD_additive_noise(audio_24khz, P=10, g_sd=2)
        assert output.shape == audio_24khz.shape


class TestDatasetIntegration:
    """Test Dataset class integration with auto_scale_rawboost."""

    def test_dataset_auto_scale_flag(self):
        """Test that Dataset properly handles auto_scale_rawboost."""
        from src.data_utils import Dataset_ASVspoof2019_train

        # Create minimal args
        args = SimpleNamespace(
            algo=5,
            N_f=5,
            nBands=5,
            minF=20,
            maxF=8000,
            minBW=100,
            maxBW=1000,
            minCoeff=10,
            maxCoeff=100,
            minG=0,
            maxG=0,
            minBiasLinNonLin=5,
            maxBiasLinNonLin=20,
            P=10,
            g_sd=2,
            SNRmin=10,
            SNRmax=40,
        )

        # Test with 16kHz (no scaling should occur)
        ds_16k = Dataset_ASVspoof2019_train(
            args=args,
            list_IDs=["test"],
            labels={"test": 0},
            base_dir="/tmp/",
            algo=5,
            sample_rate=16000,
            auto_scale_rawboost=True,
        )
        # At 16kHz, args should be wrapped but not scaled
        assert hasattr(ds_16k, "args")

        # Test with 24kHz and auto_scale=True
        ds_24k_auto = Dataset_ASVspoof2019_train(
            args=args,
            list_IDs=["test"],
            labels={"test": 0},
            base_dir="/tmp/",
            algo=5,
            sample_rate=24000,
            auto_scale_rawboost=True,
        )
        # At 24kHz with auto_scale, args should be ScaledRawBoostParams
        # Check by verifying it has the scaled values and scale metadata
        assert hasattr(ds_24k_auto.args, "scale_factor")
        assert ds_24k_auto.args.scale_factor == 1.5
        assert ds_24k_auto.args.maxF == 11900
        assert ds_24k_auto.args.minBW == 150

        # Test with 24kHz and auto_scale=False
        ds_24k_no_auto = Dataset_ASVspoof2019_train(
            args=args,
            list_IDs=["test"],
            labels={"test": 0},
            base_dir="/tmp/",
            algo=5,
            sample_rate=24000,
            auto_scale_rawboost=False,
        )
        # At 24kHz without auto_scale, args should be original
        assert ds_24k_no_auto.args == args


class TestOptunaIntegration:
    """Test Optuna search space integration."""

    def test_optuna_config_has_rawboost_params(self):
        """Verify Optuna config includes RawBoost search space."""
        from omegaconf import OmegaConf

        config_path = Path(__file__).parent.parent / "configs" / "optuna" / "default.yaml"
        cfg = OmegaConf.load(config_path)

        tier2 = cfg.get("search_space_tier2", {})

        # Check rawboost_auto_scale
        assert "rawboost_auto_scale" in tier2
        assert tier2.rawboost_auto_scale.type == "categorical"
        assert True in tier2.rawboost_auto_scale.choices
        assert False in tier2.rawboost_auto_scale.choices

        # Check rawboost_maxF
        assert "rawboost_maxF" in tier2
        assert tier2.rawboost_maxF.type == "categorical"
        assert 11900 in tier2.rawboost_maxF.choices

        # Check rawboost_bw_scale
        assert "rawboost_bw_scale" in tier2
        assert tier2.rawboost_bw_scale.type == "float"
        assert tier2.rawboost_bw_scale.low == 1.0
        assert tier2.rawboost_bw_scale.high == 2.0

    def test_optuna_sampling_simulation(self):
        """Simulate Optuna sampling of RawBoost params."""
        import optuna

        def objective(trial):
            # Simulate tier 2 sampling
            rawboost_auto_scale = trial.suggest_categorical("rawboost_auto_scale", [True, False])

            if not rawboost_auto_scale:
                rawboost_maxF = trial.suggest_categorical(
                    "rawboost_maxF", [8000, 10000, 11000, 11900]
                )
            else:
                rawboost_maxF = None

            # For MiMo frontend
            rawboost_bw_scale = trial.suggest_float("rawboost_bw_scale", 1.0, 2.0)

            # Verify values are valid
            assert rawboost_auto_scale in [True, False]
            if rawboost_maxF is not None:
                assert rawboost_maxF in [8000, 10000, 11000, 11900]
            assert 1.0 <= rawboost_bw_scale <= 2.0

            return 0.0  # Dummy objective

        # Run a few trials
        study = optuna.create_study()
        study.optimize(objective, n_trials=5, show_progress_bar=False)
        assert len(study.trials) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
