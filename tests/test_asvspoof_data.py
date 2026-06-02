from pathlib import Path

import pytest

from mimodf.data.asvspoof import ASVspoofDataSettings, build_path_plan, load_fast_eval_labels


def settings(validation_protocol="asvspoof2021_fast"):
    return ASVspoofDataSettings(
        database_path=Path("/data"),
        protocols_path=Path("/protocols"),
        track="LA",
        batch_size=4,
        eval_batch_size=8,
        num_workers=0,
        sample_rate=16000,
        cut=64600,
        rawboost_algo=6,
        rawboost_args={"N_f": 5},
        validation_protocol=validation_protocol,
    )


def test_asvspoof2021_fast_path_plan_is_explicit():
    plan = build_path_plan(settings("asvspoof2021_fast"))

    assert plan.train_protocol == Path(
        "/protocols/ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    )
    assert plan.train_audio_dir == Path("/data/ASVspoof2019_LA_train")
    assert plan.validation_trial_file == Path(
        "/protocols/ASVspoof_LA_cm_protocols/ASVspoof2021.LA.cm.eval.fast.trl.txt"
    )
    assert plan.validation_key_file == Path(
        "/protocols/ASVspoof_LA_cm_protocols/ASVspoof2021.LA.cm.eval.fast.key.txt"
    )
    assert plan.validation_audio_dir == Path("/data/ASVspoof2021_LA_eval")


def test_asvspoof2019_dev_path_plan_is_explicit():
    plan = build_path_plan(settings("asvspoof2019_dev"))

    assert plan.validation_trial_file == Path(
        "/protocols/ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
    )
    assert plan.validation_key_file is None
    assert plan.validation_audio_dir == Path("/data/ASVspoof2019_LA_dev")


def test_invalid_track_fails():
    bad = settings()
    bad = ASVspoofDataSettings(**{**bad.__dict__, "track": "PA"})

    with pytest.raises(ValueError, match="track"):
        build_path_plan(bad)


def test_load_fast_eval_labels(tmp_path):
    key = tmp_path / "fast.key.txt"
    key.write_text("- LA_E_1 - - bonafide\n- LA_E_2 - - spoof\n")

    assert load_fast_eval_labels(key) == {"LA_E_1": "bonafide", "LA_E_2": "spoof"}
