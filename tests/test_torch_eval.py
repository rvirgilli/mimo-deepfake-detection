import pytest

torch = pytest.importorskip("torch")

from mimodf.scoring.evaluate import EvaluationBatch, EvaluationItem, write_scores_from_batches
from mimodf.scoring.torch_eval import TorchBatchPredictor, TorchEvaluationError


class SumModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.was_eval = False

    def eval(self):
        self.was_eval = True
        return super().eval()

    def forward(self, batch):
        return batch.sum(dim=1)


def test_torch_batch_predictor_scores_one_value_per_input():
    model = SumModel()
    predictor = TorchBatchPredictor(model=model, score_fn=lambda output: output)

    scores = predictor([torch.tensor([1.0, 2.0]), torch.tensor([-1.0, 0.5])])

    assert scores == pytest.approx([3.0, -0.5])
    assert model.was_eval is True


def test_torch_batch_predictor_rejects_wrong_score_count():
    predictor = TorchBatchPredictor(
        model=SumModel(),
        score_fn=lambda output: torch.tensor([1.0]),
    )

    with pytest.raises(TorchEvaluationError, match="returned 1 scores for 2 inputs"):
        predictor([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])])


def test_torch_predictor_integrates_with_score_file_writer(tmp_path):
    predictor = TorchBatchPredictor(model=SumModel(), score_fn=lambda output: output)
    batch = EvaluationBatch.from_items(
        [
            EvaluationItem("utt_b", torch.tensor([2.0, 3.0])),
            EvaluationItem("utt_a", torch.tensor([1.0, 1.5])),
        ]
    )

    path = write_scores_from_batches([batch], predictor, tmp_path / "scores_LA_eval.txt")

    assert path.read_text().splitlines() == ["utt_a 2.5", "utt_b 5"]
