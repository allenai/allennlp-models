from typing import Any, Dict, List, Tuple, Union

import pytest
import torch
from allennlp.common.testing import (
    AllenNlpTestCase,
    multi_device,
    global_distributed_metric,
    run_distributed_test,
)

from allennlp_models.vision import VqaMeasure


class VqaMeasureTest(AllenNlpTestCase):
    @multi_device
    def test_vqa(self, device: str):
        vqa = VqaMeasure()
        logits = torch.tensor(
            [[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0]], device=device
        )
        labels = torch.tensor([[0], [3]], device=device)
        label_weights = torch.tensor([[1 / 3], [2 / 3]], device=device)
        vqa(logits, labels, label_weights)
        vqa_score = vqa.get_metric()["score"]
        assert vqa_score == pytest.approx((1 / 3) / 2)

    @multi_device
    def test_vqa_accumulates_and_resets_correctly(self, device: str):
        vqa = VqaMeasure()
        logits = torch.tensor(
            [[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0]], device=device
        )
        labels = torch.tensor([[0], [3]], device=device)
        labels2 = torch.tensor([[4], [4]], device=device)
        label_weights = torch.tensor([[1 / 3], [2 / 3]], device=device)

        vqa(logits, labels, label_weights)
        vqa(logits, labels, label_weights)
        vqa(logits, labels2, label_weights)
        vqa(logits, labels2, label_weights)

        vqa_score = vqa.get_metric(reset=True)["score"]
        assert vqa_score == pytest.approx((1 / 3 + 1 / 3 + 0 + 0) / 8)

        vqa(logits, labels, label_weights)
        vqa_score = vqa.get_metric(reset=True)["score"]
        assert vqa_score == pytest.approx((1 / 3) / 2)

    @multi_device
    def test_does_not_divide_by_zero_with_no_count(self, device: str):
        vqa = VqaMeasure()
        assert vqa.get_metric()["score"] == pytest.approx(0.0)

    def test_distributed_accuracy(self):
        logits = [
            torch.tensor([[0.35, 0.25, 0.1, 0.1, 0.2]]),
            torch.tensor([[0.1, 0.6, 0.1, 0.2, 0.0]]),
        ]
        labels = [torch.tensor([[0]]), torch.tensor([[3]])]
        label_weights = [torch.tensor([[1 / 3]]), torch.tensor([[2 / 3]])]
        metric_kwargs = {"logits": logits, "labels": labels, "label_weights": label_weights}
        desired_accuracy = {"score": (1 / 3) / 2}
        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            VqaMeasure(),
            metric_kwargs,
            desired_accuracy,
            exact=False,
        )

    def test_distributed_accuracy_unequal_batches(self):
        logits = [
            torch.tensor([[0.35, 0.25, 0.1, 0.1, 0.2], [0.35, 0.25, 0.1, 0.1, 0.2]]),
            torch.tensor([[0.1, 0.6, 0.1, 0.2, 0.0]]),
        ]
        labels = [torch.tensor([[0], [0]]), torch.tensor([[3]])]
        label_weights = [torch.tensor([[1], [1]]), torch.tensor([[1 / 3]])]
        metric_kwargs = {"logits": logits, "labels": labels, "label_weights": label_weights}
        desired_accuracy = {"score": (1 + 1 + 0) / 3}
        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            VqaMeasure(),
            metric_kwargs,
            desired_accuracy,
            exact=False,
        )

    def test_multiple_distributed_runs(self):
        logits = [
            torch.tensor([[0.35, 0.25, 0.1, 0.1, 0.2]]),
            torch.tensor([[0.1, 0.6, 0.1, 0.2, 0.0]]),
        ]
        labels = [torch.tensor([[0]]), torch.tensor([[3]])]
        label_weights = [torch.tensor([[1 / 3]]), torch.tensor([[2 / 3]])]
        metric_kwargs = {"logits": logits, "labels": labels, "label_weights": label_weights}
        desired_accuracy = {"score": (1 / 3) / 2}
        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            VqaMeasure(),
            metric_kwargs,
            desired_accuracy,
            exact=True,
            number_of_runs=200,
        )
