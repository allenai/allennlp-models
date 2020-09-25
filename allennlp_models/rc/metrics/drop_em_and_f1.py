from typing import Tuple, List, Union

import torch
import torch.distributed as dist

from allennlp.common.util import is_distributed

from allennlp.training.metrics.metric import Metric
from overrides import overrides

from allennlp_models.rc.tools.drop import (
    get_metrics as drop_em_and_f1,
    answer_json_to_strings,
)
from allennlp_models.rc.tools.squad import metric_max_over_ground_truths


@Metric.register("drop")
class DropEmAndF1(Metric):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computes exact match and F1 score using the official DROP
    evaluator (which has special handling for numbers and for questions with multiple answer spans,
    among other things).
    """

    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    @overrides
    def __call__(self, prediction: Union[str, List], ground_truths: List):  # type: ignore
        """
        Parameters
        ----------
        prediction: ``Union[str, List]``
            The predicted answer from the model evaluated. This could be a string, or a list of string
            when multiple spans are predicted as answer.
        ground_truths: ``List``
            All the ground truth answer annotations.
        """
        # If you wanted to split this out by answer type, you could look at [1] here and group by
        # that, instead of only keeping [0].
        ground_truth_answer_strings = [
            answer_json_to_strings(annotation)[0] for annotation in ground_truths
        ]
        exact_match, f1_score = metric_max_over_ground_truths(
            drop_em_and_f1, prediction, ground_truth_answer_strings
        )
        count = 1

        if is_distributed():
            if dist.get_backend() == "nccl":
                device = torch.cuda.current_device()
            else:
                device = torch.device("cpu")
            # Converting to int here, since we want to count the number of exact matches.
            _exact_match = torch.tensor(int(exact_match), dtype=torch.int).to(device)
            _f1_score = torch.tensor(f1_score).to(device)
            _count = torch.tensor(count).to(device)
            dist.all_reduce(_exact_match, op=dist.ReduceOp.SUM)
            dist.all_reduce(_f1_score, op=dist.ReduceOp.SUM)
            dist.all_reduce(_count, op=dist.ReduceOp.SUM)
            exact_match = _exact_match.item()
            f1_score = _f1_score.item()
            count = _count.item()

        self._total_em += exact_match
        self._total_f1 += f1_score
        self._count += count

    @overrides
    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official DROP script
        over all inputs.
        """
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return exact_match, f1_score

    @overrides
    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    def __str__(self):
        return f"DropEmAndF1(em={self._total_em}, f1={self._total_f1})"
