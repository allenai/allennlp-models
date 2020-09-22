from typing import Tuple

from overrides import overrides

from allennlp.training.metrics.metric import Metric
from allennlp_models.rc.tools import squad2


@Metric.register("squad2")
class Squad2EmAndF1(Metric):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computed exact match and F1 score using the official SQuAD2
    evaluation script.
    """

    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    @overrides
    def __call__(self, best_span_string, gold_answers):
        exact_match = max(
            squad2.compute_exact(gold_answer, best_span_string) for gold_answer in gold_answers
        )
        f1_score = max(
            squad2.compute_f1(gold_answer, best_span_string) for gold_answer in gold_answers
        )
        self._total_em += exact_match
        self._total_f1 += f1_score
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official SQuAD script
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
        return f"Squad2EmAndF1(em={self._total_em}, f1={self._total_f1})"
