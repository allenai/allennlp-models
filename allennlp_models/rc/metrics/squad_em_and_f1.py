from typing import Tuple, Union, List, cast

from allennlp.nn.util import dist_reduce_sum
from overrides import overrides

from allennlp.training.metrics.metric import Metric
from allennlp_models.rc.tools import squad


@Metric.register("squad")
class SquadEmAndF1(Metric):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computed exact match and F1 score using functions from the
    official SQuAD2 and SQuAD1.1 evaluation scripts.
    """

    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    @overrides
    def __call__(
        self,
        best_span_strings: Union[str, List[str]],
        answer_strings: Union[List[str], List[List[str]]],
    ):
        if not isinstance(best_span_strings, list):
            best_span_strings = [best_span_strings]
            answer_strings = [answer_strings]  # type: ignore

        cast(List[str], best_span_strings)
        cast(List[List[str]], answer_strings)

        assert len(best_span_strings) == len(answer_strings)

        count = len(best_span_strings)
        exact_match = 0
        f1_score = 0.0

        for prediction, gold_answers in zip(best_span_strings, answer_strings):
            exact_match += squad.metric_max_over_ground_truths(
                squad.compute_exact, prediction, gold_answers
            )
            f1_score += squad.metric_max_over_ground_truths(
                squad.compute_f1, prediction, gold_answers
            )

        # Converting to int here, since we want to count the number of exact matches.
        self._total_em += dist_reduce_sum(int(exact_match))
        self._total_f1 += dist_reduce_sum(f1_score)
        self._count += dist_reduce_sum(count)

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
        return f"SquadEmAndF1(em={self._total_em}, f1={self._total_f1})"
