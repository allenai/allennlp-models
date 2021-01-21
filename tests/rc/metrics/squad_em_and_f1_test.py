from allennlp.common.testing import (
    AllenNlpTestCase,
    global_distributed_metric,
    run_distributed_test,
)

from allennlp_models.rc.metrics import SquadEmAndF1


class SquadEmAndF1Test(AllenNlpTestCase):
    def test_squad_em_and_f1(self):
        metric = SquadEmAndF1()

        metric("this is the best span", ["this is a good span", "something irrelevant"])

        exact_match, f1_score = metric.get_metric()
        assert exact_match == 0.0
        assert f1_score == 0.75

    def test_distributed_squad_em_and_f1(self):
        best_span_strings = ["this is the best span", "this is another span"]
        answer_strings = [
            ["this is a good span", "something irrelevant"],
            ["this is another span", "this one is less perfect"],
        ]

        metric_kwargs = {"best_span_strings": best_span_strings, "answer_strings": answer_strings}
        desired_values = (1 / 2, 1.75 / 2)
        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            SquadEmAndF1(),
            metric_kwargs,
            desired_values,
            exact=True,
        )
