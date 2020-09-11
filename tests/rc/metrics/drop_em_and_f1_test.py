from allennlp.common.testing import (
    AllenNlpTestCase,
    global_distributed_metric,
    run_distributed_test,
)

from allennlp_models.rc.metrics import DropEmAndF1


class DropEmAndF1Test(AllenNlpTestCase):
    def test_drop_em_and_f1(self):
        metric = DropEmAndF1()

        metric(
            "this is the best span", [{"spans": ["this is a good span", "something irrelevant"]}]
        )
        exact_match, f1_score = metric.get_metric()
        assert exact_match == 0.0
        assert f1_score == 0.38

    def test_distributed_drop_em_and_f1(self):
        prediction = ["this is the best span", "this is another span"]
        ground_truths = [
            [{"spans": ["this is a good span", "something irrelevant"]}],
            [{"spans": ["this is another span"]}],
        ]

        metric_kwargs = {"prediction": prediction, "ground_truths": ground_truths}
        desired_values = (1 / 2, 1.38 / 2)
        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            DropEmAndF1(),
            metric_kwargs,
            desired_values,
            exact=True,
        )
