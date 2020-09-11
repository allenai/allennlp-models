import torch

from allennlp.common.testing import (
    AllenNlpTestCase,
    global_distributed_metric,
    run_distributed_test,
)

from allennlp_models.coref.metrics.mention_recall import MentionRecall


class MentionRecallTest(AllenNlpTestCase):
    def test_mention_recall(self):
        metric = MentionRecall()

        batched_top_spans = torch.tensor([[[2, 4], [1, 3]], [[5, 6], [7, 8]]])
        batched_metadata = [{"clusters": [[(2, 4), (3, 5)]]}, {"clusters": [[(5, 6), (7, 8)]]}]

        metric(batched_top_spans, batched_metadata)
        recall = metric.get_metric()
        assert recall == 0.75

    def test_distributed_mention_recall(self):
        batched_top_spans = [torch.tensor([[[2, 4], [1, 3]]]), torch.tensor([[[5, 6], [7, 8]]])]
        batched_metadata = [[{"clusters": [[(2, 4), (3, 5)]]}], [{"clusters": [[(5, 6), (7, 8)]]}]]

        metric_kwargs = {
            "batched_top_spans": batched_top_spans,
            "batched_metadata": batched_metadata,
        }
        desired_values = 0.75
        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            MentionRecall(),
            metric_kwargs,
            desired_values,
            exact=True,
        )
