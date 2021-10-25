import torch

from allennlp.common.testing import (
    multi_device,
    AllenNlpTestCase,
    global_distributed_metric,
    run_distributed_test,
)

from allennlp_models.coref.metrics.conll_coref_scores import ConllCorefScores


class ConllCorefScoresTest(AllenNlpTestCase):
    @multi_device
    def test_get_predicted_clusters(self, device: str):
        top_spans = torch.tensor([[0, 1], [4, 6], [8, 9]], device=device)
        antecedent_indices = torch.tensor([[-1, -1, -1], [0, -1, -1], [0, 1, -1]], device=device)
        predicted_antecedents = torch.tensor([-1, -1, 1], device=device)
        clusters, mention_to_cluster = ConllCorefScores.get_predicted_clusters(
            top_spans, antecedent_indices, predicted_antecedents, allow_singletons=False
        )
        assert len(clusters) == 1
        assert set(clusters[0]) == {(4, 6), (8, 9)}
        assert mention_to_cluster == {(4, 6): clusters[0], (8, 9): clusters[0]}

    def test_metric_values(self):
        top_spans = torch.tensor([[[0, 1], [4, 6], [8, 9]], [[0, 1], [4, 6], [8, 9]]])
        antecedent_indices = torch.tensor(
            [[[-1, -1, -1], [0, -1, -1], [0, 1, -1]], [[-1, -1, -1], [0, -1, -1], [0, 1, -1]]]
        )
        predicted_antecedents = torch.tensor([[-1, -1, 1], [-1, -1, 1]])

        metadata_list = [{"clusters": [((4, 6), (8, 9))]}, {"clusters": [((0, 1), (4, 6))]}]

        metric = ConllCorefScores()
        metric(top_spans, antecedent_indices, predicted_antecedents, metadata_list)

        values = metric.get_metric()
        assert values[0] == values[1] == values[2] == 0.625

    def test_distributed_metric_values(self):
        top_spans = torch.tensor([[[0, 1], [4, 6], [8, 9]]])
        antecedent_indices = torch.tensor([[[-1, -1, -1], [0, -1, -1], [0, 1, -1]]])
        predicted_antecedents = torch.tensor([[-1, -1, 1]])

        metadata_list = [[{"clusters": [((4, 6), (8, 9))]}], [{"clusters": [((0, 1), (4, 6))]}]]

        metric_kwargs = {
            "top_spans": [top_spans, top_spans],
            "antecedent_indices": [antecedent_indices, antecedent_indices],
            "predicted_antecedents": [predicted_antecedents, predicted_antecedents],
            "metadata_list": metadata_list,
        }

        desired_values = (0.625, 0.625, 0.625)

        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            ConllCorefScores(),
            metric_kwargs,
            desired_values,
            exact=True,
        )
