import torch
from overrides import overrides

from allennlp.training.metrics.metric import Metric
import torch.distributed as dist


@Metric.register("vqa")
class VqaMeasure(Metric):
    """Compute the VQA metric, as described in
    https://www.semanticscholar.org/paper/VQA%3A-Visual-Question-Answering-Agrawal-Lu/97ad70a9fa3f99adf18030e5e38ebe3d90daa2db

    In VQA, we take the answer with the highest score, and then we find out how often
    humans decided this was the right answer. The accuracy score for an answer is
    `min(1.0, human_count / 3)`.

    This metric takes the logits from the models, i.e., a score for each possible answer,
    and the labels for the question, together with their weights.
    """

    def __init__(self) -> None:
        self._sum_of_scores = 0.0
        self._score_count = 0

    @overrides
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, label_weights: torch.Tensor):
        """
        # Parameters

        logits : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, num_classes).
        labels : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, num_labels).
        label_weights : `torch.Tensor`, required.
            A tensor of floats of shape (batch_size, num_labels), giving a weight or score to
            every one of the labels.
        """

        logits, labels, label_weights = self.detach_tensors(logits, labels, label_weights)
        predictions = logits.argmax(dim=1)

        # Sum over dimension 1 gives the score per question. We care about the overall sum though,
        # so we sum over all dimensions.
        local_sum_of_scores = (
            (label_weights * (labels == predictions.unsqueeze(-1))).sum().to(torch.float32)
        )
        local_score_count = torch.tensor(labels.size(0), dtype=torch.int32, device=labels.device)

        from allennlp.common.util import is_distributed

        if is_distributed():
            dist.all_reduce(local_sum_of_scores, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_score_count, op=dist.ReduceOp.SUM)

        self._sum_of_scores += local_sum_of_scores.item()
        self._score_count += local_score_count.item()

    @overrides
    def get_metric(self, reset: bool = False):
        if self._score_count > 0:
            result = self._sum_of_scores / self._score_count
        else:
            result = 0.0
        result_dict = {"score": result}
        if reset:
            self.reset()
        return result_dict

    @overrides
    def reset(self) -> None:
        self._sum_of_scores = 0.0
        self._score_count = 0
