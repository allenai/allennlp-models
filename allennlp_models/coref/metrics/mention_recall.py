from typing import Any, Dict, List, Set, Tuple
from overrides import overrides

import torch
import torch.distributed as dist

from allennlp.common.util import is_distributed

from allennlp.training.metrics.metric import Metric


@Metric.register("mention_recall")
class MentionRecall(Metric):
    def __init__(self) -> None:
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0

    @overrides
    def __call__(
        self,  # type: ignore
        batched_top_spans: torch.Tensor,
        batched_metadata: List[Dict[str, Any]],
    ):
        num_gold_mentions = 0
        num_recalled_mentions = 0
        for top_spans, metadata in zip(batched_top_spans.tolist(), batched_metadata):

            gold_mentions: Set[Tuple[int, int]] = {
                mention for cluster in metadata["clusters"] for mention in cluster
            }
            predicted_spans: Set[Tuple[int, int]] = {(span[0], span[1]) for span in top_spans}

            num_gold_mentions += len(gold_mentions)
            num_recalled_mentions += len(gold_mentions & predicted_spans)

        if is_distributed():
            device = batched_top_spans.device
            _num_gold_mentions = torch.tensor(num_gold_mentions).to(device)
            _num_recalled_mentions = torch.tensor(num_recalled_mentions).to(device)
            dist.all_reduce(_num_gold_mentions, op=dist.ReduceOp.SUM)
            dist.all_reduce(_num_recalled_mentions, op=dist.ReduceOp.SUM)
            num_gold_mentions = _num_gold_mentions.item()
            num_recalled_mentions = _num_recalled_mentions.item()

        self._num_gold_mentions += num_gold_mentions
        self._num_recalled_mentions += num_recalled_mentions

    @overrides
    def get_metric(self, reset: bool = False) -> float:
        if self._num_gold_mentions == 0:
            recall = 0.0
        else:
            recall = self._num_recalled_mentions / self._num_gold_mentions
        if reset:
            self.reset()
        return recall

    @overrides
    def reset(self):
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0
