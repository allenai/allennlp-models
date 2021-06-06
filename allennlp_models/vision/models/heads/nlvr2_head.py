from typing import Dict, Optional

import torch
from overrides import overrides

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.heads.head import Head


@Head.register("nlvr2")
class Nlvr2Head(Head):
    def __init__(self, vocab: Vocabulary, embedding_dim: int, label_namespace: str = "labels"):
        super().__init__(vocab)

        self.label_namespace = label_namespace

        self.layer1 = torch.nn.Linear(embedding_dim * 2, embedding_dim)
        self.layer2 = torch.nn.Linear(embedding_dim, 2)

        self.activation = torch.nn.ReLU()

        from allennlp.training.metrics import CategoricalAccuracy
        from allennlp.training.metrics import FBetaMeasure

        self.accuracy = CategoricalAccuracy()
        self.fbeta = FBetaMeasure(beta=1.0, average="macro")

    @overrides
    def forward(
        self,  # type: ignore
        encoded_boxes: torch.Tensor,
        encoded_boxes_mask: torch.Tensor,
        encoded_boxes_pooled: torch.Tensor,
        encoded_text: torch.Tensor,
        encoded_text_mask: torch.Tensor,
        encoded_text_pooled: torch.Tensor,
        pooled_boxes_and_text: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        label_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pooled_boxes_and_text = pooled_boxes_and_text.transpose(0, 1)
        hidden = self.layer1(
            torch.cat((pooled_boxes_and_text[0], pooled_boxes_and_text[1]), dim=-1)
        )
        logits = self.layer2(self.activation(hidden))
        probs = torch.softmax(logits, dim=-1)

        output = {"logits": logits, "probs": probs}

        assert label_weights is None
        if label is not None:
            output["loss"] = torch.nn.functional.cross_entropy(logits, label) / logits.size(0)
            self.accuracy(logits, label)
            self.fbeta(probs, label)

        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        result = self.fbeta.get_metric(reset)
        result["accuracy"] = self.accuracy.get_metric(reset)
        return result

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if len(output_dict) <= 0:
            return output_dict
        logits = output_dict["logits"]
        entailment_answer_index = logits.argmax(-1)
        entailment_answer = [
            self.vocab.get_token_from_index(int(i), "labels") for i in entailment_answer_index
        ]
        output_dict["entailment_answer"] = entailment_answer
        return output_dict

    default_predictor = "nlvr2"
