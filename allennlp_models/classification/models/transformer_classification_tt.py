import logging
from typing import Dict, Optional

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules.transformer import TransformerEmbeddings, TransformerStack, TransformerPooler
from torch.nn import Dropout

logger = logging.getLogger(__name__)


@Model.register("transformer_classification_tt")
class TransformerClassificationTT(Model):
    """
    This class implements a classification patterned after the proposed model in
    [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al)]
    (https://api.semanticscholar.org/CorpusID:198953378).

    Parameters
    ----------
    vocab : ``Vocabulary``
    transformer_model : ``str``, optional (default=``"roberta-large"``)
        This model chooses the embedder according to this setting. You probably want to make sure this matches the
        setting in the reader.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        transformer_model: str = "roberta-large",
        num_labels: Optional[int] = None,
        label_namespace: str = "labels",
        override_weights_file: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        transformer_kwargs = {
            "model_name": transformer_model,
            "weights_path": override_weights_file,
        }
        self.embeddings = TransformerEmbeddings.from_pretrained_module(**transformer_kwargs)
        self.transformer_stack = TransformerStack.from_pretrained_module(**transformer_kwargs)
        self.pooler = TransformerPooler.from_pretrained_module(**transformer_kwargs)
        self.pooler_dropout = Dropout(p=0.1)

        self.label_tokens = vocab.get_index_to_token_vocabulary(label_namespace)
        if num_labels is None:
            num_labels = len(self.label_tokens)
        self.linear_layer = torch.nn.Linear(self.pooler.get_output_dim(), num_labels)
        self.linear_layer.weight.data.normal_(mean=0.0, std=0.02)
        self.linear_layer.bias.data.zero_()

        from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure

        self.loss = torch.nn.CrossEntropyLoss()
        self.acc = CategoricalAccuracy()
        self.f1 = FBetaMeasure()

    def forward(  # type: ignore
        self,
        text: Dict[str, torch.Tensor],
        label: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        text : ``Dict[str, torch.LongTensor]``
            From a ``TensorTextField``. Contains the text to be classified.
        label : ``Optional[torch.LongTensor]``
            From a ``LabelField``, specifies the true class of the instance

        Returns
        -------
        An output dictionary consisting of:
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised. This is only returned when `correct_alternative` is not `None`.
        logits : ``torch.FloatTensor``
            The logits for every possible answer choice
        """
        embedded_alternatives = self.embeddings(**text)
        embedded_alternatives = self.transformer_stack(
            embedded_alternatives, text["attention_mask"]
        )
        embedded_alternatives = self.pooler(embedded_alternatives.final_hidden_states)
        embedded_alternatives = self.pooler_dropout(embedded_alternatives)
        logits = self.linear_layer(embedded_alternatives)

        result = {"logits": logits, "answers": logits.argmax(1)}

        if label is not None:
            result["loss"] = self.loss(logits, label)
            self.acc(logits, label)
            self.f1(logits, label)

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        result = {"acc": self.acc.get_metric(reset)}
        for metric_name, metrics_per_class in self.f1.get_metric(reset).items():
            for class_index, value in enumerate(metrics_per_class):
                result[f"{self.label_tokens[class_index]}-{metric_name}"] = value
        return result
