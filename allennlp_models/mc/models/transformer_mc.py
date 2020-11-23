import logging
from typing import Dict, List, Optional

import torch
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model

logger = logging.getLogger(__name__)


@Model.register("transformer_mc")
class TransformerMC(Model):
    """
    This class implements a multiple choice model patterned after the proposed model in
    [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al)]
    (https://api.semanticscholar.org/CorpusID:198953378).

    It calculates a score for each sequence on top of the CLS token, and then chooses the alternative
    with the highest score.

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
        override_weights_file: Optional[str] = None,
        override_weights_strip_prefix: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)
        from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
        from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
        from allennlp.modules.seq2vec_encoders import BertPooler

        self._text_field_embedder = PretrainedTransformerEmbedder(
            transformer_model,
            override_weights_file=override_weights_file,
            override_weights_strip_prefix=override_weights_strip_prefix,
        )
        self._text_field_embedder = BasicTextFieldEmbedder({"tokens": self._text_field_embedder})
        self._pooler = BertPooler(
            transformer_model,
            override_weights_file=override_weights_file,
            override_weights_strip_prefix=override_weights_strip_prefix,
            dropout=0.1,
        )

        self._linear_layer = torch.nn.Linear(self._text_field_embedder.get_output_dim(), 1)
        self._linear_layer.weight.data.normal_(mean=0.0, std=0.02)
        self._linear_layer.bias.data.zero_()

        self._loss = torch.nn.CrossEntropyLoss()

        from allennlp.training.metrics import CategoricalAccuracy

        self._accuracy = CategoricalAccuracy()

    def forward(  # type: ignore
        self,
        alternatives: TextFieldTensors,
        correct_alternative: Optional[torch.IntTensor] = None,
        qid: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:

        """
        Parameters
        ----------
        alternatives : ``Dict[str, torch.LongTensor]``
            From a ``ListField[TextField]``. Contains a list of alternatives to evaluate for every instance.
        correct_alternative : ``Optional[torch.IntTensor]``
            From an ``IndexField``. Contains the index of the correct answer for every instance.
        qid : `Optional[List[str]]`
            A list of question IDs for the questions being processed now.

        Returns
        -------
        An output dictionary consisting of:
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised. This is only returned when `correct_alternative` is not `None`.
        logits : ``torch.FloatTensor``
            The logits for every possible answer choice
        best_alternative : ``List[int]``
            The index of the highest scoring alternative for every instance in the batch
        """
        embedded_alternatives = self._text_field_embedder(alternatives, num_wrapping_dims=1)
        flattened_embedded_alternatives = embedded_alternatives.view(
            embedded_alternatives.size(0) * embedded_alternatives.size(1),
            embedded_alternatives.size(2),
            embedded_alternatives.size(3),
        )
        flattened_pooled_alternatives = self._pooler(flattened_embedded_alternatives)
        flattened_logit_alternatives = self._linear_layer(flattened_pooled_alternatives)
        logit_alternatives = flattened_logit_alternatives.view(
            embedded_alternatives.size(0), embedded_alternatives.size(1)
        )

        result = {"logits": logit_alternatives, "best_alternative": logit_alternatives.argmax(1)}

        if correct_alternative is not None:
            correct_alternative = correct_alternative.squeeze(1)
            result["loss"] = self._loss(logit_alternatives, correct_alternative)
            self._accuracy(logit_alternatives, correct_alternative)

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "acc": self._accuracy.get_metric(reset),
        }

    default_predictor = "transformer_mc"
