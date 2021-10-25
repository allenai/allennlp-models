import logging
from typing import Dict, List, Optional

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules.transformer import TransformerEmbeddings, TransformerStack, TransformerPooler
from torch.nn import Dropout

logger = logging.getLogger(__name__)


@Model.register("transformer_mc_tt")
class TransformerMCTransformerToolkit(Model):
    """
    This class implements a multiple choice model patterned after the proposed model in
    [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al)]
    (https://api.semanticscholar.org/CorpusID:198953378).

    It is exactly like the `TransformerMC` model, except it uses the `TransformerTextField` for its input.

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
        **kwargs
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

        self.linear_layer = torch.nn.Linear(self.pooler.get_output_dim(), 1)
        self.linear_layer.weight.data.normal_(mean=0.0, std=0.02)
        self.linear_layer.bias.data.zero_()

        self.loss = torch.nn.CrossEntropyLoss()

        from allennlp.training.metrics import CategoricalAccuracy

        self.accuracy = CategoricalAccuracy()

    def forward(  # type: ignore
        self,
        alternatives: Dict[str, torch.Tensor],
        correct_alternative: Optional[torch.IntTensor] = None,
        qid: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:

        """
        Parameters
        ----------
        alternatives : ``Dict[str, torch.LongTensor]``
            From a ``ListField[TensorTextField]``. Contains a list of alternatives to evaluate for every instance.
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
        batch_size, num_alternatives, seq_length = alternatives["input_ids"].size()

        alternatives = {
            name: t.view(batch_size * num_alternatives, seq_length)
            for name, t in alternatives.items()
        }

        embedded_alternatives = self.embeddings(**alternatives)
        embedded_alternatives = self.transformer_stack(
            embedded_alternatives, alternatives["attention_mask"]
        )
        embedded_alternatives = self.pooler(embedded_alternatives.final_hidden_states)
        embedded_alternatives = self.pooler_dropout(embedded_alternatives)
        logits = self.linear_layer(embedded_alternatives)
        logits = logits.view(batch_size, num_alternatives)

        result = {"logits": logits, "best_alternative": logits.argmax(1)}

        if correct_alternative is not None:
            correct_alternative = correct_alternative.squeeze(1)
            result["loss"] = self.loss(logits, correct_alternative)
            self.accuracy(logits, correct_alternative)

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "acc": self.accuracy.get_metric(reset),
        }
