import logging
from typing import Dict, Optional

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules.transformer import TransformerEmbeddings, TransformerStack, TransformerPooler, MaskedLMHead
from torch.nn import Dropout

logger = logging.getLogger(__name__)


@Model.register("masked_lm_tt")
class MaskedLMTT(Model):
    """
    This class implements a masked LM patterned after the proposed model in
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
        self.lm_head = MaskedLMHead.from_pretrained_module(**transformer_kwargs)

        from allennlp.training.metrics import CategoricalAccuracy

        self.loss = torch.nn.CrossEntropyLoss()
        self.acc = CategoricalAccuracy()

    def forward(  # type: ignore
        self,
        text: Dict[str, torch.Tensor],
        label: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        text : ``Dict[str, torch.LongTensor]``
            From a ``TensorTextField``. Contains the text with masked out tokens.
        label : ``Optional[torch.LongTensor]``
            Specifies the true tokens at all positions, including masked positions

        Returns
        -------
        An output dictionary consisting of:
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised. This is only returned when `correct_alternative` is not `None`.
        logits : ``torch.FloatTensor``
            The logits for every possible answer choice
        """
        x = self.embeddings(**text)
        x = self.transformer_stack(x, text["attention_mask"])
        x = self.lm_head(x)

        result = {"logits": x, "answers": x.argmax(1)}

        if label is not None:
            result["loss"] = self.loss(x, label)
            self.acc(x, label)

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        result = {"acc": self.acc.get_metric(reset)}
        return result
