import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from allennlp.common.util import sanitize_wordpiece
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.nn.util import get_token_ids_from_text_field_tensors
from torch import nn

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy
from torch.nn.functional import cross_entropy

from allennlp_models.rc.models.utils import (
    get_best_span,
    replace_masked_values_with_big_negative_number,
)
from allennlp_models.rc.metrics import SquadEmAndF1

logger = logging.getLogger(__name__)


@Model.register("transformer_qa")
class TransformerQA(Model):
    """
    This class implements a reading comprehension model patterned after the proposed model in
    https://arxiv.org/abs/1810.04805 (Devlin et al), with improvements borrowed from the SQuAD model in the
    transformers project.

    It predicts start tokens and end tokens with a linear layer on top of word piece embeddings.

    Note that the metrics that the model produces are calculated on a per-instance basis only. Since there could
    be more than one instance per question, these metrics are not the official numbers on the SQuAD task. To get
    official numbers, run the script in scripts/transformer_qa_eval.py.

    Parameters
    ----------
    vocab : ``Vocabulary``
    transformer_model_name : ``str``, optional (default=``bert-base-cased``)
        This model chooses the embedder according to this setting. You probably want to make sure this is set to
        the same thing as the reader.
    """

    def __init__(
        self, vocab: Vocabulary, transformer_model_name: str = "bert-base-cased", **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._text_field_embedder = BasicTextFieldEmbedder(
            {"tokens": PretrainedTransformerEmbedder(transformer_model_name)}
        )
        self._linear_layer = nn.Linear(self._text_field_embedder.get_output_dim(), 2)

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._per_instance_metrics = SquadEmAndF1()

    def forward(  # type: ignore
        self,
        question_with_context: Dict[str, Dict[str, torch.LongTensor]],
        context_span: torch.IntTensor,
        answer_span: Optional[torch.IntTensor] = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        """
        Parameters
        ----------
        question_with_context : Dict[str, torch.LongTensor]
            From a ``TextField``. The model assumes that this text field contains the context followed by the
            question. It further assumes that the tokens have type ids set such that any token that can be part of
            the answer (i.e., tokens from the context) has type id 0, and any other token (including [CLS] and
            [SEP]) has type id 1.
        context_span : ``torch.IntTensor``
            From a ``SpanField``. This marks the span of word pieces in ``question`` from which answers can come.
        answer_span : ``torch.IntTensor``, optional
            From a ``SpanField``. This is the thing we are trying to predict - the span of text that marks the
            answer. If given, we compute a loss that gets included in the output directory.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question id, and the original texts of context, question, tokenized
            version of both, and a list of possible answers. The length of the ``metadata`` list should be the
            batch size, and each dictionary should have the keys ``id``, ``question``, ``context``,
            ``question_tokens``, ``context_tokens``, and ``answers``.

        Returns
        -------
        An output dictionary consisting of:
        span_start_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span start position.
        span_start_probs : torch.FloatTensor
            The result of ``softmax(span_start_logits)``.
        span_end_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span end position (inclusive).
        span_end_probs : torch.FloatTensor
            The result of ``softmax(span_end_logits)``.
        best_span : torch.IntTensor
            The result of a constrained inference over ``span_start_logits`` and
            ``span_end_logits`` to find the most probable span.  Shape is ``(batch_size, 2)``
            and each offset is a token index.
        best_span_scores : torch.FloatTensor
            The score for each of the best spans.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        best_span_str : List[str]
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
        """
        embedded_question = self._text_field_embedder(question_with_context)
        logits = self._linear_layer(embedded_question)
        span_start_logits, span_end_logits = logits.split(1, dim=-1)
        span_start_logits = span_start_logits.squeeze(-1)
        span_end_logits = span_end_logits.squeeze(-1)

        possible_answer_mask = torch.zeros_like(
            get_token_ids_from_text_field_tensors(question_with_context), dtype=torch.bool
        )
        for i, (start, end) in enumerate(context_span):
            possible_answer_mask[i, start : end + 1] = True

        # Replace the masked values with a very negative constant.
        span_start_logits = replace_masked_values_with_big_negative_number(
            span_start_logits, possible_answer_mask
        )
        span_end_logits = replace_masked_values_with_big_negative_number(
            span_end_logits, possible_answer_mask
        )
        span_start_probs = torch.nn.functional.softmax(span_start_logits, dim=-1)
        span_end_probs = torch.nn.functional.softmax(span_end_logits, dim=-1)
        best_spans = get_best_span(span_start_logits, span_end_logits)
        best_span_scores = torch.gather(
            span_start_logits, 1, best_spans[:, 0].unsqueeze(1)
        ) + torch.gather(span_end_logits, 1, best_spans[:, 1].unsqueeze(1))
        best_span_scores = best_span_scores.squeeze(1)

        output_dict = {
            "span_start_logits": span_start_logits,
            "span_start_probs": span_start_probs,
            "span_end_logits": span_end_logits,
            "span_end_probs": span_end_probs,
            "best_span": best_spans,
            "best_span_scores": best_span_scores,
        }

        # Compute the loss for training.
        if answer_span is not None:
            span_start = answer_span[:, 0]
            span_end = answer_span[:, 1]
            span_mask = span_start != -1
            self._span_accuracy(
                best_spans, answer_span, span_mask.unsqueeze(-1).expand_as(best_spans)
            )

            start_loss = cross_entropy(span_start_logits, span_start, ignore_index=-1)
            big_constant = min(torch.finfo(start_loss.dtype).max, 1e9)
            if torch.any(start_loss > big_constant):
                logger.critical("Start loss too high (%r)", start_loss)
                logger.critical("span_start_logits: %r", span_start_logits)
                logger.critical("span_start: %r", span_start)
                assert False

            end_loss = cross_entropy(span_end_logits, span_end, ignore_index=-1)
            if torch.any(end_loss > big_constant):
                logger.critical("End loss too high (%r)", end_loss)
                logger.critical("span_end_logits: %r", span_end_logits)
                logger.critical("span_end: %r", span_end)
                assert False

            loss = (start_loss + end_loss) / 2

            self._span_start_accuracy(span_start_logits, span_start, span_mask)
            self._span_end_accuracy(span_end_logits, span_end, span_mask)

            output_dict["loss"] = loss

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        if metadata is not None:
            best_spans = best_spans.detach().cpu().numpy()

            output_dict["best_span_str"] = []
            context_tokens = []
            for metadata_entry, best_span, cspan in zip(metadata, best_spans, context_span):
                context_tokens_for_question = metadata_entry["context_tokens"]
                context_tokens.append(context_tokens_for_question)

                best_span -= int(cspan[0])
                assert np.all(best_span >= 0)

                predicted_start, predicted_end = tuple(best_span)

                while (
                    predicted_start >= 0
                    and context_tokens_for_question[predicted_start].idx is None
                ):
                    predicted_start -= 1
                if predicted_start < 0:
                    logger.warning(
                        f"Could not map the token '{context_tokens_for_question[best_span[0]].text}' at index "
                        f"'{best_span[0]}' to an offset in the original text."
                    )
                    character_start = 0
                else:
                    character_start = context_tokens_for_question[predicted_start].idx

                while (
                    predicted_end < len(context_tokens_for_question)
                    and context_tokens_for_question[predicted_end].idx is None
                ):
                    predicted_end += 1
                if predicted_end >= len(context_tokens_for_question):
                    logger.warning(
                        f"Could not map the token '{context_tokens_for_question[best_span[1]].text}' at index "
                        f"'{best_span[1]}' to an offset in the original text."
                    )
                    character_end = len(metadata_entry["context"])
                else:
                    end_token = context_tokens_for_question[predicted_end]
                    character_end = end_token.idx + len(sanitize_wordpiece(end_token.text))

                best_span_string = metadata_entry["context"][character_start:character_end]
                output_dict["best_span_str"].append(best_span_string)

                answers = metadata_entry.get("answers")
                if len(answers) > 0:
                    self._per_instance_metrics(best_span_string, answers)
            output_dict["context_tokens"] = context_tokens
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._per_instance_metrics.get_metric(reset)
        return {
            "start_acc": self._span_start_accuracy.get_metric(reset),
            "end_acc": self._span_end_accuracy.get_metric(reset),
            "span_acc": self._span_accuracy.get_metric(reset),
            "per_instance_em": exact_match,
            "per_instance_f1": f1_score,
        }

    default_predictor = "transformer_qa"
