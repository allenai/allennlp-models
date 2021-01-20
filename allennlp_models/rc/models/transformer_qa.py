import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.functional import cross_entropy

from allennlp.common.util import sanitize_wordpiece
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.nn.util import get_token_ids_from_text_field_tensors
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy

from allennlp_models.rc.models.utils import (
    get_best_span,
    replace_masked_values_with_big_negative_number,
)
from allennlp_models.rc.metrics import SquadEmAndF1

logger = logging.getLogger(__name__)


@Model.register("transformer_qa")
class TransformerQA(Model):
    """
    Registered as `"transformer_qa"`, this class implements a reading comprehension model patterned
    after the proposed model in [Devlin et al](git@github.com:huggingface/transformers.git),
    with improvements borrowed from the SQuAD model in the transformers project.

    It predicts start tokens and end tokens with a linear layer on top of word piece embeddings.

    If you want to use this model on SQuAD datasets, you can use it with the
    [`TransformerSquadReader`](../../dataset_readers/transformer_squad#transformersquadreader)
    dataset reader, registered as `"transformer_squad"`.

    Note that the metrics that the model produces are calculated on a per-instance basis only. Since there could
    be more than one instance per question, these metrics are not the official numbers on either SQuAD task.

    To get official numbers for SQuAD v1.1, for example, you can run

    ```
    python -m allennlp_models.rc.tools.transformer_qa_eval
    ```

    # Parameters

    vocab : `Vocabulary`

    transformer_model_name : `str`, optional (default=`'bert-base-cased'`)
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
        cls_index: torch.LongTensor = None,
        answer_span: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        # Parameters

        question_with_context : `Dict[str, torch.LongTensor]`
            From a `TextField`. The model assumes that this text field contains the context followed by the
            question. It further assumes that the tokens have type ids set such that any token that can be part of
            the answer (i.e., tokens from the context) has type id 0, and any other token (including
            `[CLS]` and `[SEP]`) has type id 1.

        context_span : `torch.IntTensor`
            From a `SpanField`. This marks the span of word pieces in `question` from which answers can come.

        cls_index : `torch.LongTensor`, optional
            A tensor of shape `(batch_size,)` that provides the index of the `[CLS]` token
            in the `question_with_context` for each instance.

            This is needed because the `[CLS]` token is used to indicate that the question
            is impossible.

            If this is `None`, it's assumed that the `[CLS]` token is at index 0 for each instance
            in the batch.

        answer_span : `torch.IntTensor`, optional
            From a `SpanField`. This is the thing we are trying to predict - the span of text that marks the
            answer. If given, we compute a loss that gets included in the output directory.

        metadata : `List[Dict[str, Any]]`, optional
            If present, this should contain the question id, and the original texts of context, question, tokenized
            version of both, and a list of possible answers. The length of the `metadata` list should be the
            batch size, and each dictionary should have the keys `id`, `question`, `context`,
            `question_tokens`, `context_tokens`, and `answers`.

        # Returns

        `Dict[str, torch.Tensor]` :
            An output dictionary with the following fields:

            - span_start_logits (`torch.FloatTensor`) :
              A tensor of shape `(batch_size, passage_length)` representing unnormalized log
              probabilities of the span start position.
            - span_end_logits (`torch.FloatTensor`) :
              A tensor of shape `(batch_size, passage_length)` representing unnormalized log
              probabilities of the span end position (inclusive).
            - best_span_scores (`torch.FloatTensor`) :
              The score for each of the best spans.
            - loss (`torch.FloatTensor`, optional) :
              A scalar loss to be optimised, evaluated against `answer_span`.
            - best_span (`torch.IntTensor`, optional) :
              Provided when not in train mode and sufficient metadata given for the instance.
              The result of a constrained inference over `span_start_logits` and
              `span_end_logits` to find the most probable span.  Shape is `(batch_size, 2)`
              and each offset is a token index, unless the best span for an instance
              was predicted to be the `[CLS]` token, in which case the span will be (-1, -1).
            - best_span_str (`List[str]`, optional) :
              Provided when not in train mode and sufficient metadata given for the instance.
              This is the string from the original passage that the model thinks is the best answer
              to the question.

        """
        embedded_question = self._text_field_embedder(question_with_context)
        # shape: (batch_size, sequence_length, 2)
        logits = self._linear_layer(embedded_question)
        # shape: (batch_size, sequence_length, 1)
        span_start_logits, span_end_logits = logits.split(1, dim=-1)
        # shape: (batch_size, sequence_length)
        span_start_logits = span_start_logits.squeeze(-1)
        # shape: (batch_size, sequence_length)
        span_end_logits = span_end_logits.squeeze(-1)

        # Create a mask for `question_with_context` to mask out tokens that are not part
        # of the context.
        # shape: (batch_size, sequence_length)
        possible_answer_mask = torch.zeros_like(
            get_token_ids_from_text_field_tensors(question_with_context), dtype=torch.bool
        )
        for i, (start, end) in enumerate(context_span):
            possible_answer_mask[i, start : end + 1] = True
            # Also unmask the [CLS] token since that token is used to indicate that
            # the question is impossible.
            possible_answer_mask[i, 0 if cls_index is None else cls_index[i]] = True

        # Replace the masked values with a very negative constant since we're in log-space.
        # shape: (batch_size, sequence_length)
        span_start_logits = replace_masked_values_with_big_negative_number(
            span_start_logits, possible_answer_mask
        )
        # shape: (batch_size, sequence_length)
        span_end_logits = replace_masked_values_with_big_negative_number(
            span_end_logits, possible_answer_mask
        )

        # Now calculate the best span.
        # shape: (batch_size, 2)
        best_spans = get_best_span(span_start_logits, span_end_logits)

        # Sum the span start score with the span end score to get an overall score for the span.
        # shape: (batch_size,)
        best_span_scores = torch.gather(
            span_start_logits, 1, best_spans[:, 0].unsqueeze(1)
        ) + torch.gather(span_end_logits, 1, best_spans[:, 1].unsqueeze(1))
        best_span_scores = best_span_scores.squeeze(1)

        output_dict = {
            "span_start_logits": span_start_logits,
            "span_end_logits": span_end_logits,
            "best_span_scores": best_span_scores,
        }

        # Compute the loss.
        if answer_span is not None:
            output_dict["loss"] = self._evaluate_span(
                best_spans, span_start_logits, span_end_logits, answer_span
            )

        # Gather the string of the best span and compute the EM and F1 against the gold span,
        # if given.
        if not self.training and metadata is not None:
            (
                output_dict["best_span_str"],
                output_dict["best_span"],
            ) = self._collect_best_span_strings(best_spans, context_span, metadata, cls_index)

        return output_dict

    def _evaluate_span(
        self,
        best_spans: torch.Tensor,
        span_start_logits: torch.Tensor,
        span_end_logits: torch.Tensor,
        answer_span: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the loss against the `answer_span` and also update the span metrics.
        """
        span_start = answer_span[:, 0]
        span_end = answer_span[:, 1]
        self._span_accuracy(best_spans, answer_span)

        start_loss = cross_entropy(span_start_logits, span_start, ignore_index=-1)
        big_constant = min(torch.finfo(start_loss.dtype).max, 1e9)
        assert not torch.any(start_loss > big_constant), "Start loss too high"

        end_loss = cross_entropy(span_end_logits, span_end, ignore_index=-1)
        assert not torch.any(end_loss > big_constant), "End loss too high"

        self._span_start_accuracy(span_start_logits, span_start)
        self._span_end_accuracy(span_end_logits, span_end)

        return (start_loss + end_loss) / 2

    def _collect_best_span_strings(
        self,
        best_spans: torch.Tensor,
        context_span: torch.IntTensor,
        metadata: List[Dict[str, Any]],
        cls_index: Optional[torch.LongTensor],
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Collect the string of the best predicted span from the context metadata and
        update `self._per_instance_metrics`, which in the case of SQuAD v1.1 / v2.0
        includes the EM and F1 score.

        This returns a `Tuple[List[str], torch.Tensor]`, where the `List[str]` is the
        predicted answer for each instance in the batch, and the tensor is just the input
        tensor `best_spans` after adjustments so that each answer span corresponds to the
        context tokens only, and not the question tokens. Spans that correspond to the
        `[CLS]` token, i.e. the question was predicted to be impossible, will be set
        to `(-1, -1)`.
        """
        _best_spans = best_spans.detach().cpu().numpy()

        best_span_strings: List[str] = []
        best_span_strings_for_metric: List[str] = []
        answer_strings_for_metric: List[List[str]] = []

        for (metadata_entry, best_span, cspan, cls_ind) in zip(
            metadata,
            _best_spans,
            context_span,
            cls_index or (0 for _ in range(len(metadata))),
        ):
            context_tokens_for_question = metadata_entry["context_tokens"]

            if best_span[0] == cls_ind:
                # Predicting [CLS] is interpreted as predicting the question as unanswerable.
                best_span_string = ""
                # NOTE: even though we've "detached" 'best_spans' above, this still
                # modifies the original tensor in-place.
                best_span[0], best_span[1] = -1, -1
            else:
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

            best_span_strings.append(best_span_string)
            answers = metadata_entry.get("answers")
            if answers:
                best_span_strings_for_metric.append(best_span_string)
                answer_strings_for_metric.append(answers)

        if answer_strings_for_metric:
            self._per_instance_metrics(best_span_strings_for_metric, answer_strings_for_metric)

        return best_span_strings, best_spans

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        output = {
            "start_acc": self._span_start_accuracy.get_metric(reset),
            "end_acc": self._span_end_accuracy.get_metric(reset),
            "span_acc": self._span_accuracy.get_metric(reset),
        }
        if not self.training:
            exact_match, f1_score = self._per_instance_metrics.get_metric(reset)
            output["per_instance_em"] = exact_match
            output["per_instance_f1"] = f1_score
        return output

    default_predictor = "transformer_qa"
