import logging
from typing import Any, Dict, List, Optional

import torch
from torch.nn.functional import nll_loss

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention import MatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy

from allennlp_models.rc.dataset_readers.squad2 import SQUAD2_NO_ANSWER_TOKEN
from allennlp_models.rc.metrics import Squad2EmAndF1
from allennlp_models.rc.models.utils import get_best_span

logger = logging.getLogger(__name__)


@Model.register("bidaf_no_answer")
class BidirectionalAttentionFlowNoAnswer(Model):
    """
    This class implements Minjoon Seo's `Bidirectional Attention Flow model
    <https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Seo-Kembhavi/7586b7cca1deba124af80609327395e613a20e9d>`_
    for answering reading comprehension questions (ICLR 2017). In particular, it implements the no-answer
    variant used in Omer Levy's `Zero-Shot Relation Extraction via Reading Comprehension
    <https://www.semanticscholar.org/paper/Zero-Shot-Relation-Extraction-via-Reading-Levy-Seo/fa025e5d117929361bcf798437957762eb5bb6d4>`_`
    paper.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
    num_highway_layers : ``int``
        The number of highway layers to use in between embedding the input and passing it through
        the phrase layer.
    phrase_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and doing the bidirectional attention.
    matrix_attention : ``MatrixAttention``
        The attention function that we will use when comparing encoded passage and question
        representations.
    modeling_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between the bidirectional
        attention and predicting span start and end.
    span_end_encoder : ``Seq2SeqEncoder``
        The encoder that we will use to incorporate span start predictions into the passage state
        before predicting span end.
    dropout : ``float``, optional (default=0.2)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    no_answer_threshold : ``float``, optional (default=None)
        If the probability of no answer is greater than this value, predict no answer, regardless of whether
        it is the highest-scoring span or not.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        num_highway_layers: int,
        phrase_layer: Seq2SeqEncoder,
        matrix_attention: MatrixAttention,
        modeling_layer: Seq2SeqEncoder,
        span_end_encoder: Seq2SeqEncoder,
        dropout: float = 0.2,
        no_answer_threshold: float = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._highway_layer = TimeDistributed(
            Highway(text_field_embedder.get_output_dim(), num_highway_layers)
        )
        self._phrase_layer = phrase_layer
        self._modeling_layer = modeling_layer
        self._span_end_encoder = span_end_encoder

        encoding_dim = phrase_layer.get_output_dim()
        modeling_dim = modeling_layer.get_output_dim()
        span_start_input_dim = encoding_dim * 4 + modeling_dim
        self._matrix_attention = matrix_attention
        self._span_start_predictor = TimeDistributed(torch.nn.Linear(span_start_input_dim, 1))

        span_end_encoding_dim = span_end_encoder.get_output_dim()
        span_end_input_dim = encoding_dim * 4 + span_end_encoding_dim
        self._span_end_predictor = TimeDistributed(torch.nn.Linear(span_end_input_dim, 1))

        # Bidaf has lots of layer dimensions which need to match up - these aren't necessarily
        # obvious from the configuration files, so we check here.
        check_dimensions_match(
            modeling_layer.get_input_dim(),
            4 * encoding_dim,
            "modeling layer input dim",
            "4 * encoding dim",
        )
        check_dimensions_match(
            text_field_embedder.get_output_dim(),
            phrase_layer.get_input_dim(),
            "text field embedder output dim",
            "phrase layer input dim",
        )
        check_dimensions_match(
            span_end_encoder.get_input_dim(),
            4 * encoding_dim + 3 * modeling_dim,
            "span end encoder input dim",
            "4 * encoding dim + 3 * modeling dim",
        )

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad2_metrics = Squad2EmAndF1()
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        # If the probability of no answer goes above this value, then predict
        # "no answer". If None, ignore this.
        self.no_answer_threshold = no_answer_threshold

        initializer(self)

    def forward(  # type: ignore
        self,
        question: Dict[str, torch.LongTensor],
        passage: Dict[str, torch.LongTensor],
        span_start: torch.IntTensor = None,
        span_end: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        passage : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this passage contains the answer to the
            question, and predicts the beginning and ending positions of the answer within the
            passage.
        span_start : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            beginning position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        span_end : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            ending position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        metadata : ``List[Dict[str, Any]]``, optional
            metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question tokens, passage tokens, original passage
            text, and token offsets into the passage for each instance in the batch.  The length
            of this list should be the batch size, and each dictionary should have the keys
            ``question_tokens``, ``passage_tokens``, ``original_passage``, and ``token_offsets``.

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
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        best_span_str : List[str]
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
        """
        embedded_question = self._highway_layer(self._text_field_embedder(question))
        embedded_passage = self._highway_layer(self._text_field_embedder(passage))
        batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        question_mask = util.get_text_field_mask(question)
        passage_mask = util.get_text_field_mask(passage)
        question_lstm_mask = question_mask
        passage_lstm_mask = passage_mask

        encoded_question = self._dropout(self._phrase_layer(embedded_question, question_lstm_mask))
        encoded_passage = self._dropout(self._phrase_layer(embedded_passage, passage_lstm_mask))
        encoding_dim = encoded_question.size(-1)

        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = util.masked_softmax(passage_question_similarity, question_mask)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = util.replace_masked_values(
            passage_question_similarity, question_mask.unsqueeze(1), -1e7
        )
        # Shape: (batch_size, passage_length)
        question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # Shape: (batch_size, passage_length)
        question_passage_attention = util.masked_softmax(question_passage_similarity, passage_mask)
        # Shape: (batch_size, encoding_dim)
        question_passage_vector = util.weighted_sum(encoded_passage, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(
            batch_size, passage_length, encoding_dim
        )

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat(
            [
                encoded_passage,
                passage_question_vectors,
                encoded_passage * passage_question_vectors,
                encoded_passage * tiled_question_passage_vector,
            ],
            dim=-1,
        )

        modeled_passage = self._dropout(
            self._modeling_layer(final_merged_passage, passage_lstm_mask)
        )
        modeling_dim = modeled_passage.size(-1)

        # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim))
        span_start_input = self._dropout(torch.cat([final_merged_passage, modeled_passage], dim=-1))
        # Shape: (batch_size, passage_length)
        span_start_logits = self._span_start_predictor(span_start_input).squeeze(-1)
        # Shape: (batch_size, passage_length)
        span_start_probs = util.masked_softmax(span_start_logits, passage_mask)

        # Shape: (batch_size, modeling_dim)
        span_start_representation = util.weighted_sum(modeled_passage, span_start_probs)
        # Shape: (batch_size, passage_length, modeling_dim)
        tiled_start_representation = span_start_representation.unsqueeze(1).expand(
            batch_size, passage_length, modeling_dim
        )

        # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim * 3)
        span_end_representation = torch.cat(
            [
                final_merged_passage,
                modeled_passage,
                tiled_start_representation,
                modeled_passage * tiled_start_representation,
            ],
            dim=-1,
        )
        # Shape: (batch_size, passage_length, encoding_dim)
        encoded_span_end = self._dropout(
            self._span_end_encoder(span_end_representation, passage_lstm_mask)
        )
        # Shape: (batch_size, passage_length, encoding_dim * 4 + span_end_encoding_dim)
        span_end_input = self._dropout(torch.cat([final_merged_passage, encoded_span_end], dim=-1))
        span_end_logits = self._span_end_predictor(span_end_input).squeeze(-1)
        span_end_probs = util.masked_softmax(span_end_logits, passage_mask)
        span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e7)
        span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e7)

        best_span = get_best_span(span_start_logits, span_end_logits)
        # Calculate the probability of the best valid span.
        best_start_span_indices = best_span[:, 0]
        best_end_span_indices = best_span[:, 1]
        best_start_span_probs = span_start_probs.gather(
            1, best_start_span_indices.long().unsqueeze(-1)
        )
        best_end_span_probs = span_end_probs.gather(1, best_end_span_indices.long().unsqueeze(-1))
        best_span_probs = best_start_span_probs * best_end_span_probs

        # Get the best span among the valid answer spans (excluding the no-answer token).
        # Get the index of the no-answer token.
        no_answer_index = torch.sum(passage_mask, dim=-1) - 1

        # This is the same as passage_mask, except the last unmasked token (the no-answer token) is masked.
        valid_answer_mask = torch.ones_like(passage_mask)
        valid_answer_mask[:, no_answer_index.long()] = 0
        valid_answer_mask = passage_mask * valid_answer_mask
        best_valid_span = get_best_span(
            util.replace_masked_values(span_start_logits, valid_answer_mask, -1e7),
            util.replace_masked_values(span_end_logits, valid_answer_mask, -1e7),
        )
        # Calculate the probability of the best valid span.
        best_valid_start_span_indices = best_valid_span[:, 0]
        best_valid_end_span_indices = best_valid_span[:, 1]
        best_valid_start_span_probs = span_start_probs.gather(
            1, best_valid_start_span_indices.long().unsqueeze(-1)
        )
        best_valid_end_span_probs = span_end_probs.gather(
            1, best_valid_end_span_indices.long().unsqueeze(-1)
        )
        best_valid_span_probs = best_valid_start_span_probs * best_valid_end_span_probs

        # Calculate the log probability of no-answer
        no_answer_probs = span_start_probs.gather(
            1, no_answer_index.long().unsqueeze(-1)
        ) * span_end_probs.gather(1, no_answer_index.long().unsqueeze(-1))
        output_dict = {
            "passage_question_attention": passage_question_attention,
            "span_start_logits": span_start_logits,
            "span_start_probs": span_start_probs,
            "span_end_logits": span_end_logits,
            "span_end_probs": span_end_probs,
            "best_span": best_span,
            "best_span_probs": best_span_probs,
            "best_valid_span": best_valid_span,
            "best_valid_span_probs": best_valid_span_probs,
            "no_answer_probs": no_answer_probs,
        }

        # Compute the loss for training.
        if span_start is not None and span_end is not None:
            loss = nll_loss(
                util.masked_log_softmax(span_start_logits, passage_mask), span_start.squeeze(-1)
            )
            self._span_start_accuracy(span_start_logits, span_start.squeeze(-1))
            loss += nll_loss(
                util.masked_log_softmax(span_end_logits, passage_mask), span_end.squeeze(-1)
            )
            self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
            self._span_accuracy(best_span, torch.cat([span_start, span_end], -1))
            output_dict["loss"] = loss

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        if metadata is not None:
            output_dict["best_span_str"] = []
            question_tokens = []
            passage_tokens = []
            token_offsets = []
            for i in range(batch_size):
                question_tokens.append(metadata[i]["question_tokens"])
                passage_tokens.append(metadata[i]["passage_tokens"])
                token_offsets.append(metadata[i]["token_offsets"])
                passage_str = metadata[i]["original_passage"]
                offsets = metadata[i]["token_offsets"]
                no_answer_probability = no_answer_probs[i].item()
                if (
                    self.no_answer_threshold is not None
                    and self.no_answer_threshold < no_answer_probability
                ):
                    best_span_string = SQUAD2_NO_ANSWER_TOKEN
                else:
                    predicted_span = tuple(best_span[i].detach().cpu().numpy())
                    start_offset = offsets[predicted_span[0]][0]
                    end_offset = offsets[predicted_span[1]][1]
                    best_span_string = (passage_str + " " + SQUAD2_NO_ANSWER_TOKEN)[
                        start_offset:end_offset
                    ]
                output_dict["best_span_str"].append(best_span_string)
                answer_texts = metadata[i].get("answer_texts", [])
                if answer_texts:
                    self._squad2_metrics(best_span_string, answer_texts)
            output_dict["question_tokens"] = question_tokens
            output_dict["passage_tokens"] = passage_tokens
            output_dict["token_offsets"] = token_offsets
            if self.no_answer_threshold is not None:
                output_dict["no_answer_threshold"] = self.no_answer_threshold
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad2_metrics.get_metric(reset)
        return {
            "start_acc": self._span_start_accuracy.get_metric(reset),
            "end_acc": self._span_end_accuracy.get_metric(reset),
            "span_acc": self._span_accuracy.get_metric(reset),
            "em": exact_match,
            "f1": f1_score,
        }
