from typing import Dict, Tuple

from overrides import overrides
import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator
from allennlp.training.metrics import Perplexity
from allennlp_models.lm.modules.language_model_heads import LanguageModelHead
from allennlp_models.lm.util import BeamSearchGenerator


@Model.register("next_token_lm")
class NextTokenLM(Model):
    """
    The `NextTokenLM` embeds some input tokens, contextualizes them, then predicts the next word,
    computing a loss against known target.

    If `BeamSearch` is given, this model will predict a sequence of next tokens.

    !!! NOTE
        This was developed for use in a demo, not for training.  You *definitely* don't want to
        train a language model using this code; it would be incredibly inefficient. But it does
        compute correct gradients of the loss, however, so you can use it for interesting visualization
        of the gradients of a pretrained model, and it appears to be fast enough to sample from, at
        least for one word at a time.

    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the indexed tokens we get in `forward`.
    language_model_head : `LanguageModelHead`
        The `torch.nn.Module` that goes from the hidden states output by the contextualizer to
        logits over some output vocabulary.
    contextualizer : `Seq2SeqEncoder`, optional (default=`None`)
        Used to "contextualize" the embeddings.  This is optional because the contextualization
        might actually be done in the text field embedder.
    target_namespace : `str`, optional (default=`'bert'`)
        Namespace to use to convert predicted token ids to strings in
        `Model.make_output_human_readable`.
    dropout : `float`, optional (default=`0.0`)
        If specified, dropout is applied to the contextualized embeddings before computation of
        the softmax. The contextualized embeddings themselves are returned without dropout.
    n_best : `int`, optional (default = `5`)
        The number of best tokens to predict. If `beam_search` is given, this option is ignored.
    beam_search_generator : `BeamSearchGenerator`, optional (default = `None`)
        An optional `BeamSearchGenerator`. If given, the model will predict sequences of next
        tokens instead of just a single next token.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        language_model_head: LanguageModelHead,
        contextualizer: Seq2SeqEncoder = None,
        target_namespace: str = "bert",
        dropout: float = 0.0,
        initializer: InitializerApplicator = None,
        n_best: int = 5,
        beam_search_generator: BeamSearchGenerator = None,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder
        self._contextualizer = contextualizer
        if contextualizer:
            check_dimensions_match(
                text_field_embedder.get_output_dim(),
                contextualizer.get_input_dim(),
                "text field embedder output",
                "contextualizer input",
            )
        self._language_model_head = language_model_head
        self._target_namespace = target_namespace
        self._perplexity = Perplexity()
        self._dropout = torch.nn.Dropout(dropout)
        self._n_best = n_best
        self._beam_search_generator = beam_search_generator

        # Ensure beam_search_generator is compatable with text_field_embedder.
        if self._beam_search_generator is not None:
            self._beam_search_generator.validate_text_field_embedder(self._text_field_embedder)

        if initializer is not None:
            initializer(self)

    def forward(  # type: ignore
        self, tokens: TextFieldTensors, target_ids: TextFieldTensors = None
    ) -> Dict[str, torch.Tensor]:
        """
        Run a forward pass of the model, returning an output tensor dictionary with
        the following fields:

        - `"probabilities"`: a tensor of shape `(batch_size, n_best)` representing
          the probabilities of the predicted tokens, where `n_best`
          is either `self._n_best` or `beam_size` if using beam search.
        - `"top_indices"`: a tensor of shape `(batch_size, n_best, num_predicted_tokens)`
          containing the IDs of the predicted tokens, where `num_predicted_tokens` is just
          1 unless using beam search, in which case it depends on the parameters of the beam search.
        - `"token_ids"`: a tensor of shape `(batch_size, num_input_tokens)` containing the IDs
          of the input tokens.
        - `"loss"` (optional): the loss of the batch, only given if `target_ids` is not `None`.

        """
        output_dict = {
            "token_ids": util.get_token_ids_from_text_field_tensors(tokens),
        }

        # Shape: (batch_size, vocab_size)
        target_logits = self._next_token_scores(tokens)

        # Compute loss.
        if target_ids is not None:
            batch_size, vocab_size = target_logits.size()
            tmp = util.get_token_ids_from_text_field_tensors(target_ids)
            # In some scenarios, target_ids might be a topk list of token ids (e.g. sorted by probabilities).
            # Therefore, we need to make sure only one token per batch
            # Assume: first token in each batch is the most desirable one (e.g. highest probability)
            tmp = tmp[:, 0] if len(tmp.shape) == 2 else tmp
            assert len(tmp.shape) <= 2
            targets = tmp.view(batch_size)
            loss = torch.nn.functional.cross_entropy(target_logits, targets)
            self._perplexity(loss)
            output_dict["loss"] = loss

        if self._beam_search_generator is not None:
            # Dummy start predictions.
            # Shape: (batch_size,)
            start_predictions = torch.zeros(
                target_logits.size()[0], device=target_logits.device, dtype=torch.int
            )

            state = self._beam_search_generator.get_step_state(tokens)

            # Put this in here to avoid having to re-compute on the first step of beam search.
            state["start_target_logits"] = target_logits

            # Shape (top_indices): (batch_size, beam_size, num_predicted_tokens)
            # Shape (top_log_probs): (batch_size, beam_size)
            top_indices, top_log_probs = self._beam_search_generator.search(
                start_predictions, state, self._beam_search_step
            )

            # Shape: (batch_size, beam_size)
            top_probs = top_log_probs.exp()
        else:
            # Shape: (batch_size, vocab_size)
            probs = torch.nn.functional.softmax(target_logits, dim=-1)

            # Shape (both): (batch_size, n_best)
            # min here largely because tests use small vocab
            top_probs, top_indices = probs.topk(k=min(target_logits.size(-1), self._n_best), dim=-1)

            # Shape: (batch_size, n_best, 1)
            top_indices = top_indices.unsqueeze(-1)

        output_dict["top_indices"] = top_indices
        output_dict["probabilities"] = top_probs

        return output_dict

    def _next_token_scores(self, tokens: TextFieldTensors) -> torch.Tensor:
        """
        Get the unnormalized log probabilities of the potential next token.
        """
        # Shape: (batch_size, num_tokens, embedding_dim)
        embeddings = self._text_field_embedder(tokens)

        # Shape: (batch_size, num_tokens, encoding_dim)
        if self._contextualizer:
            mask = util.get_text_field_mask(embeddings)
            contextual_embeddings = self._contextualizer(embeddings, mask)
            final_embeddings = util.get_final_encoder_states(contextual_embeddings, mask)
        else:
            final_embeddings = embeddings[:, -1]

        # Shape: (batch_size, vocab_size)
        return self._language_model_head(self._dropout(final_embeddings))

    def _beam_search_step(
        self, predicted_tokens: torch.Tensor, state: Dict[str, torch.Tensor], step: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Step function to use with `BeamSearch`.

        `predicted_tokens` is a tensor of shape `(group_size,)` and
        `state` is a dictionary of tensors with the following fields:
        - "token_ids": shape `(group_size, num_tokens)`
        - "mask": shape `(group_size, num_tokens)`
        - "type_ids": shape `(group_size, num_tokens)`
        """
        assert self._beam_search_generator is not None

        if step == 0:
            # Shape: (group_size, vocab_size)
            start_target_logits = state.pop("start_target_logits")

            # Shape: (group_size, vocab_size)
            start_target_log_probs = torch.nn.functional.log_softmax(start_target_logits, dim=-1)

            return start_target_log_probs, state

        inputs = self._beam_search_generator.prepare_step_input(predicted_tokens, state)
        state = self._beam_search_generator.get_step_state(inputs)

        # Shape: (group_size, vocab_size)
        next_token_scores = self._next_token_scores(inputs)

        # Shape: (group_size, vocab_size)
        log_probs = torch.nn.functional.log_softmax(next_token_scores, dim=-1)

        return log_probs, state

    def get_metrics(self, reset: bool = False):
        return {"perplexity": self._perplexity.get_metric(reset=reset)}

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Collects token strings from indices, adding two fields to the `output_dict`:

        - `"top_tokens"`: a list (for each instance in the batch) of lists (for each of
          the `n` best predictions) of lists of strings (for each token in each prediction).
        - `"tokens"`: a list of list (for each instance in the batch) of strings (for each
          input token in the instance).
        """
        # Gather predicted words.
        top_tokens = []
        # shape (output_dict["top_indices"]): (batch_size, n_best, num_predicted_tokens)
        for instance in output_dict["top_indices"]:
            # shape (instance): (n_best, num_predicted_tokens)
            instance_top_words = []
            for indices in instance:
                # shape (indices): (num_predicted_tokens,)
                instance_top_words.append(
                    [
                        self.vocab.get_token_from_index(
                            index.item(), namespace=self._target_namespace
                        )
                        for index in indices
                    ]
                )
            top_tokens.append(instance_top_words)

        # Gather input tokens.
        tokens = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self.vocab.get_token_from_index(
                        token_id.item(), namespace=self._target_namespace
                    )
                    for token_id in instance_tokens
                ]
            )

        output_dict["top_tokens"] = top_tokens  # type: ignore
        output_dict["tokens"] = tokens  # type: ignore
        return output_dict

    default_predictor = "next_token_lm"
