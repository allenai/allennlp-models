from typing import Dict, Tuple

import torch

from allennlp.common.registrable import Registrable
from allennlp.data import TextFieldTensors
from allennlp.modules import TextFieldEmbedder
from allennlp.nn.beam_search import BeamSearch, StepFunctionType


class BeamSearchGenerator(Registrable):
    """
    A beam search generator for next token language models.

    This is just a wrapper around `allennlp.nn.beam_search.BeamSearch` with custom
    logic for handling the `state` dict.

    The reason we need this is because the step function that `BeamSearch` uses
    needs to know how to handle different `TextFieldTensors`, the form of which
    depends on the exact embedder class that the `NextTokenLm` uses.

    So essentially we need a different `BeamSearchGenerator` implementation
    for each different `text_field_embedder`.
    """

    def __init__(self, beam_search: BeamSearch):
        self._beam_search = beam_search

    def validate_text_field_embedder(self, text_field_embedder: TextFieldEmbedder):
        """
        This should be called after initialization to verify that the model's
        `text_field_embedder` is compatable.
        """
        raise NotImplementedError

    def get_step_state(self, inputs: TextFieldTensors) -> Dict[str, torch.Tensor]:
        """
        Create a `state` dictionary for `BeamSearch` from the `TextFieldTensors` inputs
        to the `NextTokenLm` model.

        By default this assumes the `TextFieldTensors` has a single `TokenEmbedder`,
        and just "flattens" the `TextFieldTensors` by returning the `TokenEmbedder`
        sub-dictionary.

        If you have `TextFieldTensors` with more than one `TokenEmbedder` sub-dictionary,
        you'll need to override this class.
        """
        assert len(inputs) == 1, (
            "'get_step_state()' assumes a single token embedder by default, "
            "you'll need to override this method to handle more than one"
        )

        key = list(inputs.keys())[0]

        # We can't just `return inputs[key]` because we might want to modify the state
        # dictionary (add or remove fields) without accidentally modifying the inputs
        # dictionary.
        return {k: v for (k, v) in inputs[key].items()}

    def prepare_step_input(
        self, predictions: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> TextFieldTensors:
        """
        This is like the reverse of `get_step_state()`.

        It takes `predictions` and `state` from the current step and returns
        a `TextFieldTensors` dictionary that can be fed through the embedder of the `NextTokenLm`
        model.

        This usually involves adding the predicted tokens to the proper field of the `state` dict,
        and expanding any mask tensors or other context tensors by 1 in the right dimension,
        and then unflattening the `state` so that it looks like a `TextFieldTensors` dict.
        """
        raise NotImplementedError

    def search(
        self,
        start_predictions: torch.Tensor,
        state: Dict[str, torch.Tensor],
        step_function: StepFunctionType,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calls `BeamSearch.search`, return the top predicted indices and corresponding
        log probabilities.
        """
        # Shape (top_indices): (batch_size, beam_size, num_predicted_tokens)
        # Shape (top_log_probs): (batch_size, beam_size)
        top_indices, top_log_probs = self._beam_search.search(
            start_predictions, state, step_function
        )
        return top_indices, top_log_probs
