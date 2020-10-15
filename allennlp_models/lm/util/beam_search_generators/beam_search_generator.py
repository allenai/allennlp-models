from typing import Dict, Tuple

import torch

from allennlp.common.registrable import Registrable
from allennlp.data import TextFieldTensors
from allennlp.nn.beam_search import BeamSearch, StepFunctionType


class BeamSearchGenerator(Registrable):
    """
    A beam search generator for next token language models.

    This is just a wrapper around `allennlp.nn.beam_search.BeamSearch` with custom
    logic for handling the `state` dict.
    """

    def __init__(self, beam_search: BeamSearch):
        self._beam_search = beam_search

    def get_start_state(self, tokens: TextFieldTensors) -> Dict[str, torch.Tensor]:
        """
        Get an initial state dictionary from the `tokens` input to the forward
        pass of a `NextTokenLm` model.
        """
        raise NotImplementedError

    def get_step_state(self, inputs: TextFieldTensors) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def prepare_step_input(
        self, predictions: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> TextFieldTensors:
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
