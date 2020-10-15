from typing import Dict, List

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors
from allennlp.nn.beam_search import BeamSearch
from .beam_search_generator import BeamSearchGenerator


@BeamSearchGenerator.register("transformer")
class TransformerBeamSearchGenerator(BeamSearchGenerator):
    """
    A `BeamSearchGenerator` for transformer-based `NextTokenLM` models.
    """

    def __init__(self, namespace: str, beam_search: BeamSearch) -> None:
        super().__init__(beam_search)
        self._namespace = namespace

    @overrides
    def get_start_state(
        self,
        tokens: TextFieldTensors,
    ) -> Dict[str, torch.Tensor]:
        return {k: v for (k, v) in tokens[self._namespace].items()}

    @overrides
    def get_step_state(self, inputs: TextFieldTensors) -> Dict[str, torch.Tensor]:
        return inputs[self._namespace]

    @overrides
    def prepare_step_input(
        self, predictions: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> TextFieldTensors:
        # Add `predicted_tokens` to `state["token_ids"]` and expand `state["mask"]`.
        new_token_ids: List[torch.Tensor] = []
        new_mask: List[torch.Tensor] = []
        for instance_token_ids, instance_mask, prediction in zip(
            state["token_ids"], state["mask"], predictions
        ):
            # Shape: (?,)
            masked_out = (instance_mask == False).nonzero(as_tuple=False).squeeze(-1)  # noqa: E712
            if masked_out.size()[0] > 0:
                first_mask_index = masked_out[0].item()
            else:
                first_mask_index = instance_token_ids.size()[0]

            # Shape: (batch_size, num_tokens + 1)
            new_instance_token_ids = torch.cat(
                [
                    instance_token_ids[0:first_mask_index],
                    prediction.unsqueeze(0),
                    instance_token_ids[first_mask_index:],
                ],
                dim=-1,
            )

            # Shape: (batch_size, num_tokens + 1)
            new_instance_mask = torch.cat(
                [
                    instance_mask[0:first_mask_index],
                    torch.tensor([True], device=instance_mask.device),
                    instance_mask[first_mask_index:],
                ],
                dim=-1,
            )

            new_token_ids.append(new_instance_token_ids)
            new_mask.append(new_instance_mask)

        state["token_ids"] = torch.stack(new_token_ids, 0)
        state["mask"] = torch.stack(new_mask, 0)

        # Expand `state["type_ids"]` by 1 in the last dimension, just repeating whatever the last
        # value is.
        # Shape: (group_size, num_tokens)
        type_ids = state.pop("type_ids")
        # Shape: (group_size, num_tokens + 1)
        state["type_ids"] = torch.cat([type_ids, type_ids[:, -1].unsqueeze(-1)], dim=-1)

        inputs = {self._namespace: state}

        return inputs
