from typing import Dict, List, Optional

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from .beam_search_generator import BeamSearchGenerator


@BeamSearchGenerator.register("transformer")
class TransformerBeamSearchGenerator(BeamSearchGenerator):
    """
    A `BeamSearchGenerator` for transformer-based `NextTokenLM` models.

    This can be used with any `NextTokenLM` that utilizes a single `pretrained_transformer`
    `TokenEmbedder` for it's `text_field_embedder`.
    """

    def __init__(self, *args, namespace: str = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._namespace: Optional[str] = namespace

    @overrides
    def validate_text_field_embedder(self, text_field_embedder: TextFieldEmbedder):
        assert isinstance(text_field_embedder, BasicTextFieldEmbedder)
        assert len(text_field_embedder._token_embedders) == 1
        key = list(text_field_embedder._token_embedders.keys())[0]
        assert isinstance(text_field_embedder._token_embedders[key], PretrainedTransformerEmbedder)
        self._namespace = key

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

        # The model expect input in the form of TextFieldTensors, which just has another
        # nested layer like this:
        assert self._namespace is not None, (
            "token embedder namespace could not be inferred, "
            "did you forget to call 'validate_text_field_embedder()'?"
        )
        inputs = {self._namespace: state}

        return inputs
