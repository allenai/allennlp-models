from math import floor
from typing import Sequence, Dict, Any, Optional, Iterator

import more_itertools
import torch
from allennlp.common import cached_transformers
from allennlp.data import Instance, TensorDict, allennlp_collate
from allennlp.tango.dataloader import TangoDataLoader


@TangoDataLoader.register("masked_lm")
class MaskedLMDataLoader(TangoDataLoader):
    def __init__(
        self,
        instances: Sequence[Instance],
        *,
        batch_size: int,
        tokenizer_name: str,
        input_text_field_name: str = "text",
        output_text_field_name: str = "text",
        output_label_field_name: str = "label",
        masking_fraction: float = 0.15
    ):
        self.instances = instances
        self.batch_size = batch_size
        self.input_text_field_name = input_text_field_name
        self.output_text_field_name = output_text_field_name
        self.output_label_field_name = output_label_field_name
        self.masking_fraction = masking_fraction

        tokenizer = cached_transformers.get_tokenizer(tokenizer_name)
        self.mask_token_id = tokenizer.pad_token_id
        self.mask_token_type_id = tokenizer.pad_token_type_id

    def _to_params(self) -> Dict[str, Any]:
        return {
            # We're not returning instances here.
            "batch_size": self.batch_size,
            "input_text_field_name": self.input_text_field_name,
            "output_text_field_name": self.output_text_field_name,
            "output_label_field_name": self.output_label_field_name,
            "masking_fraction": self.masking_fraction,
        }

    def num_batches_per_epoch(self) -> Optional[int]:
        return floor(len(self.instances) / self.batch_size)

    def _mask_batch(self, batch: TensorDict) -> TensorDict:
        input_tensors = batch[self.input_text_field_name]
        if not isinstance(input_tensors, dict) or "input_ids" not in input_tensors:
            raise ValueError(f"The input field to {self.__class__.__name__} is expected to be a TransformerTextField.")

        new_batch = batch.copy()
        new_batch[self.output_label_field_name] = input_tensors["token_ids"]

        masked_text_tensors = input_tensors.copy()
        mask_mask = (torch.rand_like(input_tensors["input_ids"]) * input_tensors["attention_mask"]) > self.masking_fraction
        masked_text_tensors["input_ids"] = input_tensors["input_ids"].clone()
        masked_text_tensors["input_ids"][mask_mask] = self.mask_token_id
        if "token_type_ids" in input_tensors:
            masked_text_tensors["token_type_ids"] = input_tensors["token_type_ids"].clone()
            masked_text_tensors["token_type_ids"][mask_mask] = self.mask_token_type_id
        new_batch[self.output_text_field_name] = masked_text_tensors

        return new_batch

    def __iter__(self) -> Iterator[TensorDict]:
        for batch in more_itertools.chunked(self.instances, self.batch_size):
            if len(batch) >= self.batch_size:
                yield self._mask_batch(allennlp_collate(batch))
