import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.models import Model
from allennlp.nn.util import sequence_cross_entropy_with_logits
from overrides import overrides
from transformers import BartForConditionalGeneration
from typing import Any, Dict, List, Tuple
from allennlp.nn.beam_search import BeamSearch

SPAN_START_TOKEN = '<m>'
SPAN_END_TOKEN = '</m>'
ALL_SPECIAL_TOKENS = [SPAN_START_TOKEN, SPAN_END_TOKEN]


@Model.register('squad_question_generation')
class QuestionGenerationModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 model_name: str,
                 max_decoding_steps: int = 50,
                 beam_size: int = 4) -> None:
        super().__init__(vocab)
        self.bart = BartForConditionalGeneration.from_pretrained(model_name, output_past=True)
        self.tokenizer = PretrainedTransformerTokenizer(model_name)

        # Increase the size of Bart's vocabulary to account for the new special
        # tokens that were added. Method found from https://github.com/huggingface/transformers/issues/3446
        # comment on June 12.
        vocab_size = self.bart.config.vocab_size
        self.bart.resize_token_embeddings(vocab_size + len(ALL_SPECIAL_TOKENS))

        self._start_id = self.bart.config.bos_token_id
        self._end_id = self.bart.config.eos_token_id
        self._pad_id = self.bart.config.pad_token_id

        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(
            self._end_id, max_steps=max_decoding_steps, beam_size=beam_size or 1
        )

    @overrides
    def forward(self,
                source_tokens: TextFieldTensors,
                metadata: List[Dict[str, Any]],
                target_tokens: TextFieldTensors = None) -> Dict[str, Any]:
        source_ids = source_tokens['tokens']['token_ids']
        source_mask = source_tokens['tokens']['mask']

        output_dict = {'metadata': metadata}
        if target_tokens is not None:
            # Calculate loss
            target_ids = target_tokens['tokens']['token_ids']
            target_mask = target_tokens['tokens']['mask']

            logits = self.bart(
                input_ids=source_ids,
                attention_mask=source_mask,
                decoder_input_ids=target_ids[:, :-1].contiguous(),
                decoder_attention_mask=target_mask[:, :-1].contiguous(),
                use_cache=False,
            )[0]

            # The BART paper mentions label smoothing of 0.1 for sequence generation tasks
            loss = sequence_cross_entropy_with_logits(
                logits,
                target_ids[:, 1:].contiguous(),
                target_mask[:, 1:].contiguous(),
                label_smoothing=0.1,
                average='token'
            )
            output_dict['loss'] = loss

        if not self.training:
            # Run inference: This differs from the original code which
            # includes the decoder_start_id
            initial_decoder_ids = torch.tensor(
                [[self._start_id]],
                dtype=source_ids.dtype,
                device=source_ids.device,
            ).repeat(source_ids.shape[0], 1)

            inital_state = {
                "input_ids": source_ids,
                "input_mask": source_mask,
                "encoder_states": None,
            }
            beam_result = self._beam_search.search(
                initial_decoder_ids, inital_state, self.take_step
            )

            predictions = beam_result[0]
            max_pred_indices = (
                beam_result[1].argmax(dim=-1).view(-1, 1, 1).expand(-1, -1, predictions.shape[-1])
            )
            predictions = predictions.gather(dim=1, index=max_pred_indices).squeeze(dim=1)

            output_dict["predicted_ids"] = predictions
            output_dict["log_probabilities"] = (
                beam_result[1].gather(dim=-1, index=max_pred_indices[..., 0]).squeeze(dim=-1)
            )

            self.make_output_human_readable(output_dict)

        return output_dict

    @staticmethod
    def _decoder_cache_to_dict(decoder_cache):
        cache_dict = {}
        for layer_index, layer_cache in enumerate(decoder_cache):
            for attention_name, attention_cache in layer_cache.items():
                for tensor_name, cache_value in attention_cache.items():
                    key = (layer_index, attention_name, tensor_name)
                    cache_dict[key] = cache_value
        return cache_dict

    @staticmethod
    def _dict_to_decoder_cache(cache_dict):
        decoder_cache = []
        for key, cache_value in cache_dict.items():
            # Split key and extract index and dict keys
            layer_idx, attention_name, tensor_name = key
            # Extend decoder_cache to fit layer_idx + 1 layers
            decoder_cache = decoder_cache + [{} for _ in range(layer_idx + 1 - len(decoder_cache))]
            cache = decoder_cache[layer_idx]
            if attention_name not in cache:
                cache[attention_name] = {}
            assert tensor_name not in cache[attention_name]
            cache[attention_name][tensor_name] = cache_value
        return decoder_cache

    def take_step(
            self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor], step: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if len(last_predictions.shape) == 1:
            last_predictions = last_predictions.unsqueeze(-1)

        # Only the last predictions are needed for the decoder, but we need to pad the decoder ids
        # to not mess up the positional embeddings in the decoder.
        padding_size = 0
        if step > 0:
            padding_size = step + 1
            padding = torch.full(
                (last_predictions.shape[0], padding_size),
                self._pad_id,
                dtype=last_predictions.dtype,
                device=last_predictions.device,
            )
            last_predictions = torch.cat([padding, last_predictions], dim=-1)

        decoder_cache = None
        decoder_cache_dict = {
            k: (state[k].contiguous() if state[k] is not None else None)
            for k in state
            if k not in {"input_ids", "input_mask", "encoder_states"}
        }
        if len(decoder_cache_dict) != 0:
            decoder_cache = self._dict_to_decoder_cache(decoder_cache_dict)

        log_probabilities = None
        for i in range(padding_size, last_predictions.shape[1]):
            encoder_outputs = (
                (state["encoder_states"],) if state["encoder_states"] is not None else None
            )
            outputs = self.bart(
                input_ids=state["input_ids"],
                attention_mask=state["input_mask"],
                encoder_outputs=encoder_outputs,
                decoder_input_ids=last_predictions[:, : i + 1],
                decoder_cached_states=decoder_cache,
                generation_mode=True,
                use_cache=True,
            )

            decoder_log_probabilities = F.log_softmax(outputs[0][:, 0], dim=-1)

            if log_probabilities is None:
                log_probabilities = decoder_log_probabilities
            else:
                idx = last_predictions[:, i].view(-1, 1)
                log_probabilities = decoder_log_probabilities + log_probabilities.gather(
                    dim=-1, index=idx
                )

            decoder_cache = outputs[1][1]

            state["encoder_states"] = outputs[2]

        if decoder_cache is not None:
            decoder_cache_dict = self._decoder_cache_to_dict(decoder_cache)
            state.update(decoder_cache_dict)

        return log_probabilities, state

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        # Parameters
        output_dict : `Dict[str, torch.Tensor]`
            A dictionary containing a batch of predictions with key `predictions`. The tensor should have
            shape `(batch_size, max_sequence_length)`
        # Returns
        Dict[str, Any]
            Original `output_dict` with an additional `predicted_tokens` key that maps to a list of lists of
            tokens.
        """
        predicted_ids = output_dict["predicted_ids"]
        predictions = []
        for i in range(predicted_ids.shape[0]):
            token_ids = predicted_ids[i].tolist()
            while len(token_ids) > 0 and token_ids[-1] == self._end_id:
                token_ids.pop()
            predictions.append(self.tokenizer.tokenizer.decode(token_ids).strip())
        output_dict["predicted_question"] = predictions

        return output_dict