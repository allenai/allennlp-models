from typing import Dict, Tuple, Any
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.models.model import Model
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.training.metrics import ROUGE, BLEU

from transformers import BartModel, BartForConditionalGeneration

import torch
from torch import nn
import torch.nn.functional as F


@Seq2SeqEncoder.register("bart_encoder")
class BartEncoder(Seq2SeqEncoder):
    """
    The BART encoder, implemented as a `Seq2SeqEncoder`, which assumes it operates on
    already embedded inputs.  This means that we remove the token and position embeddings
    from BART in this module.  For the typical use case of using BART to encode inputs to your
    model (where we include the token and position embeddings from BART), you should use
    `PretrainedTransformerEmbedder(bart_model_name, sub_module="encoder")` instead of this.

    # Parameters

    model_name : `str`, required
        Name of the pre-trained BART model to use. Available options can be found in
        `transformers.modeling_bart.BART_PRETRAINED_MODEL_ARCHIVE_MAP`.
    """

    def __init__(self, model_name):
        super().__init__()

        bart = BartModel.from_pretrained(model_name)
        self.hidden_dim = bart.config.hidden_size
        self.bart_encoder = bart.encoder
        self.bart_encoder.embed_tokens = lambda x: x
        self.bart_encoder.embed_positions = lambda x: torch.zeros(
            (x.shape[0], x.shape[1], self.hidden_dim), dtype=torch.float32
        )

    @overrides
    def get_input_dim(self) -> int:
        return self.hidden_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.hidden_dim

    @overrides
    def is_bidirectional(self) -> bool:
        return False

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor):
        # The first element is always the last encoder states for each input token.
        # Depending on the config, the second output will contain a list of the encoder states
        # after each transformer layer. Similarly, the third output can contain the attentions from each layer.
        # We only care about the first element.
        return self.bart_encoder(input_ids=inputs, attention_mask=mask)[0]


class _BartEncoderWrapper(nn.Module):
    """
    A wrapper class for a `Seq2SeqEncoder` allowing it to replace the encoder in `Bart`.
    This class is only used internally by `Bart`.
    """

    def __init__(
        self, encoder: Seq2SeqEncoder, embed_tokens: nn.Embedding, embed_positions: nn.Embedding
    ):
        """
        # Parameters

        encoder : `Seq2SeqEncoder`, required
            Encoder to be used by `Bart`.
        embed_tokens : `nn.Embedding`, required
            The token embedding layer of the BART model.
        embed_positions : `nn.Embedding`, required
            The positional embedding layer of the BART model.

        """
        super().__init__()
        self.encoder = encoder
        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions

        # AllenNLP Seq2SeqEncoder's don't necessarily return those and the encoder might not even use
        # Attention, thus ensure those are not expected.
        # assert not bart_config.output_attentions
        # assert not bart_config.output_hidden_states

    @overrides
    def forward(
        self,
        input_ids,
        attention_mask=None,
    ):
        x = self.embed_tokens(input_ids) + self.embed_positions(input_ids)
        encoder_states = self.encoder(x, attention_mask)
        # The last two elements are attention and history of hidden states, respectively
        return encoder_states, [], []


@Model.register("bart")
class Bart(Model):
    """
    BART model from the paper "BART: Denosing Sequence-to-Sequence Pre-training for Natural Language Generation,
    Translation, and Comprehension" (https://arxiv.org/abs/1910.13461). The Bart model here uses a language
    modeling head and thus can be used for text generation.
    """

    def __init__(
        self,
        model_name: str,
        vocab: Vocabulary,
        indexer: PretrainedTransformerIndexer = None,
        max_decoding_steps: int = 140,
        beam_size: int = 4,
        encoder: Seq2SeqEncoder = None,
    ):
        """
        # Parameters

        model_name : `str`, required
            Name of the pre-trained BART model to use. Available options can be found in
            `transformers.modeling_bart.BART_PRETRAINED_MODEL_ARCHIVE_MAP`.
        vocab : `Vocabulary`, required
            Vocabulary containing source and target vocabularies.
        indexer : `PretrainedTransformerIndexer`, optional (default = `None`)
            Indexer to be used for converting decoded sequences of ids to to sequences of tokens.
        max_decoding_steps : `int`, optional (default = `128`)
            Number of decoding steps during beam search.
        beam_size : `int`, optional (default = `5`)
            Number of beams to use in beam search. The default is from the BART paper.
        encoder : `Seq2SeqEncoder`, optional (default = `None`)
            Encoder to used in BART. By default, the original BART encoder is used.
        """
        super().__init__(vocab)
        self.bart = BartForConditionalGeneration.from_pretrained(model_name)
        self._indexer = indexer or PretrainedTransformerIndexer(model_name, namespace="tokens")

        self._start_id = self.bart.config.bos_token_id  # CLS
        self._decoder_start_id = self.bart.config.decoder_start_token_id or self._start_id
        self._end_id = self.bart.config.eos_token_id  # SEP
        self._pad_id = self.bart.config.pad_token_id  # PAD

        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(
            self._end_id, max_steps=max_decoding_steps, beam_size=beam_size or 1
        )

        self._rouge = ROUGE(exclude_indices={self._start_id, self._pad_id, self._end_id})
        self._bleu = BLEU(exclude_indices={self._start_id, self._pad_id, self._end_id})

        # Replace bart encoder with given encoder. We need to extract the two embedding layers so that
        # we can use them in the encoder wrapper
        if encoder is not None:
            assert (
                encoder.get_input_dim() == encoder.get_output_dim() == self.bart.config.hidden_size
            )
            self.bart.model.encoder = _BartEncoderWrapper(
                encoder,
                self.bart.model.encoder.embed_tokens,
                self.bart.model.encoder.embed_positions,
            )

    @overrides
    def forward(
        self, source_tokens: TextFieldTensors, target_tokens: TextFieldTensors = None
    ) -> Dict[str, torch.Tensor]:
        """
        Performs the forward step of Bart.

        # Parameters

        source_tokens : `TextFieldTensors`, required
            The source tokens for the encoder. We assume they are stored under the `tokens` key.
        target_tokens : `TextFieldTensors`, optional (default = `None`)
            The target tokens for the decoder. We assume they are stored under the `tokens` key. If no target
            tokens are given, the source tokens are shifted to the right by 1.


        # Returns

        `Dict[str, torch.Tensor]`
            During training, this dictionary contains the `decoder_logits` of shape `(batch_size,
            max_target_length, target_vocab_size)` and the `loss`. During inference, it contains `predictions`
            of shape `(batch_size, max_decoding_steps)` and `log_probabilities` of shape `(batch_size,)`.

        """
        inputs = source_tokens
        targets = target_tokens
        input_ids, input_mask = inputs["tokens"]["token_ids"], inputs["tokens"]["mask"]

        outputs = {}

        # If no targets are provided, then shift input to right by 1. Bart already does this internally
        # but it does not use them for loss calculation.
        if targets is not None:
            target_ids, target_mask = targets["tokens"]["token_ids"], targets["tokens"]["mask"]
        else:
            target_ids = input_ids[:, 1:]
            target_mask = input_mask[:, 1:]

        if self.training:
            decoder_logits = self.bart(
                input_ids=input_ids,
                attention_mask=input_mask,
                decoder_input_ids=target_ids[:, :-1].contiguous(),
                decoder_attention_mask=target_mask[:, :-1].contiguous(),
                use_cache=False,
            )[0]

            outputs["decoder_logits"] = decoder_logits

            # The BART paper mentions label smoothing of 0.1 for sequence generation tasks
            outputs["loss"] = sequence_cross_entropy_with_logits(
                decoder_logits,
                target_ids[:, 1:].contiguous(),
                target_mask[:, 1:].contiguous(),
                label_smoothing=0.1,
                average="token",
            )
        else:
            # Use decoder start id and start of sentence to start decoder
            initial_decoder_ids = torch.tensor(
                [[self._decoder_start_id, self._start_id]],
                dtype=input_ids.dtype,
                device=input_ids.device,
            ).repeat(input_ids.shape[0], 1)

            inital_state = {
                "input_ids": input_ids,
                "input_mask": input_mask,
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

            self._rouge(predictions, target_ids)
            self._bleu(predictions, target_ids)

            outputs["predictions"] = predictions
            outputs["log_probabilities"] = (
                beam_result[1].gather(dim=-1, index=max_pred_indices[..., 0]).squeeze(dim=-1)
            )

            self.make_output_human_readable(outputs)

        return outputs

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
        """
        Take step during beam search.

        # Parameters

        last_predictions : `torch.Tensor`
            The predicted token ids from the previous step. Shape: `(group_size,)`
        state : `Dict[str, torch.Tensor]`
            State required to generate next set of predictions
        step : `int`
            The time step in beam search decoding.


        # Returns

        `Tuple[torch.Tensor, Dict[str, torch.Tensor]]`
            A tuple containing logits for the next tokens of shape `(group_size, target_vocab_size)` and
            an updated state dictionary.
        """
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
                past_key_values=decoder_cache,
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

            decoder_cache = outputs[1]

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

        `Dict[str, Any]`
            Original `output_dict` with an additional `predicted_tokens` key that maps to a list of lists of
            tokens.

        """
        predictions = output_dict["predictions"]
        predicted_tokens = [None] * predictions.shape[0]
        for i in range(predictions.shape[0]):
            predicted_tokens[i] = self._indexer.indices_to_tokens(
                {"token_ids": predictions[0].tolist()}, self.vocab
            )
        output_dict["predicted_tokens"] = predicted_tokens

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if not self.training:
            metrics.update(self._rouge.get_metric(reset=reset))
            metrics.update(self._bleu.get_metric(reset=reset))
        return metrics
