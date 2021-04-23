from typing import Optional, Dict, Any

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.models.model import Model
from allennlp.modules.transformer.t5 import T5 as T5Module, T5Output, IntT, BoolT
from allennlp.training.metrics import ROUGE, BLEU


@Model.register("t5")
class T5(Model):
    def __init__(self, vocab: Vocabulary, model_name: str, **kwargs) -> None:
        super().__init__(vocab, **kwargs)
        self._model_name = model_name
        # We only instantiate this when we need it.
        self._tokenizer: Optional[PretrainedTransformerTokenizer] = None
        self.t5 = T5Module.from_pretrained_module(model_name)

        exclude_indices = {
            self.t5.pad_token_id,
            self.t5.decoder_start_token_id,
            self.t5.eos_token_id,
        }
        self._metrics = [
            ROUGE(exclude_indices=exclude_indices),
            BLEU(exclude_indices=exclude_indices),
        ]

    @property
    def tokenizer(self) -> PretrainedTransformerTokenizer:
        if self._tokenizer is None:
            self._tokenizer = PretrainedTransformerTokenizer(self._model_name)
        return self._tokenizer

    def forward(  # type: ignore
        self, source_tokens: TextFieldTensors, target_tokens: Optional[TextFieldTensors] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Performs the forward step of T5.

        # Parameters

        source_tokens : `TextFieldTensors`, required
            The source tokens for the encoder. We assume they are stored under the `tokens` key/namespace.

        target_tokens : `TextFieldTensors`, optional (default = `None`)
            The target tokens for the decoder. We assume they are also stored under the `tokens` key/namespace.
            If no target tokens are given during training / validation, the source tokens are shifted
            to the right by 1.

        # Returns

        `Dict[str, torch.Tensor]`
            Contains the `loss` when `target_tokens` is provided.
            And during prediction, includes `predictions` and `predicted_log_probs` from beam search.

        """
        input_ids, attention_mask = (
            source_tokens["tokens"]["token_ids"],
            source_tokens["tokens"]["mask"],
        )
        labels: Optional[IntT] = None
        decoder_attention_mask: Optional[BoolT] = None
        if target_tokens is not None:
            labels, decoder_attention_mask = (
                target_tokens["tokens"]["token_ids"],  # type: ignore[assignment]
                target_tokens["tokens"]["mask"],  # type: ignore[assignment]
            )
        elif self.training:
            raise ValueError("'target_tokens' required during training")

        output: T5Output = self.t5(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        output_dict: Dict[str, torch.Tensor] = {}

        if self.training:
            assert output.loss is not None
            output_dict["loss"] = output.loss
        else:
            # Shape: (batch_size, beam_size, num_tokens)
            assert output.predictions is not None
            # Shape: (batch_size, beam_size)
            assert output.predicted_log_probs is not None
            # Shape: (batch_size, num_tokens)
            output_dict["predictions"] = output.predictions[:, 0, :]
            # Shape: (batch_size, )
            output_dict["predicted_log_probs"] = output.predicted_log_probs[:, 0]

            if labels is not None:
                assert output.loss is not None
                output_dict["loss"] = output.loss

                for metric in self._metrics:
                    metric(output_dict["predictions"], labels)  # type: ignore[call-arg]

        return output_dict

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        predictions = output_dict["predictions"]
        output_dict["predicted_text"] = self.tokenizer.tokenizer.batch_decode(
            predictions, skip_special_tokens=True  # type: ignore[attr-defined]
        )
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if not self.training:
            for metric in self._metrics:
                metrics.update(metric.get_metric(reset=reset))
        return metrics
