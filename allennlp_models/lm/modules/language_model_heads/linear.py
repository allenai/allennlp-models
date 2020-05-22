from overrides import overrides
import torch

from allennlp.data import Vocabulary
from allennlp_models.lm.modules.language_model_heads.language_model_head import LanguageModelHead


@LanguageModelHead.register("linear")
class LinearLanguageModelHead(LanguageModelHead):
    """
    Uses `torch.nn.Linear` as a language model head.  Does nothing else fancy.  This was intended
    largely for testing code with small models and simple components.  It's likely that you would
    want something nicer for actually training a language model, such as tying weights with an
    input embedding, or an adaptive softmax, or something.
    """

    def __init__(self, vocab: Vocabulary, input_dim: int, vocab_namespace: str) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = vocab.get_vocab_size(vocab_namespace)
        if self.output_dim <= 0:
            raise ValueError("We can't embed into an empty vocabulary.")
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states)
