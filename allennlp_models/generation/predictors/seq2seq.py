from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register("seq2seq")
class Seq2SeqPredictor(Predictor):
    """
    Predictor for sequence to sequence models, including

    - [`ComposedSeq2Seq`](../models/composed_seq2seq.md),
    - [`SimpleSeq2Seq`](../models/simple_seq2seq.md),
    - [`CopyNetSeq2Seq`](../models/copynet_seq2seq.md),
    - [`Bart`](../models/bart.md), and
    - [`T5`](../models/t5.md).
    """

    def predict(self, source: str) -> JsonDict:
        return self.predict_json({"source": source})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"source": "..."}`.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source)

    @classmethod
    def pretrained_t5_for_generation(cls, model_name: str = "t5-base") -> "Seq2SeqPredictor":
        """
        A helper method for creating a predictor for a pretrained T5 generation model.

        # Examples

        ```python
        from allennlp_models.generation.predictors import Seq2SeqPredictor

        ARTICLE_TO_SUMMARIZE = '''
        summarize: The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building,
        and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side.
        During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest
        man-made structure in the world, a title it held for 41 years until the Chrysler Building in
        New York City was finished in 1930. It was the first structure to reach a height of 300 metres.
        Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller
        than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is
        the second tallest free-standing structure in France after the Millau Viaduct.
        '''.strip().replace(
            "\n", " "
        )
        predictor = Seq2SeqPredictor.pretrained_t5_for_generation("t5-small")
        predictor.predict(ARTICLE_TO_SUMMARIZE)
        ```
        """
        from allennlp.data import Vocabulary
        from allennlp.data.tokenizers import PretrainedTransformerTokenizer
        from allennlp.data.token_indexers import PretrainedTransformerIndexer
        from allennlp_models.generation.dataset_readers import Seq2SeqDatasetReader
        from allennlp_models.generation.models import T5

        tokenizer, token_indexer = (
            PretrainedTransformerTokenizer(model_name),
            PretrainedTransformerIndexer(model_name),
        )
        reader = Seq2SeqDatasetReader(
            source_tokenizer=tokenizer,
            source_token_indexers={"tokens": token_indexer},
            source_add_start_token=False,
            source_add_end_token=False,
            target_add_start_token=False,
            target_add_end_token=False,
        )
        vocab = Vocabulary.from_pretrained_transformer(model_name)
        model = T5(vocab, model_name)
        return cls(model, reader)
