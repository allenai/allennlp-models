from typing import Sequence, Dict

import datasets
from allennlp.common import cached_transformers
from allennlp.data import Vocabulary, Instance
from allennlp.data.fields import TransformerTextField, LabelField
from allennlp.tango.dataset import DatasetDict
from allennlp.tango.sqlite_format import SqliteDictFormat
from allennlp.tango.step import Step


@Step.register("imdb_instances")
class ImdbInstances(Step):
    DETERMINISTIC = True
    VERSION = "003"
    CACHEABLE = True

    FORMAT = SqliteDictFormat

    def run(
        self,
        tokenizer_name: str,
        max_length: int = 512,
    ) -> DatasetDict:
        tokenizer = cached_transformers.get_tokenizer(tokenizer_name)
        assert tokenizer.pad_token_type_id == 0

        def clean_text(s: str) -> str:
            return s.replace("<br />", "\n")

        # This thing is so complicated because we want to call `batch_encode_plus` with all
        # the strings at once.
        results: Dict[str, Sequence[Instance]] = {}
        for split_name, instances in datasets.load_dataset("imdb").items():
            tokenized_texts = tokenizer.batch_encode_plus(
                [clean_text(instance["text"]) for instance in instances],
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                return_token_type_ids=True,
                return_attention_mask=False,
            )

            results[split_name] = [
                Instance(
                    {
                        "text": TransformerTextField(
                            input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            padding_token_id=tokenizer.pad_token_id,
                        ),
                        "label": LabelField(instance["label"], skip_indexing=True),
                    }
                )
                for instance, input_ids, token_type_ids in zip(
                    instances, tokenized_texts["input_ids"], tokenized_texts["token_type_ids"]
                )
            ]

        # make vocab
        vocab = Vocabulary.empty()
        vocab.add_transformer_vocab(tokenizer, "tokens")
        vocab.add_tokens_to_namespace(["neg", "pos"], "labels")

        return DatasetDict(results, vocab)
