from typing import Dict

import datasets
import torch
from allennlp.common import cached_transformers
from allennlp.data import Field, Instance, Vocabulary
from allennlp.data.fields import ListField, TransformerTextField, IndexField
from allennlp.tango.dataset import DatasetDict
from allennlp.tango.step import Step


@Step.register("piqa_instances")
class PiqaInstances(Step):
    DETERMINISTIC = True
    VERSION = "004"
    CACHEABLE = True

    def run(
        self,
        tokenizer_name: str,
        max_length: int = 512,
    ) -> DatasetDict:
        tokenizer = cached_transformers.get_tokenizer(tokenizer_name)
        assert tokenizer.pad_token_type_id == 0

        dataset = {
            split_name: [
                {
                    "correct_alternative": instance["label"],
                    "alternatives": [
                        (instance["goal"], instance["sol1"]),
                        (instance["goal"], instance["sol2"]),
                    ],
                }
                for instance in instances
            ]
            for split_name, instances in datasets.load_dataset("piqa").items()
        }

        # This thing is so complicated because we want to call `batch_encode_plus` with all
        # the strings at once.
        tokenized = {
            split_name: tokenizer.batch_encode_plus(
                [alternative for instance in instances for alternative in instance["alternatives"]],
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                return_token_type_ids=True,
                return_attention_mask=False,
            )
            for split_name, instances in dataset.items()
        }

        result = {}
        for split_name, instances in dataset.items():
            tokenized_alts = tokenized[split_name]
            results_per_split = []
            for i, instance in enumerate(instances):
                alts = ListField(
                    [
                        TransformerTextField(
                            torch.tensor(tokenized_alts["input_ids"][alt_index], dtype=torch.int32),
                            torch.tensor(
                                tokenized_alts["token_type_ids"][alt_index], dtype=torch.int32
                            ),
                            padding_token_id=tokenizer.pad_token_id,
                        )
                        for alt_index in [2 * i, 2 * i + 1]
                    ]
                )
                fields: Dict[str, Field] = {"alternatives": alts}
                if instance["correct_alternative"] >= 0:
                    fields["correct_alternative"] = IndexField(
                        instance["correct_alternative"], alts
                    )
                results_per_split.append(Instance(fields))
            result[split_name] = results_per_split

        # make vocab
        vocab = Vocabulary.empty()
        vocab.add_transformer_vocab(tokenizer, "tokens")

        return DatasetDict(result, vocab)
