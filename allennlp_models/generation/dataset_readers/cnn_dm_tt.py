import random
from typing import Dict, Optional, List, Any, Iterable

from datasets import load_dataset, Dataset
from overrides import overrides
import torch.distributed as dist

from allennlp.common import cached_transformers, util
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TransformerTextField
from allennlp.data.instance import Instance


@DatasetReader.register("cnn_dm_tt")
class CNNDailyMailDatasetReaderTransformerToolkit(DatasetReader):
    """
    A CNN/Daily dataset reader for models that use the transformer toolkit.
    """

    def __init__(
        self,
        transformer_model_name: str,
        source_length_limit: int = 512,
        target_length_limit: int = 56,
        source_prefix: Optional[str] = None,
        shuffle: bool = False,
        num_preprocessing_workers: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self._tokenizer = cached_transformers.get_tokenizer(transformer_model_name)
        self._source_length_limit = source_length_limit
        self._target_length_limit = target_length_limit
        self._source_prefix = source_prefix
        self._shuffle = shuffle
        self._num_preprocessing_workers = num_preprocessing_workers

    @overrides
    def _read(self, split: str):
        dataset: Dataset = load_dataset("cnn_dailymail", version="3.0.0", split=split)  # type: ignore
        progress_prefix = ""

        if util.is_distributed():
            dataset = dataset.shard(dist.get_world_size(), dist.get_rank())

        if self._worker_info is not None and self._worker_info.num_workers > 1:
            dataset = dataset.shard(self._worker_info.num_workers, self._worker_info.id)
            progress_prefix = f"[worker {self._worker_info.id}] "

        dataset = dataset.map(
            self._batch_tokenize,
            batched=True,
            num_proc=self._num_preprocessing_workers,
            desc=progress_prefix + "Tokenizing dataset",
            remove_columns=["id"],
        )

        # Shuffling is broken in `datasets` so we have to do it our own way.
        # See https://github.com/huggingface/datasets/issues/514.
        # > dataset = dataset.shuffle(keep_in_memory=True)
        indices: Iterable[int] = range(len(dataset))
        if self._shuffle:
            indices = list(indices)
            random.shuffle(indices)

        for idx in indices:
            example = dataset[idx]
            yield Instance(
                {
                    "source": TransformerTextField(
                        **example["article"],  # type: ignore[call-overload]
                        padding_token_id=self._tokenizer.pad_token_id,
                    ),
                    "target": TransformerTextField(
                        **example["highlights"],  # type: ignore[call-overload]
                        padding_token_id=self._tokenizer.pad_token_id,
                    ),
                }
            )

    def _batch_tokenize(self, example):
        return {
            "article": self._split_batched_results(self._batch_tokenize_article(example)),
            "highlights": self._split_batched_results(self._batch_tokenize_highlights(example)),
        }

    @staticmethod
    def _split_batched_results(batched_result: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        keys, batched_values = list(zip(*batched_result.items()))
        return [{key: value for key, value in zip(keys, values)} for values in zip(*batched_values)]

    def _batch_tokenize_article(self, example):
        article_batch = example["article"]
        if self._source_prefix is not None:
            article_batch = [self._source_prefix + article for article in article_batch]
        return self._tokenizer(
            article_batch,
            max_length=self._source_length_limit,
            truncation=True,
        )

    def _batch_tokenize_highlights(self, example):
        highlights_batch = example["highlights"]
        return self._tokenizer(
            highlights_batch,
            max_length=self._target_length_limit,
            truncation=True,
        )

    @overrides
    def text_to_instance(  # type: ignore[override]
        self, article: str, highlights: str = None
    ) -> Instance:
        fields: Dict[str, Field] = {
            "source": TransformerTextField(
                **self._tokenizer(
                    article,
                    max_length=self._source_length_limit,
                    truncation=True,
                ),
                padding_token_id=self._tokenizer.pad_token_id,
            )
        }
        if highlights is not None:
            fields["target"] = TransformerTextField(
                **self._tokenizer(
                    highlights,
                    max_length=self._target_length_limit,
                    truncation=True,
                ),
                padding_token_id=self._tokenizer.pad_token_id,
            )

        return Instance(fields)
