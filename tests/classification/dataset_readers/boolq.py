# -*- coding: utf-8 -*-
from allennlp.common.util import ensure_list
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer

from allennlp_models.classification import BoolQDatasetReader
from tests import FIXTURES_ROOT


class TestBoolqReader:
    boolq_path = FIXTURES_ROOT / "classification" / "boolq.jsonl"

    def test_boolq_dataset_reader_default_setting(self):
        reader = BoolQDatasetReader()
        instances = reader.read(self.boolq_path)
        instances = ensure_list(instances)

        assert len(instances) == 5

        fields = instances[0].fields
        assert [t.text for t in fields["tokens"].tokens][:5] == [
            "Persian",
            "language",
            "--",
            "Persian",
            "(/ˈpɜːrʒən,",
        ]
        assert fields["label"].label == 1

        fields = instances[1].fields
        assert [t.text for t in fields["tokens"].tokens][:5] == [
            "Epsom",
            "railway",
            "station",
            "--",
            "Epsom",
        ]
        assert fields["label"].label == 0

    def test_boolq_dataset_reader_roberta_setting(self):
        reader = BoolQDatasetReader(
            tokenizer=PretrainedTransformerTokenizer("roberta-base", add_special_tokens=False),
            token_indexers={"tokens": PretrainedTransformerIndexer("roberta-base")},
        )
        instances = reader.read(self.boolq_path)
        instances = ensure_list(instances)

        assert len(instances) == 5

        fields = instances[0].fields
        assert [t.text for t in fields["tokens"].tokens][:5] == [
            "<s>",
            "Pers",
            "ian",
            "Ġlanguage",
            "Ġ--",
        ]
        assert [t.text for t in fields["tokens"].tokens][-5:] == [
            "Ġspeak",
            "Ġthe",
            "Ġsame",
            "Ġlanguage",
            "</s>",
        ]
        assert fields["label"].label == 1

        fields = instances[1].fields
        assert [t.text for t in fields["tokens"].tokens][:5] == [
            "<s>",
            "E",
            "ps",
            "om",
            "Ġrailway",
        ]
        assert [t.text for t in fields["tokens"].tokens][-5:] == [
            "Ġe",
            "ps",
            "om",
            "Ġstation",
            "</s>",
        ]
        assert fields["label"].label == 0
