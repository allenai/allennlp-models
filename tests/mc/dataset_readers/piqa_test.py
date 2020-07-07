import pytest

from allennlp.common.util import ensure_list

from allennlp_models.mc.dataset_readers.piqa import PiqaReader
from tests import FIXTURES_ROOT


class TestCommonsenseQaReader:
    def test_read_from_file(self):
        reader = PiqaReader(transformer_model_name="bert-base-uncased")
        instances = ensure_list(reader.read(FIXTURES_ROOT / "mc" / "piqa.jsonl"))
        assert len(instances) == 10

        instance = instances[0]
        assert len(instance.fields["alternatives"]) == 2

        alternative = instance.fields["alternatives"][0]
        token_text = [t.text for t in alternative.tokens]
        token_type_ids = [t.type_id for t in alternative.tokens]

        assert token_text[:3] == ["[CLS]", 'how', 'do']
        assert token_type_ids[:3] == [0, 0, 0]

        assert token_text[-3:] == ["dish", ".", "[SEP]"]
        assert token_type_ids[-3:] == [0, 0, 0]     # PIQA doesn't use segment IDs

        assert instance.fields["correct_alternative"] == 0

    def test_length_limit_works(self):
        length_limit = 20

        reader = PiqaReader(transformer_model_name="bert-base-uncased", length_limit=length_limit)
        instances = ensure_list(reader.read(FIXTURES_ROOT / "mc" / "piqa.jsonl"))

        assert len(instances) == 10
        for instance in instances:
            for alternative in instance.fields["alternatives"]:
                assert len(alternative) <= length_limit
