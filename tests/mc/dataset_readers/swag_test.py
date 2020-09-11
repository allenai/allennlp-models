import pytest

from allennlp.common.util import ensure_list

from allennlp_models.mc.dataset_readers.swag import SwagReader
from tests import FIXTURES_ROOT


class TestSwagReader:
    def test_read_from_file(self):
        reader = SwagReader(transformer_model_name="bert-base-uncased")
        instances = ensure_list(reader.read(FIXTURES_ROOT / "mc" / "swag.csv"))
        assert len(instances) == 11

        instance = instances[0]
        assert len(instance.fields["alternatives"]) == 4

        alternative = instance.fields["alternatives"][0]
        token_text = [t.text for t in alternative.tokens]
        token_type_ids = [t.type_id for t in alternative.tokens]

        assert token_text[:3] == ["[CLS]", "students", "lower"]
        assert token_type_ids[:3] == [0, 0, 0]

        assert token_text[-3:] == ["someone", ".", "[SEP]"]
        assert token_type_ids[-3:] == [1, 1, 1]

        assert instance.fields["correct_alternative"] == 2

    def test_length_limit_works(self):
        length_limit = 20

        reader = SwagReader(transformer_model_name="bert-base-uncased", length_limit=length_limit)
        instances = ensure_list(reader.read(FIXTURES_ROOT / "mc" / "swag.csv"))

        assert len(instances) == 11
        for instance in instances:
            for alternative in instance.fields["alternatives"]:
                assert len(alternative) <= length_limit
