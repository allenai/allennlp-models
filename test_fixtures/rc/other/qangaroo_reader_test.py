import pytest

from allennlp.common import Params
from allennlp.common.util import ensure_list

from allennlp_models.rc.other import QangarooReader
from tests import FIXTURES_ROOT


class TestQangarooReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = QangarooReader(lazy=lazy)
        instances = ensure_list(reader.read(FIXTURES_ROOT / "rc" / "qangaroo.json"))
        assert len(instances) == 2

        assert [t.text for t in instances[0].fields["candidates"][3]] == ["german", "confederation"]
        assert [t.text for t in instances[0].fields["query"]] == ["country", "sms", "braunschweig"]
        assert [t.text for t in instances[0].fields["supports"][0][:3]] == [
            "The",
            "North",
            "German",
        ]
        assert [t.text for t in instances[0].fields["answer"]] == ["german", "empire"]
        assert instances[0].fields["answer_index"].sequence_index == 4

    def test_can_build_from_params(self):
        reader = QangarooReader.from_params(Params({}))

        assert reader._token_indexers["tokens"].__class__.__name__ == "SingleIdTokenIndexer"
