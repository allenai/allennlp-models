from allennlp.common.params import Params
from allennlp.common.util import ensure_list
from allennlp.data import DatasetReader
import pytest

from allennlp_models.pair_classification import TransformerSuperGlueRteReader
from tests import FIXTURES_ROOT


class TestTransformerSuperGlueRteReader:
    def test_read_from_file_superglue_rte(self):
        reader = TransformerSuperGlueRteReader()
        instances = ensure_list(reader.read(FIXTURES_ROOT / "rc" / "superglue_rte.jsonl"))
        assert len(instances) == 4

        token_text = [t.text for t in instances[0].fields["tokens"].tokens]
        assert token_text[:3] == ["<s>", "No", "ĠWeapons"]
        assert token_text[10:14] == [".", "</s>", "</s>", "Weapons"]
        assert token_text[-3:] == ["ĠIraq", ".", "</s>"]

        assert instances[0].fields["label"].human_readable_repr() == "not_entailment"

        assert instances[0].fields["metadata"]["label"] == "not_entailment"
        assert instances[0].fields["metadata"]["index"] == 0

    def test_read_from_file_superglue_rte_no_label(self):
        reader = TransformerSuperGlueRteReader()
        instances = ensure_list(reader.read(FIXTURES_ROOT / "rc" / "superglue_rte_no_labels.jsonl"))
        assert len(instances) == 4

        token_text = [t.text for t in instances[0].fields["tokens"].tokens]
        assert token_text[:3] == ["<s>", "No", "ĠWeapons"]
        assert token_text[10:14] == [".", "</s>", "</s>", "Weapons"]
        assert token_text[-3:] == ["ĠIraq", ".", "</s>"]

        assert "label" not in instances[0].fields
        assert "label" not in instances[0].fields["metadata"]

        assert instances[0].fields["metadata"]["index"] == 0
