from allennlp.common.util import ensure_list

from allennlp_models.tagging.dataset_readers.ontonotes_ner import OntonotesNamedEntityRecognition
from tests import FIXTURES_ROOT


class TestOntonotesNamedEntityRecognitionReader:
    def test_read_from_file(self):
        conll_reader = OntonotesNamedEntityRecognition()
        instances = conll_reader.read(
            FIXTURES_ROOT / "structured_prediction" / "srl" / "conll_2012" / "subdomain"
        )
        instances = ensure_list(instances)

        fields = instances[0].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        assert tokens == [
            "Mali",
            "government",
            "officials",
            "say",
            "the",
            "woman",
            "'s",
            "confession",
            "was",
            "forced",
            ".",
        ]
        assert fields["tags"].labels == ["B-GPE", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]

        fields = instances[1].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        assert tokens == [
            "The",
            "prosecution",
            "rested",
            "its",
            "case",
            "last",
            "month",
            "after",
            "four",
            "months",
            "of",
            "hearings",
            ".",
        ]
        assert fields["tags"].labels == [
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-DATE",
            "I-DATE",
            "O",
            "B-DATE",
            "I-DATE",
            "O",
            "O",
            "O",
        ]

        fields = instances[2].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        assert tokens == ["Denise", "Dillon", "Headline", "News", "."]
        assert fields["tags"].labels == [
            "B-PERSON",
            "I-PERSON",
            "B-WORK_OF_ART",
            "I-WORK_OF_ART",
            "O",
        ]

    def test_ner_reader_can_filter_by_domain(self):
        conll_reader = OntonotesNamedEntityRecognition(domain_identifier="subdomain2")
        instances = conll_reader.read(
            FIXTURES_ROOT / "structured_prediction" / "srl" / "conll_2012"
        )
        instances = ensure_list(instances)
        assert len(instances) == 1
