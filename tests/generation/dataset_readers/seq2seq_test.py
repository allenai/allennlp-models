import tempfile
import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import ensure_list
from allennlp_models.generation import Seq2SeqDatasetReader

from tests import FIXTURES_ROOT


class TestSeq2SeqDatasetReader:
    def test_default_format(self):
        reader = Seq2SeqDatasetReader()
        instances = reader.read(str(FIXTURES_ROOT / "generation" / "seq2seq_copy.tsv"))
        instances = ensure_list(instances)

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["source_tokens"].tokens] == [
            "@start@",
            "this",
            "is",
            "a",
            "sentence",
            "@end@",
        ]
        assert [t.text for t in fields["target_tokens"].tokens] == [
            "@start@",
            "this",
            "is",
            "a",
            "sentence",
            "@end@",
        ]
        fields = instances[1].fields
        assert [t.text for t in fields["source_tokens"].tokens] == [
            "@start@",
            "this",
            "is",
            "another",
            "@end@",
        ]
        assert [t.text for t in fields["target_tokens"].tokens] == [
            "@start@",
            "this",
            "is",
            "another",
            "@end@",
        ]
        fields = instances[2].fields
        assert [t.text for t in fields["source_tokens"].tokens] == [
            "@start@",
            "all",
            "these",
            "sentences",
            "should",
            "get",
            "copied",
            "@end@",
        ]
        assert [t.text for t in fields["target_tokens"].tokens] == [
            "@start@",
            "all",
            "these",
            "sentences",
            "should",
            "get",
            "copied",
            "@end@",
        ]

    def test_source_add_start_token(self):
        reader = Seq2SeqDatasetReader(source_add_start_token=False)
        instances = reader.read(str(FIXTURES_ROOT / "generation" / "seq2seq_copy.tsv"))
        instances = ensure_list(instances)

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["source_tokens"].tokens] == [
            "this",
            "is",
            "a",
            "sentence",
            "@end@",
        ]
        assert [t.text for t in fields["target_tokens"].tokens] == [
            "@start@",
            "this",
            "is",
            "a",
            "sentence",
            "@end@",
        ]

    def test_max_length_truncation(self):
        reader = Seq2SeqDatasetReader(source_max_tokens=3, target_max_tokens=5)
        instances = reader.read(str(FIXTURES_ROOT / "generation" / "seq2seq_copy.tsv"))
        instances = ensure_list(instances)
        assert reader._source_max_exceeded == 2
        assert reader._target_max_exceeded == 1
        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["source_tokens"].tokens] == [
            "@start@",
            "this",
            "is",
            "a",
            "@end@",
        ]
        assert [t.text for t in fields["target_tokens"].tokens] == [
            "@start@",
            "this",
            "is",
            "a",
            "sentence",
            "@end@",
        ]

    def test_delimiter_parameter(self):
        reader = Seq2SeqDatasetReader(delimiter=",")
        instances = reader.read(str(FIXTURES_ROOT / "generation" / "seq2seq_copy.csv"))
        instances = ensure_list(instances)

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["source_tokens"].tokens] == [
            "@start@",
            "this",
            "is",
            "a",
            "sentence",
            "@end@",
        ]
        assert [t.text for t in fields["target_tokens"].tokens] == [
            "@start@",
            "this",
            "is",
            "a",
            "sentence",
            "@end@",
        ]
        fields = instances[2].fields
        assert [t.text for t in fields["source_tokens"].tokens] == [
            "@start@",
            "all",
            "these",
            "sentences",
            "should",
            "get",
            "copied",
            "@end@",
        ]
        assert [t.text for t in fields["target_tokens"].tokens] == [
            "@start@",
            "all",
            "these",
            "sentences",
            "should",
            "get",
            "copied",
            "@end@",
        ]

    @pytest.mark.parametrize("line", (("a\n"), ("a\tb\tc\n")))
    def test_invalid_line_format(self, line):
        with tempfile.NamedTemporaryFile("w") as fp_tmp:
            fp_tmp.write(line)
            fp_tmp.flush()
            reader = Seq2SeqDatasetReader()
            with pytest.raises(ConfigurationError):
                list(reader.read(fp_tmp.name))

    @pytest.mark.parametrize("line", (("a b\tc d\n"), ('"a b"\t"c d"\n')))
    def test_correct_quote_handling(self, line):
        with tempfile.NamedTemporaryFile("w") as fp_tmp:
            fp_tmp.write(line)
            fp_tmp.flush()
            reader = Seq2SeqDatasetReader()
            instances = reader.read(fp_tmp.name)
            instances = ensure_list(instances)
            assert len(instances) == 1
            fields = instances[0].fields
            assert [t.text for t in fields["source_tokens"].tokens] == [
                "@start@",
                "a",
                "b",
                "@end@",
            ]
            assert [t.text for t in fields["target_tokens"].tokens] == [
                "@start@",
                "c",
                "d",
                "@end@",
            ]

    def test_bad_start_or_end_symbol(self):
        with pytest.raises(ValueError, match=r"Bad start or end symbol \('BAD SYMBOL"):
            Seq2SeqDatasetReader(start_symbol="BAD SYMBOL")
