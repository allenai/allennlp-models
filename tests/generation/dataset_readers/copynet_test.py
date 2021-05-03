import numpy as np
import torch

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list
from allennlp.data import DatasetReader
from allennlp.data.fields import TensorField
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN

from allennlp_models.generation.dataset_readers import CopyNetDatasetReader

from tests import FIXTURES_ROOT


class TestCopyNetReader(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        params = Params.from_file(FIXTURES_ROOT / "generation" / "copynet" / "experiment.json")
        self.reader: CopyNetDatasetReader = DatasetReader.from_params(params["dataset_reader"])
        instances = self.reader.read(
            FIXTURES_ROOT / "generation" / "copynet" / "data" / "copyover.tsv"
        )
        self.instances = ensure_list(instances)
        self.vocab = Vocabulary.from_params(params=params["vocabulary"], instances=self.instances)

    def test_vocab_namespaces(self):
        assert self.vocab.get_vocab_size("target_tokens") > 5

    def test_instances(self):
        assert len(self.instances) == 2
        assert set(self.instances[0].fields.keys()) == set(
            (
                "source_tokens",
                "source_token_ids",
                "target_tokens",
                "target_token_ids",
                "source_to_target",
                "metadata",
            )
        )

    def test_tokens(self):
        fields = self.instances[0].fields
        assert [t.text for t in fields["source_tokens"].tokens] == [
            "these",
            "tokens",
            "should",
            "be",
            "copied",
            "over",
            ":",
            "hello",
            "world",
        ]
        assert fields["metadata"]["source_tokens"] == [
            "these",
            "tokens",
            "should",
            "be",
            "copied",
            "over",
            ":",
            "hello",
            "world",
        ]
        assert [t.text for t in fields["target_tokens"].tokens] == [
            "@start@",
            "the",
            "tokens",
            '"',
            "hello",
            "world",
            '"',
            "were",
            "copied",
            "@end@",
        ]
        assert fields["metadata"]["target_tokens"] == [
            "the",
            "tokens",
            '"',
            "hello",
            "world",
            '"',
            "were",
            "copied",
        ]

    def test_source_and_target_token_ids(self):
        source_token_ids = self.instances[0].fields["source_token_ids"].array
        target_token_ids = self.instances[0].fields["target_token_ids"].array
        assert list(source_token_ids) == [
            0,  # these
            1,  # tokens
            2,  # should
            3,  # be
            4,  # copied
            5,  # over
            6,  # :
            7,  # hello
            8,  # world
        ]
        assert list(target_token_ids) == [
            9,  # @start@
            10,  # the
            1,  # tokens
            11,  # "
            7,  # hello
            8,  # world
            11,  # "
            12,  # were
            4,  # copied
            13,  # @end@
        ]

    def test_source_to_target(self):
        source_to_target_field = self.instances[0].fields["source_to_target"]
        source_to_target_field.index(self.vocab)
        tensor = source_to_target_field.as_tensor(source_to_target_field.get_padding_lengths())
        check = np.array(
            [
                self.vocab.get_token_index("these", "target_tokens"),
                self.vocab.get_token_index("tokens", "target_tokens"),
                self.vocab.get_token_index("should", "target_tokens"),
                self.vocab.get_token_index("be", "target_tokens"),
                self.vocab.get_token_index("copied", "target_tokens"),
                self.vocab.get_token_index("over", "target_tokens"),
                self.vocab.get_token_index(":", "target_tokens"),
                self.vocab.get_token_index("hello", "target_tokens"),
                self.vocab.get_token_index("world", "target_tokens"),
            ]
        )
        np.testing.assert_equal(tensor.numpy(), check)
        assert tensor[1].item() != self.vocab.get_token_index(DEFAULT_OOV_TOKEN, "target_tokens")

    def test_text_to_instance_with_weight(self):
        instance = self.reader.text_to_instance("Hello,", "World!", weight=0.5)
        assert "weight" in instance.fields
        assert isinstance(instance.fields["weight"], TensorField)
        assert instance.fields["weight"].tensor.dtype == torch.float
