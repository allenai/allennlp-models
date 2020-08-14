from pathlib import Path
from glob import glob
import os
from typing import Dict, Tuple

import pytest

from tests import FIXTURES_ROOT
from allennlp.commands.train import TrainModel
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.params import Params
from allennlp.common.plugins import import_plugins


CONFIGS_TO_IGNORE = {
    # TODO (epwalsh): once the new data loading API is merged, try to get this model working.
    "bidirectional_language_model.jsonnet",
    # Requires some bi-directional LM archive path.
    "constituency_parser_transformer_elmo.jsonnet",
}


def find_configs():
    for item in os.walk("training_config/"):
        for pattern in ("*.json", "*.jsonnet"):
            for path in glob(os.path.join(item[0], pattern)):
                if os.path.basename(path) == "common.jsonnet":
                    continue
                yield pytest.param(
                    path,
                    marks=pytest.mark.skipif(
                        any(x in path for x in CONFIGS_TO_IGNORE), reason="ignoring"
                    ),
                )


GLOVE_PATCHES = {
    FIXTURES_ROOT
    / "glove.6B.100d.sample.txt.gz": (
        "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.100d.txt.gz",
    ),
    FIXTURES_ROOT
    / "glove.6B.300d.sample.txt.gz": (
        "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.300d.txt.gz",
        "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
        "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.lower.converted.zip",
    ),
}


def patch_glove(params):
    for key, value in params.items():
        if isinstance(value, str):
            for patch, patch_targets in GLOVE_PATCHES.items():
                if value in patch_targets:
                    params[key] = str(patch)
        elif isinstance(value, Params):
            patch_glove(value)


DATASET_PATCHES: Dict[Path, Tuple[str, ...]] = {
    FIXTURES_ROOT
    / "structured_prediction"
    / "srl"
    / "conll_2012": ("SRL_TRAIN_DATA_PATH", "SRL_VALIDATION_DATA_PATH"),
    FIXTURES_ROOT
    / "structured_prediction"
    / "example_ptb.trees": ("PTB_TRAIN_PATH", "PTB_DEV_PATH", "PTB_TEST_PATH"),
    FIXTURES_ROOT
    / "structured_prediction"
    / "dependencies.conllu": ("PTB_DEPENDENCIES_TRAIN", "PTB_DEPENDENCIES_VAL"),
    FIXTURES_ROOT
    / "structured_prediction"
    / "semantic_dependencies"
    / "dm.sdp": ("SEMEVAL_TRAIN", "SEMEVAL_DEV", "SEMEVAL_TEST"),
    FIXTURES_ROOT / "tagging" / "conll2003.txt": ("NER_TRAIN_DATA_PATH", "NER_TEST_DATA_PATH"),
    FIXTURES_ROOT / "mc" / "swag.csv": ("SWAG_TRAIN", "SWAG_DEV", "SWAG_TEST"),
    FIXTURES_ROOT / "rc" / "drop.json": ("DROP_TRAIN", "DROP_DEV"),
    FIXTURES_ROOT / "lm" / "language_model" / "sentences.txt": ("BIDIRECTIONAL_LM_TRAIN_PATH",),
    FIXTURES_ROOT / "rc" / "squad.json": ("SQUAD_TRAIN", "SQUAD_DEV"),
    FIXTURES_ROOT
    / "coref"
    / "coref.gold_conll": ("COREF_TRAIN_DATA_PATH", "COREF_DEV_DATA_PATH", "COREF_TEST_DATA_PATH",),
    FIXTURES_ROOT
    / "structured_prediction"
    / "srl"
    / "conll_2012"
    / "subdomain": ("CONLL_TRAIN_DATA_PATH", "CONLL_DEV_DATA_PATH"),
    FIXTURES_ROOT
    / "tagging"
    / "conll2003.txt": (
        "NER_TRAIN_DATA_PATH",
        "NER_TEST_DATA_PATH",
        "NER_TEST_A_PATH",
        "NER_TEST_B_PATH",
    ),
    FIXTURES_ROOT
    / "generation"
    / "bart"
    / "data"
    / "url_lists"
    / "all_train.txt": ("CNNDM_TRAIN", "CNNDM_DEV",),
}


@pytest.mark.pretrained_config_test
class TestAllenNlpPretrainedModelConfigs(AllenNlpTestCase):
    @classmethod
    def setup_class(cls):
        # Make sure all the classes we need are registered.
        import_plugins()

        # Patch dataset paths.
        for dataset_patch, patch_targets in DATASET_PATCHES.items():
            for patch_target in patch_targets:
                os.environ[patch_target] = str(dataset_patch)

    @pytest.mark.parametrize("path", find_configs())
    def test_pretrained_configs(self, path):
        params = Params.from_file(
            path,
            params_overrides="{"
            "'trainer.cuda_device': -1, "
            "'trainer.num_epochs': 2, "
            "'dataset_reader.max_instances': 4, "
            "'dataset_reader.lazy': false, "
            "}",
        )

        # Patch any pretrained glove files with smaller fixtures.
        patch_glove(params)

        # Remove unnecessary keys.
        for key in ("random_seed", "numpy_seed", "pytorch_seed", "distributed"):
            if key in params:
                del params[key]

        # Just make sure the train loop can be instantiated.
        TrainModel.from_params(params=params, serialization_dir=self.TEST_DIR, local_rank=0)
