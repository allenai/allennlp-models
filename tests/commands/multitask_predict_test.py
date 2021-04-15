# These test should really be in the core repo, but they are here because the multitask model is
# here.
import json
import os
import pathlib
import shutil
import sys
import tempfile

import pytest
from allennlp.commands import main
from allennlp.common.checks import ConfigurationError

from allennlp.common.testing import AllenNlpTestCase

from tests import FIXTURES_ROOT


class TestMultitaskPredict(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.classifier_model_path = FIXTURES_ROOT / "vision" / "vilbert_multitask" / "model.tar.gz"
        self.classifier_data_path = FIXTURES_ROOT / "vision" / "vilbert_multitask" / "dataset.json"
        self.tempdir = pathlib.Path(tempfile.mkdtemp())
        self.infile = self.tempdir / "inputs.txt"
        self.outfile = self.tempdir / "outputs.txt"

    def test_works_with_multitask_model(self):
        sys.argv = [
            "__main__.py",  # executable
            "predict",  # command
            str(self.classifier_model_path),
            str(self.classifier_data_path),
            "--output-file",
            str(self.outfile),
            "--silent",
        ]

        main()

        assert os.path.exists(self.outfile)

        with open(self.outfile, "r") as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 3
        for result in results:
            assert "vqa_best_answer" in result.keys() or "ve_entailment_answer" in result.keys()

        shutil.rmtree(self.tempdir)

    def test_using_dataset_reader_works_with_specified_multitask_head(self):
        sys.argv = [
            "__main__.py",  # executable
            "predict",  # command
            str(self.classifier_model_path),
            "unittest",  # "path" of the input data, but it's not really a path for VQA
            "--output-file",
            str(self.outfile),
            "--silent",
            "--use-dataset-reader",
            "--multitask-head",
            "vqa",
        ]

        main()

        assert os.path.exists(self.outfile)

        with open(self.outfile, "r") as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 3
        for result in results:
            assert "vqa_best_answer" in result.keys()

        shutil.rmtree(self.tempdir)

    def test_using_dataset_reader_fails_with_missing_parameter(self):
        sys.argv = [
            "__main__.py",  # executable
            "predict",  # command
            str(self.classifier_model_path),
            "unittest",  # "path" of the input data, but it's not really a path for VQA
            "--output-file",
            str(self.outfile),
            "--silent",
            "--use-dataset-reader",
        ]

        with pytest.raises(ConfigurationError):
            main()
