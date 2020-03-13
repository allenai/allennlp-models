import os
import subprocess

import torch
from allennlp.data import Vocabulary
from torch.testing import assert_allclose

from allennlp.common.testing import AllenNlpTestCase, multi_device
from allennlp.training.metrics import SpanBasedF1Measure

from allennlp_models.syntax.srl.util import write_bio_formatted_tags_to_file
from tests import PROJECT_ROOT


class SpanBasedF1Test(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        vocab = Vocabulary()
        vocab.add_token_to_namespace("O", "tags")
        vocab.add_token_to_namespace("B-ARG1", "tags")
        vocab.add_token_to_namespace("I-ARG1", "tags")
        vocab.add_token_to_namespace("B-ARG2", "tags")
        vocab.add_token_to_namespace("I-ARG2", "tags")
        vocab.add_token_to_namespace("B-V", "tags")
        vocab.add_token_to_namespace("I-V", "tags")
        vocab.add_token_to_namespace("U-ARG1", "tags")
        vocab.add_token_to_namespace("U-ARG2", "tags")
        vocab.add_token_to_namespace("B-C-ARG1", "tags")
        vocab.add_token_to_namespace("I-C-ARG1", "tags")
        vocab.add_token_to_namespace("B-ARGM-ADJ", "tags")
        vocab.add_token_to_namespace("I-ARGM-ADJ", "tags")

        # BMES.
        vocab.add_token_to_namespace("B", "bmes_tags")
        vocab.add_token_to_namespace("M", "bmes_tags")
        vocab.add_token_to_namespace("E", "bmes_tags")
        vocab.add_token_to_namespace("S", "bmes_tags")

        self.vocab = vocab

    @multi_device
    def test_span_f1_matches_perl_script_for_continued_arguments(self, device: str):
        bio_tags = ["B-ARG1", "O", "B-C-ARG1", "B-V", "B-ARGM-ADJ", "O"]
        sentence = ["Mark", "and", "Matt", "were", "running", "fast", "."]

        gold_indices = [self.vocab.get_token_index(x, "tags") for x in bio_tags]
        gold_tensor = torch.tensor([gold_indices], device=device)
        prediction_tensor = torch.rand([1, 6, self.vocab.get_vocab_size("tags")], device=device)
        mask = torch.tensor([[True, True, True, True, True, True, True, True, True]], device=device)

        # Make prediction so that it is exactly correct.
        for i, tag_index in enumerate(gold_indices):
            prediction_tensor[0, i, tag_index] = 1

        metric = SpanBasedF1Measure(self.vocab, "tags")
        metric(prediction_tensor, gold_tensor, mask)
        metric_dict = metric.get_metric()

        # We merged the continued ARG1 label into a single span, so there should
        # be exactly 1 true positive for ARG1 and nothing present for C-ARG1
        assert metric._true_positives["ARG1"] == 1
        # The labels containing continuation references get merged into
        # the labels that they continue, so they should never appear in
        # the precision/recall counts.
        assert "C-ARG1" not in metric._true_positives.keys()
        assert metric._true_positives["V"] == 1
        assert metric._true_positives["ARGM-ADJ"] == 1

        assert_allclose(metric_dict["recall-ARG1"], 1.0)
        assert_allclose(metric_dict["precision-ARG1"], 1.0)
        assert_allclose(metric_dict["f1-measure-ARG1"], 1.0)
        assert_allclose(metric_dict["recall-V"], 1.0)
        assert_allclose(metric_dict["precision-V"], 1.0)
        assert_allclose(metric_dict["f1-measure-V"], 1.0)
        assert_allclose(metric_dict["recall-ARGM-ADJ"], 1.0)
        assert_allclose(metric_dict["precision-ARGM-ADJ"], 1.0)
        assert_allclose(metric_dict["f1-measure-ARGM-ADJ"], 1.0)
        assert_allclose(metric_dict["recall-overall"], 1.0)
        assert_allclose(metric_dict["precision-overall"], 1.0)
        assert_allclose(metric_dict["f1-measure-overall"], 1.0)

        # Check that the number of true positive ARG1 labels is the same as the perl script's output:
        gold_file_path = os.path.join(self.TEST_DIR, "gold_conll_eval.txt")
        prediction_file_path = os.path.join(self.TEST_DIR, "prediction_conll_eval.txt")
        with open(gold_file_path, "w") as gold_file, open(
            prediction_file_path, "w"
        ) as prediction_file:
            # Use the same bio tags as prediction vs gold to make it obvious by looking
            # at the perl script output if something is wrong.
            write_bio_formatted_tags_to_file(
                gold_file, prediction_file, 4, sentence, bio_tags, bio_tags
            )
        # Run the official perl script and collect stdout.
        perl_script_command = [
            "perl",
            str(PROJECT_ROOT / "allennlp_models" / "syntax" / "srl" / "srl-eval.pl"),
            prediction_file_path,
            gold_file_path,
        ]
        stdout = subprocess.check_output(perl_script_command, universal_newlines=True)
        stdout_lines = stdout.split("\n")
        # Parse the stdout of the perl script to find the ARG1 row (this happens to be line 8).
        num_correct_arg1_instances_from_perl_evaluation = int(
            [token for token in stdout_lines[8].split(" ") if token][1]
        )
        assert num_correct_arg1_instances_from_perl_evaluation == metric._true_positives["ARG1"]
