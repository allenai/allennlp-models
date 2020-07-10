import logging

from allennlp.data import DatasetReader
from overrides import overrides

from allennlp_models.mc.dataset_readers.transformer_mc import TransformerMCReader

logger = logging.getLogger(__name__)


@DatasetReader.register("commonsenseqa")
class CommonsenseQaReader(TransformerMCReader):
    """
    Reads the input data for the CommonsenseQA dataset (https://arxiv.org/abs/1811.00937).
    """

    @overrides
    def _read(self, file_path: str):
        from allennlp.common.file_utils import cached_path

        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        from allennlp.common.file_utils import json_lines_from_file

        for json in json_lines_from_file(file_path):
            choices = [(choice["label"], choice["text"]) for choice in json["question"]["choices"]]
            correct_choice = [
                i for i, (label, _) in enumerate(choices) if label == json["answerKey"]
            ][0]
            yield self.text_to_instance(
                json["id"], json["question"]["stem"], [c[1] for c in choices], correct_choice
            )
