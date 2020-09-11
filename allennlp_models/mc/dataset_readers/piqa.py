import logging

from allennlp.data import DatasetReader
from overrides import overrides

from allennlp_models.mc.dataset_readers.transformer_mc import TransformerMCReader

logger = logging.getLogger(__name__)


@DatasetReader.register("piqa")
class PiqaReader(TransformerMCReader):
    """
    Reads the input data for the PIQA dataset (https://arxiv.org/abs/1911.11641).
    """

    @overrides
    def _read(self, file_path: str):
        import re

        labels_path = re.sub(r"\.jsonl$", "-labels.lst", file_path, 1)
        if labels_path == file_path:
            raise ValueError(
                "Could not determine file name for the labels corresponding to %s.", file_path
            )

        from allennlp.common.file_utils import cached_path

        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        from allennlp.common.file_utils import json_lines_from_file

        json_lines = json_lines_from_file(file_path)

        labels_path = cached_path(labels_path)
        from allennlp.common.file_utils import text_lines_from_file

        logger.info("Reading labels at %s", labels_path)
        labels_lines = text_lines_from_file(labels_path)

        for qid, (json, label) in enumerate(zip(json_lines, labels_lines)):
            goal = json["goal"]
            sol1 = json["sol1"]
            sol2 = json["sol2"]
            label = int(label)
            yield self.text_to_instance(str(qid), goal, [sol1, sol2], label)
