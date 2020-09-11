import logging

from allennlp.data import DatasetReader
from overrides import overrides

from allennlp_models.mc.dataset_readers.transformer_mc import TransformerMCReader

logger = logging.getLogger(__name__)


@DatasetReader.register("swag")
class SwagReader(TransformerMCReader):
    """
    Reads the input data for the SWAG dataset (https://arxiv.org/abs/1808.05326).
    """

    @overrides
    def _read(self, file_path: str):
        from allennlp.common.file_utils import cached_path

        file_path = cached_path(file_path)
        logger.info("Reading file at %s", file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            import csv

            for line_number, line in enumerate(csv.reader(f)):
                if line_number == 0:
                    continue

                yield self.text_to_instance(
                    qid=line[1], start=line[3], alternatives=line[7:11], label=int(line[11])
                )
