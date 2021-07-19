import logging

from allennlp.data import DatasetReader

from allennlp_models.mc.dataset_readers.piqa import PiqaReader
from allennlp_models.mc.dataset_readers.transformer_mc_tt import (
    TransformerMCReaderTransformerToolkit,
)

logger = logging.getLogger(__name__)


@DatasetReader.register("piqa_tt")
class PiqaReaderTransformerToolkit(TransformerMCReaderTransformerToolkit, PiqaReader):
    pass
