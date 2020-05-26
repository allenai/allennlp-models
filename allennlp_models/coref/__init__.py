"""
Coreference resolution is defined as follows: given a document, find and cluster entity mentions.
"""

from allennlp_models.coref.dataset_readers.conll import ConllCorefReader
from allennlp_models.coref.dataset_readers.preco import PrecoReader
from allennlp_models.coref.dataset_readers.winobias import WinobiasReader
from allennlp_models.coref.models.coref import CoreferenceResolver
from allennlp_models.coref.predictors.coref import CorefPredictor
