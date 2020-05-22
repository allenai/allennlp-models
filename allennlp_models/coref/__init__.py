"""
Coreference resolution is defined as follows: given a document, find and cluster entity mentions.
"""

from allennlp_models.coref.dataset_readers.conll_reader import ConllCorefReader
from allennlp_models.coref.dataset_readers.preco_reader import PrecoReader
from allennlp_models.coref.dataset_readers.winobias_reader import WinobiasReader
from allennlp_models.coref.models.coref_model import CoreferenceResolver
from allennlp_models.coref.predictors.coref_predictor import CorefPredictor
