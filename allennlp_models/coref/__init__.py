"""
Coreference resolution is defined as follows: given a document, find and cluster entity mentions.
"""

from allennlp_models.coref.conll_reader import ConllCorefReader
from allennlp_models.coref.preco_reader import PrecoReader
from allennlp_models.coref.winobias_reader import WinobiasReader
from allennlp_models.coref.coref_model import CoreferenceResolver
from allennlp_models.coref.coref_predictor import CorefPredictor
