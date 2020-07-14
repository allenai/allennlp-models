from typing import Optional

from allennlp.models import load_archive
from allennlp.predictors import Predictor, TextClassifierPredictor

from allennlp_models.coref import CorefPredictor
from allennlp_models.mc.predictors import TransformerMCPredictor
from allennlp_models.pair_classification import DecomposableAttentionPredictor
from allennlp_models.rc import ReadingComprehensionPredictor
from allennlp_models.structured_prediction import (
    SemanticRoleLabelerPredictor,
    OpenIePredictor,
    ConstituencyParserPredictor,
    BiaffineDependencyParserPredictor,
)
from allennlp_models.tagging.predictors import SentenceTaggerPredictor


# flake8: noqa: E501


def _load_predictor(archive_file: str, predictor_name: Optional[str] = None) -> Predictor:
    """
    Helper to load the desired predictor from the given archive.
    """
    archive = load_archive(archive_file)
    return Predictor.from_archive(archive, predictor_name)


def bert_srl_shi_2019() -> SemanticRoleLabelerPredictor:
    predictor = _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.07.14.tar.gz"
    )
    return predictor


def bidirectional_attention_flow_seo_2017() -> ReadingComprehensionPredictor:
    """
    Reading Comprehension

    Based on `BiDAF (Seo et al, 2017) <https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Comprehen-Seo-Kembhavi/007ab5528b3bd310a80d553cccad4b78dc496b02>`_
    """
    predictor = _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/bidaf-model-2020.03.19.tar.gz"
    )
    return predictor


def naqanet_dua_2019() -> ReadingComprehensionPredictor:
    predictor = _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/naqanet-2020.02.19.tar.gz"
    )
    return predictor


def open_information_extraction_stanovsky_2018() -> OpenIePredictor:
    predictor = _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz"
    )
    return predictor


def decomposable_attention_with_elmo_parikh_2017() -> DecomposableAttentionPredictor:
    """
    Textual Entailment

    Based on `Parikh et al, 2017 <https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Languag-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27>`_
    """
    predictor = _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/decomposable-attention-elmo-2020.04.09.tar.gz"
    )
    return predictor


def neural_coreference_resolution() -> CorefPredictor:
    """
    Coreference Resolution
    """
    predictor = _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
    )
    return predictor


def named_entity_recognition_with_elmo_peters_2018() -> SentenceTaggerPredictor:
    """
    Named Entity Recognition

    Based on `Deep contextualized word representations <https://arxiv.org/abs/1802.05365>`_
    """
    predictor = _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz"
    )
    return predictor


def fine_grained_named_entity_recognition() -> SentenceTaggerPredictor:
    """
    Fine Grained Named Entity Recognition
    """
    predictor = _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/fine-grained-ner.2020-06-24.tar.gz"
    )
    return predictor


def fine_grained_named_entity_recognition_transformer() -> SentenceTaggerPredictor:
    """
    Fine Grained Named Entity Recognition with the transformer
    """
    predictor = _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/fgner_transformer.2020-07-14.tar.gz"
    )
    return predictor


def span_based_constituency_parsing_with_elmo_joshi_2018() -> ConstituencyParserPredictor:
    """
    Constituency Parsing

    Based on `Minimal Span Based Constituency Parser (Stern et al, 2017) <https://www.semanticscholar.org/paper/A-Minimal-Span-Based-Neural-Constituency-Parser-Stern-Andreas/593e4e749bd2dbcaf8dc25298d830b41d435e435>`_ but with ELMo embeddings
    """
    predictor = _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz"
    )
    return predictor


def biaffine_parser_universal_dependencies_todzat_2017() -> BiaffineDependencyParserPredictor:
    """
    Biaffine Dependency Parser (Stanford Dependencies)

    Based on `Dozat and Manning, 2017 <https://arxiv.org/pdf/1611.01734.pdf>`_
    """
    predictor = _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz"
    )
    return predictor


def esim_nli_with_elmo_chen_2017() -> DecomposableAttentionPredictor:
    """
    ESIM

    Based on `Enhanced LSTM for Natural Language Inference <https://arxiv.org/pdf/1609.06038.pdf>`_ and uses ELMo
    """
    predictor = _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/decomposable-attention-elmo-2020.04.09.tar.gz"
    )
    return predictor


def glove_sst() -> TextClassifierPredictor:
    return _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/basic_stanford_sentiment_treebank-2020.06.09.tar.gz"
    )


def roberta_sst() -> TextClassifierPredictor:
    return _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/sst-roberta-large-2020.06.08.tar.gz"
    )


def roberta_mnli() -> DecomposableAttentionPredictor:
    return _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/mnli_roberta-2020.06.09.tar.gz",
        "textual-entailment",
    )


def roberta_snli() -> DecomposableAttentionPredictor:
    return _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/snli_roberta-2020.06.09.tar.gz",
        "textual-entailment",
    )


def roberta_piqa() -> TransformerMCPredictor:
    return _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/piqa.2020-07-08.tar.gz"
    )


def roberta_commonsenseqa() -> TransformerMCPredictor:
    return _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/commonsenseqa.2020-07-08.tar.gz"
    )


def roberta_swag() -> TransformerMCPredictor:
    return _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/swag.2020-07-08.tar.gz"
    )
