#!/usr/bin/env python

"""
Ensures allennlp and models versions are the same.
"""

from allennlp.version import VERSION as CORE_VERSION
from allennlp_models.version import VERSION as MODELS_VERSION


assert CORE_VERSION == MODELS_VERSION, f"core: {CORE_VERSION}, models: {MODELS_VERSION}"
