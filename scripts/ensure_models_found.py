#!/usr/bin/env python

"""
Ensures models are automatically found by allennlp.
"""
import logging

from allennlp.common.plugins import import_plugins
from allennlp.models import Model

import allennlp_models #Should ideally fail with some error.

logging.basicConfig(level=logging.INFO)

import_plugins()
Model.by_name("copynet_seq2seq")
