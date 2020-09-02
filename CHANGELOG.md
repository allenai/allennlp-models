# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed

- Fixed evaluation of metrics when using distributed setting.
- Fixed a bug introduced in 1.0 where the SRL model did not reproduce the original result.

## [v1.1.0rc4](https://github.com/allenai/allennlp-models/releases/tag/v1.1.0rc4) - 2020-08-21

### Added

- Added regression tests for training configs that run on a scheduled workflow.
- Added a test for the pretrained sentiment analysis model.
- Added way for questions from quora dataset to be concatenated like the sequences in the SNLI dataset.

## [v1.1.0rc3](https://github.com/allenai/allennlp-models/releases/tag/v1.1.0rc3) - 2020-08-12

### Fixed

- Fixed `GraphParser.get_metrics` so that it expects a dict from `F1Measure.get_metric`.
- `CopyNet` and `SimpleSeq2Seq` models now work with AMP.
- Made the SST reader a little more strict in the kinds of input it accepts.



## [v1.1.0rc2](https://github.com/allenai/allennlp-models/releases/tag/v1.1.0rc2) - 2020-07-31

### Changed

- Updated to PyTorch 1.6.

### Fixed

- Updated the RoBERTa SST config to make proper use of the CLS token
- Updated RoBERTa SNLI and MNLI pretrained models for latest `transformers` version

### Added

- Added BART model
- Added `ModelCard` and related classes. Added model cards for all the pretrained models.
- Added a field `registered_predictor_name` to `ModelCard`.
- Added a method `load_predictor` to `allennlp_models.pretrained`.
- Added support to multi-layer decoder in simple seq2seq model.


## [v1.1.0rc1](https://github.com/allenai/allennlp-models/releases/tag/v1.1.0rc1) - 2020-07-14


### Fixed

- Updated the BERT SRL model to be compatible with the new huggingface tokenizers.
- `CopyNetSeq2Seq` model now works with pretrained transformers.
- A bug with `NextTokenLM` that caused simple gradient interpreters to fail.
- A bug in `training_config` of `qanet` and `bimpm` that used the old version of `regularizer` and `initializer`.
- The fine-grained NER transformer model did not survive an upgrade of the transformers library, but it is now fixed.
- Fixed many minor formatting issues in docstrings. Docs are now published at [https://docs.allennlp.org/models/](https://docs.allennlp.org/models/).

### Changed

- `CopyNetDatasetReader` no longer automatically adds `START_TOKEN` and `END_TOKEN`
  to the tokenized source. If you want these in the tokenized source, it's up to
  the source tokenizer.

### Added

- Added two models for fine-grained NER
- Added a category for multiple choice models, including a few reference implementations
- Implemented manual distributed sharding for SNLI dataset reader.


## [v1.0.0](https://github.com/allenai/allennlp-models/releases/tag/v1.0.0) - 2020-06-16

No additional note-worthy changes since rc6.

## [v1.0.0rc6](https://github.com/allenai/allennlp-models/releases/tag/v1.0.0rc6) - 2020-06-11

### Changed

- Removed deprecated `"simple_seq2seq"` predictor

### Fixed

- Replaced `deepcopy` of `Instance`s with new `Instance.duplicate()` method.
- A bug where pretrained sentence taggers would fail to be initialized because some of the models
  were not imported.
- A bug in some RC models that would cause mixed precision training to crash when using NVIDIA apex.
- Predictor names were inconsistently switching between dashes and underscores. Now they all use underscores.

### Added

- Added option to SemanticDependenciesDatasetReader to not skip instances that have no arcs, for validation data
- Added a default predictors to several models
- Added sentiment analysis models to pretrained.py
- Added NLI models to pretrained.py

## [v1.0.0rc5](https://github.com/allenai/allennlp-models/releases/tag/v1.0.0rc5) - 2020-05-14

### Changed

- Moved the models into categories based on their format

### Fixed

- Made `transformer_qa` predictor accept JSON input with the keys "question" and "passage" to be consistent with the `reading_comprehension` predictor.

### Added

- `conllu` dependency (previously part of `allennlp`'s dependencies)

## [v1.0.0rc4](https://github.com/allenai/allennlp-models/releases/tag/v1.0.0rc4) - 2020-05-14

We first introduced this `CHANGELOG` after release `v1.0.0rc4`, so please refer to the GitHub release
notes for this and earlier releases.
