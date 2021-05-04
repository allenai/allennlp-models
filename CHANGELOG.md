# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Added tests for checklist suites for SQuAD-style reading comprehension models (`bidaf`), and textual entailment models (`decomposable_attention` and `esim`).
- Added an optional "weight" parameter to `CopyNetSeq2Seq.forward()` for calculating a weighted loss instead of the simple average over the
  the negative log likelihoods for each instance in the batch.
- Added a way to initialize the `SrlBert` model without caching/loading pretrained transformer weights.
  You need to set the `bert_model` parameter to the dictionary form of the corresponding `BertConfig` from HuggingFace.
  See [PR #257](https://github.com/allenai/allennlp-models/pull/257) for more details.


## [v2.4.0](https://github.com/allenai/allennlp-models/releases/tag/v2.4.0) - 2021-04-22

### Added

- Added `T5` model for generation.
- Added a classmethod constructor on `Seq2SeqPredictor`: `.pretrained_t5_for_generation()`.
- Added a parameter called `source_prefix` to `CNNDailyMailDatasetReader`. This is useful with T5, for example, by setting `source_prefix` to "summarization: ".
- Tests for `VqaMeasure`.
- Distributed tests for `ConllCorefScores` and `SrlEvalScorer` metrics.

### Fixed

- `pretrained.load_predictor()` now allows for loading model onto GPU.
- `VqaMeasure` now calculates correctly in the distributed case.
- `ConllCorefScores` now calculates correctly in the distributed case.
- `SrlEvalScorer` raises an appropriate error if run in the distributed setting.

### Changed

- Updated `registered_predictor_name` to `null` in model cards for the models where it was the same as the default predictor.


## [v2.3.0](https://github.com/allenai/allennlp-models/releases/tag/v2.3.0) - 2021-04-14

### Fixed

- Fixed bug in `experiment_from_huggingface.jsonnet` and `experiment.jsonnet` by changing `min_count` to have key `labels` instead of `answers`. Resolves failure of model checks that involve calling `_extend` in `vocabulary.py`
- `TransformerQA` now outputs span probabilities as well as scores.
- `TransformerQAPredictor` now implements `predictions_to_labeled_instances`, which is required for the interpret module.

### Added

- Added script that produces the coref training data.
- Added tests for using `allennlp predict` on multitask models.
- Added reader and training config for RoBERTa on SuperGLUE's Recognizing Textual Entailment task

## [v2.2.0](https://github.com/allenai/allennlp-models/releases/tag/v2.2.0) - 2021-03-26

### Added

- Evaluating RC task card and associated LERC model card
- Compatibility with PyTorch 1.8
- Allows the order of examples in the task cards to be specified explicitly
- Dataset reader for SuperGLUE BoolQ

### Changed

- Add option `combine_input_fields` in `SnliDatasetReader` to support only having "non-entailment" and "entailment" as output labels.
- Made all the models run on AllenNLP 2.1
- Add option `ignore_loss_on_o_tags` in `CrfTagger` to set the flag outside its forward function.
- Add `make_output_human_readable` for pair classification models (`BiMpm`, `DecomposableAttention`, and `ESIM`).

### Fixed

- Fixed https://github.com/allenai/allennlp/issues/4745.
- Updated `QaNet` and `NumericallyAugmentedQaNet` models to remove bias for layers that are followed by normalization layers.
- Updated the model cards for `rc-naqanet`, `vqa-vilbert` and `ve-vilbert`.
- Predictors now work for the vilbert-multitask model.
- Support unlabeled instances in `SnliDatasetReader`.


## [v2.1.0](https://github.com/allenai/allennlp-models/releases/tag/v2.1.0) - 2021-02-24


### Changed

- `coding_scheme` parameter is now deprecated in `Conll2000DatasetReader`, please use `convert_to_coding_scheme` instead.

### Added

- BART model now adds a `predicted_text` field in `make_output_human_readable` that has the cleaned text corresponding to `predicted_tokens`.

### Fixed
 
- Made `label` parameter in `TransformerMCReader.text_to_instance` optional with default of `None`.
- Updated many of the models for version 2.1.0. Fixed and re-trained many of the models.


## [v2.0.1](https://github.com/allenai/allennlp-models/releases/tag/v2.0.1) - 2021-02-01

### Fixed

- Fixed `OpenIePredictor.predict_json` so it treats auxiliary verbs as verbs
  when the language is English.


## [v2.0.0](https://github.com/allenai/allennlp-models/releases/tag/v2.0.0) - 2021-01-27

### Fixed

- Made the training configs compatible with the tensorboard logging changes in the main repo


## [v2.0.0rc1](https://github.com/allenai/allennlp-models/releases/tag/v2.0.0rc1) - 2021-01-21

### Added

- Dataset readers, models, metrics, and training configs for VQAv2, GQA, and Visual Entailment

### Fixed

- Fixed `training_configs/pair_classification/bimpm.jsonnet` and `training_configs/rc/dialog_qa.jsonnet`
  to work with new data loading API.
- Fixed the potential for a dead-lock when training the `TransformerQA` model on multiple GPUs
  when nodes receive different sized batches.
- Fixed BART. This implementation had some major bugs in it that caused poor performance during prediction.

### Removed

- Moving `ModelCard` and `TaskCard` abstractions out of the models repository.

### Changed

- `master` branch renamed to `main`
- `SquadEmAndF1` metric can now also accept a batch of predictions and corresponding answers (instead of a single one)
  in the form of list (for each).


## [v1.3.0](https://github.com/allenai/allennlp-models/releases/tag/v1.2.2) - 2020-12-15

### Fixed

- Fix an index bug in  BART prediction.
- Add `None` check in `PrecoReader`'s `text_to_instance()` method. 
- Fixed `SemanticRoleLabelerPredictor.tokens_to_instances` so it treats auxiliary verbs as verbs
  when the language is English

### Added

- Added link to source code to API docs.
- Information updates for remaining model cards (also includes the ones in demo, but not in the repository).

### Changed

- Updated `Dockerfile.release` and `Dockerfile.commit` to work with different CUDA versions.
- Changes required for the `transformers` dependency update to version 4.0.1.

### Fixed

- Added missing folder for `taskcards` in setup.py


## [v1.2.2](https://github.com/allenai/allennlp-models/releases/tag/v1.2.2) - 2020-11-17

### Changed

- Changed AllenNLP dependency for releases to allow for a range of versions, instead
  of being pinned to an exact version.
- There will now be multiple Docker images pushed to Docker Hub for releases, each
  corresponding to a different supported CUDA version (currently just 10.2 and 11.0).

### Fixed

- Fixed `pair-classification-esim` pretrained model.
- Fixed `ValueError` error message in `Seq2SeqDatasetReader`.
- Better check for start and end symbols in `Seq2SeqDatasetReader` that doesn't fail for BPE-based tokenizers.

### Added

- Added `short_description` field to `ModelCard`. 
- Information updates for all model cards.


## [v1.2.1](https://github.com/allenai/allennlp-models/releases/tag/v1.2.1) - 2020-11-10

### Added

- Added the `TaskCard` class and task cards for common tasks.
- Added a test for the interpret functionality

### Changed

- Added more information to model cards for pair classification models (`pair-classification-decomposable-attention-elmo`, `pair-classification-roberta-snli`, `pair-classification-roberta-mnli`, `pair-classification-esim`).

### Fixed

- Fixed TransformerElmo config to work with the new AllenNLP
- Pinned the version of torch more tightly to make AMP work
- Fixed the somewhat fragile Bidaf test


## [v1.2.0](https://github.com/allenai/allennlp-models/releases/tag/v1.2.0) - 2020-10-29

### Changed

- Updated docstring for Transformer MC.
- Added more information to model cards for multiple choice models (`mc-roberta-commonsenseqa`,
`mc-roberta-piqa`, and `mc-roberta-swag`).

### Fixed

- Fixed many training configs to work out-of-the box. These include the configs for `bart_cnn_dm`, `swag`, `bidaf`, `bidaf_elmo`,
  `naqanet`, and `qanet`.
- Fixed minor bug in MaskedLanguageModel, where getting token ids used hard-coded assumptions (that
  could be wrong) instead of our standard utility function.


## [v1.2.0rc1](https://github.com/allenai/allennlp-models/releases/tag/v1.2.0rc1) - 2020-10-22

### Added

- Added dataset reader support for SQuAD 2.0 with both the `SquadReader` and `TransformerSquadReader`.
- Updated the SQuAD v1.1 metric to work with SQuAD 2.0 as well.
- Updated the `TransformerQA` model to work for SQuAD 2.0.
- Added official support for Python 3.8.
- Added a json template for model cards.
- Added `training_config` as a field in model cards.
- Added a `BeamSearchGenerator` registrable class which can be provided to a `NextTokenLM` model
  to utilize beam search for predicting a sequence of tokens, instead of a single next token.
  `BeamSearchGenerator` is an abstract class, so a concrete registered implementation needs to be used.
  One implementation is provided so far: `TransformerBeamSearchGenerator`, registered as `transformer`,
  which will work with any `NextTokenLM` that uses a `PretrainedTransformerEmbedder`.
- Added an `overrides` parameter to `pretrained.load_predictor()`.

### Changed

- `rc-transformer-qa` pretrained model is now an updated version trained on SQuAD v2.0.
- `skip_invalid_examples` parameter in SQuAD dataset readers has been deprecated. Please use
  `skip_impossible_questions` instead.

### Fixed

- Fixed `lm-masked-language-model` pretrained model.
- Fixed BART for latest `transformers` version.
- Fixed BiDAF predictor and BiDAF predictor tests.
- Fixed a bug with `Seq2SeqDatasetReader` that would cause an exception when
  the desired behavior is to not add start or end symbols to either the source or the target
  and the default `start_symbol` or `end_symbol` are not part of the tokenizer's vocabulary.


## [v1.1.0](https://github.com/allenai/allennlp-models/releases/tag/v1.1.0) - 2020-09-08

### Fixed

- Updated `LanguageModelTokenEmbedder` to allow allow multiple token embedders, but only use first with non-empty type
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
