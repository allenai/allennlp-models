<div align="center">
    <br>
    <a href="https://github.com/allenai/allennlp">
      <img src="https://raw.githubusercontent.com/allenai/allennlp/main/docs/img/allennlp-logo-dark.png" width="400"/>
    </a>
    <br>
    <br>
    <p>
    Officially supported AllenNLP models.
    </p>
    <hr/>
</div>
<p align="center">
    <a href="https://github.com/allenai/allennlp-models/actions">
        <img alt="Build" src="https://github.com/allenai/allennlp-models/workflows/CI/badge.svg?event=push&branch=main">
    </a>
    <a href="https://pypi.org/project/allennlp-models/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/allennlp-models">
    </a>
    <a href="https://github.com/allenai/allennlp-models/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/allenai/allennlp-models.svg?color=blue&cachedrop">
    </a>
    <a href="https://codecov.io/gh/allenai/allennlp">
        <img alt="Codecov" src="https://codecov.io/gh/allenai/allennlp/branch/main/graph/badge.svg">
    </a>
</p>
<br/>

<div align="center">
❗️ To file an issue, please open a ticket on <a href="https://github.com/allenai/allennlp/issues/new/choose">allenai/allennlp</a> and tag it with "Models". ❗️
</div>

## In this README

- [About](#about)
    - [Tasks and components](#tasks-and-components)
    - [Pre-trained models](#pre-trained-models)
- [Installing](#installing)
    - [From PyPI](#from-pypi)
    - [From source](#from-source)
    - [Using Docker](#using-docker)

## About

This repository contains the components - such as [`DatasetReader`](https://docs.allennlp.org/main/api/data/dataset_readers/dataset_reader/#datasetreader), [`Model`](https://docs.allennlp.org/main/api/models/model/#model), and [`Predictor`](https://docs.allennlp.org/main/api/predictors/predictor/#predictor) classes - for applying [AllenNLP](https://github.com/allenai/allennlp) to a wide variety of NLP [tasks](#tasks-and-components).
It also provides an easy way to download and use [pre-trained models](#pre-trained-models) that were trained with these components.

### Tasks and components

This is an overview of the tasks supported by the AllenNLP Models library along with the corresponding components provided, organized by category. For a more comprehensive overview, see the [AllenNLP Models documentation](https://docs.allennlp.org/models/main/) or the [Paperswithcode page](https://paperswithcode.com/lib/allennlp).

- [**Classification**](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/classification)
  
    Classification tasks involve predicting one or more labels from a predefined set to assign to each input. Examples include Sentiment Analysis, where the labels might be `{"positive", "negative", "neutral"}`, and Binary Question Answering, where the labels are `{True, False}`.

    🛠 **Components provided:** Dataset readers for various datasets, including [BoolQ](https://docs.allennlp.org/models/main/models/classification/dataset_readers/boolq/) and [SST](https://docs.allennlp.org/models/main/models/classification/dataset_readers/stanford_sentiment_tree_bank/), as well as a [Biattentive Classification Network](https://docs.allennlp.org/models/main/models/classification/models/biattentive_classification_network/) model.

- [**Coreference Resolution**](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/coref)

    Coreference resolution tasks require finding all of the expressions in a text that refer to common entities.

    <div align="center">
    <a href="https://nlp.stanford.edu/projects/coref.shtml"><img src="https://nlp.stanford.edu/projects/corefexample.png" width="300" /></a>
    </div>

    See [nlp.stanford.edu/projects/coref](https://nlp.stanford.edu/projects/coref.shtml) for more details.

    🛠 **Components provided:** A general [Coref](https://docs.allennlp.org/models/main/models/coref/models/coref/) model and several dataset readers.

- [**Generation**](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/generation)

    This is a broad category for tasks such as Summarization that involve generating unstructered and often variable-length text.

    🛠 **Components provided:** Several Seq2Seq models such a [Bart](https://docs.allennlp.org/models/main/models/generation/models/bart/), [CopyNet](https://docs.allennlp.org/models/main/models/generation/models/copynet_seq2seq/), and a general [Composed Seq2Seq](https://docs.allennlp.org/models/main/models/generation/models/copynet_seq2seq/), along with corresponding dataset readers.

- [**Language Modeling**](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/lm)

    Language modeling tasks involve learning a probability distribution over sequences of tokens.

    🛠 **Components provided:** Several language model implementations, such as a [Masked LM](https://docs.allennlp.org/models/main/models/lm/models/masked_language_model/) and a [Next Token LM](https://docs.allennlp.org/models/main/models/lm/models/next_token_lm/).

- [**Multiple Choice**](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/mc)

    Multiple choice tasks require selecting a correct choice among alternatives, where the set of choices may be different for each input. This differs from classification where the set of choices is predefined and fixed across all inputs.

    🛠 **Components provided:** A [transformer-based multiple choice model](https://docs.allennlp.org/models/main/models/mc/models/transformer_mc/) and a handful of dataset readers for specific datasets.

- [**Pair Classification**](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/pair_classification)

    Pair classification is another broad category that contains tasks such as Textual Entailment, which is to determine whether, for a pair of sentences, the facts in the first sentence imply the facts in the second.

    🛠 **Components provided:** Dataset readers for several datasets, including [SNLI](https://docs.allennlp.org/models/main/models/pair_classification/dataset_readers/snli/) and [Quora Paraphrase](https://docs.allennlp.org/models/main/models/pair_classification/dataset_readers/quora_paraphrase/).

- [**Reading Comprehension**](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/rc)

    Reading comprehension tasks involve answering questions about a passage of text to show that the system understands the passage.

    🛠 **Components provided:** Models such as [BiDAF](https://docs.allennlp.org/models/main/models/rc/models/bidaf/) and a [transformer-based QA model](https://docs.allennlp.org/models/main/models/rc/models/transformer_qa/), as well as readers for datasets such as [DROP](https://docs.allennlp.org/models/main/models/rc/dataset_readers/drop/), [QuAC](https://docs.allennlp.org/models/main/models/rc/dataset_readers/quac/), and [SQuAD](https://docs.allennlp.org/models/main/models/rc/dataset_readers/squad/).

- [**Structured Prediction**](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/structured_prediction)

    Structured prediction includes tasks such as Semantic Role Labeling (SRL), which is for determining the latent predicate argument structure of a sentence and providing representations that can answer basic questions about sentence meaning, including who did what to whom, etc.

    🛠 **Components provided:** Dataset readers for [Penn Tree Bank](https://docs.allennlp.org/models/main/models/structured_prediction/dataset_readers/penn_tree_bank/), [OntoNotes](https://docs.allennlp.org/models/main/models/structured_prediction/dataset_readers/srl/), etc., and several models including one for [SRL](https://docs.allennlp.org/models/main/models/structured_prediction/models/srl/) and a very general [graph parser](https://docs.allennlp.org/models/main/models/structured_prediction/models/graph_parser/).

- [**Sequence Tagging**](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/tagging)

    Sequence tagging tasks include Named Entity Recognition (NER) and Fine-grained NER.

    🛠 **Components provided:** A [Conditional Random Field model](https://docs.allennlp.org/models/main/models/tagging/models/crf_tagger/) and dataset readers for datasets such as  [CoNLL-2000](https://docs.allennlp.org/models/main/models/tagging/dataset_readers/conll2000/), [CoNLL-2003](https://docs.allennlp.org/models/main/models/tagging/dataset_readers/conll2003/), [CCGbank](https://docs.allennlp.org/models/main/models/tagging/dataset_readers/ccgbank/), and [OntoNotes](https://docs.allennlp.org/models/main/models/tagging/dataset_readers/ontonotes_ner/).

- [**Text + Vision**](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/vision)

    This is a catch-all category for any text + vision multi-modal tasks such Visual Question Answering (VQA), the task of generating a answer in response to a natural language question about the contents of an image.

    🛠 **Components provided:** Several models such as a [ViLBERT model for VQA](https://docs.allennlp.org/models/main/models/vision/models/vilbert_vqa/) and one for [Visual Entailment](https://docs.allennlp.org/models/main/models/vision/models/visual_entailment/), along with corresponding dataset readers. 

### Pre-trained models

Every pretrained model in AllenNLP Models has a corresponding [`ModelCard`](https://docs.allennlp.org/main/api/common/model_card/#modelcard) in the [`allennlp_models/modelcards/`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards) folder.
Many of these models are also hosted on the [AllenNLP Demo](https://demo.allennlp.org) and the [AllenNLP Project Gallery](https://gallery.allennlp.org/).

To programmatically list the available models, you can run the following from a Python session:

```python
>>> from allennlp_models import pretrained
>>> print(pretrained.get_pretrained_models())
```

The output is a dictionary that maps the model IDs to their `ModelCard`:

```
{'structured-prediction-srl-bert': <allennlp.common.model_card.ModelCard object at 0x14a705a30>, ...}
```

You can load a `Predictor` for any of these models with the [`pretrained.load_predictor()`](https://docs.allennlp.org/models/main/models/pretrained/#load_predictor) helper.
For example:

```python
>>> pretrained.load_predictor("mc-roberta-swag")
```

Here is a list of pre-trained models currently available.

<!-- This section is automatically generated, do not edit by hand! If you need to udpate it, run the script 'scripts/update_readme_model_list.py' -->

- [`coref-spanbert`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/coref-spanbert.json) - Higher-order coref with coarse-to-fine inference (with SpanBERT embeddings).
- [`evaluate_rc-lerc`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/evaluate_rc-lerc.json) - A BERT model that scores candidate answers from 0 to 1.
- [`generation-bart`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/generation-bart.json) - BART with a language model head for generation.
- [`glove-sst`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/glove-sst.json) - LSTM binary classifier with GloVe embeddings.
- [`lm-masked-language-model`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/lm-masked-language-model.json) - BERT-based masked language model
- [`lm-next-token-lm-gpt2`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/lm-next-token-lm-gpt2.json) - OpenAI's GPT-2 language model that generates the next token.
- [`mc-roberta-commonsenseqa`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/mc-roberta-commonsenseqa.json) - RoBERTa-based multiple choice model for CommonSenseQA.
- [`mc-roberta-piqa`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/mc-roberta-piqa.json) - RoBERTa-based multiple choice model for PIQA.
- [`mc-roberta-swag`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/mc-roberta-swag.json) - RoBERTa-based multiple choice model for SWAG.
- [`nlvr2-vilbert`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/nlvr2-vilbert-head.json) - ViLBERT-based model for Visual Entailment.
- [`nlvr2-vilbert`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/nlvr2-vilbert.json) - ViLBERT-based model for Visual Entailment.
- [`pair-classification-binary-gender-bias-mitigated-roberta-snli`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/pair-classification-binary-gender-bias-mitigated-roberta-snli.json) - RoBERTa finetuned on SNLI with binary gender bias mitigation.
- [`pair-classification-decomposable-attention-elmo`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/pair-classification-decomposable-attention-elmo.json) - The decomposable attention model (Parikh et al, 2017) combined with ELMo embeddings trained on SNLI.
- [`pair-classification-esim`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/pair-classification-esim.json) - Enhanced LSTM trained on SNLI.
- [`pair-classification-roberta-mnli`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/pair-classification-roberta-mnli.json) - RoBERTa finetuned on MNLI.
- [`pair-classification-roberta-rte`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/pair-classification-roberta-rte.json) - A pair classification model patterned after the proposed model in Devlin et al, fine-tuned on the SuperGLUE RTE corpus
- [`pair-classification-roberta-snli`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/pair-classification-roberta-snli.json) - RoBERTa finetuned on SNLI.
- [`rc-bidaf-elmo`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/rc-bidaf-elmo.json) - BiDAF model with ELMo embeddings instead of GloVe.
- [`rc-bidaf`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/rc-bidaf.json) - BiDAF model with GloVe embeddings.
- [`rc-naqanet`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/rc-naqanet.json) - An augmented version of QANet that adds rudimentary numerical reasoning ability, trained on DROP (Dua et al., 2019), as published in the original DROP paper.
- [`rc-nmn`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/rc-nmn.json) - A neural module network trained on DROP.
- [`rc-transformer-qa`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/rc-transformer-qa.json) - A reading comprehension model patterned after the proposed model in Devlin et al, with improvements borrowed from the SQuAD model in the transformers project
- [`roberta-sst`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/roberta-sst.json) - RoBERTa-based binary classifier for Stanford Sentiment Treebank
- [`semparse-nlvr`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/semparse-nlvr.json) - The model is a semantic parser trained on Cornell NLVR.
- [`semparse-text-to-sql`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/semparse-text-to-sql.json) - This model is an implementation of an encoder-decoder architecture with LSTMs and constrained type decoding trained on the ATIS dataset.
- [`semparse-wikitables`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/semparse-wikitables.json) - The model is a semantic parser trained on WikiTableQuestions.
- [`structured-prediction-biaffine-parser`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/structured-prediction-biaffine-parser.json) - A neural model for dependency parsing using biaffine classifiers on top of a bidirectional LSTM.
- [`structured-prediction-constituency-parser`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/structured-prediction-constituency-parser.json) - Constituency parser with character-based ELMo embeddings
- [`structured-prediction-srl-bert`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/structured-prediction-srl-bert.json) - A BERT based model (Shi et al, 2019) with some modifications (no additional parameters apart from a linear classification layer)
- [`structured-prediction-srl`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/structured-prediction-srl.json) - A reimplementation of a deep BiLSTM sequence prediction model (Stanovsky et al., 2018)
- [`tagging-elmo-crf-tagger`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/tagging-elmo-crf-tagger.json) - NER tagger using a Gated Recurrent Unit (GRU) character encoder as well as a GRU phrase encoder, with GloVe embeddings.
- [`tagging-fine-grained-crf-tagger`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/tagging-fine-grained-crf-tagger.json) - This model identifies a broad range of 16 semantic types in the input text. It is a reimplementation of Lample (2016) and uses a biLSTM with a CRF layer, character embeddings and ELMo embeddings.
- [`tagging-fine-grained-transformer-crf-tagger`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/tagging-fine-grained-transformer-crf-tagger.json) - Fine-grained NER model
- [`ve-vilbert`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/ve-vilbert.json) - ViLBERT-based model for Visual Entailment.
- [`vgqa-vilbert`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/vgqa-vilbert.json) - ViLBERT (short for Vision-and-Language BERT), is a model for learning task-agnostic joint representations of image content and natural language.
- [`vqa-vilbert`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards/vqa-vilbert.json) - ViLBERT (short for Vision-and-Language BERT), is a model for learning task-agnostic joint representations of image content and natural language.

<!-- End automatically generated section -->


## Installing

### From PyPI

`allennlp-models` is available on PyPI. To install with `pip`, just run

```bash
pip install allennlp-models
```

Note that the `allennlp-models` package is tied to the [`allennlp` core package](https://pypi.org/projects/allennlp-models). Therefore when you install the models package you will get the corresponding version of `allennlp` (if you haven't already installed `allennlp`). For example,

```bash
pip install allennlp-models==2.2.0
pip freeze | grep allennlp
# > allennlp==2.2.0
# > allennlp-models==2.2.0
```

### From source

If you intend to install the models package from source, then you probably also want to [install `allennlp` from source](https://github.com/allenai/allennlp#installing-from-source).
Once you have `allennlp` installed, run the following within the same Python environment:

```bash
git clone https://github.com/allenai/allennlp-models.git
cd allennlp-models
ALLENNLP_VERSION_OVERRIDE='allennlp' pip install -e .
pip install -r dev-requirements.txt
```

The `ALLENNLP_VERSION_OVERRIDE` environment variable ensures that the `allennlp` dependency is unpinned so that your local install of `allennlp` will be sufficient. If, however, you haven't installed `allennlp` yet and don't want to manage a local install, just omit this environment variable and `allennlp` will be installed from the main branch on GitHub.

Both `allennlp` and `allennlp-models` are developed and tested side-by-side, so they should be kept up-to-date with each other. If you look at the GitHub Actions [workflow for `allennlp-models`](https://github.com/allenai/allennlp-models/actions), it's always tested against the main branch of `allennlp`. Similarly, `allennlp` is always tested against the main branch of `allennlp-models`.

### Using Docker

Docker provides a virtual machine with everything set up to run AllenNLP--
whether you will leverage a GPU or just run on a CPU.  Docker provides more
isolation and consistency, and also makes it easy to distribute your
environment to a compute cluster.

Once you have [installed Docker](https://docs.docker.com/engine/installation/) you can either use a [prebuilt image from a release](https://hub.docker.com/r/allennlp/models) or build an image locally with any version of `allennlp` and `allennlp-models`.

If you have GPUs available, you also need to install the [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) runtime.

To build an image locally from a specific release, run

```bash
docker build \
    --build-arg RELEASE=1.2.2 \
    --build-arg CUDA=10.2 \
    -t allennlp/models - < Dockerfile.release
```

Just replace the `RELEASE` and `CUDA` build args with what you need. You can check [the available tags](https://hub.docker.com/r/allennlp/allennlp/tags)
on Docker Hub to see which CUDA versions are available for a given `RELEASE`.

Alternatively, you can build against specific commits of `allennlp` and `allennlp-models` with

```bash
docker build \
    --build-arg ALLENNLP_COMMIT=d823a2591e94912a6315e429d0fe0ee2efb4b3ee \
    --build-arg ALLENNLP_MODELS_COMMIT=01bc777e0d89387f03037d398cd967390716daf1 \
    --build-arg CUDA=10.2 \
    -t allennlp/models - < Dockerfile.commit
```

Just change the `ALLENNLP_COMMIT` / `ALLENNLP_MODELS_COMMIT` and `CUDA` build args to the desired commit SHAs and CUDA versions, respectively.

Once you've built your image, you can run it like this:

```bash
mkdir -p $HOME/.allennlp/
docker run --rm --gpus all -v $HOME/.allennlp:/root/.allennlp allennlp/models
```

> Note: the `--gpus all` is only valid if you've installed the nvidia-docker runtime.
