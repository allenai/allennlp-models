<div align="center">
    <br>
    <img src="https://raw.githubusercontent.com/allenai/allennlp/main/docs/img/allennlp-logo-dark.png" width="400"/>
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

## About

This repository contains the components - such as [`DatasetReader`](https://docs.allennlp.org/main/api/data/dataset_readers/dataset_reader/#datasetreader), [`Model`](https://docs.allennlp.org/main/api/models/model/#model), and [`Predictor`](https://docs.allennlp.org/main/api/predictors/predictor/#predictor) classes - for applying [AllenNLP](https://github.com/allenai/allennlp) to a wide variety of NLP [tasks](#tasks-and-components).
It also provides an easy way to download and use [pre-trained models](#pre-trained-models) that were trained with these components.

### Tasks and components

This is an overview of the tasks supported by the AllenNLP Models library along with the corresponding components provided, organized by category. For a more comprehensive overview, see the [AllenNLP Models documentation](https://docs.allennlp.org/models/main/) or the [Paperswithcode page](https://paperswithcode.com/lib/allennlp).

- [**Classification**](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/classification)
  
  Classification is a broad category that contains many more specific tasks such as Sentiment Analysis and Binary Question Answering.

  🛠 **Components provided:** Dataset readers for various datasets, including [BoolQ](https://docs.allennlp.org/models/main/models/classification/dataset_readers/boolq/) and [SST](https://docs.allennlp.org/models/main/models/classification/dataset_readers/stanford_sentiment_tree_bank/), as well as a [Biattentive Classification Network](https://docs.allennlp.org/models/main/models/classification/models/biattentive_classification_network/) model.

- [**Coreference Resolution**](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/coref)

  Coreference resolution is the task of finding all expressions that refer to the same entity in a text. It is an important step for many higher level NLP tasks that involve natural language understanding such as document summarization, question answering, and information extraction.

  🛠 **Components provided:** A general [Coref](https://docs.allennlp.org/models/main/models/coref/models/coref/) model and several dataset readers.

- [**Generation**](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/generation)

  This is a category for tasks such as Summarization that involve generating unstructered and often variable-length text.

  🛠 **Components provided:** Several Seq2Seq models such a [Bart](https://docs.allennlp.org/models/main/models/generation/models/bart/), [CopyNet](https://docs.allennlp.org/models/main/models/generation/models/copynet_seq2seq/), and a general [Composed Seq2Seq](https://docs.allennlp.org/models/main/models/generation/models/copynet_seq2seq/), along with corresponding dataset readers.

- [**Language Modeling**](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/lm)

  Language Modeling tasks involve learning a probability distribution over sequences of tokens.

  🛠 **Components provided:** Several language model implementations, such as a [Masked LM](https://docs.allennlp.org/models/main/models/lm/models/masked_language_model/) and a [Next Token LM](https://docs.allennlp.org/models/main/models/lm/models/next_token_lm/).

- [**Multiple Choice**](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/mc)

  Multiple Choice tasks are tasks that involve selecting a correct answer out of a set of possible answers, given some input.

  🛠 **Components provided:** A [transformer-based multiple choice model](https://docs.allennlp.org/models/main/models/mc/models/transformer_mc/) and a handful of dataset readers for specific datasets.

- [**Pair Classification**](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/pair_classification)

  Pair classification is another broad category that contains more specific tasks such as Textual Entailment, which is the task to determine whether, for a pair of sentences, the facts in the first sentence imply the facts in the second.

  🛠 **Components provided:** Dataset readers for several datasets, including [SNLI](https://docs.allennlp.org/models/main/models/pair_classification/dataset_readers/snli/) and [Quora Paraphrase](https://docs.allennlp.org/models/main/models/pair_classification/dataset_readers/quora_paraphrase/).

- [**Reading Comprehension**](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/rc)

  Reading comprehension is the task of answering questions about a passage of text to show that the system understands the passage.

  🛠 **Components provided:** Models such as [BiDAF](https://docs.allennlp.org/models/main/models/rc/models/bidaf/) and a [transformer-based QA model](https://docs.allennlp.org/models/main/models/rc/models/transformer_qa/), as well as readers for datasets such as [DROP](https://docs.allennlp.org/models/main/models/rc/dataset_readers/drop/), [QuAC](https://docs.allennlp.org/models/main/models/rc/dataset_readers/quac/), and [SQuAD](https://docs.allennlp.org/models/main/models/rc/dataset_readers/squad/).

- [**Structured Prediction**](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/structured_prediction)

  Structured Prediction includes tasks such as Semantic Role Labeling (SRL), which is the task of determining the latent predicate argument structure of a sentence and providing representations that can answer basic questions about sentence meaning, including who did what to whom, etc.

  🛠 **Components provided:** Dataset readers for [Penn Tree Bank](https://docs.allennlp.org/models/main/models/structured_prediction/dataset_readers/penn_tree_bank/), [OntoNotes](https://docs.allennlp.org/models/main/models/structured_prediction/dataset_readers/srl/), etc., and several models including one for [SRL](https://docs.allennlp.org/models/main/models/structured_prediction/models/srl/) and a very general [graph parser](https://docs.allennlp.org/models/main/models/structured_prediction/models/graph_parser/).

- [**Sequence Tagging**](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/tagging)

  Sequence tagging tasks include Named Entity Recognition (NER) and Fine-grained NER.

  🛠 **Components provided:** A [Conditional Random Field model](https://docs.allennlp.org/models/main/models/tagging/models/crf_tagger/) and dataset readers for datasets such as  [CoNLL-2000](https://docs.allennlp.org/models/main/models/tagging/dataset_readers/conll2000/), [CoNLL-2003](https://docs.allennlp.org/models/main/models/tagging/dataset_readers/conll2003/), [CCGbank](https://docs.allennlp.org/models/main/models/tagging/dataset_readers/ccgbank/), and [OntoNotes](https://docs.allennlp.org/models/main/models/tagging/dataset_readers/ontonotes_ner/).

- [**Text + Vision**](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/vision)

  This is a catch-all category for any text + vision multi-modal tasks such Visual Question Answering (VQA), the task of generating a answer in response to a natural language question about the contents of an image.

  🛠 **Components provided:** Several models such as a [ViLBERT model for VQA](https://docs.allennlp.org/models/main/models/vision/models/vilbert_vqa/) and one for [Visual Entailment](https://docs.allennlp.org/models/main/models/vision/models/visual_entailment/), along with corresponding dataset readers. 

### Pre-trained models

Every pretrained model in AllenNLP Models has a corresponding [`ModelCard`](https://docs.allennlp.org/main/api/common/model_card/#modelcard) in the [`allennlp_models/modelcards/`](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards) folder.
Many of these models are also hosted on the [AllenNLP Demo](https://demo.allennlp.org).

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

## Installing

### From PyPI

`allennlp-models` is available on PyPI. To install with `pip`, just run

```bash
pip install --pre allennlp-models
```

Note that the `allennlp-models` package is tied to the [`allennlp` core package](https://pypi.org/projects/allennlp-models). Therefore when you install the models package you will get the corresponding version of `allennlp` (if you haven't already installed `allennlp`). For example,

```bash
pip install allennlp-models==1.0.0rc3
pip freeze | grep allennlp
# > allennlp==1.0.0rc3
# > allennlp-models==1.0.0rc3
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

Just replace the `RELEASE` and `CUDA` build args what you need. Currently only CUDA 10.2 and 11.0 are officially supported.

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
