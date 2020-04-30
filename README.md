<div align="center">
    <br>
    <img src="https://raw.githubusercontent.com/allenai/allennlp/master/docs/img/allennlp-logo-dark.png" width="400"/>
    <p>
    Officially supported AllenNLP models.
    </p>
    <hr/>
</div>
<p align="center">
    <a href="https://github.com/allenai/allennlp-models/actions">
        <img alt="Build" src="https://github.com/allenai/allennlp-models/workflows/CI/badge.svg?event=push&branch=master">
    </a>
    <a href="https://pypi.org/project/allennlp-models/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/allennlp-models">
    </a>
    <a href="https://github.com/allenai/allennlp-models/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/allenai/allennlp-models.svg?color=blue&cachedrop">
    </a>
    <a href="https://codecov.io/gh/allenai/allennlp">
        <img alt="Codecov" src="https://codecov.io/gh/allenai/allennlp/branch/master/graph/badge.svg">
    </a>
</p>
<br/>

<div align="center">
❗️ To file an issue, please open a ticket on https://github.com/allenai/allennlp and tag it with "Models". ❗️
</div>

## Installing

### From PyPI

`allennlp-models` is available on PyPI. To install with `pip`, just run

```
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

```
git clone https://github.com/allenai/allennlp-models.git
cd allennlp-models
ALLENNLP_VERSION_OVERRIDE='allennlp' pip install -e .
pip install -r dev-requirements.txt
```

The `ALLENNLP_VERSION_OVERRIDE` flag ensures that your local install of `allennlp` won't be overridden. Otherwise `allennlp` will be installed from the latest commit to the master branch on GitHub.

Both `allennlp` and `allennlp-models` are developed and tested side-by-side, so they should be kept up-to-date with each other. If you look at the GitHub Actions [workflow for `allennlp-models`](https://github.com/allenai/allennlp-models/actions), it's always tested against the master branch of `allennlp`. Similarly, `allennlp` is always tested against the master branch of `allennlp-models`.
