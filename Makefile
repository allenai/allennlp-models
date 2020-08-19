VERSION = $(shell python ./scripts/get_version.py current --minimal)

SRC = allennlp_models

MD_DOCS_ROOT = docs/
MD_DOCS_API_ROOT = $(MD_DOCS_ROOT)models/
MD_DOCS_SRC = $(filter-out %/__init__.py $(SRC)/version.py,$(shell find $(SRC) -type f -name '*.py' | grep -v -E 'tests/'))
MD_DOCS = $(subst .py,.md,$(subst $(SRC)/,$(MD_DOCS_API_ROOT),$(MD_DOCS_SRC)))
MD_DOCS_CMD = python scripts/py2md.py
MD_DOCS_CONF = mkdocs.yml
MD_DOCS_CONF_SRC = mkdocs-skeleton.yml
MD_DOCS_TGT = site/
MD_DOCS_EXTRAS = $(addprefix $(MD_DOCS_ROOT),README.md CHANGELOG.md)

DOCKER_TAG = latest
DOCKER_RUN_CMD = docker run --rm \
		-v $$HOME/.allennlp:/root/.allennlp \
		-v $$HOME/.cache/torch:/root/.cache/torch \
		-v $$HOME/nltk_data:/root/nltk_data
ALLENNLP_COMMIT_SHA = $(shell git ls-remote https://github.com/epwalsh/allennlp data-loading | cut -f 1)

ifeq ($(shell uname),Darwin)
ifeq ($(shell which gsed),)
$(error Please install GNU sed with 'brew install gnu-sed')
else
SED = gsed
endif
else
SED = sed
endif

.PHONY : version
version :
	@echo AllenNLP Models $(VERSION)

.PHONY : clean
clean :
	rm -rf $(MD_DOCS_TGT)
	rm -rf $(MD_DOCS_API_ROOT)
	rm -f $(MD_DOCS_ROOT)*.md
	rm -rf .pytest_cache/
	rm -rf allennlp_models.egg-info/
	rm -rf dist/
	rm -rf build/
	find . | grep -E '(\.mypy_cache|__pycache__|\.pyc|\.pyo$$)' | xargs rm -rf

.PHONY : lint
lint :
	flake8

.PHONY : format
format :
	black --check .

.PHONY : typecheck
typecheck :
	mypy allennlp_models tests --ignore-missing-imports --no-strict-optional --no-site-packages

.PHONY : test
test :
	pytest --color=yes -rf --durations=40 -m "not pretrained_model_test" -m "not pretrained_config_test"

.PHONY : gpu-test
gpu-test :
	pytest --color=yes -v -rf -m gpu

.PHONY : test-with-cov
test-with-cov :
	pytest --color=yes -rf --durations=40 \
			-m "not pretrained_model_test" \
			-m "not pretrained_config_test" \
			--cov-config=.coveragerc \
			--cov=allennlp_models/ \
			--cov-report=xml

.PHONY : test-pretrained
test-pretrained :
	pytest -v -n2 --forked --color=yes --durations=10 -m "pretrained_model_test"

.PHONY : test-configs
test-configs :
	pytest -v -n2 --forked --color=yes --durations=10 -m "pretrained_config_test"

.PHONY : build-all-api-docs
build-all-api-docs : scripts/py2md.py
	@PYTHONPATH=./ $(MD_DOCS_CMD) $(subst /,.,$(subst .py,,$(MD_DOCS_SRC))) -o $(MD_DOCS)

.PHONY : build-docs
build-docs : build-all-api-docs $(MD_DOCS_CONF) $(MD_DOCS) $(MD_DOCS_EXTRAS)
	mkdocs build

.PHONY : serve-docs
serve-docs : build-all-api-docs $(MD_DOCS_CONF) $(MD_DOCS) $(MD_DOCS_EXTRAS)
	mkdocs serve --dirtyreload

.PHONY : update-docs
update-docs : $(MD_DOCS) $(MD_DOCS_EXTRAS)

$(MD_DOCS_ROOT)README.md : README.md
	cp $< $@
	# Alter the relative path of the README image for the docs.
	$(SED) -i '1s/docs/./' $@

$(MD_DOCS_ROOT)%.md : %.md
	cp $< $@

scripts/py2md.py :
	wget https://raw.githubusercontent.com/allenai/allennlp/master/scripts/py2md.py -O $@

$(MD_DOCS_CONF) : $(MD_DOCS_CONF_SRC) $(MD_DOCS)
	@PYTHONPATH=./ python scripts/build_docs_config.py $@ $(MD_DOCS_CONF_SRC) $(MD_DOCS_ROOT) $(MD_DOCS_API_ROOT)

$(MD_DOCS_API_ROOT)%.md : $(SRC)/%.py scripts/py2md.py
	mkdir -p $(shell dirname $@)
	$(MD_DOCS_CMD) $(subst /,.,$(subst .py,,$<)) --out $@

.PHONY :
docker-image :
	docker build \
		--pull \
		--build-arg ALLENNLP_VERSION=$(VERSION) \
		-f Dockerfile \
		-t allennlp/models:v$(VERSION) .

.PHONY : docker-test-image
docker-test-image :
	docker build --pull -f Dockerfile.test --build-arg ALLENNLP_COMMIT_SHA=$(ALLENNLP_COMMIT_SHA) -t allennlp-models/test:$(DOCKER_TAG) .

.PHONY : docker-test-run
docker-test-run :
	$(DOCKER_RUN_CMD) --gpus 2 allennlp-models/test:$(DOCKER_TAG) $(ARGS)
