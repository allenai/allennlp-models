VERSION = $(shell python ./scripts/get_version.py current --minimal)
DOCKER_TAG = latest
DOCKER_RUN_CMD = docker run --rm \
		-v $$HOME/.allennlp:/root/.allennlp \
		-v $$HOME/.cache/torch:/root/.cache/torch \
		-v $$HOME/nltk_data:/root/nltk_data
ALLENNLP_COMMIT_SHA = $(shell git ls-remote https://github.com/allenai/allennlp master | cut -f 1)

.PHONY : version
version :
	@echo AllenNLP Models $(VERSION)

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
	pytest --color=yes -rf --durations=40 -m "not pretrained_model_test"

.PHONY : gpu-test
gpu-test :
	pytest --color=yes -v -rf -m gpu

.PHONY : test-with-cov
test-with-cov :
	pytest --color=yes -rf --durations=40 -m "not pretrained_model_test" \
			--cov-config=.coveragerc \
			--cov=allennlp_models/ \
			--cov-report=xml

.PHONY : test-pretrained
test-pretrained :
	pytest -v --color=yes -m "pretrained_model_test"

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
