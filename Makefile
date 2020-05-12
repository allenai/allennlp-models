DOCKER_TAG = latest
DOCKER_RUN_CMD = docker run --rm -v $$HOME/.allennlp:/root/.allennlp

.PHONY : version
version :
	@python -c 'from allennlp_models.version import VERSION; print(f"AllenNLP Models v{VERSION}")'

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
	pytest --color=yes -rf --cov-config=.coveragerc --cov=allennlp_models/ --durations=40 -m "not pretrained_model_test"

.PHONY : test-pretrained
test-pretrained :
	pytest -v --color=yes -m "pretrained_model_test"

.PHONY : docker-test-image
docker-test-image :
	docker build --pull -f Dockerfile.test -t allennlp-models/test:$(DOCKER_TAG) .

.PHONY : docker-test-run
docker-test-run :
	$(DOCKER_RUN_CMD) --gpus 2 allennlp-models/test:$(DOCKER_TAG) $(ARGS)
