.PHONY : lint
lint :
	flake8 -v
	black -v --check .

.PHONY : typecheck
typecheck :
	mypy allennlp_models --ignore-missing-imports --no-strict-optional --no-site-packages

.PHONY : test
test :
	pytest --color=yes -rf --durations=40

.PHONY : test-with-cov
test-with-cov :
	pytest --color=yes -rf --cov-config=.coveragerc --cov=allennlp_models/ --durations=40
