.PHONY : lint
lint :
	flake8 -v
	black -v --check .

.PHONY : typecheck
typecheck :
	mypy allennlp_models --ignore-missing-imports --no-strict-optional --no-site-packages

.PHONY : test
test :
	pytest -v --color=yes

.PHONY : test-with-cov
test-with-cov :
	pytest -v --color=yes --cov=allennlp_models/ --cov-report=xml
