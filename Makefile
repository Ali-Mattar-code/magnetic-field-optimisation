.PHONY: install test lint validate reproduce app

install:
	pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check .

validate:
	python -m magfield.cli validate

reproduce:
	python -m magfield.cli reproduce --config configs/quick_reproduction.yaml

app:
	streamlit run app/app.py

