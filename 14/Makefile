.PHONY: all
all: requirements.txt

requirements.txt: poetry.lock
	poetry export --format requirements.txt --output $@

.PHONY: run
run:
	poetry run streamlit run app.py