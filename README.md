Welcome to EPIONCHO-IBM

# Installation

`poetry install`
`poetry run pre-commit install`

# Testing

`poetry run python test.py`

# With snakeviz
poetry run python -m cProfile -o program.prof test_new.py; poetry run  snakeviz program.prof -s
