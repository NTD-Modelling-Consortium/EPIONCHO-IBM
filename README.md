Welcome to EPIONCHO-IBM

# Requirements

Epioncho-IBM requires **Python 3.10** or newer, as well as [Poetry](https://python-poetry.org) installed and internet connection with access to [GitHub](github.com).

# Installation

1. First, check if you have pip installed (`pip3 --version`). It's very likely that you do already. If you don't run: `python3 -m ensurepip`
2. If you don't have Poetry installed yet, you can either follow the [official guidline](https://python-poetry.org/docs/#installation) or the following steps. The commands below will install poetry for your user only. It will be available in `$HOME/.local/bin`. It's often useful to add this path to `$PATH` so that it can be accessed just by typing `poetry` in the terminal.
   ```bash
   pip3 install poetry # installing in the user's home directory
   echo "export PATH=$HOME/.local/bin:$PATH" >> ~/.bashrc # Adding to $PATH
   ```
3. If you want to install poetry for all of the users instead, you should run `sudo pip3 install poetry` instead. You then don't need to worry about `$PATH`.
4. Now we are ready to checkout the project and install all of the dependencies
   ```bash
   git clone https://github.com/NTD-Modelling-Consortium/EPIONCHO-IBM.git
   # if the above does not work, try
   git clone git@github.com:NTD-Modelling-Consortium/EPIONCHO-IBM.git

   # optional select a desired branch: git checkout <branch>
   cd EPIONCHO-IBM
   poetry install
   ```
5. If you are expecting to make code contribution, run the following to make sure the code conforms to the coding guidlines:
   ```bash
   poetry run pre-commit install
   ```
6. Check that everything works as expected:
   Run Benchmark Tests
   ```bash
   cd tests
   poetry run pytest
   ```

   Run Test Simulations:
   ```bash
   # go back to main folder
   cd ../
   poetry run python examples/testing_modules/test_new_changes.py
   ```
   Then go to `test_outputs/compare_model_to_r.R`, run the full script, and see the result by opening `test_outputs/comparison_plot.png`

# Running simulations and testing

## Running simulations

You can run execute any python file by prefixing it with `poetry run python`. This ensures that the installed python environment is used. For example, to run the integration tests:

```bash
poetry run python examples/testing_modules/test_new_changes.py
```

Further examples of how simulations can be run are in `examples/...`. A template for simulations can be found in `examples/simulation_template.py`

## Testing

The tests have three parts - 
Benchmarking, Unit Testing, and Integration Testing

Both Benchmark and Unit Tests can be run with pytest. Always run theses tests from within the tests directory.
```bash
cd tests/
poetry run pytest
```
The Integration testing requires running the python files in `poetry run python examples/testing_modules/....`. `test_new_changes_hdf5.py` is still a work-in-progress test.

### Benchmark Tests
The benchmarking tests are in `tests/benchmark/`, where we will simulate the current version of the model, and compare the mean outputs with different populations and end_times with a previously generated mean and standard deviation under the same parameters.

To re-generate the means and standard deviations, you can run 
```bash
poetry run tests/benchmark/generate_benchmarks.py
```
This command will primarily need to be run if the model is changed significantly, new outputs are added, or an additional benchmark test is added.
Any benchmark tests should be developed with a short run-time in mind.
Any config items for the benchmark tests are in `tests/benchmark/constants.py` or `tests/conftest.py`
See [pytest documentation](https://docs.pytest.org/en/latest/contents.html) for more details on how to develop tests.

The tests in `tests/old_benchmark` utilize a custom wrapper of pytest, called `pytest-trust-random`. The benchmark generation of this package is broken as of this writing, so no additional tests can be added.

### Unit Tests
These tests are found in `tests/epioncho_tests/`, use pytest, and are mainly just unit tests.

### Integration Tests
These tests are found in `examples/testing_modules/`. They are meant to test the full model under varying scenarios, and make sure the microfilarial prevalence is the same as what we expect based on previously accepted simulations. 

# Profiling

Profiling is a way of finding out how the execution time of all of the functions in a program. You can run the following to see an execution graph of a given program:

```bash
poetry run python -m cProfile -o /tmp/program.prof examples/test_new.py; poetry run snakeviz /tmp/program.prof -s # opens a browser
```
