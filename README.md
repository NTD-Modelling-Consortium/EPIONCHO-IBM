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
   git clone git@github.com:dreamingspires/EPIONCHO-IBM.git
   # optionally select a desired branch: git checkout <branch>
   cd EPIONCHO-IBM
   poetry install
   ```
5. If you are expecting to make code contribution, run the following to make sure the code conforms to the coding guidlines:
   ```bash
   poetry run pre-commit install
   ```
6. Check that everything works as expected:
   ```bash
   poetry run python examples/test_new.py
   cd tests
   poetry run pytest
   ```

# Running simulations and testing

As per above, you can run tests by executing:

```bash
cd tests
poetry run pytest
```

You can run execute any python file by prefixing it with `poetry run python`. This ensures that the installed python environment is used. For example:

```bash
poetry run python examples/test.py
```

# Profiling

Profiling is a way of finding out how the execution time of all of the functions in a program. You can run the following to see an execution graph of a given program:

```bash
poetry run python -m cProfile -o /tmp/program.prof examples/test_new.py
poetry run snakeviz /tmp/program.prof -s # opens a browser
```
