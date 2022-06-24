def pytest_addoption(parser):
    parser.addoption(
        "--generatebenchmark",
        dest="genbenchmark",
        action="store_true",
        help="my option: type1 or type2",
    )
