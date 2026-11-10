"""pytest options: `--gpu` runs tests marked `gpu`."""


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", help="run tests marked gpu")


def pytest_collection_modifyitems(config, items):
    import pytest

    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="needs --gpu and the model on this machine")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
