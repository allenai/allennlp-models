import pathlib

PROJECT_ROOT = (pathlib.Path(__file__).parent / "..").resolve()  # pylint: disable=no-member
TESTS_ROOT = PROJECT_ROOT / "tests"
FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"
