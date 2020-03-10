from allennlp_models import version


class TestVersion:
    def test_version_exists(self):
        assert version.VERSION.startswith(version.VERSION_SHORT)
