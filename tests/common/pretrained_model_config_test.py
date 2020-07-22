from allennlp.common.testing import AllenNlpTestCase
from allennlp_models.common.pretrained_model_config import PretrainedModelConfiguration, ModelCard
from allennlp.models import Model


class TestPretrainedModelConfiguration(AllenNlpTestCase):
    def test_init(self):
        fake_model_card = ModelCard()
        config = PretrainedModelConfiguration(
            id="fake_id",
            name="Fake Name",
            description="Model's description",
            last_trained_on="2020.07.21",
            archive_file="fake.tar.gz",
            predictor_name="fake_predictor",
            overrides={},
            model_card=fake_model_card,
        )

        assert config.archive_file == config._storage_location + "fake.tar.gz"

    def test_init_dict_no_model(self):
        config = PretrainedModelConfiguration.from_dict({"id": "fake_id", "name": "Fake Name"})

        assert config.id == "fake_id"
        assert config.name == "Fake Name"

    def test_init_dict_registered_model(self):
        @Model.register("fake-model")
        class FakeModel(Model):
            """
            This is a fake model with a docstring.

            # Parameters

            fake_param1: str
            fake_param2: int
            """

            def forward(self, **kwargs):
                return {}

        config = PretrainedModelConfiguration.from_dict({"id": "fake-model"})

        assert config.name == "FakeModel"
        assert config.description == "This is a fake model with a docstring."

    def test_init_dict_model(self):
        class FakeModel(Model):
            """
            This is a fake model with a docstring.

            # Parameters

            fake_param1: str
            fake_param2: int
            """

            def forward(self, **kwargs):
                return {}

        config = PretrainedModelConfiguration.from_dict({"id": "fake-model"}, FakeModel)

        assert config.name == "FakeModel"
        assert config.description == "This is a fake model with a docstring."

    def test_init_dict_model_override(self):
        class FakeModel(Model):
            """
            This is a fake model with a docstring.

            # Parameters

            fake_param1: str
            fake_param2: int
            """

            def forward(self, **kwargs):
                return {}

        config = PretrainedModelConfiguration.from_dict(
            {"id": "fake-model", "description": "This is the fake model trained on a dataset."},
            FakeModel,
        )

        assert config.description == "This is the fake model trained on a dataset."
