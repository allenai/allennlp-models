from allennlp.common.testing import AllenNlpTestCase
from allennlp_models.common import model_card as mc
from allennlp.models import Model


class TestPretrainedModelConfiguration(AllenNlpTestCase):
    def test_init(self):
        model_card = mc.add_pretrained_model(
            id="fake_id",
            name="Fake Name",
            model_details="Model's description",
            archive_file="fake.tar.gz",
            predictor_name="fake_predictor",
            overrides={}
        )

        assert model_card.id == "fake_id"
        assert model_card.name == "Fake Name"
        assert model_card.archive_file == mc.STORAGE_LOCATION + "fake.tar.gz"
        assert model_card.predictor_name == "fake_predictor"
        assert model_card.model_details.description == "Model's description"

    def test_init_registered_model(self):
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

        model_card = mc.add_pretrained_model(**{"id": "fake-model"})

        assert model_card.name == "FakeModel"
        assert model_card.model_details.description == "This is a fake model with a docstring."

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

        model_card = mc.add_pretrained_model(**{"id": "fake-model", "model_class": FakeModel})

        assert model_card.name == "FakeModel"
        assert model_card.model_details.description == "This is a fake model with a docstring."

    def test_init_registered_model_override(self):
        @Model.register("fake-model-2")
        class FakeModel(Model):
            """
            This is a fake model with a docstring.

            # Parameters

            fake_param1: str
            fake_param2: int
            """

            def forward(self, **kwargs):
                return {}

        model_card = mc.add_pretrained_model(
            **{"id": "fake-model-2", "model_details": "This is the fake model trained on a dataset.",
               "model_class": FakeModel}
        )

        assert model_card.model_details.description == "This is the fake model trained on a dataset."

    def test_init_model_card_info_obj(self):
        @Model.register("fake-model-3")
        class FakeModel(Model):
            """
            This is a fake model with a docstring.

            # Parameters

            fake_param1: str
            fake_param2: int
            """

            def forward(self, **kwargs):
                return {}

        intended_use = mc.IntendedUse("Use 1", "User 1")

        model_card = mc.add_pretrained_model(**{"id": "fake-model-3", 
                                           "intended_use": intended_use})

        model_card_dict = model_card.to_dict()
        assert model_card.name == "FakeModel"

        for key, val in intended_use.__dict__.items():
            if val:
                assert key in model_card_dict
            else:
                assert key not in model_card_dict
