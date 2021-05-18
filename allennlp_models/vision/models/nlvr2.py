import logging
from typing import Dict, Optional, List, Any

from overrides import overrides
import numpy as np
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules.transformer import (
    TransformerEmbeddings,
    ImageFeatureEmbeddings,
    BiModalEncoder,
)
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import FBetaMeasure

from allennlp_models.vision.models.vision_text_model import VisionTextModel

logger = logging.getLogger(__name__)


@Model.register("nlvr2")
@Model.register("nlvr2_from_huggingface", constructor="from_huggingface_model_name")
class Nlvr2Model(VisionTextModel):
    """
    Model for visual entailment task based on the paper
    [Visual Entailment: A Novel Task for Fine-Grained Image Understanding]
    (https://api.semanticscholar.org/CorpusID:58981654).

    # Parameters

    vocab : `Vocabulary`
    text_embeddings : `TransformerEmbeddings`
    image_embeddings : `ImageFeatureEmbeddings`
    encoder : `BiModalEncoder`
    pooled_output_dim : `int`
    fusion_method : `str`, optional (default = `"mul"`)
    dropout : `float`, optional (default = `0.1`)
    label_namespace : `str`, optional (default = `labels`)
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_embeddings: TransformerEmbeddings,
        image_embeddings: ImageFeatureEmbeddings,
        encoder: BiModalEncoder,
        pooled_output_dim: int,
        fusion_method: str = "mul",
        dropout: float = 0.1,
        label_namespace: str = "labels",
        *,
        ignore_text: bool = False,
        ignore_image: bool = False,
    ) -> None:

        super().__init__(
            vocab,
            text_embeddings,
            image_embeddings,
            encoder,
            pooled_output_dim,
            fusion_method,
            dropout,
            label_namespace,
            is_multilabel=False,
        )

        self.pooled_output_dim = pooled_output_dim

        self.layer1 = torch.nn.Linear(pooled_output_dim * 2, pooled_output_dim)
        self.layer2 = torch.nn.Linear(pooled_output_dim, 2)

        self.activation = torch.nn.ReLU()

        self.accuracy = CategoricalAccuracy()
        self.fbeta = FBetaMeasure(beta=1.0, average="macro")

    @overrides
    def forward(
        self,  # type: ignore
        box_features: torch.Tensor,
        box_coordinates: torch.Tensor,
        box_mask: torch.Tensor,
        hypothesis: TextFieldTensors,
        label: Optional[torch.Tensor] = None,
        identifier: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = box_features.shape[0]

        box_features = box_features.transpose(0, 1)
        box_coordinates = box_coordinates.transpose(0, 1)
        box_mask = box_mask.transpose(0, 1)

        pooled_outputs1 = self.backbone(
            box_features[0], box_coordinates[0], box_mask[0], hypothesis
        )["pooled_boxes_and_text"]
        pooled_outputs2 = self.backbone(
            box_features[1], box_coordinates[1], box_mask[1], hypothesis
        )["pooled_boxes_and_text"]

        # TODO: concatenate these correctly
        hidden = self.layer1(torch.cat((pooled_outputs1, pooled_outputs2), dim=-1))

        # Shape: (batch_size, num_labels)
        logits = self.layer2(self.activation(hidden))

        # Shape: (batch_size, num_labels)
        probs = torch.softmax(logits, dim=-1)

        outputs = {"logits": logits, "probs": probs}
        outputs = self._compute_loss_and_metrics(batch_size, outputs, label)

        return outputs

    @overrides
    def _compute_loss_and_metrics(
        self,
        batch_size: int,
        outputs: torch.Tensor,
        label: torch.Tensor,
    ):
        if label is not None:
            outputs["loss"] = (
                torch.nn.functional.cross_entropy(outputs["logits"], label) / batch_size
            )
            self.accuracy(outputs["logits"], label)
            self.fbeta(outputs["probs"], label)
        return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.fbeta.get_metric(reset)
        accuracy = self.accuracy.get_metric(reset)
        metrics.update({"accuracy": accuracy})
        return metrics

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        batch_labels = []
        for batch_index, batch in enumerate(output_dict["probs"]):
            labels = np.argmax(batch, axis=-1)
            batch_labels.append(labels)
        output_dict["labels"] = batch_labels
        return output_dict

    default_predictor = "nlvr2"
