import logging
from typing import Dict

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules.transformer import (
    TransformerEmbeddings,
    ImageFeatureEmbeddings,
    BiModalEncoder,
)
from allennlp.training.metrics import CategoricalAccuracy
from torch.nn import CrossEntropyLoss

from allennlp_models.vision.models.vision_text_model import VisionTextModel

logger = logging.getLogger(__name__)


@Model.register("vilbert_ir")
@Model.register("vilbert_ir_from_huggingface", constructor="from_huggingface_model_name")
class ImageRetrievalVilbert(VisionTextModel):
    """
    Model for image retrieval task based on the VilBERT paper.

    # Parameters

    vocab : `Vocabulary`
    text_embeddings : `TransformerEmbeddings`
    image_embeddings : `ImageFeatureEmbeddings`
    encoder : `BiModalEncoder`
    pooled_output_dim : `int`
    fusion_method : `str`, optional (default = `"mul"`)
    dropout : `float`, optional (default = `0.1`)
    label_namespace : `str`, optional (default = `answers`)
    k: `int`, optional (default = `1`)
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
        k: int = 1,
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
            is_multilabel=False,
            ignore_text=ignore_text,
            ignore_image=ignore_image,
        )
        self.classifier = torch.nn.Linear(pooled_output_dim, 1)

        self.accuracy = CategoricalAccuracy()
        self.loss = CrossEntropyLoss()

        self.k = k

    @overrides
    def forward(
        self,  # type: ignore
        box_features: torch.Tensor,
        box_coordinates: torch.Tensor,
        box_mask: torch.Tensor,
        caption: TextFieldTensors,
        label: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size = box_features.shape[0]

        if self.training:
            # Shape: (batch_size, num_images, pooled_output_dim)
            pooled_output = self.backbone(box_features, box_coordinates, box_mask, caption)[
                "pooled_boxes_and_text"
            ]

            # Shape: (batch_size, num_images)
            logits = self.classifier(pooled_output).squeeze(-1)
            probs = torch.softmax(logits, dim=-1)

            outputs = {"logits": logits, "probs": probs}
            outputs = self._compute_loss_and_metrics(batch_size, outputs, label)

            return outputs

        else:
            with torch.no_grad():
                # Shape: (batch_size, num_images, pooled_output_dim)
                pooled_output = self.backbone(box_features, box_coordinates, box_mask, caption)[
                    "pooled_boxes_and_text"
                ]

                # Shape: (batch_size, num_images)
                scores = self.classifier(pooled_output).squeeze(-1)

                # Shape: (batch_size, k)
                _, indices = scores.topk(self.k, dim=-1)

                # Shape: (batch_size)
                pre_logits = torch.sum((indices == label.reshape(-1, 1)), dim=-1).float()

                # Shape: (batch_size, 2)
                # 1st column == 1 if we found the image in the top k, 2nd column == 1 if we didn't
                logits = torch.stack((pre_logits, (pre_logits == 0).float()), dim=1)

                outputs = {"logits": logits}
                outputs = self._compute_loss_and_metrics(
                    batch_size, outputs, torch.zeros(batch_size).long().to(logits.device)
                )

                return outputs

    @overrides
    def _compute_loss_and_metrics(
        self,
        batch_size: int,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ):
        outputs["loss"] = self.loss(outputs["logits"], labels)
        self.accuracy(outputs["logits"], labels)
        return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy.get_metric(reset),
        }

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return output_dict

    default_predictor = "vilbert_ir"
