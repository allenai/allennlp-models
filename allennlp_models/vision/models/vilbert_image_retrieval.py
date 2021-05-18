# Vilbert takes in:
# 1. Image token?
# 2. image features
# 3. clasification token
# 4. word tokens
# 5. separation token
# (in multi-task training) 6. task token

# Vilbert outputs:
# h_img, h_cls, h_sep
# we will treat h_img and h_cls as "holistic image and text representations" (respectively)
# to get a score, we:
# 1. element-wise multiply h_img and h_cls
# 2. then multiply the result by weights to get a number
# 3. then softmax to get a probability?

# idea for instances:
# offline, calculate 3 hard negatives for every caption/image pair
# I think the hard negatives are *images*, not other captions
# "For efficiency, you can calculate the L2 distance between image feature and caption feature and use that"
# this will look like a multiple choice setup, I think
import logging
from typing import Dict, Optional

from overrides import overrides
import faiss
import numpy as np
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules.transformer import (
    TransformerEmbeddings,
    ImageFeatureEmbeddings,
    BiModalEncoder,
)
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import FBetaMeasure

from allennlp_models.vision.models.vision_text_model import VisionTextModel

logger = logging.getLogger(__name__)


@Model.register("vilbert_ir")
@Model.register("ir_vilbert_from_huggingface", constructor="from_huggingface_model_name")
class ImageRetrievalVilbert(VisionTextModel):
    # TODO: fix
    """
    Model for VQA task based on the VilBERT paper.

    # Parameters

    vocab : `Vocabulary`
    text_embeddings : `TransformerEmbeddings`
    image_embeddings : `ImageFeatureEmbeddings`
    encoder : `BiModalEncoder`
    pooled_output_dim : `int`
    fusion_method : `str`, optional (default = `"sum"`)
    dropout : `float`, optional (default = `0.1`)
    label_namespace : `str`, optional (default = `answers`)
    """

    # TODO: fix?
    def __init__(
        self,
        vocab: Vocabulary,
        text_embeddings: TransformerEmbeddings,
        image_embeddings: ImageFeatureEmbeddings,
        encoder: BiModalEncoder,
        pooled_output_dim: int,
        fusion_method: str = "mul",
        dropout: float = 0.1,
        label_namespace: str = "answers",
        *,
        ignore_text: bool = False,
        ignore_image: bool = False,
        k: int = 1
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
            ignore_text=ignore_text,
            ignore_image=ignore_image,
        )

        from allennlp.training.metrics import F1MultiLabelMeasure
        from allennlp_models.vision.metrics.vqa import VqaMeasure

        self.classifier = torch.nn.Linear(pooled_output_dim, 1)

        self.accuracy = CategoricalAccuracy()
        self.fbeta = FBetaMeasure(beta=1.0, average="macro")

        self.k = k
        # self.f1_metric = F1MultiLabelMeasure(average="micro")
        # self.vqa_metric = VqaMeasure()

    # TODO: fix
    @overrides
    def forward(
        self,  # type: ignore
        box_features: torch.Tensor,
        box_coordinates: torch.Tensor,
        box_mask: torch.Tensor,
        # hard_negative_features: torch.Tensor,
        # hard_negative_coordinates: torch.Tensor,
        # hard_negative_masks: torch.Tensor,
        caption: TextFieldTensors,
        label: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size = box_features.shape[0]
        num_images = box_features.shape[0]

        # Reshape inputs to feed into VilBERT
        box_features = box_features.transpose(0, 1)
        box_coordinates = box_coordinates.transpose(0, 1)
        box_mask = box_mask.transpose(0, 1)

        if self.training:
            # Shape: (batch_size, 4)
            logits = self.classifier(
                # Shape: (batch_size, 4, pooled_output_dim)
                torch.stack(
                    [
                        self.backbone(box_features[0], box_coordinates[0], box_mask[0], caption)[
                            "pooled_boxes_and_text"
                        ],
                        self.backbone(box_features[1], box_coordinates[1], box_mask[1], caption)[
                            "pooled_boxes_and_text"
                        ],
                        self.backbone(box_features[2], box_coordinates[2], box_mask[2], caption)[
                            "pooled_boxes_and_text"
                        ],
                        self.backbone(box_features[3], box_coordinates[3], box_mask[3], caption)[
                            "pooled_boxes_and_text"
                        ],
                    ],
                    dim=1,
                )
            ).squeeze(-1)

            probs = torch.softmax(logits, dim=1)

            outputs = {"logits": logits, "probs": probs}
            outputs = self._compute_loss_and_metrics(batch_size, outputs, label)

            return outputs

        else:
            vilbert_outputs = []
            for i in range(num_images):
                curr_box_features = box_features[i]
                curr_box_coordinates = box_coordinates[i]
                curr_box_mask = box_mask[i]

                vilbert_outputs.append(
                    # Shape: (batch_size, pooled_output_dim)
                    self.backbone(curr_box_features, curr_box_coordinates, curr_box_mask, caption)[
                        "pooled_boxes_and_text"
                    ]
                )

            # Shape: (batch_size, num_images, pooled_output_dim)
            stacked_outputs = torch.stack(vilbert_outputs, dim=1)

            # Shape: (batch_size, k)
            scores = self.classifier(stacked_outputs).squeeze(-1)

            # Shapes: (batch_size, k)
            values, indices = scores.topk(self.k, dim=-1)

            # Shape: (batch_size)
            logits = torch.sum(indices == labels.reshape(-1. 1), dim=-1)

            # probs = torch.softmax(logits, dim=1)

            outputs = {"logits": logits} # , "probs": probs}
            outputs = self._compute_loss_and_metrics(batch_size, outputs, torch.ones(batch_size))

            return outputs


    # TODO: fix
    @overrides
    def _compute_loss_and_metrics(
        self,
        batch_size: int,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ):
        # TODO: make sure this is right
        # idea: the correct image for a caption i is image_i
        # labels = torch.from_numpy(np.arange(0, batch_size))
        # labels = labels.to(outputs["logits"].device)

        outputs["loss"] = torch.nn.functional.cross_entropy(outputs["logits"], labels) / batch_size
        self.accuracy(outputs["logits"], labels)
        # print("fbeta info:")
        # print(outputs["probs"].size())
        # print(labels.size())
        # self.fbeta(outputs["probs"], labels)
        return outputs

    # TODO: fix
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # metrics = self.fbeta.get_metric(reset)
        # accuracy = self.accuracy.get_metric(reset)
        # metrics.update({"accuracy": accuracy})
        # return metrics
        return {
            "accuracy": self.accuracy.get_metric(reset),
        }

    # TODO: fix
    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        batch_tokens = []
        for batch_index, batch in enumerate(output_dict["probs"]):
            tokens = {}
            for i, prob in enumerate(batch):
                tokens[self.vocab.get_token_from_index(i, self.label_namespace)] = float(prob)
            batch_tokens.append(tokens)
        output_dict["tokens"] = batch_tokens
        return output_dict

    default_predictor = "vilbert_ir"
