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
        batch_size, num_images, num_boxes, feature_dimension = box_features.shape

        box_features = box_features.view(batch_size * num_images, num_boxes, feature_dimension)
        box_coordinates = box_coordinates.view(
            batch_size * num_images, box_coordinates.shape[2], box_coordinates.shape[3]
        )
        box_mask = box_mask.view(batch_size * num_images, box_mask.shape[2])

        # Shape: (batch_size * num_images, pooled_output_dim)
        pooled_output = self.backbone(box_features, box_coordinates, box_mask, caption)[
            "pooled_boxes_and_text"
        ]

        pooled_output = pooled_output.view(batch_size, num_images, pooled_output.shape[1])

        # Shape: (batch_size, num_images)
        logits = self.classifier(pooled_output).squeezse(-1)

        if self.training:
            probs = torch.softmax(logits, dim=1)

            outputs = {"logits": logits, "probs": probs}
            outputs = self._compute_loss_and_metrics(batch_size, outputs, label)

            return outputs

        else:
            # Shape: (batch_size, k)
            _, indices = scores.topk(self.k, dim=-1)

            # Shape: (batch_size)
            pre_logits = torch.sum((indices == label.reshape(-1, 1)), dim=-1).float()
            # Shape: (batch_size, 2)
            # 0-th column == 1 if we found the image in the top k, 1st column == 1 if we didn't
            logits = torch.stack((pre_logits, (pre_logits == 0).float()), dim=1)

            outputs = {"logits": logits}
            outputs = self._compute_loss_and_metrics(
                batch_size, outputs, torch.zeros(batch_size).long().to(logits.device)
            )

            return outputs

        #################

        batch_size = box_features.shape[0]
        num_images = box_features.shape[1]

        # TODO: We actually want to roll up the dimensions to get e.g. (batch_size * num_images, num_boxes, feature_dim)
        # Then we only have to feed it into VilBERT once
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
            if num_images != 1000:
                print("num images:")
                print(num_images)
                logger.info("num images:")
                logger.info(num_images)

            # Shapes: (batch_size, k)
            values, indices = scores.topk(self.k, dim=-1)

            # Shape: (batch_size)
            # logits = (indices == label.reshape(-1, 1)).float()
            # Shape: (batch_size)
            pre_logits = torch.sum((indices == label.reshape(-1, 1)), dim=-1).float()
            to_append = (pre_logits == 0).float()
            # Shape: (batch_size, 2)
            # 0-th column == 1 if we found the image in the top k, 1st column == 1 if we didn't
            logits = torch.stack((pre_logits, to_append), dim=1)

            # probs = torch.softmax(logits, dim=1)

            outputs = {"logits": logits}  # , "probs": probs}
            # This is slightly wrong. I have labels that correspond to the image index, and then top k
            # scores. I think I should go back to getting the int-version bool mask thing, summing them, and then
            # do something with those? Right now the labels I'm passing in here mean nothing.
            # Ex:
            # A row in the logits tensor could be [0, 1, 0, 0] and this means we found the image in the top k=4 images
            # The label really is just a placeholder for "find the image", but putting label=1 for this row would
            # tell the model we didn't find it. I think I should sum up the top k image ints to get either 0 or 1
            # and then use a single class loss? That might work? It's a binary question of "was this image in the top k"
            # Should be good now? Need to test out above changes
            outputs = self._compute_loss_and_metrics(
                batch_size, outputs, torch.zeros(batch_size).long().to(logits.device)
            )

            return outputs

    # TODO: fix
    @overrides
    def _compute_loss_and_metrics(
        self,
        batch_size: int,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ):
        outputs["loss"] = torch.nn.functional.cross_entropy(outputs["logits"], labels) / batch_size
        self.accuracy(outputs["logits"], labels)
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
