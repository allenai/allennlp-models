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

from allennlp_models.vision.models.vision_text_model import VisionTextModel

logger = logging.getLogger(__name__)

# TODO: I think we might actually want to use the output of vilbert, maybe?
# IE calculate h_img and h_cls (like in the paper), and then do these calculations using those vectors?
# input shapes: (batch_size, embedding_dim)
# output shape: (batch_size, num_neighbors)
def calculate_hard_negatives(image_embeddings, text_embeddings, num_neighbors: int = 3):
    # for each image, find the 100 nearest neighbors
    # then, find 3 images with best l2 distances
    index = faiss.IndexFlatL2(list(image_embeddings)[1])
    index.add(image_embeddings)  # TODO: I think we need to rotate this input
    k = 10
    # D, neighbors = index.search(processed_images, k)
    D, neighbors = index.search(text_embeddings, k)
    for image_embedding, text_embedding, curr_neighbors in zip(
        image_embeddings, text_embeddings, neighbors
    ):
        for neighbor in curr_neighbors:
            # calculate L2 distance between neighbor and caption tokens
            caption_vec = caption_dict["caption"]


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
        fusion_method: str = "sum",
        dropout: float = 0.1,
        label_namespace: str = "answers",
        *,
        ignore_text: bool = False,
        ignore_image: bool = False
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
            is_multilabel=True,
            ignore_text=ignore_text,
            ignore_image=ignore_image,
        )

        from allennlp.training.metrics import F1MultiLabelMeasure
        from allennlp_models.vision.metrics.vqa import VqaMeasure

        self.classifier = torch.nn.Linear(pooled_output_dim, 1)

        self.f1_metric = F1MultiLabelMeasure(average="micro")
        self.vqa_metric = VqaMeasure()

    # TODO: fix
    @overrides
    def forward(
        self,  # type: ignore
        box_features: torch.Tensor,
        box_coordinates: torch.Tensor,
        box_mask: torch.Tensor,
        caption: TextFieldTensors,
    ) -> Dict[str, torch.Tensor]:

        batch_size = box_features.size(0)

        # collect outputs for this batch
        backbone_outputs = self.backbone(box_features, box_coordinates, box_mask, caption)

        # Get embeddings (do we want to just take the mean?)
        # Shape: (batch_size, pooled_output_dim)
        text_embeddings = backbone_outputs["encoded_text_pooled"]
        # Shape: (batch_size, pooled_output_dim)
        image_embeddings = backbone_outputs["encoded_boxes_pooled"]

        # TODO: do stuff with this
        logits = self.classifier(
            text_embeddings.unsqueeze(0) * image_embeddings.unsqueeze(1)
        ).squeeze(-1)
        probs = torch.softmax(logits, dim=-1)
        # for text_embedding in text_embeddings:
        #     curr_logits = self.classifier(text_embedding * image_embeddings)
        #     curr_probs = torch.softmax(curr_logits, dim=0)

        # Multiply each text embedding te_i by each image embedding ie_j
        # and then by weights w to get a score for each text/image pair

        # TODO: this is copied, change to fit this model
        # Shape: (batch_size, batch_size)?
        # TODO: change `backbone_outputs["pooled_boxes_and_text"]` to use backbone_output embeddings + hard negative info
        # logits = self.classifier(backbone_outputs["pooled_boxes_and_text"])

        # # Shape: (batch_size, num_labels)
        # if self.is_multilabel:
        #     probs = torch.sigmoid(logits)
        # else:
        #     probs = torch.softmax(logits, dim=-1)

        outputs = {"logits": logits, "probs": probs}
        outputs = self._compute_loss_and_metrics(batch_size, outputs)

        return outputs

    # TODO: fix
    @overrides
    def _compute_loss_and_metrics(
        self,
        batch_size: int,
        outputs: torch.Tensor,
    ):
        # TODO: make sure this is right
        # idea: the correct image for a caption i is image_i
        labels = torch.from_numpy(np.arange(0, batch_size))

        print(labels.device)
        labels.to(outputs["logits"].device)
        print(labels.device)

        outputs["loss"] = torch.nn.functional.cross_entropy(outputs["logits"], labels) / batch_size
        self.accuracy(outputs["logits"], labels)
        self.fbeta(outputs["probs"], labels)
        return outputs

    # TODO: fix
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.fbeta.get_metric(reset)
        accuracy = self.accuracy.get_metric(reset)
        metrics.update({"accuracy": accuracy})
        return metrics

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
