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
# output shape: (batch_size, )
def calculate_hard_negatives(image_embeddings, text_embeddings):
    # for each image, find the 100 nearest neighbors
    # then, find 3 images with best l2 distances
    index = faiss.IndexFlatL2(list(image_embeddings)[1])
    index.add(processed_images)
    k = 5  # 100
    D, neighbors = index.search(processed_images, k)
    for caption_dict, processed_image, curr_neighbors in zip(
        caption_dicts, processed_images, neighbors
    ):
        for neighbor in curr_neighbors:
            # calculate L2 distance between neighbor and caption tokens
            caption_vec = caption_dict["caption"]


@Model.register("image_retrieval_vilbert")
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

        self.f1_metric = F1MultiLabelMeasure(average="micro")
        self.vqa_metric = VqaMeasure()

    # TODO: fix
    @overrides
    def forward(
        self,  # type: ignore
        box_features: torch.Tensor,
        box_coordinates: torch.Tensor,
        box_mask: torch.Tensor,
        question: TextFieldTensors,
        labels: Optional[torch.Tensor] = None,
        label_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        # collect outputs for this batch
        backbone_outputs = self.backbone(box_features, box_coordinates, box_mask, text)

        # calculate hard negatives for this batch
        # Shape: (batch_size, 4)
        candidate_images = calculate_hard_negatives(
            torch.mean(backbone_outputs["encoded_text"], 1),
            torch.mean(backbone_outputs["encoded_boxes"], 1),
        )

        # then, use them to do the normal calculations

        # TODO: this is copied, change to fit this model
        # Shape: (batch_size, num_labels)
        # TODO: change `backbone_outputs["pooled_boxes_and_text"]` to use backbone_output embeddings + hard negative info
        logits = self.classifier(backbone_outputs["pooled_boxes_and_text"])

        # Shape: (batch_size, num_labels)
        if self.is_multilabel:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=-1)

        outputs = {"logits": logits, "probs": probs}
        outputs = self._compute_loss_and_metrics(batch_size, outputs, labels, label_weights)

        return outputs

    # TODO: fix
    @overrides
    def _compute_loss_and_metrics(
        self,
        batch_size: int,
        outputs: torch.Tensor,
        label: torch.Tensor,
        label_weights: Optional[torch.Tensor] = None,
    ):

        # TODO: implement loss and metrics

        return outputs

    # TODO: fix
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        result = self.f1_metric.get_metric(reset)
        result["vqa_score"] = self.vqa_metric.get_metric(reset)["score"]
        return result

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

    default_predictor = "vilbert_vqa"
