import logging
from os import PathLike
from typing import (
    Dict,
    # List,
    Union,
    Optional,
    Tuple,
    Iterable,
)
import json

from overrides import overrides
import torch
from torch import Tensor

from allennlp.common.lazy import Lazy
from allennlp.common.file_utils import cached_path
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ArrayField, LabelField, ListField, TextField
from allennlp.data.image_loader import ImageLoader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.modules.vision.grid_embedder import GridEmbedder
from allennlp.modules.vision.region_detector import RegionDetector

from allennlp_models.vision.dataset_readers import utils
from allennlp_models.vision.dataset_readers.vision_reader import VisionReader

logger = logging.getLogger(__name__)


@DatasetReader.register("vgqa")
class VGQAReader(VisionReader):
    """
    Parameters
    ----------
    image_dir: `str`
        Path to directory containing `png` image files.
    image_loader: `ImageLoader`
        The image loader component used to load the images.
    image_featurizer: `Lazy[GridEmbedder]`
        The backbone image processor (like a ResNet), whose output will be passed to the region
        detector for finding object boxes in the image.
    region_detector: `Lazy[RegionDetector]`
        For pulling out regions of the image (both coordinates and features) that will be used by
        downstream models.
    answer_vocab: `Union[Vocabulary, str]`, optional
        The vocabulary to use for answers. The reader will look into the `"answers"` namespace
        in the vocabulary to find possible answers.
        If this is given, the reader only outputs instances with answers contained in this vocab.
        If this is not given, the reader outputs all instances with all answers.
        If this is a URL or filename, we will download a previously saved vocabulary from there.
    feature_cache_dir: `Union[str, PathLike]`, optional
        An optional directory to cache the featurized images in. Featurizing images takes a long
        time, and many images are duplicated, so we highly recommend to use this cache.
    tokenizer: `Tokenizer`, optional
        The `Tokenizer` to use to tokenize the text. By default, this uses the tokenizer for
        `"bert-base-uncased"`.
    token_indexers: `Dict[str, TokenIndexer]`, optional
        The `TokenIndexer` to use. By default, this uses the indexer for `"bert-base-uncased"`.
    cuda_device: `Union[int, torch.device]`, optional
        Either a torch device or a GPU number. This is the GPU we'll use to featurize the images.
    max_instances: `int`, optional
        For debugging, you can use this parameter to limit the number of instances the reader
        returns.
    image_processing_batch_size: `int`
        The number of images to process at one time while featurizing. Default is 8.
    """

    def __init__(
        self,
        image_dir: Optional[Union[str, PathLike]] = None,
        *,
        image_loader: Optional[ImageLoader] = None,
        image_featurizer: Optional[Lazy[GridEmbedder]] = None,
        region_detector: Optional[Lazy[RegionDetector]] = None,
        answer_vocab: Optional[Union[Vocabulary, str]] = None,
        feature_cache_dir: Optional[Union[str, PathLike]] = None,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Optional[Dict[str, TokenIndexer]] = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        max_instances: Optional[int] = None,
        image_processing_batch_size: int = 8,
        write_to_cache: bool = True,
    ) -> None:
        run_featurization = image_loader and image_featurizer and region_detector
        if image_dir is None and run_featurization:
            raise ValueError(
                "Because of the size of the image datasets, we don't download them automatically. "
                "Please go to https://visualgenome.org/api/v0/api_home.html, download the datasets you need, "
                "and set the image_dir parameter to point to your download location. This dataset "
                "reader does not care about the exact directory structure. It finds the images "
                "wherever they are."
            )

        super().__init__(
            image_dir,
            image_loader=image_loader,
            image_featurizer=image_featurizer,
            region_detector=region_detector,
            feature_cache_dir=feature_cache_dir,
            tokenizer=tokenizer,
            token_indexers=token_indexers,
            cuda_device=cuda_device,
            max_instances=max_instances,
            image_processing_batch_size=image_processing_batch_size,
            write_to_cache=write_to_cache,
        )

        # read answer vocab
        if answer_vocab is None:
            self.answer_vocab = None
        else:
            if isinstance(answer_vocab, str):
                answer_vocab = cached_path(answer_vocab, extract_archive=True)
                answer_vocab = Vocabulary.from_files(answer_vocab)
            self.answer_vocab = frozenset(
                utils.preprocess_answer(a)
                for a in answer_vocab.get_token_to_index_vocabulary("answers").keys()
            )

    @overrides
    def _read(self, file_path: str):
        # if the splits are using slicing syntax, honor it
        question_slice, file_path = utils.get_data_slice(file_path)

        file_path = cached_path(file_path, extract_archive=True)

        logger.info("Reading file at %s", file_path)
        questions = []
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        for data in dataset:
            for qa in data["qas"]:
                questions.append(qa)
        questions = questions[question_slice]

        question_dicts = list(self.shard_iterable(questions))
        processed_images: Iterable[Optional[Tuple[Tensor, Tensor]]]
        if self.produce_featurized_images:
            # It would be much easier to just process one image at a time, but it's faster to process
            # them in batches. So this code gathers up instances until it has enough to fill up a batch
            # that needs processing, and then processes them all.

            filenames = [f"{question_dict['image_id']}.jpg" for question_dict in question_dicts]
            logger.info("images size: %s", len(self.images))
            try:
                processed_images = self._process_image_paths(
                    self.images[filename] for filename in filenames
                )
            except KeyError as e:
                print(self.images)
                missing_id = e.args[0]
                raise KeyError(
                    missing_id,
                    f"We could not find an image with the id {missing_id}. "
                    "Because of the size of the image datasets, we don't download them automatically. "
                    "Please go to https://visualqa.org/download.html, download the datasets you need, "
                    "and set the image_dir parameter to point to your download location. This dataset "
                    "reader does not care about the exact directory structure. It finds the images "
                    "wherever they are.",
                )
        else:
            processed_images = [None for _ in range(len(question_dicts))]

        logger.info("Reading the dataset")
        failed_instances_count = 0
        attempted_instances_count = 0
        for qa, processed_image in zip(question_dicts, processed_images):
            question = qa["question"]
            answer = utils.preprocess_answer(qa["answer"])
            qa_id = qa["qa_id"]

            instance = self.text_to_instance(
                qa_id,
                question,
                answer,
                processed_image,
            )

            attempted_instances_count += 1
            if instance is not None:
                yield instance
            else:
                failed_instances_count += 1

        failed_instances_fraction = failed_instances_count / attempted_instances_count
        logger.warning(f"{failed_instances_fraction*100:.0f}% of instances failed.")

    @overrides
    def text_to_instance(
        self,  # type: ignore
        qa_id: int,
        question: str,
        answer: Optional[str],
        image: Union[str, Tuple[Tensor, Tensor]],
        use_cache: bool = True,
        keep_impossible_questions: bool = True,
    ) -> Optional[Instance]:
        question_field = TextField(self._tokenizer.tokenize(question), None)

        fields: Dict[str, Field] = {
            "question": question_field,
        }

        if isinstance(image, str):
            features, coords = next(self._process_image_paths([image], use_cache=use_cache))
        else:
            features, coords = image

        fields["box_features"] = ArrayField(features)
        fields["box_coordinates"] = ArrayField(coords)
        fields["box_mask"] = ArrayField(
            features.new_ones((features.shape[0],), dtype=torch.bool),
            padding_value=False,
            dtype=torch.bool,
        )

        if answer is not None:
            labels_fields = []
            weights = []
            if (not self.answer_vocab or answer in self.answer_vocab) or keep_impossible_questions:
                labels_fields.append(LabelField(answer, label_namespace="answers"))
                weights.append(1.0)

            if len(labels_fields) <= 0:
                return None

            fields["label_weights"] = ArrayField(torch.tensor(weights))
            fields["labels"] = ListField(labels_fields)

        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance["question"].token_indexers = self._token_indexers  # type: ignore
