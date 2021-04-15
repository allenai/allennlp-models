import logging
from collections import Counter
from functools import lru_cache
from os import PathLike
from typing import (
    Dict,
    List,
    Union,
    Optional,
    MutableMapping,
    NamedTuple,
    Tuple,
    Iterable,
)
import json
import re

from overrides import overrides
import torch
from torch import Tensor

from allennlp.common.lazy import Lazy
from allennlp.common.file_utils import cached_path, LocalCacheResource
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ArrayField, LabelField, ListField, TextField
from allennlp.data.image_loader import ImageLoader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.modules.vision.grid_embedder import GridEmbedder
from allennlp.modules.vision.region_detector import RegionDetector

from allennlp_models.vision.dataset_readers.vision_reader import VisionReader

logger = logging.getLogger(__name__)

@lru_cache(maxsize=None)
def preprocess_answer(answer: str) -> str:
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(",", "")
    return answer


def get_score(count: int) -> float:
    return min(1.0, count / 3)


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
    multiple_answers_per_question: `bool`
        VQA questions have multiple answers. By default, we use all of them, and give more
        points to the more common answer. But VQA also has a special answer, the so-called
        "multiple choice answer". If this is set to `False`, we only use that answer.
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
        multiple_answers_per_question: bool = True,
        write_to_cache: bool = True,
    ) -> None:
        run_featurization = image_loader and image_featurizer and region_detector
        if image_dir is None and run_featurization:
            raise ValueError(
                "Because of the size of the image datasets, we don't download them automatically. "
                "Please go to https://visualqa.org/download.html, download the datasets you need, "
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
                preprocess_answer(a)
                for a in answer_vocab.get_token_to_index_vocabulary("answers").keys()
            )

        if self.produce_featurized_images:
            # normalize self.images some more
            # At this point, self.images maps filenames to full paths, but we want to map image ids to full paths.
            filename_re = re.compile(r".*(\d{12})\.((jpg)|(png))")

            def id_from_filename(filename: str) -> Optional[int]:
                match = filename_re.fullmatch(filename)
                if match is None:
                    return None
                return int(match.group(1))

            self.images = {
                id_from_filename(name): full_path for name, full_path in self.images.items()
            }
            if None in self.images:
                del self.images[None]

        self.multiple_answers_per_question = multiple_answers_per_question

    # todo: implement this
    def process_answer(self, answer):
        return answer

    @overrides
    def _read(self, file_path: str)):
        file_path = cached_path(file_path, extract_archive=True)

        logger.info("Reading file at %s", file_path)
        yielded_relation_count = 0
        with open_compressed(file_path) as dataset_file:
            dataset = json.load(dataset_file)

        # todo: process images
        processed_images = {}

        logger.info("Reading the dataset")
        for elem in dataset:
            for qa in self.shard_iterable(elem["qas"]):
                question = qa['question']
                answer = self.process_answer(qa['answer'])
                qa_id = qa['qa_id']
                processed_image = processed_images[qa['image_id']]

                instance = self.text_to_instance(
                    qa_id,
                    question,
                    answer,
                    processed_image,
                    is_training=True,
                )

                yield instance

    @overrides
    def text_to_instance(
        self,  # type: ignore
        question: str,
        image: Union[str, Tuple[Tensor, Tensor]],
        answer_counts: Optional[MutableMapping[str, int]] = None,
        *,
        use_cache: bool = True,
    ) -> Optional[Instance]:
        tokenized_question = self._tokenizer.tokenize(question)
        question_field = TextField(tokenized_question, None)

        fields: Dict[str, Field] = {
            "question": question_field,
        }

        if image is not None:
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

        if answer_counts is not None:
            answer_fields = []
            weights = []

            for answer, count in answer_counts.items():
                if self.answer_vocab is None or answer in self.answer_vocab:
                    answer_fields.append(LabelField(answer, label_namespace="answers"))
                    weights.append(get_score(count))

            if len(answer_fields) <= 0:
                return None

            fields["labels"] = ListField(answer_fields)
            fields["label_weights"] = ArrayField(torch.tensor(weights))

        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance["question"].token_indexers = self._token_indexers  # type: ignore

    def _get_answers_by_question_id(self, split):
        answers_by_question_id = {}
        if split.annotations is not None:
            # Pre-processing the annotations is time-consuming, so we don't want to
            # have to re-do it each time we call read(). So we cache this result.
            annotations_path = cached_path(split.annotations, extract_archive=True)
            with LocalCacheResource(split.annotations + "-cache", annotations_path) as cache:
                if cache.cached():
                    logger.info(
                        "Reading annotation answer counts from cache at %s",
                        cache.path,
                    )
                    with cache.reader() as f:
                        answers_by_question_id = json.load(f)
                else:
                    logger.info("Calculating annotation answer counts...")
                    with open(annotations_path) as f:
                        annotations = json.load(f)
                    for a in annotations["annotations"]:
                        qid = a["question_id"]
                        answer_counts: MutableMapping[str, int] = Counter()
                        if self.multiple_answers_per_question:
                            for answer in (answer_dict["answer"] for answer_dict in a["answers"]):
                                answer_counts[preprocess_answer(answer)] += 1
                        else:
                            answer_counts[preprocess_answer(a["multiple_choice_answer"])] = 1
                        answers_by_question_id[str(qid)] = answer_counts
                    logger.info("Caching annotation answer counts to %s", cache.path)
                    with cache.writer() as f:
                        json.dump(answers_by_question_id, f)
        return answers_by_question_id
