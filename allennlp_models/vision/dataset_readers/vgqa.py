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

# todo: put these in utils
contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
manual_map = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]
period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile(r"(\d)(\,)(\d)")
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def process_punctuation(inText: str) -> str:
    outText = inText
    for p in punct:
        if (p + " " in inText or " " + p in inText) or (comma_strip.search(inText) is not None):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(input: str) -> str:
    output = []
    for word in input.lower().split():
        word = manual_map.get(word, word)
        if word not in articles:
            output.append(word)
        else:
            pass
    for index, word in enumerate(output):
        if word in contractions:
            output[index] = contractions[word]
    return " ".join(output)


# todo: also remove words like a/the/an/etc?
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
        multiple_answers_per_question: bool = False,
        write_to_cache: bool = True,
    ) -> None:
        run_featurization = image_loader and image_featurizer and region_detector
        if image_dir is None and run_featurization:
            # todo: update
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

            if None in self.images:
                del self.images[None]

        self.multiple_answers_per_question = multiple_answers_per_question

    @overrides
    def _read(self, file_path: str):
        # if the splits are using slicing syntax, honor it
        slice_match = re.match(r"(.*)\[([0123456789:]*)]", file_path)
        if slice_match is None:
            question_slice = slice(None, None, None)
        else:
            split_name = slice_match[1]
            slice_args = [int(a) if len(a) > 0 else None for a in slice_match[2].split(":")]
            question_slice = slice(*slice_args)

        file_path = cached_path(file_path.split("[")[0], extract_archive=True)

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
            try:
                logger.warning(f"file path for image: {filename}")
                logger.warning(f"file path for image: {self.images[filename]}")
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
            answer = preprocess_answer(qa["answer"])
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
        answer: str,
        image: Union[str, Tuple[Tensor, Tensor]],
        use_cache: bool = True,
    ) -> Optional[Instance]:
        question_field = TextField(self._tokenizer.tokenize(question), None)

        fields: Dict[str, Field] = {
            "question": question_field,
        }

        if image is None:
            return None

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

        if answer is not None:
            labels_fields = []
            weights = []
            if not self.answer_vocab or answer["answer"] in self.answer_vocab:
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
