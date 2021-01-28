from pathlib import Path
from typing import Dict, Optional, List
import logging
import os
import glob
import hashlib
import ftfy

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)


@DatasetReader.register("cnn_dm")
class CNNDailyMailDatasetReader(DatasetReader):
    """
    Reads the CNN/DailyMail dataset for text summarization.

    The output of `read` is a list of `Instance` s with the fields:
        source_tokens : `TextField` and
        target_tokens : `TextField`

    # Parameters

    source_tokenizer : `Tokenizer`, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to `SpacyTokenizer()`.
    target_tokenizer : `Tokenizer`, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to `source_tokenizer`.
    source_token_indexers : `Dict[str, TokenIndexer]`, optional
        Indexers used to define input (source side) token representations. Defaults to
        `{"tokens": SingleIdTokenIndexer()}`.
    target_token_indexers : `Dict[str, TokenIndexer]`, optional
        Indexers used to define output (target side) token representations. Defaults to
        `source_token_indexers`.
    source_max_tokens : `int`, optional
        Maximum number of tokens in source sequence.
    target_max_tokens : `int`, optional
        Maximum number of tokens in target sequence.
    """

    def __init__(
        self,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        source_max_tokens: Optional[int] = None,
        target_max_tokens: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self._source_tokenizer = source_tokenizer or SpacyTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens

    @staticmethod
    def _hashhex(url):
        h = hashlib.sha1()
        h.update(url)
        return h.hexdigest()

    @staticmethod
    def _sanitize_story_line(line):
        line = ftfy.fix_encoding(line)

        sentence_endings = [".", "!", "?", "...", "'", "`", '"', ")", "\u2019", "\u201d"]

        # CNN stories always start with "(CNN)"
        if line.startswith("(CNN)"):
            line = line[len("(CNN)") :]

        # Highlight are essentially bullet points and don't have proper sentence endings
        if line[-1] not in sentence_endings:
            line += "."

        return line

    @staticmethod
    def _read_story(story_path: str):
        article: List[str] = []
        summary: List[str] = []
        highlight = False

        with open(story_path, "r") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue

                if line == "@highlight":
                    highlight = True
                    continue
                line = CNNDailyMailDatasetReader._sanitize_story_line(line)
                (summary if highlight else article).append(line)

        return " ".join(article), " ".join(summary)

    @staticmethod
    def _strip_extension(filename: str) -> str:
        return os.path.splitext(filename)[0]

    @overrides
    def _read(self, file_path: str):
        # Reset exceeded counts
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0

        url_file_path = cached_path(file_path, extract_archive=True)
        data_dir = os.path.join(os.path.dirname(url_file_path), "..")
        cnn_stories_path = os.path.join(data_dir, "cnn_stories")
        dm_stories_path = os.path.join(data_dir, "dm_stories")

        cnn_stories = {Path(s).stem for s in glob.glob(os.path.join(cnn_stories_path, "*.story"))}
        dm_stories = {Path(s).stem for s in glob.glob(os.path.join(dm_stories_path, "*.story"))}

        with open(url_file_path, "r") as url_file:
            for url in self.shard_iterable(url_file):
                url = url.strip()

                url_hash = self._hashhex(url.encode("utf-8"))

                if url_hash in cnn_stories:
                    story_base_path = cnn_stories_path
                elif url_hash in dm_stories:
                    story_base_path = dm_stories_path
                else:
                    raise ConfigurationError(
                        "Story with url '%s' and hash '%s' not found" % (url, url_hash)
                    )

                story_path = os.path.join(story_base_path, url_hash) + ".story"
                article, summary = self._read_story(story_path)

                if len(article) == 0 or len(summary) == 0 or len(article) < len(summary):
                    continue

                yield self.text_to_instance(article, summary)

    @overrides
    def text_to_instance(
        self, source_sequence: str, target_sequence: str = None
    ) -> Instance:  # type: ignore
        tokenized_source = self._source_tokenizer.tokenize(source_sequence)
        if self._source_max_tokens is not None and len(tokenized_source) > self._source_max_tokens:
            tokenized_source = tokenized_source[: self._source_max_tokens]

        source_field = TextField(tokenized_source)
        if target_sequence is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_sequence)
            if (
                self._target_max_tokens is not None
                and len(tokenized_target) > self._target_max_tokens
            ):
                tokenized_target = tokenized_target[: self._target_max_tokens]
            target_field = TextField(tokenized_target)
            return Instance({"source_tokens": source_field, "target_tokens": target_field})
        else:
            return Instance({"source_tokens": source_field})

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["source_tokens"]._token_indexers = self._source_token_indexers  # type: ignore
        if "target_tokens" in instance.fields:
            instance.fields["target_tokens"]._token_indexers = self._target_token_indexers  # type: ignore
