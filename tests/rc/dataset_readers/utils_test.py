import pytest

from allennlp_models.rc.dataset_readers.utils import char_span_to_token_span


@pytest.mark.parametrize(
    "token_offsets, character_span, expected_result",
    [
        ([(0, 3), (4, 4), (5, 8)], (5, 8), ((2, 2), False)),
        ([(0, 3), (4, 4), (5, 8)], (4, 8), ((1, 2), False)),
        ([(0, 3), (4, 4), (5, 8)], (0, 8), ((0, 2), False)),
        ([(0, 3), (4, 4), (5, 8)], (1, 8), ((0, 2), True)),
        ([(0, 3), (4, 4), (5, 8)], (7, 8), ((2, 2), True)),
        ([(0, 3), (4, 4), (5, 8)], (7, 9), ((2, 2), True)),
    ],
)
def test_char_span_to_token_span(token_offsets, character_span, expected_result):
    assert char_span_to_token_span(token_offsets, character_span) == expected_result


def test_char_span_to_token_span_throws():
    with pytest.raises(ValueError):
        char_span_to_token_span([(0, 3), (4, 4), (5, 8)], (7, 19))
