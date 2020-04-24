import pytest

from allennlp_models.rc.common.reader_utils import char_span_to_token_span


@pytest.mark.parametrize(
    "token_offsets, character_span, expected_result",
    [
        ([(0, 3), (4, 4), (5, 8)], (5, 8), ((2, 2), False)),
        ([(0, 3), (4, 4), (5, 8)], (4, 8), ((1, 2), False)),
        ([(0, 3), (4, 4), (5, 8)], (0, 8), ((0, 2), False)),
        ([(0, 3), (4, 4), (5, 8)], (1, 8), ((0, 2), True)),
        ([(0, 3), (4, 4), (5, 8)], (7, 8), ((2, 2), True)),
    ],
)
def test_char_span_to_token_span(token_offsets, character_span, expected_result):
    assert char_span_to_token_span(token_offsets, character_span) == expected_result
