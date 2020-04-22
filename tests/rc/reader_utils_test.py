import pytest

from allennlp_models.rc.common.reader_utils import char_span_to_token_span


@pytest.mark.parametrize(
    "token_offsets, character_span, expected_result",
    [
        ([(0, 3), (3, 4), (4, 8)], (3, 7), ((1, 2), True)),
        ([(0, 3), (3, 4), (4, 8)], (0, 7), ((0, 2), True)),
        ([(0, 3), (3, 4), (4, 8)], (1, 7), ((0, 2), False)),
    ],
)
def test_char_span_to_token_span(token_offsets, character_span, expected_result):
    assert char_span_to_token_span(token_offsets, character_span) == expected_result
