import torch
from torch.nn.parallel.data_parallel import DataParallel

from allennlp.common.testing import AllenNlpTestCase, requires_multi_gpu

from allennlp_models.rc.modules.seq2seq_encoders.stacked_self_attention import (
    StackedSelfAttentionEncoder,
)


class TestStackedSelfAttention(AllenNlpTestCase):
    def test_get_dimension_is_correct(self):
        encoder = StackedSelfAttentionEncoder(
            input_dim=9,
            hidden_dim=12,
            projection_dim=6,
            feedforward_hidden_dim=5,
            num_layers=3,
            num_attention_heads=3,
        )
        assert encoder.get_input_dim() == 9
        # hidden_dim + projection_dim
        assert encoder.get_output_dim() == 12

    def test_stacked_self_attention_can_run_foward(self):
        # Correctness checks are elsewhere - this is just stacking
        # blocks which are already well tested, so we just check shapes.
        encoder = StackedSelfAttentionEncoder(
            input_dim=9,
            hidden_dim=12,
            projection_dim=9,
            feedforward_hidden_dim=5,
            num_layers=3,
            num_attention_heads=3,
        )
        inputs = torch.randn([3, 5, 9])
        encoder_output = encoder(inputs, None)
        assert list(encoder_output.size()) == [3, 5, 12]

    @requires_multi_gpu
    def test_stacked_self_attention_can_run_foward_on_multiple_gpus(self):
        encoder = StackedSelfAttentionEncoder(
            input_dim=9,
            hidden_dim=12,
            projection_dim=9,
            feedforward_hidden_dim=5,
            num_layers=3,
            num_attention_heads=3,
        ).to(0)
        parallel_encoder = DataParallel(encoder, device_ids=[0, 1])
        inputs = torch.randn([3, 5, 9]).to(0)
        encoder_output = parallel_encoder(inputs, None)
        assert list(encoder_output.size()) == [3, 5, 12]
