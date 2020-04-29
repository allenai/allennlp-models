from allennlp.data import Vocabulary
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer

# from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import GruSeq2SeqEncoder
from allennlp.nn.util import move_to_device

import torch

# from transformers import BartConfig

from allennlp_models.seq2seq import Bart


# TODO: remove this file, it's just for testing.


def main():
    bart_model_name = "bart-large-cnn"

    indexer = PretrainedTransformerIndexer(bart_model_name, namespace="tokens")
    tokenizer = indexer._allennlp_tokenizer

    data = (
        "We present BART, a denoising autoencoder for pretraining sequence-to-sequence models. BART is "
        "trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to "
        "reconstruct the original text. It uses a standard Tranformer-based neural machine translation "
        "architecture which, despite its simplicity, can be seen as generalizing BERT (due to the "
        "bidirectional encoder), GPT (with the left-to-right decoder), and many other more recent pretraining "
        "schemes. We evaluate a number of noising approaches, finding the best performance by both randomly "
        "shuffling the order of the original sentences and using a novel in-filling scheme, where spans of "
        "text are replaced with a single mask token. BART is particularly effective when fine tuned for text "
        "generation but also works well for comprehension tasks. It matches the performance of RoBERTa with "
        "comparable training resources on GLUE and SQuAD, achieves new stateof-the-art results on a range of "
        "abstractive dialogue, question answering, and summarization tasks, with gains of up to 6 ROUGE. BART "
        "also provides a 1.1 BLEU increase over a back-translation system for machine translation, "
        "with only target language pretraining. We also report ablation experiments that replicate other "
        "pretraining schemes within the BART framework, to better measure which factors most influence "
        "end-task performance."
    )

    summary = (
        "BART is a denoising autoencoder for pretraining sequence-to-sequence models. BART is trained by "
        "corrupting text and learning to reconstruct it. It uses a standard Transformer-based architecture."
    )

    data_tokens = tokenizer.tokenize(data)
    vocab = Vocabulary.empty()
    inputs = indexer.tokens_to_indices(data_tokens, vocab)
    inputs = indexer.as_padded_tensor_dict(inputs, indexer.get_padding_lengths(inputs))
    summary_tokens = tokenizer.tokenize(summary)
    targets = indexer.tokens_to_indices(summary_tokens, vocab)
    targets = indexer.as_padded_tensor_dict(targets, indexer.get_padding_lengths(targets))

    # Add batch dimension
    for k in inputs.keys():
        inputs[k] = inputs[k].unsqueeze(0)
        targets[k] = targets[k].unsqueeze(0)

    inputs = move_to_device(inputs, 0)
    targets = move_to_device(targets, 0)

    # cfg = BartConfig.from_pretrained(bart_model_name)
    # encoder = GruSeq2SeqEncoder(cfg.d_model, cfg.d_model)

    # Some testing
    bart_allennlp = Bart(
        bart_model_name, vocab, max_decoding_steps=64, beam_size=5  # , encoder=encoder
    ).to("cuda")
    bart_allennlp.eval()
    with torch.no_grad():
        outputs = bart_allennlp({"tokens": inputs})
        bart_allennlp.make_output_human_readable(outputs)
        print(
            tokenizer.tokenizer.convert_tokens_to_string(
                [t.text for t in outputs["predicted_tokens"][0]]
            )
        )
    return


if __name__ == "__main__":
    main()
