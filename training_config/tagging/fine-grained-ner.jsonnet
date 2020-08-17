local train_data = std.extVar("CONLL_TRAIN_DATA_PATH");
local dev_data = std.extVar("CONLL_DEV_DATA_PATH");
// local train_data = "/net/nfs.corp/allennlp/dirkg/data/conll-formatted-ontonotes-5.0/data/train";
// local dev_data = "/net/nfs.corp/allennlp/dirkg/data/conll-formatted-ontonotes-5.0/data/development";
// local train_data = "/Users/dirkg/Documents/data/conll-formatted-ontonotes-5.0/data/train";
// local dev_data = "/Users/dirkg/Documents/data/conll-formatted-ontonotes-5.0/data/development";

{
    "dataset_reader": {
        "type": "ontonotes_ner",
        "coding_scheme": "BIOUL",
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            },
            "token_characters": {
                "type": "characters"
            },
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        }
    },
    "train_data_path": train_data,
    "validation_data_path": dev_data,
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 64
        }
    },
    "model": {
        "type": "crf_tagger",
        "dropout": 0.5,
        "encoder": {
            "type": "stacked_bidirectional_lstm",
            "hidden_size": 200,
            "input_size": 1202,
            "num_layers": 2,
            "recurrent_dropout_probability": 0.5,
            "use_highway": true
        },
        "feedforward": {
            "activations": "tanh",
            "dropout": 0.5,
            "hidden_dims": 400,
            "input_dim": 400,
            "num_layers": 1
        },
        "include_start_end_transitions": false,
        "initializer": {
          "regexes": [
            [
                ".*tag_projection_layer.*weight",
                {
                    "type": "xavier_uniform"
                }
            ],
            [
                ".*tag_projection_layer.*bias",
                {
                    "type": "zero"
                }
            ],
            [
                ".*feedforward.*weight",
                {
                    "type": "xavier_uniform"
                }
            ],
            [
                ".*feedforward.*bias",
                {
                    "type": "zero"
                }
            ],
            [
                ".*weight_ih.*",
                {
                    "type": "xavier_uniform"
                }
            ],
            [
                ".*weight_hh.*",
                {
                    "type": "orthogonal"
                }
            ],
            [
                ".*bias_ih.*",
                {
                    "type": "zero"
                }
            ],
            [
                ".*bias_hh.*",
                {
                    "type": "lstm_hidden_bias"
                }
            ]
          ]
        },
        "label_encoding": "BIOUL",
        "regularizer": {
          "regexes": [
            [
                "scalar_parameters",
                {
                    "alpha": 0.001,
                    "type": "l2"
                }
            ]
          ]
        },
        "text_field_embedder": {
          "token_embedders": {
            "elmo": {
                "type": "elmo_token_embedder",
                "do_layer_norm": false,
                "dropout": 0
            },
            "token_characters": {
                "type": "character_encoding",
                "embedding": {
                    "embedding_dim": 25,
                    "sparse": true,
                    "vocab_namespace": "token_characters"
                },
                "encoder": {
                    "type": "lstm",
                    "hidden_size": 128,
                    "input_size": 25,
                    "num_layers": 1
                }
            },
            "tokens": {
                "type": "embedding",
                "embedding_dim": 50,
                "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.50d.txt.gz",
                "sparse": true,
                "trainable": true
            }
          }
        },
        "verbose_metrics": true
    },
    "trainer": {
        "grad_norm": 5,
        "num_epochs": 30,
        "optimizer": {
            "type": "dense_sparse_adam",
            "lr": 0.001
        },
        "patience": 25,
        "validation_metric": "+f1-measure-overall"
    }
}
