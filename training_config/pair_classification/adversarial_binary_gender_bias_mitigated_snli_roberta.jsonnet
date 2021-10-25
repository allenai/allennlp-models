local transformer_model = "roberta-large";
local transformer_dim = 1024;

{
  "dataset_reader": {
    "type": "snli",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model,
      "add_special_tokens": false
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": 512
      }
    }
  },
  "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_train.jsonl",
  "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_dev.jsonl",
  "model": {
    "type": "adversarial_bias_mitigator",
    "predictor": {
      "_pretrained": {
        "archive_file": "https://storage.googleapis.com/allennlp-public-models/snli-roberta.2021-03-11.tar.gz",
        "module_path": "",
        "freeze": false
      }
    },
    "adversary": {
        "type": "feedforward_regression_adversary",
        "feedforward": {
            "input_dim": 3,
            "num_layers": 1,
            "hidden_dims": 1,
            "activations": "linear"
        }
    },
    "bias_direction": {
        "type": "two_means",
        "seed_word_pairs_file": "https://raw.githubusercontent.com/tolga-b/debiaswe/4c3fa843ffff45115c43fe112d4283c91d225c09/data/definitional_pairs.json",
        "tokenizer": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
          "max_length": 512
        }
    },
    "predictor_output_key": "probs"
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 32
    }
  },
  "trainer": {
    "num_epochs": 10,
    "validation_metric": "+accuracy",
    "callbacks": [
        "adversarial_bias_mitigator_backward"
    ],
    "optimizer": {
        "type": "multi",
        "optimizers": {
            "predictor": {
                "type": "adam",
                "lr": 1e-5
            },
            "adversary": {
                "type": "adam",
                "lr": 1e-5
            },
            "default": {
                "type": "adam",
                "lr": 1e-5
            }
        },
        "parameter_groups": [
            [
                [
                    "^predictor"
                ],
                {
                    "optimizer_name": "predictor"
                }
            ],
            [
                [
                    "^adversary"
                ],
                {
                    "optimizer_name": "adversary"
                }
            ]
        ]
    }
  }
}