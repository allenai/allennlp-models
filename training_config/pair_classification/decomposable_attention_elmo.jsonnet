// Configuraiton for a textual entailment model based on:
//  Parikh, Ankur P. et al. “A Decomposable Attention Model for Natural Language Inference.” EMNLP (2016).
{
  "dataset_reader": {
    "type": "snli",
    "token_indexers": {
      "elmo": {
        "type": "elmo_characters"
      }
    },
    "tokenizer": {
      "end_tokens": ["@@NULL@@"]
    }
  },
  "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_train.jsonl",
  "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_dev.jsonl",
  "model": {
    "type": "decomposable_attention",
    "text_field_embedder": {
      "token_embedders": {
        "elmo": {
            "type": "elmo_token_embedder",
            "do_layer_norm": false,
            "dropout": 0.2
        }
      }
    },
    "attend_feedforward": {
      "input_dim": 1024,
      "num_layers": 2,
      "hidden_dims": 200,
      "activations": "relu",
      "dropout": 0.2
    },
    "matrix_attention": {
      "type": "dot_product"
    },
    "compare_feedforward": {
      "input_dim": 2048,
      "num_layers": 2,
      "hidden_dims": 200,
      "activations": "relu",
      "dropout": 0.2
    },
    "aggregate_feedforward": {
      "input_dim": 400,
      "num_layers": 2,
      "hidden_dims": [200, 3],
      "activations": ["relu", "linear"],
      "dropout": [0.2, 0.0]
    },
     "initializer": {
       "regexes": [
         [".*linear_layers.*weight", {"type": "xavier_normal"}],
         [".*token_embedder_tokens\\._projection.*weight", {"type": "xavier_normal"}]
       ]
     }
   },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 64
    }
  },
  "trainer": {
    "num_epochs": 140,
    "patience": 20,
    "cuda_device": -1,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    }
  }
}
