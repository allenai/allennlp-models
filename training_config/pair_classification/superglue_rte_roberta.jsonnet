local transformer_model = "roberta-large";
local transformer_dim = 1024;

{
  "dataset_reader":{
    "type": "transformer_superglue_rte"
  },
  "train_data_path": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/RTE.zip!RTE/train.jsonl",
  "validation_data_path": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/RTE.zip!RTE/val.jsonl",
  "test_data_path": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/RTE.zip!RTE/test.jsonl",
  "model": {
    "type": "basic_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
          "max_length": 512
        }
      }
    },
    "seq2vec_encoder": {
       "type": "cls_pooler",
       "embedding_dim": transformer_dim,
    },
    "feedforward": {
      "input_dim": transformer_dim,
      "num_layers": 1,
      "hidden_dims": transformer_dim,
      "activations": "tanh"
    },
    "dropout": 0.1,
    "namespace": "tags"
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 32
    }
  },
  "trainer": {
    "num_epochs": 7,
    "validation_metric": "+accuracy",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-5,
      "weight_decay": 0.1,
    }
  }
}
