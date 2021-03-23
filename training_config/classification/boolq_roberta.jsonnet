local transformer_model = "roberta-large";
local transformer_dim = 1024;

{
  "dataset_reader":{
    "type": "boolq",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
      }
    },
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model,
    }
  },
  "train_data_path": "https://storage.googleapis.com/allennlp-public-data/BoolQ.zip!BoolQ/train.jsonl",
  "validation_data_path": "https://storage.googleapis.com/allennlp-public-data/BoolQ.zip!BoolQ/val.jsonl",
  "test_data_path": "https://storage.googleapis.com/allennlp-public-data/BoolQ.zip!BoolQ/test.jsonl",
  "model": {
    "type": "basic_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
        }
      }
    },
    "seq2vec_encoder": {
       "type": "bert_pooler",
       "pretrained_model": transformer_model,
       "dropout": 0.1,
    },
    "namespace": "tags",
    "num_labels": 2,
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": ["tokens"],
      "batch_size" : 2
    }
  },
  "trainer": {
    "num_epochs": 10,
    "validation_metric": "+accuracy",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": 10,
      "num_steps_per_epoch": 3088,
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 1e-5,
      "weight_decay": 0.1,
    },
    "num_gradient_accumulation_steps": 16,
  },
}
