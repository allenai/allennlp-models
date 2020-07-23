// Configuration for a RoBERTa sentiment analysis classifier, using the binary Stanford Sentiment Treebank (Socher at al. 2013).

local transformer_model = "roberta-large";
local transformer_dim = 1024;

{
  "dataset_reader":{
    "type": "sst_tokens",
    "use_subtrees": true,
    "granularity": "2-class",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model
      }
    },
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model
    }
  },
  "validation_dataset_reader": self.dataset_reader + {
    "use_subtrees": false
  },
  "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/train.txt",
  "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/dev.txt",
  "test_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/test.txt",
  "model": {
    "type": "basic_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model
        }
      }
    },
    "seq2vec_encoder": {
       "type": "bert_pooler",
       "pretrained_model": transformer_model,
       "dropout": 0.1,
    },
    "namespace": "tags"
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": ["tokens"],
      "batch_size" : 32
    }
  },
  "trainer": {
    "num_epochs": 20,
    "patience": 5,
    "validation_metric": "+accuracy",
    "learning_rate_scheduler": {
      "type": "linear_with_warmup",
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "weight_decay": 0.01,
      "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
      "lr": 1e-5,
      "eps": 1e-8,
      "correct_bias": true
    },
  }
}
