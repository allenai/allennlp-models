local bert_model = "bert-base-uncased";

{
    "dataset_reader": {
      "type": "srl",
      "bert_model_name": bert_model,
    },

    "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "batch_size" : 32
      }
    },

    // "train_data_path": "/net/nfs.corp/allennlp/data/ontonotes/conll-formatted-ontonotes-5.0/data/train",
    // "validation_data_path": "/net/nfs.corp/allennlp/data/ontonotes/conll-formatted-ontonotes-5.0/data/development",
    "train_data_path": std.extVar("SRL_TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("SRL_VALIDATION_DATA_PATH"),

    "model": {
        "type": "srl_bert",
        "embedding_dropout": 0.1,
        "bert_model": bert_model,
    },

    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 5e-5,
            "correct_bias": false,
            "weight_decay": 0.01,
            "parameter_groups": [
              [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}],
            ],
        },

        "learning_rate_scheduler": {
            "type": "slanted_triangular",
        },
        "checkpointer": {
            "keep_most_recent_by_count": 2,
        },
        "grad_norm": 1.0,
        "num_epochs": 15,
        "validation_metric": "+f1-measure-overall",
    },
}
