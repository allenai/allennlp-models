local bert_model = "bert-base-cased";

// This is not really a training config. We train on a tiny amount of data and rely mostly on the pre-trained weights
// from BERT.

{
  "dataset_reader": {
    "type": "masked_language_modeling",
    "model_name": bert_model,
  },
  "train_data_path": "test_fixtures/lm/language_model/sentences.txt",
  "validation_data_path": "test_fixtures/lm/language_model/sentences.txt",
  "model": {
    "type": "masked_language_model",
    "target_namespace": "tags",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": bert_model
        }
      }
    },
    "language_model_head": {
      "type": "bert",
      "model_name": bert_model
    }
  },
  "data_loader": {
    "batch_size": 32
  },
  "trainer": {
    "num_epochs": 1,
    "cuda_device" : -1,
    "optimizer": {
      "type": "sgd",
      "lr": 0.01
    }
  }
}
