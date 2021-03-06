local model_name = "epwalsh/bert-xsmall-dummy";
{
  "dataset_reader": {
    "type": "vqav2",
    "image_dir": "test_fixtures/vision/images/vqav2",
    "image_loader": "torch",
    "image_featurizer": "null",
    "region_detector": {
      "type": "random",
      "seed": 322
    },
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": model_name
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": model_name
      }
    }
  },
  "train_data_path": "unittest",
  "validation_data_path": "unittest",
  "vocabulary": {"min_count": {"answers": 2}},
  "datasets_for_vocab_creation": ["train"],
  "model": {
    "type": "vqa_vilbert_from_huggingface",
    "model_name": model_name,
    "image_feature_dim": 10,
    "image_num_hidden_layers": 1,
    "image_hidden_size": 20,
    "image_num_attention_heads": 1,
    "image_intermediate_size": 5,
    "image_attention_dropout": 0.0,
    "image_hidden_dropout": 0.0,
    "image_biattention_id": [0, 1],
    "image_fixed_layer": 0,

    "text_biattention_id": [0, 1],
    "text_fixed_layer": 0,

    "combined_hidden_size": 20,
    "combined_num_attention_heads": 2,

    "pooled_output_dim": 20,
    "fusion_method": "sum",
    "pooled_dropout": 0.0,
  },
  "data_loader": {
    "batch_size": 32
  },
  "trainer": {
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 0.00005
    },
    "num_epochs": 1,
  },
}
