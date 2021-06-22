local model_name = "epwalsh/bert-xsmall-dummy";
{
  "dataset_reader": {
    "type": "flickr30k",
    "image_dir": "test_fixtures/vision/images/flickr30k",
    "data_dir": "test_fixtures/vision/flickr30k/sentences",
    "image_loader": "torch",
    "image_featurizer": "null",
    "featurize_captions": false,
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
  "train_data_path": "test_fixtures/vision/flickr30k/tiny-dev.txt",
  "validation_data_path": "test_fixtures/vision/flickr30k/tiny-dev.txt",
  "model": {
    "type": "vilbert_ir_from_huggingface",
    "model_name": model_name,
    "image_feature_dim": 10,
    "image_num_hidden_layers": 1,
    "image_hidden_size": 200,
    "image_num_attention_heads": 1,
    "image_intermediate_size": 50,
    "image_attention_dropout": 0.0,
    "image_hidden_dropout": 0.0,
    "image_biattention_id": [0, 1],
    "image_fixed_layer": 0,

    "text_biattention_id": [0, 1],
    "text_fixed_layer": 0,

    "combined_hidden_size": 200,
    "combined_num_attention_heads": 4,

    "pooled_output_dim": 100,
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
  }
}
