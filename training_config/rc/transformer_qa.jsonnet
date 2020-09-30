local transformer_model = 'bert-base-cased';

local epochs = 3;
local batch_size = 8;
local length_limit = 384;

{
  "dataset_reader": {
    "type": "transformer_squad2",
    "transformer_model_name": transformer_model,
    "length_limit": length_limit,
    // "max_instances": 1000,  // debug setting
  },
  "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/squad/squad-train-v2.0.json",
  "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/squad/squad-dev-v2.0.json",
  "vocabulary": {
    "type": "empty",
  },
  "model": {
    "type": "transformer_qa",
    "transformer_model_name": transformer_model,
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": batch_size,
    }
  },
  "trainer": {
    "optimizer": {
      "type": "huggingface_adamw",
      "weight_decay": 0.0,
      "parameter_groups": [
        [["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}],
      ],
      "lr": 2e-5,
      "eps": 1e-8
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": epochs,
      "cut_frac": 0.1,
    },
    "grad_clipping": 1.0,
    "num_epochs": epochs,
  },
  "random_seed": 42,
  "numpy_seed": 42,
  "pytorch_seed": 42,
}
