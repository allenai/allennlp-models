local transformer_model = 'bert-base-cased';

local epochs = 3;
local batch_size = 8;

{
  "dataset_reader": {
      "type": "transformer_squad",
      "transformer_model_name": transformer_model,
      "skip_invalid_examples": true,
      //"max_instances": 200  // debug setting
  },
  "validation_dataset_reader": self.dataset_reader + {
      "skip_invalid_examples": false,
  },
  "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/squad/squad-train-v1.1.json",
  "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/squad/squad-dev-v1.1.json",
  "model": {
      "type": "transformer_qa",
      "transformer_model_name": transformer_model,
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": batch_size
    }
  },
  "trainer": {
    "optimizer": {
      "type": "huggingface_adamw",
      "weight_decay": 0.0,
      "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
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
    "cuda_device": -1
  },
  "random_seed": 42,
  "numpy_seed": 42,
  "pytorch_seed": 42,
}
