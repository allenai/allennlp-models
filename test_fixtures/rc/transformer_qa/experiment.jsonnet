local transformer_model = "albert-base-v2";
local epochs = 2;
local batch_size = 3;

// This defines the number of instances, not the number of questions. One question can end up as
// multiple instances.
local number_of_train_instances = 5;
local number_of_dev_instances = 5;

{
    "dataset_reader": {
        "type": "transformer_squad",
        "transformer_model_name": transformer_model,
        "skip_invalid_examples": true,
    },
    "validation_dataset_reader": self.dataset_reader + {
        "skip_invalid_examples": false,
    },
    "train_data_path": "test_fixtures/rc/squad.json",
    "validation_data_path": "test_fixtures/rc/squad.json",
    "model": {
        "type": "transformer_qa",
        "transformer_model_name": transformer_model,
    },
    "data_loader": {
        "batch_size": batch_size
    },
    "trainer": {
      "optimizer": {
        "type": "huggingface_adamw",
        "weight_decay": 0.0,
        "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
        "lr": 5e-5,
        "eps": 1e-8
      },
      "learning_rate_scheduler": {
        "type": "slanted_triangular",
        "num_epochs": epochs,
        "cut_frac": 0,
        "num_steps_per_epoch": std.ceil(number_of_train_instances / batch_size),
      },
      "grad_clipping": 1.0,
      "num_epochs": epochs,
      "cuda_device": -1
    },
}
