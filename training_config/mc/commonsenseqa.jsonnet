local transformer_model = 'roberta-large';

local epochs = 20;
local batch_size = 64;

local gpu_batch_size = 4;
local gradient_accumulation_steps = batch_size / gpu_batch_size;

{
  "dataset_reader": {
      "type": "commonsenseqa",
      "transformer_model_name": transformer_model,
      //"max_instances": 200  // debug setting
  },
  "train_data_path": "https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl",
  "validation_data_path": "https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl",
  //"test_data_path": "https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl"
  "model": {
      "type": "transformer_mc",
      "transformer_model": transformer_model,
  },
  "data_loader": {
    "shuffle": true,
    "batch_size": gpu_batch_size
  },
  "trainer": {
    "optimizer": {
      "type": "huggingface_adamw",
      "weight_decay": 0.01,
      "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
      "lr": 1e-5,
      "eps": 1e-8,
      "correct_bias": true
    },
    "learning_rate_scheduler": {
      "type": "linear_with_warmup",
      "warmup_steps": 100
    },
    // "grad_norm": 1.0,
    "num_epochs": epochs,
    "num_gradient_accumulation_steps": gradient_accumulation_steps,
    "patience": 3,
    "validation_metric": "+acc",
  },
  "random_seed": 42,
  "numpy_seed": 42,
  "pytorch_seed": 42,
}
