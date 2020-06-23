local transformer_model = 'roberta-base';

local epochs = std.extVar('epochs');
local batch_size = std.extVar('batch_size');
local weight_decay = std.extVar('weight_decay');
local lr = std.extVar('lr');
local cut_frac = std.extVar('cut_frac');
local grad_norm = std.extVar('grad_norm');
local correct_bias = std.extVar('correct_bias');

{
  "dataset_reader": {
      "type": "piqa",
      "transformer_model_name": transformer_model,
      //"max_instances": 200  // debug setting
  },
  "train_data_path": "https://yonatanbisk.com/piqa/data/train.jsonl",
  "validation_data_path": "https://yonatanbisk.com/piqa/data/valid.jsonl",
  "model": {
      "type": "transformer_mc",
      "transformer_model": transformer_model,
  },
  "data_loader": {
    "sampler": "random",
    "batch_size": batch_size
  },
  "trainer": {
    "optimizer": {
      "type": "huggingface_adamw",
      "weight_decay": weight_decay,
      "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
      "lr": lr,
      "eps": 1e-8,
      "correct_bias": true
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": cut_frac,
    },
    "grad_norm": grad_norm,
    "num_epochs": epochs,
    "cuda_device": 0
  },
  "random_seed": 42,
  "numpy_seed": 42,
  "pytorch_seed": 42,
}
