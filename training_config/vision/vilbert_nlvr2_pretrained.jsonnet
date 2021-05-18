local model_name = "bert-large-uncased";
local num_gpus = 1;
local effective_batch_size = 64;
local gpu_batch_size = effective_batch_size / num_gpus;
local num_epochs = 10;
local patience = 5;
local num_gradient_accumulation_steps = effective_batch_size / gpu_batch_size / std.max(1, num_gpus);
local num_instances = 86373;

{
  "dataset_reader": {
    "type": "nlvr2",
    "image_dir": "/net/nfs2.allennlp/data/vision/nlvr2/images",
    "feature_cache_dir": "/net/nfs2.allennlp/data/vision/nlvr2/feature_cache",
    "image_loader": "torch",
    "image_featurizer": "resnet_backbone",
    "region_detector": "faster_rcnn",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": model_name
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": model_name
      }
    },
    "image_processing_batch_size": 16,
    // "max_instances": 1000
  },
  "train_data_path": "train",
  "validation_data_path": "dev",
  "test_data_path": "test",
  "evaluate_on_test": true,
  "model": {
    "type": "nlvr2_from_huggingface",
    "model_name": model_name,
    "image_feature_dim": 1024,
    "image_hidden_size": 1024,
    "image_num_attention_heads": 8,
    "image_num_hidden_layers": 6,
    "combined_hidden_size": 1024,
    "combined_num_attention_heads": 8,
    "pooled_output_dim": 1024,
    "image_intermediate_size": 1024,
    "image_attention_dropout": 0.1,
    "image_hidden_dropout": 0.1,
    "image_biattention_id": [0, 1, 2, 3, 4, 5],
    "text_biattention_id": [6, 7, 8, 9, 10, 11],
    "text_fixed_layer": 0,
    "image_fixed_layer": 0,
    "fusion_method": "mul"
  },
  "data_loader": {
    "batch_size": gpu_batch_size,
    "shuffle": true,
  },
  [if num_gpus > 1 then "distributed"]: {
    "cuda_devices": std.range(0, num_gpus - 1)
    #"cuda_devices": std.repeat([-1], num_gpus)  # Use this for debugging on CPU
  },
  "trainer": {
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-5,
      "weight_decay": 0.01,
    },
    "learning_rate_scheduler": {
      "type": "linear_with_warmup",
      "warmup_steps" : std.ceil(0.1 * num_instances * num_epochs * num_gradient_accumulation_steps / effective_batch_size)
      // "warmup_steps": 5000
    },
    "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
    "validation_metric": "+accuracy",
    "num_epochs": num_epochs,
    "patience": patience
  },
}
