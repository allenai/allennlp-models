local model_name = "bert-base-uncased";
local vocab_size = 30522;     // for bert-*-uncased models
//local vocab_size = 28996;   // for bert-*-cased models
local num_gpus = 1;
local gpu_batch_size = 16;
local effective_batch_size = gpu_batch_size * num_gpus;
local num_epochs = 40;
local patience = 5;
local num_instances = 148915;
local num_gradient_accumulation_steps = 128 / effective_batch_size;

local dataset = "data";

{
  "dataset_reader": {
    "type": "flickr30k",
    "image_dir": "/net/nfs2.allennlp/data/vision/flickr30k/images/",
    "data_dir": "https://github.com/BryanPlummer/flickr30k_entities/raw/master/annotations.zip!Sentences/",
    "feature_cache_dir": "/net/nfs2.allennlp/data/vision/flickr30k/feature_cache",
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
    // "max_instances": 1000, // DEBUG
    "image_processing_batch_size": 16,
    "featurize_captions": true,
  },
  "validation_dataset_reader": self.dataset_reader {
    "is_evaluation": true,
  },
  "train_data_path": "https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/train.txt",
  "validation_data_path": "https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/val.txt",
  "test_data_path": "https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/test.txt",
  "evaluate_on_test": true,
  "model": {
    "type": "vilbert_ir",
    "text_embeddings": {
      "vocab_size": vocab_size,
      "embedding_size": 1024,
      "pad_token_id": 0,
      "max_position_embeddings": 512,
      "type_vocab_size": 2,
      "dropout": 0.1
    },
    "image_embeddings": {
      "feature_size": 1024,
      "embedding_size": 1024
    },
    "encoder": {
      # text
      "hidden_size1": 1024,
      "num_hidden_layers1": 24,
      "intermediate_size1": 4096,
      "num_attention_heads1": 16,
      "attention_dropout1": 0.1,
      "hidden_dropout1": 0.1,
      "biattention_id1": [18, 19, 20, 21, 22, 23],
      "fixed_layer1": 0,

      # vision
      "hidden_size2": 1024,
      "num_hidden_layers2": 6,
      "intermediate_size2": 1024,
      "num_attention_heads2": 8,
      "attention_dropout2": 0.1,
      "hidden_dropout2": 0.1,
      "biattention_id2": [0, 1, 2, 3, 4, 5],
      "fixed_layer2": 0,

      "combined_num_attention_heads": 8,
      "combined_hidden_size": 1024,
      "activation": "gelu",
    },
    "pooled_output_dim": 1024,
    "fusion_method": "mul",
  },
  "data_loader": {
    "batch_size": gpu_batch_size,
    "shuffle": true,
    // "batches_per_epoch": 3 * std.ceil(num_instances / gpu_batch_size),
  },
  "validation_data_loader": {
    "batch_size": 1,
    // "batches_per_epoch": 5000,
  },
  [if num_gpus > 1 then "distributed"]: {
    "cuda_devices": std.range(0, num_gpus - 1)
    #"cuda_devices": std.repeat([-1], num_gpus)  # Use this for debugging on CPU
  },
  "trainer": {
    // "callbacks": [
    //     {
    //         // "batch_size_interval": 100,
    //         "project": "allennlp-testing",
    //         "should_log_learning_rate": true,
    //         "should_log_parameter_statistics": true,
    //         "summary_interval": 100,
    //         "type": "wandb"
    //     }
    // ],
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-5,
      "correct_bias": true,
      "weight_decay": 0.01,
      "parameter_groups": [
        // [["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}], // can't use both at the same time
        // smaller learning rate for the pretrained weights
        [["^embeddings\\.", "^encoder.layers1\\.", "^t_pooler\\."], {"lr": 2e-6}]
      ],
    },
    "learning_rate_scheduler": {
      "type": "linear_with_warmup",
      "warmup_steps" : std.ceil(0.1 * num_instances * num_epochs * num_gradient_accumulation_steps / effective_batch_size),
      // "warmup_steps": 5000,
    },
    "validation_metric": ["+top_1_acc", "+top_5_acc", "+top_10_acc"],
    "patience": patience,
    "num_epochs": num_epochs,
    "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
  },
  "random_seed": 13034431,
  "numpy_seed": 13034431,
  "pytorch_seed": 13034431,
}
