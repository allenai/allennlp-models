local model_name = "bert-base-uncased";
local vocab_size = 30522;     // for bert-*-uncased models
//local vocab_size = 28996;   // for bert-*-cased models
local num_gpus = 1;
local gpu_batch_size = 32;
local effective_batch_size = gpu_batch_size * num_gpus;
local num_epochs = 20;
local patience = 5;
local num_instances = 158915;
local num_gradient_accumulation_steps = 1;

local construct_vocab = false;
local dataset = "data";

// local vocabulary = if construct_vocab then {
//       // read the files to construct the vocab
//       "min_count": {"answers": 5}
//     } else {
//       // TODO: update
//       // read the constructed vocab
//       "type": "from_files",
//       # todo: upload vocab to google
//       // "directory": "https://storage.googleapis.com/allennlp-public-data/vqav2/vilbert_vqa_%s.%s.vocab.tar.gz",
//       "directory": "/home/jacobm/model-output/vgqa-vocab/output.tar.gz",
//     };

{
  "dataset_reader": {
    "type": "flickr30k",
    "image_dir": "/net/nfs2.allennlp/data/vision/flickr30k/images/",
    "data_dir": "/net/nfs2.allennlp/data/vision/flickr30k/captions/",
    [if !construct_vocab then "feature_cache_dir"]: "/net/nfs2.allennlp/data/vision/flickr30k/feature_cache",
    #"image_dir": std.format("/Users/dirkg/Documents/data/vision/vqa/%s", dataset),
    #[if !construct_vocab then "feature_cache_dir"]: std.format("/Users/dirkg/Documents/data/vision/vqa/%s/feature_cache", dataset),
    [if !construct_vocab then "image_loader"]: "torch",
    [if !construct_vocab then "image_featurizer"]: "resnet_backbone",
    [if !construct_vocab then "region_detector"]: "faster_rcnn",
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
    // TODO: comment this out
    "max_instances": 1000,
    "image_processing_batch_size": 16,
    // "answer_vocab": if construct_vocab then null else vocabulary,
    // "multiple_answers_per_question": !construct_vocab,
  },
  // "validation_dataset_reader": self.dataset_reader {
    // "answer_vocab": null    // make sure we don't skip unanswerable questions during validation
  // },
  // "vocabulary": vocabulary,
  "train_data_path": "https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/train.txt",
  "validation_data_path": "https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/val.txt",
  "test_data_path": "https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/test.txt",
  "evaluate_on_test": true,
  "model": {
    "type": "vilbert_ir",
    // "model_name": model_name,
    // "image_feature_dim": 1024,
    // "image_hidden_size": 1024,
    // "image_num_attention_heads": 8,
    // "image_num_hidden_layers": 6,
    // "combined_hidden_size": 1024,
    // "combined_num_attention_heads": 8,
    // "pooled_output_dim": 1024,
    // "image_intermediate_size": 1024,
    // "image_attention_dropout": 0.1,
    // "image_hidden_dropout": 0.1,
    // "image_biattention_id": [0, 1, 2, 3, 4, 5],
    // "text_biattention_id": [6, 7, 8, 9, 10, 11],
    // "text_fixed_layer": 0,
    // "image_fixed_layer": 0,
    // "fusion_method": "mul",
    // "ignore_text": false, # debug setting
    // "ignore_image": false, # debug setting
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
      "lr": 5e-5,
      "correct_bias": true,
      "weight_decay": 0.01,
      "parameter_groups": [
        // [["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}], // can't use both at the same time
        // smaller learning rate for the pretrained weights
        [["^embeddings\\.", "^encoder.layers1\\.", "^t_pooler\\."], {"lr": 5e-6}]
      ],
    },
    "learning_rate_scheduler": {
      "type": "linear_with_warmup",
      "warmup_steps" : std.ceil(0.1 * num_instances * num_epochs * num_gradient_accumulation_steps / effective_batch_size)
      // "warmup_steps": 5000
    },
    "validation_metric": "+accuracy",
    "patience": patience,
    "num_epochs": num_epochs,
    "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
  },
  "random_seed": 13034431,
  "numpy_seed": 13034431,
  "pytorch_seed": 13034431,
}
