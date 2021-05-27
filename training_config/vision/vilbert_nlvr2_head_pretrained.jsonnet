local model_name = "bert-large-uncased";
local num_gpus = 1;
local effective_batch_size = 64;
local gpu_batch_size = effective_batch_size / num_gpus;
local num_epochs = 10;
local patience = 5;
local num_gradient_accumulation_steps = effective_batch_size / gpu_batch_size / std.max(1, num_gpus);
local num_instances = 86373;

local construct_vocab = false;

local vocabulary = if construct_vocab then {
      // read the files to construct the vocab
      "min_count": {"answers": 9}
    } else {
      // read the constructed vocab
      "type": "from_files",
      "directory": std.format(
        "https://storage.googleapis.com/allennlp-public-data/vilbert/vilbert_multitask.%s.vocab.tar.gz",
        model_name)
    };

local reader_common = {
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
    "max_instances": 1000, # DEBUG
    "image_processing_batch_size": 16,
};

{
  "dataset_reader": {
    "type": "multitask",
    "readers": {
      "nlvr2": reader_common {
        "type": "nlvr2",
        "image_dir": "/net/nfs2.allennlp/data/vision/nlvr2/images",
        [if !construct_vocab then "feature_cache_dir"]: "/net/nfs2.allennlp/data/vision/nlvr2/feature_cache",
      }
    }
  },
  "vocabulary": vocabulary,
  "train_data_path": {
    "nlvr2": "train",
  },
  "validation_data_path": {
    "nlvr2": "dev",
  },
  "test_data_path": {
    "nlvr2": "test",
  },
  "model": {
    "type": "multitask",
    "arg_name_mapping": {
      "backbone": {"question": "text", "hypothesis": "text"}
    },
    "backbone": {
      "type": "vilbert_from_huggingface",
      "model_name": model_name,
      "image_feature_dim": 1024,
      "image_num_hidden_layers": 6,
      "image_hidden_size": 1024,
      "image_num_attention_heads": 8,
      "image_intermediate_size": 1024,
      "image_attention_dropout": 0.1,
      "image_hidden_dropout": 0.1,
      "image_biattention_id": [0, 1, 2, 3, 4, 5],
      "text_biattention_id": [6, 7, 8, 9, 10, 11],
      "text_fixed_layer": 0,
      "image_fixed_layer": 0,
      "combined_hidden_size": 1024,
      "combined_num_attention_heads": 8,
      "pooled_output_dim": 1024,
      "fusion_method": "mul"
    },
    "heads": {
      "nlvr2": {
        "type": "nlvr2",
        "embedding_dim": 1024
      },
    }
  },
  "data_loader": {
    "type": "multitask",
    "scheduler": {
        "batch_size": gpu_batch_size,
    },
    "shuffle": true,
    //[if !construct_vocab then "max_instances_in_memory"]: 1024*16
  },
  [if num_gpus > 1 then "distributed"]: {
    "cuda_devices": std.range(0, num_gpus - 1)
    //"cuda_devices": std.repeat([-1], num_gpus)  # Use this for debugging on CPU
  },
  // Don't train if we're just constructing vocab. The results would be confusing.
  [if !construct_vocab then "trainer"]: {
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 4e-5,
      "correct_bias": true,
      "weight_decay": 0.01,
      "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
    },
    "learning_rate_scheduler": {
      "type": "linear_with_warmup",
      "warmup_steps" : std.ceil(0.1 * num_instances * num_epochs * num_gradient_accumulation_steps / effective_batch_size)
    },
    "validation_metric": ["+gqa_vqa", "+vqa_vqa", "+ve_acc"],
    "patience": patience,
    "num_epochs": num_epochs,
    "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
  },
  "random_seed": 876170670,
  "numpy_seed": 876170670,
  "pytorch_seed": 876170670,
}
