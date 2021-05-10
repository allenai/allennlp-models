local model_name = "bert-base-uncased";
local vocab_size = 30522;     // for bert-*-uncased models
//local vocab_size = 28996;   // for bert-*-cased models
local effective_batch_size = 128;
local num_gpus = 4;
local gpu_batch_size = effective_batch_size / num_gpus;
local num_epochs = 20;
local patience = 5;
local num_instances = 1062451;
local num_gradient_accumulation_steps = effective_batch_size / gpu_batch_size / std.max(1, num_gpus);

local construct_vocab = false;
local dataset = "data";

local vocabulary = if construct_vocab then {
      // read the files to construct the vocab
      "min_count": {"answers": 9}
    } else {
      // read the constructed vocab
      "type": "from_files",
      "directory": "https://storage.googleapis.com/allennlp-public-data/vgqa/vilbert_vgqa_bert-base-uncased.vocab.tar.gz",
    };

{
  "dataset_reader": {
    "type": "vgqa",
    "image_dir": "/net/nfs2.allennlp/data/vision/visual_genome/1.2",
    [if !construct_vocab then "feature_cache_dir"]: "/net/nfs2.allennlp/data/vision/visual_genome/1.2/feature_cache",
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
    #"max_instances": 1000,
    "image_processing_batch_size": 16,
    "answer_vocab": if construct_vocab then null else vocabulary,
  },
  "validation_dataset_reader": self.dataset_reader {
    "answer_vocab": null    // make sure we don't skip unanswerable questions during validation
  },
  "vocabulary": vocabulary,
  "train_data_path": "https://visualgenome.org/static/data/dataset/question_answers.json.zip!question_answers.json[5000:]",
  "validation_data_path": "https://visualgenome.org/static/data/dataset/question_answers.json.zip!question_answers.json[:5000]",
  "model": {
    "type": "vqa_vilbert_from_huggingface",
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
    "fusion_method": "mul",
    "ignore_text": false, # debug setting
    "ignore_image": false, # debug setting
  },
  "data_loader": {
    "batch_size": gpu_batch_size,
    "shuffle": true,
  },
  [if num_gpus > 1 then "distributed"]: {
    "cuda_devices": std.range(0, num_gpus - 1)
    #"cuda_devices": std.repeat([-1], num_gpus)  # Use this for debugging on CPU
  },
  // Don't train if we're just constructing vocab. The results would be confusing.
  [if !construct_vocab then "trainer"]: {
    // "callbacks": [
    //     {
    //         "batch_size_interval": 100,
    //         "project": "allennlp-testing",
    //         "should_log_learning_rate": true,
    //         "should_log_parameter_statistics": true,
    //         "summary_interval": 100,
    //         "type": "wandb"
    //     }
    // ],
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 4e-5,
      "correct_bias": true,
      "weight_decay": 0.01,
      "parameter_groups": [
        // [["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}], // can't use both at the same time
        // smaller learning rate for the pretrained weights
        [["^embeddings\\.", "^encoder.layers1\\.", "^t_pooler\\."], {"lr": 4e-6}]
      ],
    },
    "learning_rate_scheduler": {
      "type": "linear_with_warmup",
      "warmup_steps" : std.ceil(0.1 * num_instances * num_epochs * num_gradient_accumulation_steps / effective_batch_size)
    },
    "validation_metric": "+vqa_score",
    "patience": patience,
    "num_epochs": num_epochs,
    "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
  },
  "random_seed": 13034431,
  "numpy_seed": 13034431,
  "pytorch_seed": 13034431,
}
