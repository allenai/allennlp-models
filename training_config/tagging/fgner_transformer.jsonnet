local data_dir = std.extVar("CONLL_DATA_PATH");
// local data_dir = "/net/nfs.corp/allennlp/dirkg/data/conll-formatted-ontonotes-5.0/data";
// local data_dir = "/Users/dirkg/Documents/data/conll-formatted-ontonotes-5.0/data";

local transformer_model = "roberta-base";
local epochs = 3;
local batch_size = 8;
local max_length = 512;

{
    "dataset_reader": {
        "type": "ontonotes_ner",
        "coding_scheme": "BIOUL",
        "token_indexers": {
          "tokens": {
            "type": "pretrained_transformer_mismatched",
            "model_name": transformer_model,
            "max_length": max_length
          },
        },
    },
    "train_data_path": data_dir + "/train",
    "validation_data_path": data_dir + "/development",
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": batch_size
        }
    },
    "model": {
        "type": "crf_tagger",
        "encoder": {
            "type": "pass_through",
        },
        "include_start_end_transitions": false,
        "label_encoding": "BIOUL",
        "text_field_embedder": {
          "token_embedders": {
            "tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": transformer_model,
                "max_length": max_length
            }
          }
        },
        "verbose_metrics": true
    },
    "trainer": {
        "optimizer": {
          "type": "huggingface_adamw",
          "weight_decay": 0.0,
          "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
          "lr": 1e-5,
          "eps": 1e-8
        },
        "learning_rate_scheduler": {
          "type": "slanted_triangular",
          "cut_frac": 0.05,
        },
        "grad_norm": 1.0,
        "num_epochs": epochs,
        "cuda_device": -1,
        "validation_metric": "+f1-measure-overall"
    }
}
