
local debug = true;

local transformer_model = if debug then "roberta-base" else "roberta-large";

local training_common = {
    "type": "training",
    "training_split": "train",
    "validation_split": "validation",
    [if debug then "limit_batches_per_epoch"]: 7,
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
};

{
    "steps": {
        "dataset": {
            "type": "dataset_remix",
            "input": {
                "type": "imdb_instances",
                "tokenizer_name": transformer_model
            },
            "new_splits": {
                 "train": "train[:20000]",
                 "validation": "train[20000:]",
                 "test": "test"
            },
            "keep_old_splits": false,
            "shuffle_before": true,
            "random_seed": 1
        },

        # baseline
        "trained_model_baseline": training_common {
            "dataset": { "ref": "dataset" },
            "data_loader": {
                "batch_size": 32
            },
            "model": {
              "type": "transformer_classification_tt",
              "transformer_model": transformer_model,
            },
            "num_epochs": if debug then 3 else 20,
            "patience": 3,
        },
        "evaluation_baseline": {
            "type": "evaluation",
            "dataset": { "ref": "dataset" },
            "model": { "ref": "trained_model_baseline" },
            "split": "test",
            [if debug then "data_loader"]: {
                "type": "max_batches",
                "max_batches_per_epoch": 7,
                "inner": {
                    "batch_size": 5
                }
            },
        },

        # TAPT
        "dataset_only_text": {
            "type": "text_fields_only",
            "input": { "ref": "dataset" },
            "min_length": 5
        },
        "pretrained_model_tapt": training_common {
            "dataset": { "ref": "dataset_only_text" },
            "data_loader": {
                "type": "masked_lm",
                "batch_size": 32,
                "tokenizer_name": transformer_model,
            },
            "model": {
                "type": "masked_lm_tt",
                "transformer_model": transformer_model
            },
            "num_epochs": 100
        },
        "trained_model_tapt": self.trained_model_baseline {
            "model": {
                "type": "transformer_classification_tt",
                "transformer_model": { "ref": "pretrained_model_tapt" }
            }
        },
        "evaluation_tapt": self.evaluation_baseline {
            "model": { "ref": "trained_model_tapt" }
        }
    }
}
