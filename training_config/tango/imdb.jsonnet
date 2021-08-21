
local debug = true;

local transformer_model = if debug then "roberta-base" else "roberta-large";

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
            "keep_old_splits": false
        },
        "trained_model": {
            "type": "training",
            "dataset": {"ref": "dataset"},
            "training_split": "train",
            [if !debug then "data_loader"]: {
                "batch_size": 32
            },
            [if debug then "data_loader"]: {
                "type": "max_batches",
                "max_batches_per_epoch": 7,
                "inner": {
                    "batch_size": 5
                }
            },
            "model": {
              "type": "transformer_classification_tt",
              "transformer_model": transformer_model,
            },
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
            "num_epochs": if debug then 3 else 20,
            "patience": 3,
        },
        "evaluation": {
            "type": "evaluation",
            "dataset": { "ref": "dataset" },
            "model": { "ref": "trained_model" },
            "split": "test",
            [if debug then "data_loader"]: {
                "type": "max_batches",
                "max_batches_per_epoch": 7,
                "inner": {
                    "batch_size": 5
                }
            },
        }
    }
}
