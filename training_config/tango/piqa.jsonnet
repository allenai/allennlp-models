local transformer_model = "bert-base-cased";

local debug = true;

{
    "steps": {
        "dataset": {
            "type": "piqa_instances",
            "tokenizer_name": transformer_model
        },
        "trained_model": {
            "type": "training",
            "dataset": {"ref": "dataset"},
            "training_split": "train",
            "data_loader": {
                "type": "max_batches",
                [if debug then "max_batches_per_epoch"]: 7,
                "inner": {
                    "batch_size": if debug then 5 else 32
                }
            },
            "model": {
              "type": "transformer_mc_tt",
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
            "validation_metric": "+acc",
        },
        "evaluation": {
            "type": "evaluation",
            "dataset": {
                "type": "ref",
                "ref": "dataset"
            },
            "model": {
                "type": "ref",
                "ref": "trained_model"
            },
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
