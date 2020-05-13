{
    "train_data_path": "",
    "validation_data_path": "",
    "dataset_reader": {
        "type": "allennlp_models.seq2seq.Seq2SeqDatasetReader",
        "source_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": "bart-large",
            "add_special_tokens": false
        },
        "source_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": "bart-large",
                "namespace": "tokens"
            }
        },

        "start_symbol": "<s>",
        "end_symbol": "</s>",
        "source_max_tokens": 1022,
        "target_max_tokens": 1022
    },
    "model": {
        "type": "bart",
        "model_name": "bart-large"
    },
    "data_loader": {
        "batch_sampler": {
            "type": "max_tokens_sampler",
            "max_tokens": 2048,
            "sorting_keys": ["source_tokens"]
        }
    },
    "trainer": {
        "num_epochs": 32,
        "optimizer": {
            "type": "adam",
            "lr": 3e-5,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
        "learning_rate_scheduler": {
            "type": "polynomial_decay",
            "total_steps": 20000,
            "warmup_steps": 500
        },
        "tensorboard_writer": {
            "summary_interval": 4,
            "should_log_learning_rate": true
        },
        "num_gradient_accumulation_steps": 4,
        "cuda_device": 0
    }
}
