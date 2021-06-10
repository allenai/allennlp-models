local model_name = "sshleifer/bart-tiny-random";
local data_base_url = "test_fixtures/generation/bart/data/";

{
    "train_data_path": data_base_url + "/url_lists/all_train.txt",
    "validation_data_path": data_base_url + "/url_lists/all_val.txt",
    "dataset_reader": {
        "type": "cnn_dm",
        "source_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": model_name
        },
        "source_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": model_name,
                "namespace": "tokens"
            }
        },
        "source_max_tokens": 1022,
        "target_max_tokens": 54,
    },
    "model": {
        "type": "bart",
        "model_name": model_name,
        "beam_search": {
            "max_steps": 140,
            "beam_size": 4
        },
    },
    "data_loader": {
        "batch_size": 2,
        "shuffle": true
    },
    "trainer": {
        "num_epochs": 1,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 3e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "correct_bias": true
        },
        "learning_rate_scheduler": {
            "type": "polynomial_decay",
        },
        "grad_norm": 1.0,
        "run_confidence_checks": false
    }
}
