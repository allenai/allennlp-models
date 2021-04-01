local model_name = "patrickvonplaten/t5-tiny-random";
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
        "source_max_tokens": 512,
        "target_max_tokens": 54,
    },
    "model": {
        "type": "t5",
        "model_name": model_name
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
        "enable_default_callbacks": false
    }
}
