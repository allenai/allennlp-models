local model_name = "facebook/bart-large";
local data_base_url = "https://storage.googleapis.com/allennlp-public-data/cnndm-combined-data-2020.07.13.tar.gz";

{
    "train_data_path": data_base_url + "!cnndm-combined-data-2020.07.13/url_lists/all_train.txt",
    "validation_data_path": data_base_url + "!cnndm-combined-data-2020.07.13/url_lists/all_val.txt",
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
        // "max_instances": 1000 // DEBUG setting
    },
    "model": {
        "type": "bart",
        "model_name": model_name
    },
    "data_loader": {
        "batch_size": 4,
        "shuffle": true
    },
    "trainer": {
        "num_epochs": 3,
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
        "tensorboard_writer": {
            "summary_interval": 4,
            "should_log_learning_rate": true
        },
        "grad_norm": 1.0,
    }
}
