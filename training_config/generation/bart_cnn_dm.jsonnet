local model_name = "facebook/bart-large";
local data_base_url = "https://storage.googleapis.com/allennlp-public-data/cnndm-combined-data-2020.07.13.tar.gz";
local train_data = data_base_url + "!cnndm-combined-data-2020.07.13/url_lists/all_train.txt";
local dev_data = data_base_url + "!cnndm-combined-data-2020.07.13/url_lists/all_val.txt";

{
    "train_data_path": train_data,
    "validation_data_path": dev_data,
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
        "model_name": model_name,
        "beam_search": {
            "max_steps": 140,
            "beam_size": 4
        },
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
        "grad_norm": 1.0,
    }
}
