local target_namespace = "target_tokens";
local transformer_model = "bert-base-cased";
local hidden_size = 768;

{
    "dataset_reader": {
        "type": "copynet_seq2seq",
        "target_namespace": target_namespace,
        "source_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": transformer_model,
        },
        "target_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": transformer_model,
            "add_special_tokens": false,
        },
        "source_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": transformer_model,
            },
        },
        "add_start_and_end_tokens": false,
    },
    "train_data_path": "test_fixtures/generation/copynet/data/copyover.tsv",
    "validation_data_path": "test_fixtures/generation/copynet/data/copyover.tsv",
    "model": {
        "type": "copynet_seq2seq",
        "source_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": transformer_model,
                    "train_parameters": false,
                },
            }
        },
        "encoder": {
            "type": "pass_through",
            "input_dim": hidden_size,
        },
        "attention": {
            "type": "bilinear",
            "vector_dim": hidden_size,
            "matrix_dim": hidden_size,
        },
        "target_embedding_dim": 10,
        "beam_size": 3,
        "max_decoding_steps": 20,
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size" : 8,
        }
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
        },
        "num_epochs": 2,
        "cuda_device": -1,
    }
}
