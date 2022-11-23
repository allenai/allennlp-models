local target_namespace = "target_tokens";
local transformer_model = "test_fixtures/bert-xsmall-dummy";
local hidden_size = 20;

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
        "beam_search": {
            "max_steps": 20,
            "beam_size": 3,
        },
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size" : 64,
        }
    },
    "trainer": {
        "optimizer": {
            "type": "adamw",
            "weight_decay": 0.0,
            "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
            "lr": 5e-5,
            "eps": 1e-8
        },
        "num_epochs": 2,
        "cuda_device": -1,
    }
}
