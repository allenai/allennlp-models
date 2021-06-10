local bert_model = "epwalsh/bert-xsmall-dummy";

{
    "dataset_reader":{
        "type":"srl",
        "bert_model_name": bert_model,
    },
    "train_data_path": "test_fixtures/structured_prediction/srl",
    "validation_data_path": "test_fixtures/structured_prediction/srl",
    "model": {
        "type": "srl_bert",
        "bert_model": bert_model,
        "embedding_dropout": 0.0
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 5,
            "padding_noise": 0.0
        }
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "checkpointer": {
            "keep_most_recent_by_count": 1
        },
        "num_epochs": 3,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": -1
    }
}
