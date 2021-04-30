local bert_model = "test_fixtures/bert-xsmall-dummy";

# Take from test_fixtures/bert-xsmall-dummy/config.json
local bert_config = {
  "architectures": [
    "BertModel"
  ],
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 20,
  "initializer_range": 0.02,
  "intermediate_size": 40,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 1,
  "num_hidden_layers": 1,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "vocab_size": 250
};

{
    "dataset_reader":{
        "type":"srl",
        "bert_model_name": bert_model,
    },
    "train_data_path": "test_fixtures/structured_prediction/srl",
    "validation_data_path": "test_fixtures/structured_prediction/srl",
    "model": {
        "type": "srl_bert",
        "bert_model": bert_config,
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
            "num_serialized_models_to_keep": 1
        },
        "num_epochs": 3,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": -1
    }
}
