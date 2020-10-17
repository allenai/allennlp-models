local transformer_model = "test_fixtures/bert-xsmall-dummy";
local epochs = 2;
local batch_size = 3;

{
    "dataset_reader": {
        "type": "transformer_squad",
        "transformer_model_name": transformer_model,
    },
    "train_data_path": "test_fixtures/rc/squad2.json",
    "validation_data_path": "test_fixtures/rc/squad2.json",
    "model": {
        "type": "transformer_qa",
        "transformer_model_name": transformer_model,
    },
    "data_loader": {
        "batch_size": batch_size
    },
    "trainer": {
      "optimizer": {
        "type": "huggingface_adamw",
        "weight_decay": 0.0,
        "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
        "lr": 5e-5,
        "eps": 1e-8
      },
      "grad_clipping": 1.0,
      "num_epochs": epochs,
      "cuda_device": -1
    },
}
