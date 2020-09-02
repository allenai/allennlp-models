local bert_model="facebook/bart-large";

{
  "dataset_reader": {
      "type": "squad_question_generation",
      "model_name": bert_model,
      "lazy": false,
      // "max_instances": 200  // debug setting
  },

  "train_data_path": "test_fixtures/rc/squad.json",
  "validation_data_path": "test_fixtures/rc/squad.json",

  "model": {
      "type": "squad_question_generation",
      "model_name": bert_model,
      "beam_size": 1,
  },

  "data_loader": {
      "batch_sampler": {
          "type": "basic",
          "sampler": {"type": "random"},
          "batch_size": 4,
          "drop_last": false,
      },
  },

  "trainer": {
      "checkpointer": {
        "num_serialized_models_to_keep": 1
      },
      "num_epochs": 1,
      "cuda_device": 0,
      "grad_norm": 1,
      "optimizer": {
        "type": "huggingface_adamw",
        "lr": 3e-5,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "correct_bias": true
      }
  },
  "random_seed": 42,
  "numpy_seed": 42,
  "pytorch_seed": 42
}