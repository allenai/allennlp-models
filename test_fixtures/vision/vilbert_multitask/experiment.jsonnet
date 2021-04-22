local model_name = "epwalsh/bert-xsmall-dummy";

local vocabulary = {
      "type": "from_files",
      "directory": "https://storage.googleapis.com/allennlp-public-data/vilbert/vilbert_multitask.bert-base-uncased.vocab.tar.gz"
    };

local reader_common = {
    "image_loader": "torch",
    "image_featurizer": "null",
    "region_detector": {
      "type": "random",
      "seed": 322
    },
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": model_name
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": model_name
      }
    }
};

{
  "dataset_reader": {
    "type": "multitask",
    "readers": {
      "vqa": reader_common {
        "type": "vqav2",
        "image_dir": "test_fixtures/vision/images/vqav2",
        "answer_vocab": vocabulary,
        "multiple_answers_per_question": true
      },
      "ve": reader_common {
        "type": "visual-entailment",
        "image_dir": "test_fixtures/vision/images/visual_entailment",
      }
    }
  },
  "validation_dataset_reader": self.dataset_reader {
    "readers": super.readers {
      "vqa": super.vqa {
        "answer_vocab": null    // make sure we don't skip unanswerable questions during validation
      }
    }
  },
  "vocabulary": vocabulary,
  "train_data_path": {
    "vqa": "unittest",
    "ve": "test_fixtures/vision/visual_entailment/sample_pairs.jsonl",
  },
  "validation_data_path": {
    "vqa": "unittest",
    "ve": "test_fixtures/vision/visual_entailment/sample_pairs.jsonl",
  },
  "model": {
    "type": "multitask",
    "arg_name_mapping": {
      "backbone": {"question": "text", "hypothesis": "text"}
    },
    "backbone": {
      "type": "vilbert_from_huggingface",
      "model_name": model_name,

      "image_feature_dim": 10,
      "image_num_hidden_layers": 1,
      "image_hidden_size": 20,
      "image_num_attention_heads": 1,
      "image_intermediate_size": 5,
      "image_attention_dropout": 0.0,
      "image_hidden_dropout": 0.0,
      "image_biattention_id": [0, 1],
      "image_fixed_layer": 0,

      "text_biattention_id": [0, 1],
      "text_fixed_layer": 0,

      "combined_hidden_size": 20,
      "combined_num_attention_heads": 2,

      "pooled_output_dim": 20,
      "fusion_method": "sum",
    },
    "heads": {
      "vqa": {
        "type": "vqa",
        "embedding_dim": 20
      },
      "ve": {
        "type": "visual_entailment",
        "embedding_dim": 20
      }
    }
  },
  "data_loader": {
    "type": "multitask",
    "scheduler": { "batch_size": 2 }
  },
  "trainer": {
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 4e-5,
    },
    "validation_metric": ["+vqa_vqa", "+ve_acc"],
    "patience": 1,
    "num_epochs": 3,
  }
}
