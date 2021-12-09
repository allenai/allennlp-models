// =================== Configurable Settings ======================

local debug = true;

local model_name = if debug then "t5-small" else "t5-11b";

local batch_size_per_gpu = if debug then 4 else 1;

// To train "t5-11b" you will probably need 8 GPUs.
local num_gpus = 8;

// This is probably necessary for t5-11b unless you have more than 8 GPUs.
local activation_checkpointing = true;

// Set to `false` if you want to skip validation.
local validate = true;

// AMP is currently unusably slow with t5-11b, which may be due to a bug bug within
// FairScale, but I'm not sure yet.
local use_amp = false;

// These are reasonable defaults.
local source_length = 512;
local target_length = 54;

// Set to `true` to log to Weights & Biases.
local use_wandb = false;

// ================================================================

// ------ !! You probably don't need to edit below here !! --------

local data_base_url = "https://storage.googleapis.com/allennlp-public-data/cnndm-combined-data-2020.07.13.tar.gz";
local train_data = data_base_url + "!cnndm-combined-data-2020.07.13/url_lists/all_train.txt";
local dev_data = data_base_url + "!cnndm-combined-data-2020.07.13/url_lists/all_val.txt";

local dataset_reader = {
    "type": "cnn_dm",
    "source_tokenizer": {
        "type": "pretrained_transformer",
        "model_name": model_name,
    },
    "source_token_indexers": {
        "tokens": {
            "type": "pretrained_transformer",
            "model_name": model_name,
            "namespace": "tokens",
        }
    },
    "source_max_tokens": source_length,
    "target_max_tokens": target_length,
    "source_prefix": "summarize: ",
};

local data_loader = {
    "batch_size": batch_size_per_gpu,
    "shuffle": true,
};

local wandb_callback = {
    "type": "wandb",
    "project": "allennlp-t5",
    "entity": "allenai-team1",
    "watch_model": false,
    "summary_interval": 1,
    "should_log_parameter_statistics": false,
    "should_log_learning_rate": false,
};

{
    "train_data_path": train_data,
    [if validate then "validation_data_path"]: dev_data,
    "dataset_reader": dataset_reader + {
        [if debug then "max_instances"]: batch_size_per_gpu * 40,
    },
    "validation_dataset_reader": dataset_reader + {
        "max_instances": if debug then batch_size_per_gpu * 4 else batch_size_per_gpu * 10,
    },
    "model": {
        "type": "t5",
        "model_name": model_name,
        "beam_search": {
            "beam_size": 3,
            "max_steps": if debug then 5 else 50,
        },
        [if activation_checkpointing then "checkpoint_wrapper"]: {
            "type": "fairscale",
            "offload_to_cpu": true
        },
    },
    "data_loader": data_loader + {
        [if !debug then "max_instances_in_memory"]: batch_size_per_gpu * 128,
        [if !debug then "num_workers"]: 1,
    },
    "validation_data_loader": data_loader,
    "vocabulary": {
        "type": "empty",
    },
    "trainer": {
        "use_amp": use_amp,
        [if use_amp then "grad_scaling"]: false,  # TODO: use grad scaling once it's fixed in FairScale.
        "num_epochs": 3,
        "optimizer": {
            "type": "huggingface_adafactor",
        },
        "grad_norm": 1.0,
        [if use_wandb then "callbacks"]: [wandb_callback],
    },
    [if num_gpus > 1 then "distributed"]: {
        "cuda_devices": std.range(0, num_gpus - 1),
        "ddp_accelerator": {
            "type": "fairscale_fsdp",
            "mixed_precision": use_amp,
        },
    },
}
