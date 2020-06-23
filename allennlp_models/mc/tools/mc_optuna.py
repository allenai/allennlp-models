import optuna

def objective(trial: optuna.Trial) -> float:
    trial.suggest_int('epochs', 1, 5)
    trial.suggest_int('gradient_accumulation_steps', 1, 16)
    trial.suggest_float('weight_decay', 0, 0.1)
    trial.suggest_float('lr', 1e-5/3, 5e-5, log=True)
    trial.suggest_float('cut_frac', 0, 0.1)
    trial.suggest_float('grad_norm', 0, 2.0)

    executor = optuna.integration.allennlp.AllenNLPExecutor(
        trial=trial,  # trial object
        config_file="training_config/mc/piqa-optuna.jsonnet",
        serialization_dir=f"models/piqa-optuna/{trial.number}",
        metrics="best_validation_acc",
        include_package="allennlp_models"
    )
    return executor.run()


if __name__ == '__main__':
    study = optuna.create_study(
        storage=None,
        sampler=optuna.samplers.TPESampler(seed=24),
        study_name="piqa-optuna",
        direction="maximize",
    )

    timeout = 60 * 60 * 10  # timeout (sec): 60*60*10 sec => 10 hours
    study.optimize(
        objective,
        n_jobs=1,  # number of processes in parallel execution
        n_trials=60,  # number of trials to train a model
        timeout=timeout,  # threshold for executing time (sec)
    )

    optuna.integration.allennlp.dump_best_config("training_config/mc/piqa-optuna.jsonnet", "best_piqa-optuna.json", study)
