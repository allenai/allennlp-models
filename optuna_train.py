"""

Simple example to use AllenNLPExecutor.
There is no parameter to be tuned here.

"""

import optuna


config_file = "training_config/rc/transformer_qa.jsonnet"
result_path = "result/trial_{}"
metric = "best_validation_per_instance_em"


def objective(trial: optuna.Trial) -> float:
    # We will declare search space of hyperparameters using trial.suggest here
    executor = optuna.integration.allennlp.AllenNLPExecutor(
        trial,  # trial object
        config_file,  # jsonnet path
        result_path.format(trial.number),  # directory for snapshots and logs
        metric,  # metric which you want to track
        include_package="allennlp_models"  # same as `--include-package` in allennlp
    )
    return executor.run()


if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_jobs=1,  # number of processes in parallel execution
        n_trials=1,  # number of trials to train a model
        timeout=None,  # threshold for executing time
    )
