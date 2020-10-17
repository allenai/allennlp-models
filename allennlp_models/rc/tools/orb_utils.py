from typing import List, Tuple
from allennlp_models.rc.tools.squad import get_metric_score as get_metric_squad
from allennlp_models.rc.tools.drop import get_metrics as drop_metrics
from allennlp_models.rc.tools.narrativeqa import get_metric_score as get_metric_narrativeqa


def get_metric_drop(predicted: str, ground_truths: List[str]) -> Tuple[float, float]:
    em_scores = []
    f1_scores = []
    for ground_truth in ground_truths:
        exact_match, f1 = drop_metrics(predicted, ground_truth)
        em_scores.append(exact_match)
        f1_scores.append(f1)
    return max(em_scores), max(f1_scores)


def update_extractive_metrics(metrics, dataset_name, exact_match, f1):
    metrics[dataset_name]["exact_match"] = (
        metrics[dataset_name]["exact_match"] + exact_match
        if "exact_match" in metrics[dataset_name]
        else exact_match
    )
    metrics[dataset_name]["f1"] = (
        metrics[dataset_name]["f1"] + f1 if "f1" in metrics[dataset_name] else f1
    )
    metrics[dataset_name]["total"] = (
        metrics[dataset_name]["total"] + 1 if "total" in metrics[dataset_name] else 1
    )
    return metrics


def update_abstractive_metrics(
    metrics, bleu_1_score, bleu_4_score, meteor_score, rouge_f, rouge_p, rouge_r
):
    metrics["narrativeqa"]["bleu_1"] = (
        metrics["narrativeqa"]["bleu_1"] + bleu_1_score
        if "bleu_1" in metrics["narrativeqa"]
        else bleu_1_score
    )
    metrics["narrativeqa"]["bleu_4"] = (
        metrics["narrativeqa"]["bleu_4"] + bleu_4_score
        if "bleu_4" in metrics["narrativeqa"]
        else bleu_4_score
    )
    metrics["narrativeqa"]["meteor"] = (
        metrics["narrativeqa"]["meteor"] + meteor_score
        if "meteor" in metrics["narrativeqa"]
        else meteor_score
    )
    metrics["narrativeqa"]["rouge_f"] = (
        metrics["narrativeqa"]["rouge_f"] + rouge_f
        if "rouge_f" in metrics["narrativeqa"]
        else rouge_f
    )
    metrics["narrativeqa"]["rouge_p"] = (
        metrics["narrativeqa"]["rouge_p"] + rouge_p
        if "rouge_p" in metrics["narrativeqa"]
        else rouge_p
    )
    metrics["narrativeqa"]["rouge_r"] = (
        metrics["narrativeqa"]["rouge_r"] + rouge_r
        if "rouge_r" in metrics["narrativeqa"]
        else rouge_r
    )
    metrics["narrativeqa"]["total"] = (
        metrics["narrativeqa"]["total"] + 1 if "total" in metrics["narrativeqa"] else 1
    )
    return metrics


def evaluate_dataset(dataset_name, prediction, ground_truths, metrics):
    prediction = prediction[0] if isinstance(prediction, list) else prediction
    if dataset_name in [
        "squad1",
        "squad2",
        "ropes",
        "newsqa",
        "duorc",
        "squad1_syn",
        "ropes_syn",
        "newsqa_syn",
        "duorc_syn",
    ]:
        exact_match, f1 = get_metric_squad(prediction, [truth[0] for truth in ground_truths])
        metrics = update_extractive_metrics(metrics, dataset_name, exact_match, f1)
    elif dataset_name in ["drop", "quoref", "drop_syn", "quoref_syn"]:
        exact_match, f1 = get_metric_drop(prediction, [truth[0] for truth in ground_truths])
        metrics = update_extractive_metrics(metrics, dataset_name, exact_match, f1)
    elif dataset_name == "narrativeqa":
        prediction = prediction[0] if isinstance(prediction, list) else prediction
        ground_truths = [truth[0] for truth in ground_truths]
        bleu1, bleu4, meteor, rouge_f, rouge_p, rouge_r = get_metric_narrativeqa(
            prediction, ground_truths
        )
        metrics = update_abstractive_metrics(
            metrics, bleu1, bleu4, meteor, rouge_f, rouge_p, rouge_r
        )
    else:
        print("Incorrect dataset name at :{0}".format(dataset_name))
        raise ValueError

    return metrics
