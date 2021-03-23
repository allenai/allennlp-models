import json
import logging
import time
from typing import Iterable, List, Set

from allennlp.common.checks import check_for_gpu
from allennlp.data import Instance
from allennlp.predictors import Predictor

from tqdm import tqdm

from allennlp_models.rc.metrics import SquadEmAndF1

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import allennlp_models.rc  # noqa F401: Needed to register the registrables.
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Evaluation for SQuAD 1.1")
    parser.add_argument("--cuda-device", type=int, default=-1)
    parser.add_argument("--qa-model", type=str)
    parser.add_argument(
        "--input-file",
        type=str,
        default="https://allennlp.s3.amazonaws.com/datasets/squad/squad-dev-v1.1.json",
    )
    args = parser.parse_args()

    # Read inputs
    check_for_gpu(args.cuda_device)
    predictor = Predictor.from_path(
        args.qa_model, predictor_name="transformer_qa", cuda_device=args.cuda_device
    )
    instances = predictor._dataset_reader.read(args.input_file)

    # We have to make sure we put instances with the same qid all into the same batch.
    def batch_instances_by_qid(instances: Iterable[Instance]) -> Iterable[List[Instance]]:
        current_qid = None
        current_batch = []
        for instance in instances:
            instance_qid = instance["metadata"]["id"]
            if current_qid is None:
                current_qid = instance_qid
            if instance_qid == current_qid:
                current_batch.append(instance)
            else:
                yield current_batch
                current_batch = [instance]
                current_qid = instance_qid
        if len(current_batch) > 0:
            yield current_batch

    def make_batches(
        instances: Iterable[Instance], batch_size: int = 64
    ) -> Iterable[List[Instance]]:
        current_batch: List[Instance] = []
        for qid_instances in batch_instances_by_qid(instances):
            if len(qid_instances) + len(current_batch) < batch_size:
                current_batch.extend(qid_instances)
            else:
                if len(current_batch) > 0:
                    yield current_batch
                current_batch = qid_instances
        if len(current_batch) > 0:
            yield current_batch

    # Run model and evaluate results
    last_logged_scores_time = time.monotonic()
    ids_seen: Set[str] = set()
    metric = SquadEmAndF1()
    answers = {}
    for batch in make_batches(tqdm(instances, desc="Evaluating instances")):
        gold_answers = {
            instance["metadata"]["id"]: instance["metadata"]["answers"] for instance in batch
        }
        for result in predictor.predict_batch_instance(batch):
            assert result["id"] not in ids_seen
            ids_seen.add(result["id"])
            gold_answer = gold_answers[result["id"]]
            if len(gold_answer) == 0:
                gold_answer = [""]  # no-answer case
            metric(result["best_span_str"], gold_answer)
            answers[result["id"]] = result["best_span_str"]
        if time.monotonic() - last_logged_scores_time > 30:
            exact_match, f1_score = metric.get_metric()
            logger.info(json.dumps({"em": exact_match, "f1": f1_score}))
            last_logged_scores_time = time.monotonic()

    # Print results
    exact_match, f1_score = metric.get_metric()
    print(json.dumps(answers))
    print(json.dumps({"em": exact_match, "f1": f1_score}))
