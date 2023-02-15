from typing import List, Tuple
import argparse
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from better_events.better_core import BetterDocument
from better_events.better_utilities import load_documents
from tasks.task_minion import TriggerClassificationMinionTask
from metrics import TokenClassificationMetrics


def get_triggers_better(
        doc: BetterDocument, task: str, remove_duplicates: bool
) -> List[Tuple[str, int, int]]:
    triggers = []
    for sent in doc.sentences:
        events = sent.abstract_events if task == 'abstract' else sent.basic_events
        for event in events:
            event_type = event.event_type
            for anchor in event.anchors:
                start_char = anchor.grounded_span.full_span.start_char
                end_char = anchor.grounded_span.full_span.end_char
                triggers.append((event_type, start_char, end_char))

    if remove_duplicates:
        # sort by start_char
        triggers = sorted(set(triggers), key=lambda x: x[1])
    return triggers


def get_metrics_better(gold: str, system: str, task: str):

    gold_data = load_documents(filename=gold)
    system_data = load_documents(filename=system)

    gold_n = 0
    system_n = 0
    correct_n = 0
    for doc_id, gold_doc in gold_data.items():
        system_doc = system_data[doc_id]

        gold_triggers = get_triggers_better(doc=gold_doc, task=task, remove_duplicates=True)
        system_triggers = get_triggers_better(doc=system_doc, task=task, remove_duplicates=True)

        gold_n += len(gold_triggers)
        system_n += len(system_triggers)
        for g in gold_triggers:
            for s in system_triggers:
                if s == g:
                    correct_n += 1

    try:
        precision = correct_n / system_n
        recall = correct_n / gold_n
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        precision = 0
        recall = 0
        f1 = 0

    return TokenClassificationMetrics(
        acc_score=0., precision=precision, recall=recall, f1=f1, class_report=None
    )


def get_metrics_minion(gold: str, system: str):

    task = TriggerClassificationMinionTask()
    gold_data = task.read_examples_from_file(file_path=gold, augment=False)
    system_data = task.read_examples_from_file(file_path=system, augment=False)

    assert len(gold_data) == len(system_data)
    gold_labels = [d.labels for d in gold_data]
    system_labels = [d.labels for d in system_data]

    precision = precision_score(gold_labels, system_labels)
    recall = recall_score(gold_labels, system_labels)
    f1 = f1_score(gold_labels, system_labels)
    return TokenClassificationMetrics(
        acc_score=0., precision=precision, recall=recall, f1=f1, class_report=None
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold")
    parser.add_argument("--system")
    parser.add_argument("--task")
    args = parser.parse_args()

    if "minion" not in args.task:
        metrics = get_metrics_better(gold=args.gold, system=args.system, task=args.task)
    else:
        metrics = get_metrics_minion(gold=args.gold, system=args.system)
    print(f"precision = {metrics.precision}, recall = {metrics.recall}, f1 = {metrics.f1}")


if __name__ == '__main__':
    main()
