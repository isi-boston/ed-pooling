import argparse
import os
import numpy as np
from score_triggers import get_metrics_better, get_metrics_minion


TASKS = ['abstract', 'phase1', 'phase2', 'en-ace', 'en-minion']
TASK_TO_LANGS = {
    'abstract': ['en', 'ar', 'fa', 'ko'],
    'phase1': ['en', 'ar'],
    'phase2': ['en', 'fa'],
    'en-ace': ['en', 'ar'],
    'en-minion': ["en", "es", "pt", "pl", "tr", "hi", "ja", "ko"]
}
POOLING_STRATEGIES = ["first_token", "last_token", "average", "idf", "attention"]

GOLD_FILES = {
    'abstract': {
        'en': 'data/abstract/abstract.original-english.analysis.augment_gold.en.json',
        'ar': 'data/abstract/original-arabic.abstract-final.augment-simple-tok.ar.json',
        'fa': 'data/abstract/abstract.original-farsi.abstract-final.fa.json',
        'ko': 'data/abstract/abstract-phase3.original-korean.sample.augment.ko.json',
    },
    'phase1': {
        'en': 'data/phase1/basic-phase1.original-english.analysis.augment_gold.en.json',
        'ar': 'data/phase1/original-arabic.basic-final.augment-simple-tok.ar.json'
    },
    'phase2': {
        'en': 'data/phase2/basic-phase2.original-english.analysis.augment_gold.en.json',
        'fa': 'data/phase2/basic.original-farsi.basic-final.augment.fa.json'
    },
    'en-ace': {
        'en': 'data/en-ace/en_test.jhu.better.json',
        'ar': 'data/ar-ace/ar_test.jhu.better-split-80.json'
    },
    'en-minion': {lang: f'data/{lang}-minion/test.json' for lang in TASK_TO_LANGS["en-minion"]}
}

SYSTEM_FILES = {
    'abstract': {
        'en': 'preds.json',
        'ar': 'original-arabic.abstract-final.augment-simple-tok.ar.preds.json',
        'fa': 'abstract.original-farsi.abstract-final.fa.preds.resolveconflictalloverlaps.json',
        'ko': 'abstract-phase3.original-korean.sample.augment.ko.preds.json',
    },
    'phase1': {
        'en': 'preds.json',
        'ar': 'original-arabic.basic-final.augment-simple-tok.ar.preds.json'
    },
    'phase2': {
        'en': 'preds.json',
        'fa': 'basic.original-farsi.basic-final.augment.fa.preds.resolveconflictalloverlaps.json'
    },
    'en-ace': {
        'en': 'preds.json',
        'ar': 'ar_test.jhu.better-split-80.preds.json'
    },
    "en-minion": {lang: f"{lang}-test.preds.json" for lang in TASK_TO_LANGS["en-minion"]}
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_lang")
    parser.add_argument("--model_name")
    parser.add_argument("--seeds", nargs='+', type=int)
    args = parser.parse_args()

    header_rep = ','.join(['Task', 'Lang'] + POOLING_STRATEGIES)
    print(header_rep)

    for task in TASKS:
        for pred_lang in TASK_TO_LANGS[task]:
            score_rep = f'{task},{pred_lang}'
            gold_file = GOLD_FILES[task][pred_lang]
            for strategy in POOLING_STRATEGIES:
                all_metrics = []
                for seed in args.seeds:
                    expt_dir = f"{task}_{args.train_lang}_{strategy}_{args.model_name}_{seed}"
                    expt_dir = os.path.join(f'expts/issue-1/{task}', expt_dir)
                    system_file = os.path.join(expt_dir, SYSTEM_FILES[task][pred_lang])
                    if 'minion' not in task:
                        metrics = get_metrics_better(gold=gold_file, system=system_file, task=task)
                    else:
                        metrics = get_metrics_minion(gold=gold_file, system=system_file)
                    all_metrics.append(metrics)

                all_f1 = [m.f1 * 100. for m in all_metrics]
                mean_f1, std_f1 = np.mean(all_f1), np.std(all_f1)
                mean_f1, std_f1 = np.round(mean_f1, decimals=1), np.round(std_f1, decimals=1)
                score_rep += f',{mean_f1} ({std_f1})'
            print(score_rep)


if __name__ == '__main__':
    main()
