import argparse
from importlib import import_module
from transformers import AutoTokenizer


def get_task(task_type):

    # task
    task_modules = [
        'tasks.task_trigger', 'tasks.task_minion',
    ]
    token_classification_task = None
    for mod in task_modules:
        try:
            module = import_module(mod)
            token_classification_task = getattr(module, task_type)()
        except AttributeError:
            continue
    return token_classification_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--task_type")
    parser.add_argument("--model_name_or_path")
    parser.add_argument("--do_lowercase", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        do_lower_case=args.do_lowercase
    )
    task = get_task(args.task_type)

    examples = task.read_examples_from_file(file_path=args.input, augment=False)
    num_words = 0
    num_tokens = 0
    for example in examples:
        words = example.words
        labels = example.labels
        words = [w for w, l in zip(words, labels) if l != 'O']
        tokens = []
        for w in words:
            tokens.extend(tokenizer.tokenize(w))
        num_words += len(words)
        num_tokens += len(tokens)

    print(f"{args.input}\t{num_tokens / num_words}")


if __name__ == '__main__':
    main()
