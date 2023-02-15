import argparse
import numpy as np
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--model_name_or_path")
    parser.add_argument("--max_line_count", type=int)
    parser.add_argument("--do_lowercase", action="store_true")
    args = parser.parse_args()

    token_counts = {}
    words_count = 0
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        do_lower_case=args.do_lowercase
    )
    total_line_count = 0
    for file_input in args.inputs:
        if total_line_count >= args.max_line_count:
            break
        with open(file_input, encoding="utf-8") as f:
            for line in f:
                if total_line_count >= args.max_line_count:
                    break
                total_line_count += 1
                line = line.rstrip()
                words = line.split()
                for word in words:
                    words_count += 1
                    tokens = tokenizer.tokenize(word)
                    tokens = set(tokens)
                    for token in tokens:
                        token_counts[token] = token_counts.get(token, 0) + 1

    with open(args.output, 'w', encoding="utf-8") as f:
        for tok, count in token_counts.items():
            idf = np.log(words_count / count)
            print(f"{tok}\t{idf}", file=f)


if __name__ == "__main__":
    main()
