import argparse
import json
from transformers import AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--model")
    parser.add_argument("--threshold", type=int, default=128)
    args = parser.parse_args()

    # get huggingface tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    total_words, words_dropped = 0, 0
    total_labels, labels_dropped = 0, 0
    num_multi_word_trigger = 0
    label_set = set()
    with open(args.input_file, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            words, labels = data["tokens"], data["labels"]
            total_words += len(words)
            total_labels += sum([lab != 'O' for lab in labels])

            word_to_tokens = [tokenizer.tokenize(w) for w in words]
            assert len(words) == len(word_to_tokens) == len(labels)
            current_seq_length = 0
            max_words = 0
            wtl = []
            for word, toks, lab in zip(words, word_to_tokens, labels):
                current_seq_length += len(toks)
                wtl.append((word, toks, lab))
                if current_seq_length < args.threshold:
                    max_words += 1
                else:
                    words_dropped += 1
                    if lab != 'O':
                        labels_dropped += 1

            num_multi_word_trigger += sum([lab[:2] not in ['O', 'B_'] for lab in labels])
            label_set.update([lab[2:] if len(lab) > 2 else lab for lab in labels])

    # dump
    print(
        f"total_words = {total_words}, "
        f"words_dropped = {words_dropped}, "
        f"total_labels = {total_labels}, "
        f"labels_dropped = {labels_dropped}, "
        f"labels dropped = {labels_dropped/total_labels}, "
        f"num_multi_word_trigger = {num_multi_word_trigger}, "
        f"label_set = {sorted(list(label_set))}"
    )
