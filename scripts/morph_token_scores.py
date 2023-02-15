import argparse
import json


def get_scores(flags, aggregation_strategy):
    flags = [1. if f else 0. for f in flags]

    if not flags or all([f == 0. for f in flags]):
        return flags

    if aggregation_strategy == 'mean':
        scores = [f / sum(flags) for f in flags]
    elif aggregation_strategy == 'first':
        first_index = flags.index(1.)
        scores = [0. for _ in flags]
        scores[first_index] = 1.
    elif aggregation_strategy == 'last':
        final_index = max([index for index, item in enumerate(flags) if item == 1.])
        scores = [0. for _ in flags]
        scores[final_index] = 1.
    else:
        raise NotImplementedError
    return scores


def get_morph_scores(input_file, aggregation_strategy):

    data = json.load(open(input_file, encoding="utf-8"))
    morph_scores = {}
    differences = 0
    for entry in data:
        words, tokenized, mlm_stem_flags = \
            entry["words"], entry["tokenized"], entry["mlm_stem_flags"]
        assert len(words) == len(tokenized) == len(mlm_stem_flags)
        for word, toks, flags in zip(words, tokenized, mlm_stem_flags):
            assert len(toks) == len(flags)
            flags = get_scores(flags=flags, aggregation_strategy=aggregation_strategy)
            if word not in morph_scores:
                morph_scores[word] = {"tokens": toks, "flags": flags}
            else:
                assert len(morph_scores[word]["tokens"]) == len(toks)
                assert all([t == m for t, m in zip(toks, morph_scores[word]["tokens"])])
                if any([t != m for t, m in zip(flags, morph_scores[word]["flags"])]):
                    print(
                        f"{entry['doc_id']} "
                        f"{entry['sent_id']} "
                        f"{entry['text']} "
                        f"{toks} "
                        f"{flags} "
                        f"{morph_scores[word]}"
                    )
                    differences += 1
    return morph_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--aggregate")
    args = parser.parse_args()

    morph_scores = {}
    for input_file in args.inputs:
        current_morph_scores = get_morph_scores(
            input_file=input_file, aggregation_strategy=args.aggregate
        )
        morph_scores.update(current_morph_scores)

    json.dump(
        morph_scores, indent=4, ensure_ascii=False, fp=open(args.output, 'w', encoding='utf-8')
    )


if __name__ == "__main__":
    main()
