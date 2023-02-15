from typing import Any, Dict, List, Optional

from better_events.better_core import (
    BetterSentence,
    SimpleGroundedSpan,
    GroundedSpan,
    BetterSpan,
    BetterSpanSet,
    ScoredBetterSpan,
)


def remove_lone_i(labels: List[str]) -> List[str]:

    """
    ['B-x', 'B-x', 'I-x', 'I-o', 'B-z', 'O', 'B-d']
     -> ['B-x', 'B-x', 'I-x', 'O', 'B-z', 'O', 'B-d']
    """

    current_root = None
    for idx, lab in enumerate(labels):
        if lab == 'O':
            current_root = None
            continue
        else:
            root = lab[2:]
            if lab[:2] == 'B-':
                current_root = root
                continue
            elif lab[:2] == 'I-':
                if current_root is None or current_root != root:
                    labels[idx] = 'O'
                    current_root = None
    return labels


def get_predicted_spans(
        predictions: List[str]
) -> List[Dict[str, Any]]:

    spans = []
    k = 0
    current_span = None
    current_span_to_be_added = False
    while k < len(predictions):
        if predictions[k] == 'O' and current_span_to_be_added:
            current_span['end'] = k - 1
            spans.append(current_span)
            current_span_to_be_added = False
        elif predictions[k].startswith('B-'):
            if current_span_to_be_added:
                current_span['end'] = k - 1
                spans.append(current_span)
            current_span = {
                'type': predictions[k].split('-', maxsplit=1)[-1],
                'start': k,
                'end': k  # inclusive
            }
            current_span_to_be_added = True
        k += 1

    if current_span_to_be_added:
        current_span['end'] = len(predictions) - 1
        spans.append(current_span)

    return spans


def get_better_grounded_span(
        sentence: BetterSentence,
        span: Dict[str, Any],
        mention_idx: Optional[str] = None
) -> GroundedSpan:
    tokens = sentence.tokens[span['start']:span['end'] + 1]

    full_span = SimpleGroundedSpan(
        sentence.doc_text,
        start_char=tokens[0].doc_character_span[0],
        end_char=tokens[-1].doc_character_span[1],
        start_token=span['start'],
        end_token=span['end']
    )

    return GroundedSpan(
        sent_id=int(sentence.sent_id),
        full_span=full_span,
        head_span=full_span,
        mention_id=mention_idx
    )


def get_better_span_set(
        sentence: BetterSentence,
        span: Dict[str, Any],
        score: float,
        mention_id: Optional[str] = None
) -> BetterSpanSet:

    grounded_span = get_better_grounded_span(sentence, span, mention_idx=mention_id)
    better_span = BetterSpan(
        grounded_span.full_span.text,
        grounded_span.head_span.text,
        grounded_span
    )
    scored_better_span = ScoredBetterSpan(better_span, score=score)
    span_set = BetterSpanSet([scored_better_span])
    return span_set
