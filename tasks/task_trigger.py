from typing import Dict, List
import json

from better_events.better_core import (
    BetterDocument,
    BetterSentence,
    BetterSpanSet,
    BetterEvent,
    ABSTRACT_EVENT_TYPE,
    HELPFUL_HARMFUL,
    MATERIAL_VERBAL,
)
from tasks.task_base import Guid, InputExample, TokenClassificationTask
from tasks.task_labels import (
    get_bio_labels,
    convert_ids_to_labels,
    BetterBasicTriggerLabels,
    BetterAbstractTriggerLabels,
    BetterAbstractTriggerLabelsWithQuad,
    AceTriggerLabels,
    VALID_BASIC_EVENT_TYPES_PHASE_1,
    BASIC_EVENT_TYPES_PHASE_2_ONLY,
    BASIC_EVENT_TYPES_PHASE_3_ONLY,
    MinionLabels
)
from prediction_object import SeqPredictionObject
from tasks.task_utils import get_better_span_set, get_predicted_spans


class TriggerClassificationTask(TokenClassificationTask):

    def get_labels(self) -> List[str]:
        raise NotImplementedError

    def get_label_map(self) -> Dict[int, str]:
        raise NotImplementedError

    def get_events(self, sent: BetterSentence) -> List[BetterEvent]:
        raise NotImplementedError

    def empty_events(self, sent: BetterSentence) -> None:
        raise NotImplementedError

    def get_event_type(self, event) -> str:
        raise NotImplementedError

    def construct_event_from_span_set(
            self,
            idx: str,
            event_type: str,
            span_set: BetterSpanSet
    ) -> BetterEvent:
        raise NotImplementedError

    def add_event_to_sentence(self, sentence: BetterSentence, event: BetterEvent) -> None:
        # Will mutate sentence
        raise NotImplementedError

    def read_examples_from_file(
            self,
            file_path: str,
            augment: bool
    ) -> List[InputExample]:

        examples = []
        data = json.load(open(file_path, encoding='utf-8'))
        for doc_id, doc in data.items():
            better_doc = BetterDocument.from_json(doc)
            for sent in better_doc.sentences:

                if sent.sentence_type not in {'Sentence', 'Headline'}:
                    continue

                # words
                words = [t.text for t in sent.tokens]

                # labels
                labels = ['O'] * len(words)
                events = self.get_events(sent)
                for event in events:
                    event_type = self.get_event_type(event)
                    for scored_span in event.anchors:
                        start = scored_span.span.grounded_span.full_span.start_token
                        end = scored_span.span.grounded_span.full_span.end_token + 1
                        labels[start] = f'B-{event_type}'
                        for idx in range(start + 1, end):
                            labels[idx] = f'I-{event_type}'

                # guid
                guid = Guid(sent_id=sent.sent_id, doc_id=doc_id)

                examples.append(
                    InputExample(
                        words=words,
                        labels=labels,
                        guid=guid,
                        skip_transformer_tokenization=better_doc.properties.get(
                            'skip_mlm_tokenization', False
                        )
                    )
                )
        return examples

    def write_predictions_to_file(
            self,
            predictions: List[SeqPredictionObject],
            orig_filename: str,
            out_filename: str
    ) -> None:

        with open(orig_filename, "r", encoding='utf-8') as better_f:
            data = json.load(better_f)

        better_docs = {}
        for doc_id, doc in data.items():
            better_docs[doc_id] = BetterDocument.from_json(doc)
            # Get rid of all events in this original document;
            # we're going to replace them with predicted events
            for sent in better_docs[doc_id].sentences:
                self.empty_events(sent)

        examples = [p.example for p in predictions]
        prediction_ids = [p.preds for p in predictions]
        prediction_probs = [p.pred_prob for p in predictions]
        prediction_labels = convert_ids_to_labels(
            label_map=self.get_label_map(),
            ids=prediction_ids
        )

        for example, pred_labels, pred_probs in zip(examples, prediction_labels, prediction_probs):
            sent_id = example.guid.sent_id
            doc_id = example.guid.doc_id
            doc = better_docs[doc_id]
            sentence = doc.sentences_by_id[sent_id]

            spans = get_predicted_spans(pred_labels)
            for id_counter, span in enumerate(spans):
                span_set = get_better_span_set(
                    sentence,
                    span,
                    score=pred_probs[span['start']]  # score is the score of B- label.
                )
                event_idx = '_'.join([str(doc_id), str(sent_id), str(id_counter)])
                event = self.construct_event_from_span_set(event_idx, span['type'], span_set)
                self.add_event_to_sentence(sentence, event)
            doc.sentences_by_id[sent_id] = sentence

        # Transform to dictionaries in preparation for serialization to JSON
        for doc_id, doc in better_docs.items():
            better_docs[doc_id] = better_docs[doc_id].to_dict()

        # Always pretty print for now
        with open(out_filename, 'w', encoding='utf-8') as outfile:
            outfile.write(json.dumps(better_docs, indent=2, ensure_ascii=False))


class BetterBasicTrigger(TriggerClassificationTask):
    def __init__(self):
        self.labels = get_bio_labels(self.supported_event_types())
        self.label_map = {i: label for i, label in enumerate(self.labels)}

    def get_labels(self) -> List[str]:
        return self.labels

    def get_label_map(self) -> Dict[int, str]:
        return self.label_map

    def supported_event_types(self):
        return BetterBasicTriggerLabels

    def get_events(self, sent: BetterSentence) -> List[BetterEvent]:

        # We may have a file with P1 and P2 events, but may want to read only P1 events.
        supported_events = []
        for ev in sent.basic_events:
            if ev.event_type in self.supported_event_types():
                supported_events.append(ev)
        return supported_events

    def empty_events(self, sent: BetterSentence) -> None:
        # Remove only the events which are supported by this class.
        # For events which are not supported:
        #     1) For each anchor, a new event is created
        #     2) The arguments are removed
        not_supported_events = []
        for ev in sent.basic_events:
            if ev.event_type not in self.supported_event_types():
                for idx, anchor in enumerate(ev.anchors):
                    new_ev = self.construct_event_from_span_set(
                        f"{ev.event_id}-{idx}",
                        ev.event_type,
                        BetterSpanSet([anchor])
                    )
                    not_supported_events.append(new_ev)
        sent.basic_events = not_supported_events

    def get_event_type(self, event) -> str:
        return event.event_type

    def construct_event_from_span_set(
            self,
            idx: str,
            event_type: str,
            span_set: BetterSpanSet
    ) -> BetterEvent:
        event = BetterEvent(
            event_id=idx,
            event_type=event_type,
            properties={},
            anchors=span_set,
            arguments=[],
            event_arguments=[],
            state_of_affairs=None
        )
        return event

    def add_event_to_sentence(self, sentence: BetterSentence, event: BetterEvent) -> None:
        # add only if event does not overlap with any existing event in the sentence.

        assert len(event.anchors.spans) == 1
        overlaps_with_existing_event = False
        for ev in sentence.basic_events:
            for anchor in ev.anchors:
                if anchor.grounded_span.head_span.overlaps(
                        event.anchors.spans[0].grounded_span.head_span
                ):
                    overlaps_with_existing_event = True
                    break
        if not overlaps_with_existing_event:
            sentence.basic_events.append(event)


class BetterBasicTriggerP1Only(BetterBasicTrigger):
    def __init__(self):
        super(BetterBasicTriggerP1Only, self).__init__()

    def supported_event_types(self):
        return VALID_BASIC_EVENT_TYPES_PHASE_1


class BetterBasicTriggerP2Only(BetterBasicTrigger):
    def __init__(self):
        super(BetterBasicTriggerP2Only, self).__init__()

    def supported_event_types(self):
        return BASIC_EVENT_TYPES_PHASE_2_ONLY


class BetterBasicTriggerP3Only(BetterBasicTrigger):
    def __init__(self):
        super(BetterBasicTriggerP3Only, self).__init__()

    def supported_event_types(self):
        return BASIC_EVENT_TYPES_PHASE_3_ONLY


class BetterAbstractTrigger(TriggerClassificationTask):
    def __init__(self):
        self.labels = get_bio_labels(BetterAbstractTriggerLabels)
        self.label_map = {i: label for i, label in enumerate(self.labels)}

    def get_labels(self) -> List[str]:
        return self.labels

    def get_label_map(self) -> Dict[int, str]:
        return self.label_map

    def get_events(self, sent: BetterSentence) -> List[BetterEvent]:
        return sent.abstract_events

    def empty_events(self, sent: BetterSentence) -> None:
        sent.abstract_events = []

    def get_event_type(self, event) -> str:
        return event.event_type

    def construct_event_from_span_set(
            self,
            idx: str,
            event_type: str,
            span_set: BetterSpanSet
    ) -> BetterEvent:
        event = BetterEvent(
            event_id=idx,
            event_type=ABSTRACT_EVENT_TYPE,
            properties={},
            anchors=span_set,
            arguments=[],
            event_arguments=[],
            state_of_affairs=None
        )
        return event

    def add_event_to_sentence(self, sentence: BetterSentence, event: BetterEvent) -> None:
        sentence.abstract_events.append(event)


class BetterAbstractTriggerWithQuad(TriggerClassificationTask):
    def __init__(self):
        self.labels = get_bio_labels(BetterAbstractTriggerLabelsWithQuad)
        self.label_map = {i: label for i, label in enumerate(self.labels)}

    def get_labels(self) -> List[str]:
        return self.labels

    def get_label_map(self) -> Dict[int, str]:
        return self.label_map

    def get_events(self, sent: BetterSentence) -> List[BetterEvent]:
        return sent.abstract_events

    def empty_events(self, sent: BetterSentence) -> None:
        sent.abstract_events = []

    def get_event_type(self, event) -> str:
        event_type = f'{event.properties[HELPFUL_HARMFUL]}+' \
            f'{event.properties[MATERIAL_VERBAL]}'
        return event_type

    def construct_event_from_span_set(
            self,
            idx: str,
            event_type: str,
            span_set: BetterSpanSet
    ) -> BetterEvent:
        help_harm, mat_verb = event_type.split("+")
        event = BetterEvent(
            event_id=idx,
            event_type=ABSTRACT_EVENT_TYPE,
            properties={HELPFUL_HARMFUL: help_harm, MATERIAL_VERBAL: mat_verb},
            anchors=span_set,
            arguments=[],
            event_arguments=[],
            state_of_affairs=None
        )
        return event

    def add_event_to_sentence(self, sentence: BetterSentence, event: BetterEvent) -> None:
        sentence.abstract_events.append(event)


class AceTrigger(TriggerClassificationTask):
    def __init__(self):
        self.labels = get_bio_labels(AceTriggerLabels)
        self.label_map = {i: label for i, label in enumerate(self.labels)}

    def get_labels(self) -> List[str]:
        return self.labels

    def get_label_map(self) -> Dict[int, str]:
        return self.label_map

    def get_events(self, sent: BetterSentence) -> List[BetterEvent]:
        return sent.basic_events

    def empty_events(self, sent: BetterSentence) -> None:
        sent.basic_events = []

    def get_event_type(self, event) -> str:
        return event.event_type

    def construct_event_from_span_set(
            self,
            idx: str,
            event_type: str,
            span_set: BetterSpanSet
    ) -> BetterEvent:
        event = BetterEvent(
            event_id=idx,
            event_type=event_type,
            properties={},
            anchors=span_set,
            arguments=[],
            event_arguments=[],
            state_of_affairs=None
        )
        return event

    def add_event_to_sentence(self, sentence: BetterSentence, event: BetterEvent) -> None:
        sentence.basic_events.append(event)


class MinionTriggerBetterFormat(TriggerClassificationTask):
    def __init__(self):
        self.labels = get_bio_labels(MinionLabels)
        self.label_map = {i: label for i, label in enumerate(self.labels)}

    def get_labels(self) -> List[str]:
        return self.labels

    def get_label_map(self) -> Dict[int, str]:
        return self.label_map

    def get_events(self, sent: BetterSentence) -> List[BetterEvent]:
        return sent.basic_events

    def empty_events(self, sent: BetterSentence) -> None:
        sent.basic_events = []

    def get_event_type(self, event) -> str:
        return event.event_type

    def construct_event_from_span_set(
            self,
            idx: str,
            event_type: str,
            span_set: BetterSpanSet
    ) -> BetterEvent:
        event = BetterEvent(
            event_id=idx,
            event_type=event_type,
            properties={},
            anchors=span_set,
            arguments=[],
            event_arguments=[],
            state_of_affairs=None
        )
        return event

    def add_event_to_sentence(self, sentence: BetterSentence, event: BetterEvent) -> None:
        sentence.basic_events.append(event)


if __name__ == "__main__":
    task_ = BetterBasicTriggerP2Only()
    examples_ = task_.read_examples_from_file(
       "../c_1127260902.augment.zh.json",
        augment=False
    )
    print(examples_[0].words, examples_[0].labels)
