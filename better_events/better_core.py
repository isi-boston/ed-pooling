import unicodedata
from collections import defaultdict
from itertools import combinations
from typing import List, Optional, Iterable, Mapping, Dict, Tuple, AbstractSet,\
    Any, Union, Callable, DefaultDict

import sys
from attr import attrib, attrs, asdict, validators

from better_events.better_annotation import (
    AnnotatedText,
    AnnotationSet,
    StructuralElement,
    ISO_1_TO_ISO_2
)
from better_events.better_validation import validate_granular_arg_role, validate_irrealis, \
    validate_argument_role, validate_event_type, VALID_MENTION_TYPES, NOMINAL, \
    validate_mention_type, NAME, PRONOUN, EVENT_ANCHOR, STRUCTURAL_SENTENCE, \
    validate_granular_arg_property, TEMPLATE_ANCHOR, VALID_GRANULAR_ARG_PROPERTIES, \
    VALID_GRANULAR_ARG_ROLES, EXCLUDED_BASIC_EVENT_TYPES, is_guessed_event_argument_role
from better_events.propositions import Proposition

HELPFUL_HARMFUL = "helpful-harmful"
MATERIAL_VERBAL = "material-verbal"

ABSTRACT_EVENT_TYPE = "abstract"

ABSTRACT_TASK = "abstract"
BASIC_TASK = "basic"
GRANULAR_TASK = "granular"

VALID_EVENT_TASKS = [ABSTRACT_TASK, BASIC_TASK, GRANULAR_TASK]

AGENT = "agent"
PATIENT = "patient"
MONEY = "money"
REF_EVENT = "ref-event"


@attrs(frozen=True, slots=True, hash=True, cache_hash=True, eq=False)
class DocumentText:
    """Inspired by flexnlp DocumentText, but without text/raw_text
       distinction. Used to make top-level document text available
       to low-level components without gratuitous and error-prone
       string copying. """
    text: str = attrib(validator=validators.instance_of(str))

    def span_text(self, start: int, end_incl: int):
        return self.text[start: end_incl + 1]

    def __eq__(self, other) -> bool:
        if self is other:
            return True

        return (
            isinstance(other, DocumentText)
            and hash(self) == hash(other)
            and self.text == other.text
        )


@attrs(frozen=True)
class BetterDependency:
    source_index: int = attrib()
    source_label: str = attrib()
    target_index: int = attrib()
    target_label: str = attrib()
    edge_label: str = attrib()

    @classmethod
    def from_json(cls, json_dict: Dict):
        # noinspection PyArgumentList
        return cls(
            json_dict["source_index"],
            json_dict["source_label"],
            json_dict["target_index"],
            json_dict["target_label"],
            json_dict["edge_label"],
        )

    def to_dict(self) -> Mapping:
        return asdict(self)

    def __str__(self):
        return "{}:{} --{}--> {}:{}".format(
            self.source_index,
            self.source_label,
            self.edge_label,
            self.target_index,
            self.target_label,
        )

    # Create a copy with modified word indices.
    def offset_variant(self, word_offset: int) -> "BetterDependency":
        return BetterDependency(
            self.source_index + word_offset,
            self.source_label,
            self.target_index + word_offset,
            self.target_label,
            self.edge_label,
        )

    @staticmethod
    def spans_for_set(dependencies: Iterable['BetterDependency']):
        return [
            tuple(sorted([dep.source_index, dep.target_index]))
            for dep in dependencies
        ]


@attrs(frozen=True, cmp=False, repr=False)
class SimpleGroundedSpan:
    """start/end tokens in a sentence, character start/end
    relative to document"""
    doc_text: DocumentText = attrib()
    # offsets are INCLUSIVE
    start_char: int = attrib()
    end_char: int = attrib()
    start_token: Optional[int] = attrib()
    end_token: Optional[int] = attrib()

    @property
    def text(self):
        return self.doc_text.span_text(self.start_char, self.end_char)

    def __attrs_post_init__(self):
        if self.start_token is not None or self.end_token is not None:
            if self.start_token is None or self.end_token is None:
                raise ValueError("Token start or end was not initialized")
        if (
            self.start_char < 0
            or self.end_char < 0
            or (self.start_token is not None and self.start_token < 0)
            or (self.end_token is not None and self.end_token < 0)
        ):
            raise ValueError("char or token index must be >= 0")
        if self.start_char > self.end_char or (
            self.start_token is not None
            and self.end_token is not None
            and self.start_token > self.end_token
        ):
            raise ValueError("start and end are mismatched (start > end")

    def __eq__(self, other: 'SimpleGroundedSpan'):
        # OK if text differs, that could be a weird artifact of something
        return self.start_char == other.start_char and self.end_char == other.end_char

    def __lt__(self, other: 'SimpleGroundedSpan'):
        return (self.start_char, self.end_char) < (other.start_char, other.end_char)

    def __hash__(self):
        return hash(self.doc_text) ^ hash(self.start_char) ^ \
            hash(self.end_char) ^ hash(self.start_token) ^ hash(self.end_token)

    def with_new_tokens(self, start_tok: int, end_tok: int) -> "SimpleGroundedSpan":
        return SimpleGroundedSpan(
            self.doc_text, self.start_char, self.end_char, start_tok, end_tok
        )

    def contains(self, other: "SimpleGroundedSpan"):
        return self.start_char <= other.start_char and self.end_char >= other.end_char

    def overlaps(self, other: "SimpleGroundedSpan"):
        if other.start_char <= self.start_char <= other.end_char:
            return True
        if self.start_char <= other.start_char <= self.end_char:
            return True
        return False

    @classmethod
    def from_json(cls, doc_text: DocumentText, json_dict: Dict):
        # noinspection PyArgumentList
        return cls(
            doc_text,
            json_dict["start_char"],
            json_dict["end_char"],
            json_dict.get("start_token", None),
            json_dict.get("end_token", None),
        )

    def to_dict(self) -> Mapping:
        results = {"text": self.text,
                   "start_char": self.start_char,
                   "end_char": self.end_char,
                   "start_token": self.start_token,
                   "end_token": self.end_token}
        return results

    def __str__(self):
        return "{} tok:[{}:{}] char:[{}:{}]".format(
            self.text, self.start_token, self.end_token, self.start_char, self.end_char
        )

    def __repr__(self):
        return str(self)

    def offset_variant(self, word_offset: int) -> "SimpleGroundedSpan":
        return SimpleGroundedSpan(
            self.doc_text,
            self.start_char,
            self.end_char,
            self.start_token + word_offset,
            self.end_token + word_offset,
        )


@attrs(frozen=True, cmp=False)
class GroundedSpan:
    """Both a full & head span, plus an optional mention ID"""

    sent_id: int = attrib(converter=int)
    full_span: SimpleGroundedSpan = attrib()
    head_span: SimpleGroundedSpan = attrib()
    mention_id: Optional[str] = attrib()

    @classmethod
    def from_json(cls, doc_text: DocumentText, json_dict: Dict):
        # noinspection PyArgumentList
        return cls(
            json_dict.get("sent_id"),
            SimpleGroundedSpan.from_json(doc_text, json_dict["full_span"]),
            SimpleGroundedSpan.from_json(doc_text, json_dict["head_span"]),
            json_dict.get("mention_id", None),
        )

    def with_new_mention_id(self, mention_id: Optional[str]) -> "GroundedSpan":
        return GroundedSpan(
            self.sent_id, self.full_span, self.head_span, mention_id
        )

    def with_new_head(self, head_span: SimpleGroundedSpan) -> "GroundedSpan":
        return GroundedSpan(
            self.sent_id, self.full_span, head_span, self.mention_id
        )

    def with_new_full(self, full_span: SimpleGroundedSpan) -> "GroundedSpan":
        return GroundedSpan(
            self.sent_id, full_span, self.head_span, self.mention_id
        )

    def with_new_head_and_full(
        self, head_span: SimpleGroundedSpan, full_span: SimpleGroundedSpan
    ) -> "GroundedSpan":
        return GroundedSpan(self.sent_id, full_span, head_span, self.mention_id)

    def to_dict(self) -> Mapping:
        results = {"sent_id": self.sent_id,
                   "full_span": self.full_span.to_dict(),
                   "head_span": self.head_span.to_dict(),
                   "mention_id": self.mention_id}
        return results

    def __str__(self):
        return "{} ({}) (sent:{}, ment:{})".format(
            self.full_span, self.head_span, self.sent_id, self.mention_id
        )

    def __hash__(self):
        return hash(self.sent_id) ^ hash(self.full_span) ^ hash(self.head_span)

    def __eq__(self, other: "GroundedSpan"):
        return self.full_span == other.full_span and \
               self.head_span == other.head_span

    def __lt__(self, other: "GroundedSpan"):
        return self.full_span < other.full_span

    def give_max_span(self) -> Iterable[int]:
        return self.full_span.start_token, self.full_span.end_token

    def offset_variant(self, word_offset: int, sent_id: int = None) -> "GroundedSpan":
        if sent_id is None:
            sent_id = self.sent_id
        # noinspection PyArgumentList
        return self.__class__(
            sent_id,
            self.full_span.offset_variant(word_offset),
            self.head_span.offset_variant(word_offset),
            self.mention_id,
        )


@attrs(frozen=True, hash=False, cmp=False)
class BetterSpan:
    """Text & head_text, plus an optional grounded span"""

    text: str = attrib()
    head_text: str = attrib()
    grounded_span: Optional[GroundedSpan] = attrib()

    @classmethod
    def from_json(cls, doc_text: DocumentText, json_dict: Dict):
        gs = None
        if "grounded_span" in json_dict:
            gs = GroundedSpan.from_json(doc_text, json_dict["grounded_span"])
        # noinspection PyArgumentList
        return cls(json_dict["text"], json_dict["head_text"], gs)

    def with_new_mention_id(self, mention_id: Optional[str]) -> "BetterSpan":
        if not self.grounded_span:
            return self
        new_gs = self.grounded_span.with_new_mention_id(mention_id)
        return BetterSpan(self.text, self.head_text, new_gs)

    def with_new_head(self, head_span: SimpleGroundedSpan) -> "BetterSpan":
        new_gs = self.grounded_span.with_new_head(head_span)
        return BetterSpan(self.text, head_span.text, new_gs)

    def with_new_full(self, full_span: SimpleGroundedSpan) -> "BetterSpan":
        new_gs = self.grounded_span.with_new_full(full_span)
        return BetterSpan(full_span.text, self.head_text, new_gs)

    def with_new_head_and_full(
        self, head_span: SimpleGroundedSpan, full_span: SimpleGroundedSpan
    ) -> "BetterSpan":
        new_gs = self.grounded_span.with_new_head_and_full(head_span, full_span)
        return BetterSpan(full_span.text, head_span.text, new_gs)

    @classmethod
    def from_grounded_span(cls, gs: GroundedSpan):
        return BetterSpan(gs.full_span.text, gs.head_span.text, gs)

    def to_dict(self) -> Dict:
        results = {
            "text": self.text,
            "head_text": self.head_text,
        }
        if self.grounded_span:
            results["grounded_span"] = self.grounded_span.to_dict()
        return results

    def __str__(self):
        # Does NOT return grounding
        return "{} ({})".format(self.text, self.head_text)

    def __eq__(self, other: "BetterSpan"):
        if self.grounded_span and other.grounded_span:
            return self.grounded_span == other.grounded_span
        return self.text == other.text and self.head_text == other.head_text

    def __lt__(self, other: "BetterSpan"):
        if self.grounded_span and other.grounded_span:
            return self.grounded_span < other.grounded_span
        return self.text < other.text

    def __hash__(self):
        if self.grounded_span:
            return hash(self.grounded_span)
        return hash(self.text) ^ hash(self.head_text)

    def offset_variant(self, word_offset: int, sent_id: int) -> "BetterSpan":
        if self.grounded_span is None:
            out_gs = None
        else:
            out_gs = self.grounded_span.offset_variant(word_offset, sent_id)
        # noinspection PyArgumentList
        return self.__class__(self.text, self.head_text, out_gs)


@attrs(frozen=True, hash=False, cmp=False)
class ScoredBetterSpan:
    span: BetterSpan = attrib()
    score: Optional[float] = attrib(default=None)

    @classmethod
    def from_json(cls, doc_text: DocumentText, json_dict: Dict):
        span = BetterSpan.from_json(doc_text, json_dict)
        return cls(span, json_dict.get("score", None))

    def with_new_mention_id(self, mention_id: Optional[str]) -> 'ScoredBetterSpan':
        return ScoredBetterSpan(self.span.with_new_mention_id(mention_id), self.score)

    @property
    def text(self):
        return self.span.text

    @property
    def head_text(self):
        return self.span.head_text

    @property
    def grounded_span(self):
        return self.span.grounded_span

    def __hash__(self):
        return hash(self.span)

    def __lt__(self, other: "ScoredBetterSpan"):
        return self.span < other.span

    def to_dict(self):
        span_dict = self.span.to_dict()
        if self.score is not None:
            span_dict["score"] = self.score
        return span_dict

    def __str__(self):
        return f"{str(self.span)} score: {self.score}"

    def offset_variant(
            self, word_offset: int, sent_id: int
    ) -> "ScoredBetterSpan":
        out_s = self.span.offset_variant(word_offset, sent_id)
        # noinspection PyArgumentList
        return self.__class__(out_s, self.score)


@attrs(frozen=True, cmp=False)
class BetterSpanSet:
    """A set of BetterSpans that all correspond
    to the same real-world 'thing'. """

    spans: List[ScoredBetterSpan] = attrib(
        converter=lambda ls: sorted(list(set(ls)))
    )

    @classmethod
    def from_json(cls, doc_text: DocumentText, json_dict: Dict):
        # noinspection PyArgumentList
        return cls(
            [ScoredBetterSpan.from_json(doc_text, x) for x in json_dict["spans"]]
        )

    def to_dict(self) -> Mapping:
        return {"spans": [x.to_dict() for x in sorted(self.spans)]}

    def to_mitre(self):
        """Used to turn abstract events data to mitre format,
         since it isn't possible to serialize gold Mentions there.
         Converts back to EXCLUSIVE span boundaries: [) """

        spans = []
        for s in sorted(self.spans):
            span_dict = {}
            if s.grounded_span:
                if s.grounded_span.full_span:
                    span_dict["start"] = s.grounded_span.full_span.start_char
                    span_dict["end"] = s.grounded_span.full_span.end_char + 1
                if s.grounded_span.head_span:
                    span_dict["hstart"] = s.grounded_span.head_span.start_char
                    span_dict["hend"] = s.grounded_span.head_span.end_char + 1
                else:
                    span_dict["no-head"] = True

            span_dict["string"] = s.text
            span_dict["hstring"] = s.head_text

            spans.append(span_dict)
        return spans

    def __iter__(self):
        return iter(self.spans)

    def __getitem__(self, idx) -> ScoredBetterSpan:
        return self.spans[idx]

    def __len__(self):
        return len(self.spans)

    def __eq__(self, other: 'BetterSpanSet'):
        """Defined so that a span set behaves like a *set* (order of spans
        is irrelevant)"""
        if not isinstance(other, BetterSpanSet):
            raise ValueError
        for span in self.spans:
            if span not in other.spans:
                return False
        return True

    def __hash__(self):
        val = 1
        for span in self.spans:
            val ^= hash(span.grounded_span)
        return val

    def __lt__(self, other: "BetterSpanSet"):
        return self.spans < other.spans

    def __str__(self):
        return ";".join([str(x) for x in self.spans])

    def offset_variant(self, word_offset: int, sent_id: int) -> "BetterSpanSet":
        # noinspection PyArgumentList
        return self.__class__(
            [span.offset_variant(word_offset, sent_id) for span in self.spans]
        )

    def get_entities(self, entities: List['BetterEntity']) -> List['BetterEntity']:
        entity_counts = defaultdict(int)
        entity_mapping = {}
        for m_id in self.get_mention_ids():
            for ent in entities:
                if m_id in ent.mentions:
                    entity_counts[ent.entity_id] += 1
                    entity_mapping[ent.entity_id] = ent
        # Sort in order of frequency
        results = []
        for val, key in sorted([(val, key) for key, val in entity_counts.items()], reverse=True):
            results.append(entity_mapping[key])
        return results

    def get_mention_ids(self) -> AbstractSet[str]:
        mention_ids = set()
        for s in self.spans:
            if not s.grounded_span or s.grounded_span.mention_id is None:
                continue
            mention_ids.add(s.grounded_span.mention_id)
        return mention_ids

    def get_entity_id(
        self,
        entities: List["BetterEntity"],
        mentions: Dict[str, "BetterMention"],
        id_counter: Optional[int] = None,
    ):
        if id_counter is not None:
            return f"e-{id_counter}"

        for entity in entities:
            res = entity.spanset_in_entity(self, mentions)
            if res:
                return res

        print(
            f"WARNING: Could not find requested entity id for BetterSpanSet "
            f"{self.__str__()} - Most likely one of the spans is included in "
            f"two different span sets in the annotation"
        )
        return "e-???????"


@attrs(frozen=True)
class BetterEventArgument:
    # These are ONLY used for granular arguments
    # Otherwise circular references ensure
    role: str = attrib(validator=validate_granular_arg_role)
    basic_event: 'BetterEvent' = attrib()
    irrealis: Optional[str] = attrib(validator=validate_irrealis)
    time_attachments: Optional[List[BetterSpanSet]] = attrib()
    score: Optional[float] = attrib()

    def __str__(self):
        anchors = [x.text for x in self.basic_event.anchors]
        return "{}: {}".format(self.role, " * ".join(anchors))

    @classmethod
    def from_json(cls, doc_text: DocumentText, json_dict: Dict, basic_event_map: Mapping[str, 'BetterEvent']):
        time_attachments = json_dict.get("time-attachments", None)
        if time_attachments is not None:
            time_attachments = [BetterSpanSet.from_json(doc_text, spanset)
                                for spanset in time_attachments]

        event_id = json_dict["event_id"]
        event = basic_event_map.get(event_id, None)
        if not event:
            warning = f"VERY IMPORTANT WARNING: No basic event found for {event_id}"
            sys.stderr.write(warning + "\n")
            print("REPEAT", warning)

        # noinspection PyArgumentList
        return cls(
            role=json_dict["role"],
            basic_event=event,
            irrealis=json_dict.get("irrealis", None),
            time_attachments=time_attachments,
            score=json_dict.get("score", None)
        )

    def to_dict(self) -> Mapping:
        output = {
            "role": self.role,
            "event_id": self.basic_event.event_id,
            "irrealis": self.irrealis
        }
        if self.time_attachments:
            output['time-attachments'] = [
                span.to_dict() for span in self.time_attachments
            ]
        if self.score is not None:
            output['score'] = self.score
        return output

    def with_new_spans(
            self,
            new_anchor_spans: List[ScoredBetterSpan],
            new_args: List['BetterArgument'],
            new_time_attachments: List[BetterSpanSet]
    ):
        return BetterEventArgument(
            self.role,
            BetterEvent(
                self.basic_event.event_id,
                self.basic_event.event_type,
                self.basic_event.properties,
                BetterSpanSet(new_anchor_spans),
                new_args,
                [],  # event_arguments
                self.basic_event.state_of_affairs

            ),
            self.irrealis,
            new_time_attachments,
            self.score
        )

    def with_new_event(self, new_event: 'BetterEvent'):
        return BetterEventArgument(
            self.role,
            new_event,
            self.irrealis,
            self.time_attachments,
            self.score
        )


@attrs(frozen=True, cmp=False)
class BetterArgument:
    role: str = attrib(validator=validate_argument_role)
    span_set: BetterSpanSet = attrib()
    irrealis: Optional[str] = attrib(validator=validate_irrealis)
    time_attachments: Optional[List[BetterSpanSet]] = attrib()
    mitre_coref_id: Optional[str] = attrib()

    @classmethod
    def from_json(cls, doc_text: DocumentText, json_dict: Dict):
        time_attachments = json_dict.get("time-attachments", None)
        if time_attachments is not None:
            time_attachments = [BetterSpanSet.from_json(doc_text, spanset)
                                for spanset in time_attachments]

        # noinspection PyArgumentList
        return cls(
            role=json_dict["role"],
            span_set=BetterSpanSet.from_json(doc_text, json_dict["span_set"]),
            irrealis=json_dict.get("irrealis", None),
            time_attachments=time_attachments,
            mitre_coref_id=json_dict.get("mitre_coref_id", None)
        )

    def to_dict(self) -> Mapping:
        output = {
            "role": self.role,
            "span_set": self.span_set.to_dict(),
            "irrealis": self.irrealis,
            "mitre_coref_id": self.mitre_coref_id
        }
        if self.time_attachments:
            output['time-attachments'] = [
                span.to_dict() for span in self.time_attachments
            ]
        return output

    def __str__(self):
        return "{}: {}".format(self.role, self.span_set)

    def __eq__(self, other: "BetterArgument"):
        return self.role == other.role and self.span_set == other.span_set \
          and self.irrealis == other.irrealis and \
          self.time_attachments == other.time_attachments

    def __lt__(self, other: "BetterArgument"):
        if self.role == other.role:
            return self.span_set < other.span_set
        return self.role < other.role

    def __hash__(self):
        return hash(self.span_set) ^ hash(self.role)

    def with_new_spans(self, spans: List[ScoredBetterSpan]) -> "BetterArgument":
        return BetterArgument(
            role=self.role,
            span_set=BetterSpanSet(spans),
            irrealis=self.irrealis,
            time_attachments=self.time_attachments,
            mitre_coref_id=self.mitre_coref_id
        )

    def with_new_mitre_coref_id(self, mitre_coref_id: str) -> "BetterArgument":
        return BetterArgument(
            role=self.role,
            span_set=self.span_set,
            irrealis=self.irrealis,
            time_attachments=self.time_attachments,
            mitre_coref_id=mitre_coref_id
        )

    def offset_variant(self, word_offset: int, sent_id: int) -> "BetterArgument":
        # noinspection PyArgumentList
        return self.__class__(
            role=self.role,
            span_set=self.span_set.offset_variant(word_offset, sent_id),
            irrealis=self.irrealis,
            time_attachments=self.time_attachments,
            mitre_coref_id=self.mitre_coref_id
        )

    def with_irrealis(self, irrealis: str) -> "BetterArgument":
        return BetterArgument(
            role=self.role,
            span_set=self.span_set,
            irrealis=irrealis,
            time_attachments=self.time_attachments,
            mitre_coref_id=self.mitre_coref_id
        )


@attrs(frozen=True, hash=False, cmp=False)
class BetterEvent:

    event_id: str = attrib()
    event_type: str = attrib(validator=validate_event_type)
    properties: Dict[str, str] = attrib()
    anchors: BetterSpanSet = attrib()
    arguments: List[BetterArgument] = attrib()
    # These are ONLY used for granular events pointing to basic events;
    # otherwise circular references ensure
    event_arguments: List[BetterEventArgument] = attrib()
    state_of_affairs: Optional[bool] = attrib()

    @classmethod
    def from_json(cls, doc_text: DocumentText, json_dict: Dict, basic_event_map: Mapping[str, 'BetterEvent']):
        # Note: basic_event_map can and should be null when constructing basic events
        #       basic_event_map should be populated when constructing granular events or sub-events

        # noinspection PyArgumentList
        return cls(
            str(json_dict["event_id"]),
            json_dict["event_type"],
            json_dict["properties"],
            BetterSpanSet.from_json(doc_text, json_dict["anchors"]),
            [BetterArgument.from_json(doc_text, x) for x in json_dict["arguments"]],
            [BetterEventArgument.from_json(doc_text, x, basic_event_map)
             for x in json_dict.get("event_arguments", [])],
            json_dict.get("state-of-affairs", None),
        )

    def to_dict(self) -> Mapping:
        result = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "properties": self.properties,
            "anchors": self.anchors.to_dict(),
            "arguments": [x.to_dict() for x in sorted(self.arguments)],
            "event_arguments": [x.to_dict() for x in self.event_arguments],
        }
        if self.state_of_affairs is not None:
            result["state-of-affairs"] = self.state_of_affairs
        return result

    def __str__(self):
        text = "{}/{}: anchors={}".format(
            self.event_id,
            self.event_type,
            self.anchors)
        if self.arguments:
            text += "; arguments=".format(";".join([str(x) for x in self.arguments]))
        if self.event_arguments:
            text += "; event_arguments=".format(";".join([str(x) for x in self.event_arguments]))
        return text

    def __eq__(self, other: "BetterEvent"):
        props = True
        props = props and \
            self.properties.get(HELPFUL_HARMFUL, None) == other.properties.get(HELPFUL_HARMFUL, None)
        props = props and \
            self.properties.get(MATERIAL_VERBAL, None) == other.properties.get(MATERIAL_VERBAL, None)
        return props and self.event_type == other.event_type and \
            self.anchors == other.anchors and \
            self.arguments == other.arguments and \
            self.event_arguments == other.event_arguments

    def __lt__(self, other: "BetterEvent"):
        return self.event_id < other.event_id

    def __hash__(self):
        val = 1
        for arg in self.arguments:
            val ^= hash(arg)
        for ev_arg in self.event_arguments:
            val ^= hash(ev_arg)
        if HELPFUL_HARMFUL in self.properties:
            val ^= hash(self.properties[HELPFUL_HARMFUL])
        if MATERIAL_VERBAL in self.properties:
            val ^= hash(self.properties[MATERIAL_VERBAL])
        return val ^ hash(self.event_type) ^ hash(self.anchors)

    def to_html(self):
        anchor_heads = [a.span.grounded_span.head_span for a in self.anchors]
        anchor_head_texts = " * ".join([ah.text for ah in anchor_heads])
        result = f"{self.event_id}/{self.event_type}: {anchor_head_texts}"
        if self.arguments:
            result += "<ul>"
            args_to_print = []
            for arg in self.arguments:
                arg_heads = [a.span.grounded_span.head_span for a in arg.span_set]
                arg_head_texts = " * ".join([ah.text for ah in arg_heads])
                args_to_print.append(f"<li><i>{arg.role}</i>: {arg_head_texts}</li>")
            result += "".join(sorted(args_to_print))
            result += "</ul>"
        return result

    def with_new_anchors(self, anchors: BetterSpanSet) -> 'BetterEvent':
        return BetterEvent(
            self.event_id,
            self.event_type,
            self.properties,
            anchors,
            self.arguments,
            self.event_arguments,
            self.state_of_affairs
        )

    def with_new_event_type(self, event_type: str) -> 'BetterEvent':
        return BetterEvent(
            self.event_id,
            event_type,
            self.properties,
            self.anchors,
            self.arguments,
            self.event_arguments,
            self.state_of_affairs
        )

    def with_new_properties(self, properties: Dict[str, str]) -> 'BetterEvent':
        return BetterEvent(
            self.event_id,
            self.event_type,
            properties,
            self.anchors,
            self.arguments,
            self.event_arguments,
            self.state_of_affairs
        )

    def with_new_arguments(self, arguments: List[BetterArgument]) -> 'BetterEvent':
        return BetterEvent(
            self.event_id,
            self.event_type,
            self.properties,
            self.anchors,
            arguments,
            self.event_arguments,
            self.state_of_affairs
        )

    def with_new_event_arguments(self, event_arguments: List[BetterEventArgument]) -> 'BetterEvent':
        return BetterEvent(
            self.event_id,
            self.event_type,
            self.properties,
            self.anchors,
            self.arguments,
            event_arguments,
            self.state_of_affairs
        )

    def give_span(self) -> Tuple[int, int]:
        starts = []
        ends = []
        for span in self.anchors.spans:
            if span.grounded_span is not None:
                starts.append(span.grounded_span.full_span.start_token)
                ends.append(span.grounded_span.full_span.end_token)

        for argument in self.arguments:
            for span in argument.span_set.spans:
                if span.grounded_span is not None:
                    starts.append(span.grounded_span.full_span.start_token)
                    ends.append(span.grounded_span.full_span.end_token)

        return min(starts), max(ends)

    def offset_variant(self, word_offset: int, sent_id: int) -> "BetterEvent":
        # noinspection PyArgumentList
        return self.__class__(
            self.event_id,
            self.event_type,
            self.properties.copy(),
            self.anchors.offset_variant(word_offset, sent_id),
            [
                argument.offset_variant(word_offset, sent_id)
                for argument in self.arguments
            ],
            self.event_arguments,
            self.state_of_affairs
        )


def get_mitre_mention_type(mention_type: str) -> str:
    if mention_type in VALID_MENTION_TYPES:
        return mention_type
    if mention_type in ['desc', 'descriptor']:
        return NOMINAL
    return mention_type


@attrs(frozen=True, hash=False)
class BetterMention:
    mention_id: str = attrib()
    mention_type: str = attrib(converter=get_mitre_mention_type, validator=validate_mention_type)
    grounded_span: GroundedSpan = attrib()
    entity_type: Optional[str] = attrib()
    properties: Mapping[str, Union[str, int, bool]] = attrib()

    @classmethod
    def from_json(cls, doc_text: DocumentText, json_dict: Dict):
        # noinspection PyArgumentList
        return cls(
            json_dict["mention_id"],
            json_dict["mention_type"],
            GroundedSpan.from_json(doc_text, json_dict["grounded_span"]),
            json_dict.get("entity_type", None),
            json_dict.get("properties", {}),
        )

    def is_valid_mitre_mention(self):
        return self.mention_type in [NAME, NOMINAL, PRONOUN, EVENT_ANCHOR, TEMPLATE_ANCHOR]

    def with_new_entity_type(self, entity_type: Optional[str]) -> 'BetterMention':
        # NOTE: Does NOT change mention ID!
        return BetterMention(self.mention_id, self.mention_type, self.grounded_span,
                             entity_type, self.properties)

    def with_new_mention_and_entity_type(self, mention_type: str, entity_type: Optional[str]) -> 'BetterMention':
        # NOTE: Does NOT change mention ID!
        return BetterMention(self.mention_id, mention_type, self.grounded_span,
                             entity_type, self.properties)

    def with_new_grounded_span(self, grounded_span: GroundedSpan) -> 'BetterMention':
        # NOTE: Does NOT change mention ID!
        return BetterMention(self.mention_id, self.mention_type, grounded_span,
                             self.entity_type, self.properties)

    @property
    def head_span(self):
        return self.grounded_span.head_span

    @property
    def full_span(self):
        return self.grounded_span.full_span

    @property
    def sent_id(self):
        return self.grounded_span.sent_id

    def to_dict(self) -> Mapping:
        return {
            "mention_id": self.mention_id,
            "mention_type": self.mention_type,
            "grounded_span": self.grounded_span.to_dict(),
            "entity_type": self.entity_type,
            "properties": self.properties,
        }

    def to_mitre(self) -> Mapping:
        mitre_dict = {
            "end": self.grounded_span.full_span.end_char + 1,
            "hend": self.grounded_span.head_span.end_char + 1,
            "hstart": self.grounded_span.head_span.start_char,
            "hstring": self.head_span.text,
            "start": self.full_span.start_char,
            "string": self.full_span.text,
        }
        if self.mention_type:
            if self.mention_type == 'LIST':
                print("WARNING: Deal with lists!")
            mitre_dict["synclass"] = self.mention_type
        return mitre_dict

    def __str__(self):
        return "{}/{}/{}: {}".format(
            self.mention_id, self.mention_type,
            self.entity_type, self.grounded_span
        )

    def __hash__(self):
        return hash(self.__str__())

    def offset_variant(
            self, word_offset: int, sent_id: int = None
    ) -> "BetterMention":
        if not self.grounded_span:
            return self
        # noinspection PyArgumentList
        return self.__class__(
            self.mention_id,
            self.mention_type,
            self.grounded_span.offset_variant(word_offset, sent_id),
            self.entity_type,
            self.properties
        )


@attrs(frozen=True)
class BetterEntity:
    entity_type: Optional[str] = attrib()
    mentions: List[str] = attrib()
    # in the event that the entity is taken from annotated data,
    # we store its spanset id here
    entity_id: Optional[str] = attrib(default=None)
    # these are the 'includes-relations' from the mitre input
    subset_coreferants: List[str] = attrib(default=list())

    @classmethod
    def from_json(cls, json_dict: Dict):
        # noinspection PyArgumentList
        return cls(
            json_dict["entity_type"],
            json_dict["mentions"],
            json_dict.get("entity_id", None),
            json_dict.get("subset_coreferants", list()),
        )

    def to_dict(self) -> Mapping:
        return asdict(self)

    def to_mitre(self, mentions_by_id: Dict[str, BetterMention]) -> Mapping:
        mention_spans = []
        for mention_id in self.mentions:
            if mention_id in mentions_by_id:
                mention = mentions_by_id[mention_id]
                if not mention.grounded_span:
                    continue
                if mention.is_valid_mitre_mention():
                    mention_spans.append(mention.to_mitre())
        return {
            "spans": mention_spans, "ssid": self.entity_id
        }

    def __str__(self):
        return "{}: {}".format(self.entity_type, " ".join(self.mentions))

    def spanset_in_entity(
        self, spanset: BetterSpanSet, mentions_by_id: Dict[str, BetterMention]
    ) -> Optional[str]:
        """ returns None if spanset contents do not match this entity"""
        mentions_in_spanset = []
        for span in spanset:
            mention_found = False
            for mention_id, mention in mentions_by_id.items():
                # More permissive matching (via mention id) to allow for
                # expanded spans to match.
                # TODO: find better way to make input spanset and
                #       mentions_by_id consistent after span expansion
                if (span.grounded_span.mention_id ==
                        mention.grounded_span.mention_id
                        or span.grounded_span == mention.grounded_span):
                    mention_found = True
                    mentions_in_spanset.append(mention_id)
                    break
            if not mention_found:
                return None
        for mention_id in mentions_in_spanset:
            if mention_id not in self.mentions:
                return None
        return self.entity_id


@attrs(frozen=True)
class BetterToken:
    text: str = attrib()
    # inclusive span!
    doc_character_span: Tuple[int, int] = attrib()

    def __str__(self):
        return self.text

    def __len__(self):
        return len(self.text)

    def to_dict(self):
        return {
            "text": self.text,
            "start": self.doc_character_span[0],
            "end": self.doc_character_span[1],
        }

    @classmethod
    def from_json(cls, json_dict: Dict):
        return cls(json_dict["text"], (json_dict["start"], json_dict["end"]))


class BetterSentence:
    def __init__(
        self,
        *,
        doc_text: DocumentText,
        sent_id: int,
        task_type: str,
        tokens: List[BetterToken],
        pos_tags: List[str],
        dependencies: Iterable[BetterDependency],
        mentions: List[BetterMention],
        propositions: Iterable[Proposition],
        original_document_character_span: SimpleGroundedSpan,
        sentence_type: str,
        abstract_events: List[BetterEvent],
        basic_events: List[BetterEvent],
        relations: List[BetterEvent],
        original_sent_id: Optional[int],
        sent_split_index: Optional[int],
        translation: str = "",
        span_projections: Mapping[SimpleGroundedSpan, str] = {}
    ):
        self.doc_text: DocumentText = doc_text
        self.sent_id = int(sent_id)
        self._tokens: List[BetterToken] = []
        self.task_type: str = task_type
        self.original_sent_id: int = original_sent_id
        self.sent_split_index: int = sent_split_index
        self.original_document_character_span: SimpleGroundedSpan = original_document_character_span
        self.sentence_type: str = sentence_type

        self.pos_tags: List[str] = pos_tags
        self.dependencies: Iterable[BetterDependency] = dependencies
        self.mentions: List[BetterMention] = mentions
        self.mentions_by_id: Dict[str, BetterMention] = {m.mention_id: m for m in mentions}
        self.propositions: Iterable[Proposition] = propositions

        self.abstract_events: List[BetterEvent] = abstract_events
        self.basic_events: List[BetterEvent] = basic_events
        self.relations: List[BetterEvent] = relations

        self.translation = translation
        self.span_projections = span_projections

        # ground mentions and events to
        self.set_tokens(tokens)

        # validation:
        if self.abstract_events and self.basic_events:
            raise ValueError(
                "BetterSentence can't (reasonably) represent abstract event "
                "and basic event annotations at the same time"
            )

        if self.tokens and self.pos_tags and len(self.tokens) != \
                len(self.pos_tags):
            raise ValueError("Mismatched number of tokens and pos_tags")

    @property
    def tokens(self):
        return self._tokens

    def set_tokens_for_mention(self, mention: BetterMention) -> BetterMention:
        # Import has to be here to avoid circular imports
        from better_events.better_mapper import BetterMapper
        if not mention.grounded_span:
            return mention
        new_full_span = None
        if mention.grounded_span.full_span.text:
            new_full_span = BetterMapper.ground_to_tokens_sgs(
                self, mention.grounded_span.full_span,
            )
        new_head_span = None
        if mention.grounded_span.head_span.text:
            new_head_span = BetterMapper.ground_to_tokens_sgs(
                self, mention.grounded_span.head_span,
            )
        return BetterMention(
            mention.mention_id,
            mention.mention_type,
            GroundedSpan(
                mention.grounded_span.sent_id,
                new_full_span,
                new_head_span,
                mention.mention_id
            ),
            mention.entity_type,
            mention.properties
        )

    def set_tokens(self, tokens: List[BetterToken]):
        self._tokens = tokens
        if not tokens:
            return

        from better_events.better_mapper import BetterMapper

        # also reconcile existing mentions and events if applicable
        updated_mentions = [self.set_tokens_for_mention(m)
                            for m in self.mentions]

        self.mentions = updated_mentions
        self.mentions_by_id = {
            ment.mention_id: ment for ment in updated_mentions
        }
        self.abstract_events = [BetterMapper.ground_event_to_tokens(None, self, e)
                                for e in self.abstract_events]
        self.basic_events = [BetterMapper.ground_event_to_tokens(None, self, e)
                             for e in self.basic_events]
        self.relations = [BetterMapper.ground_event_to_tokens(None, self, e)
                          for e in self.relations]

    def get_offset_in_sentence(self, token: BetterToken) -> Tuple[int, int]:
        assert token in self.tokens
        start_doc_index = self.original_document_character_span.start_char
        return (
            token.doc_character_span[0] - start_doc_index,
            token.doc_character_span[1] - start_doc_index,
        )

    def get_token_index(self, token: BetterToken) -> int:
        if token not in self.tokens:
            raise ValueError(
                "Attempting to look up index for token not in BetterSentence"
            )
        for i, tok in enumerate(self.tokens):
            if tok == token:
                return i

    @classmethod
    def derive_subsentence(
        cls,
        source_sent: "BetterSentence",
        doc_text: DocumentText,
        start_word: int,
        end_word: int,
        new_sent_id: int,
        split_id: int,
        abstract_events: List[BetterEvent],
        basic_events: List[BetterEvent],
        relations: List[BetterEvent],
        mentions: List[BetterMention],
    ):
        if start_word == 0:
            new_text_begin = source_sent.original_document_character_span.start_char
        else:
            new_text_begin = source_sent.tokens[start_word].doc_character_span[
                0]

        if end_word == -1 + len(source_sent.tokens):
            new_text_end = source_sent.original_document_character_span.end_char
        else:
            # Include trailing whitespace by going up to the beginning of the next token.
            new_text_end = source_sent.tokens[end_word + 1].doc_character_span[
                               0] - 1

        return cls(
            doc_text=source_sent.doc_text,
            sent_id=new_sent_id,
            tokens=source_sent.tokens[start_word: 1 + end_word],
            sentence_type=source_sent.sentence_type,
            pos_tags=[],  # We don't try to restore pos tags after splitting (decimal point interpreted as period)
            dependencies=[],  # We don't try to restore dependencies after splitting sentences
            mentions=mentions,
            propositions=[],  # We don't try to restore propositions after splitting sentences
            abstract_events=abstract_events,
            basic_events=basic_events,
            relations=relations,
            original_document_character_span=SimpleGroundedSpan(
                doc_text,
                start_char=new_text_begin,
                end_char=new_text_end,
                start_token=start_word,
                end_token=end_word,
            ),
            original_sent_id=source_sent.sent_id,
            sent_split_index=split_id,
            task_type=source_sent.task_type,
            translation="",  # We don't try to preserve translations/projections after splitting
            span_projections={},  # We don't try to preserve translations/projections after splitting
        )

    @classmethod
    def recombine_sentences(cls, split_sents: List["BetterSentence"]):
        for sent in split_sents:
            assert (
                sent.original_sent_id is not None
                and sent.original_sent_id == split_sents[0].original_sent_id
            )
            assert sent.sent_split_index is not None

        split_sents = sorted(split_sents, key=lambda x: x.sent_split_index)
        new_tokens = []
        new_sent = cls(
            doc_text=split_sents[0].doc_text,
            sent_id=split_sents[0].original_sent_id,
            task_type=split_sents[0].task_type,
            tokens=[],
            sentence_type=split_sents[0].sentence_type,
            pos_tags=[],
            dependencies=[],
            propositions=[],
            mentions=[],
            original_document_character_span=SimpleGroundedSpan(
                split_sents[0].doc_text,
                start_char=split_sents[0].original_document_character_span.start_char,
                end_char=split_sents[-1].original_document_character_span.end_char,
                start_token=None,
                end_token=None,
            ),
            original_sent_id=None,
            sent_split_index=None,
            abstract_events=[],
            basic_events=[],
            relations=[],
            translation="",
            span_projections={}
        )

        for sent in split_sents:
            word_offset = len(new_sent.tokens)

            new_tokens += sent.tokens[:]

            new_sent.pos_tags += sent.pos_tags[:]
            new_sent.dependencies += [
                d.offset_variant(word_offset) for d in sent.dependencies
            ]

            # Note: We do not currently try to recombine propositions

            new_sent.mentions += [
                m.offset_variant(word_offset, sent.original_sent_id)
                for m in sent.mentions
            ]
            new_sent.abstract_events += [
                ev.offset_variant(word_offset, sent.original_sent_id)
                for ev in sent.abstract_events
            ]
            new_sent.basic_events += [
                ev.offset_variant(word_offset, sent.original_sent_id)
                for ev in sent.basic_events
            ]
            new_sent.relations += [
                ev.offset_variant(word_offset, sent.original_sent_id)
                for ev in sent.relations
            ]

        new_sent.set_tokens(new_tokens)
        return new_sent

    @classmethod
    def from_json(cls, doc_text: DocumentText, json_dict: Dict):
        abstract_events = [BetterEvent.from_json(doc_text, ev, {})
                           for ev in json_dict.get("abstract_events", [])]
        basic_events = [BetterEvent.from_json(doc_text, ev, {})
                        for ev in json_dict.get("basic_events", [])
                        if ev["event_type"] not in EXCLUDED_BASIC_EVENT_TYPES]
        relations = [BetterEvent.from_json(doc_text, ev, {})
                     for ev in json_dict.get("relations", [])]
        dependencies = [BetterDependency.from_json(x) for x in json_dict.get("dependencies", [])]
        mentions = [BetterMention.from_json(doc_text, x) for x in json_dict.get("mentions", [])]
        propositions = [Proposition.from_json(x) for x in json_dict.get("propositions", [])]
        tokens = [BetterToken.from_json(x) for x in json_dict.get("tokens", [])]

        translation = json_dict.get("translation", "")

        span_projections = {
            SimpleGroundedSpan.from_json(doc_text, d['span']): d['projected_text']
            for d in json_dict.get("span_projections", [])
        }

        # Build from proposition IDs
        proposition_mapping = {p.prop_id: p for p in propositions}
        for p in propositions:
            p.arguments = [arg.with_instantiated_prop(proposition_mapping) for arg in p.arguments]

        return cls(
            doc_text=doc_text,
            sent_id=json_dict["sent_id"],
            task_type=json_dict["task_type"],
            tokens=tokens,
            pos_tags=json_dict.get("pos_tags", []),
            dependencies=dependencies,
            mentions=mentions,
            propositions=propositions,
            original_document_character_span=SimpleGroundedSpan.from_json(
                doc_text,
                json_dict["original_document_character_span"]
            ),
            sentence_type=json_dict["sentence_type"],
            abstract_events=abstract_events,
            basic_events=basic_events,
            relations=relations,
            original_sent_id=json_dict.get("original_sent_id", None),
            sent_split_index=json_dict.get("sent_split_index", None),
            translation=translation,
            span_projections=span_projections
        )
        # noinspection PyArgumentList

    def to_dict(self) -> Mapping:
        return {
            "sent_id": self.sent_id,
            "task_type": self.task_type,
            "text": self.original_document_character_span.text,
            "tokens": [x.to_dict() for x in self.tokens],
            "pos_tags": self.pos_tags,
            "dependencies": [x.to_dict() for x in self.dependencies],
            "mentions": [x.to_dict() for x in self.mentions],
            "propositions": [x.to_dict() for x in self.propositions],
            "abstract_events": [x.to_dict() for x in self.abstract_events],
            "basic_events": [x.to_dict() for x in self.basic_events],
            "relations": [x.to_dict() for x in self.relations],
            "original_sent_id": self.original_sent_id,
            "sent_split_index": self.sent_split_index,
            "original_document_character_span":
                self.original_document_character_span.to_dict(),
            "sentence_type": self.sentence_type,
            "translation": self.translation,
            "span_projections": [
                {'span': key.to_dict(), 'projected_text': value}
                for key, value in self.span_projections.items()
            ]
        }

    def to_mitre_abstract_events(self, entities: List[BetterEntity]) -> Mapping:
        events = {}
        spans_by_id = {}
        span_id_counter = 1
        for e in self.abstract_events:
            event_dict = {
                "agents": [],
                "anchors": "",
                "eventid": e.event_id,
                "patients": [],
            }
            if HELPFUL_HARMFUL in e.properties:
                event_dict["helpful-harmful"] = e.properties[HELPFUL_HARMFUL]
            if MATERIAL_VERBAL in e.properties:
                event_dict["material-verbal"] = e.properties[MATERIAL_VERBAL]
            anchor_ss_id = e.anchors.get_entity_id(
                entities, self.mentions_by_id, span_id_counter
            ).replace("e", "ss")
            if anchor_ss_id not in spans_by_id:
                spans_by_id[anchor_ss_id] = e.anchors.to_mitre()
                span_id_counter += 1
            event_dict["anchors"] = anchor_ss_id

            for arg in e.arguments:
                arg_span_id = arg.span_set.get_entity_id(
                    entities, self.mentions_by_id, span_id_counter
                ).replace("e", "ss")
                if arg_span_id not in spans_by_id:
                    spans_by_id[arg_span_id] = arg.span_set.to_mitre()
                    span_id_counter += 1
                if arg.role == AGENT:
                    event_dict["agents"].append(arg_span_id)
                elif arg.role == PATIENT:
                    event_dict["patients"].append(arg_span_id)

            events[e.event_id] = event_dict

        for span_set_id, span_set in spans_by_id.items():
            new_span_set = {"spans": span_set, "ssid": span_set_id}
            spans_by_id[span_set_id] = new_span_set

        return {"abstract-events": {"events": events, "span-sets": spans_by_id}}

    def to_mitre_basic_events(self, doc: 'BetterDocument'):
        output_events = {}
        for e in self.basic_events:

            # MITRE scorer doesn't like integers, make sure these IDs all serialize as str

            agents = set()
            patients = set()
            money_args = set()
            ref_events = set()
            for arg in e.arguments:
                if arg.role == REF_EVENT:
                    ref_events.add(str(arg.mitre_coref_id))
                elif arg.role == AGENT:
                    agents.add(str(arg.mitre_coref_id))
                elif arg.role == PATIENT:
                    patients.add(str(arg.mitre_coref_id))
                elif arg.role == MONEY:
                    money_args.add(str(arg.mitre_coref_id))
                else:
                    if arg.role not in VALID_GRANULAR_ARG_ROLES:
                        print("Unrecognized role for argument of basic event:", arg.role)

            anchor_entities = e.anchors.get_entities(doc.mitre_span_sets)
            if not anchor_entities:
                print(f"WARNING: No anchor entity for event {e.event_id} in {doc.doc_id}! This "
                      f"should have been taken care of in BetterCoref!")
            if len(anchor_entities) > 1:
                print("WARNING: More than one anchor entity for event!")

            if not anchor_entities:
                anchor_entity_id = None
            else:
                anchor_entity_id = sorted(anchor_entities)[0].entity_id

            event_dict = {
                "agents": list(sorted(agents)),
                "anchors": str(anchor_entity_id),
                "eventid": str(e.event_id),
                "event-type": e.event_type,
                "patients": list(sorted(patients)),
                "money": list(sorted(money_args)),
                "ref-events": list(sorted(ref_events)),
                "state-of-affairs": e.state_of_affairs,
            }
            output_events[e.event_id] = event_dict

        return output_events

    def __str__(self):
        return self.original_document_character_span.text

    def give_dependency_spans(self) -> Iterable[Iterable[int]]:
        return BetterDependency.spans_for_set(self.dependencies)

    def give_mention_spans(self) -> Iterable[Iterable[int]]:
        return [
            mention.grounded_span.give_max_span() for mention in self.mentions
        ]

    def tokens_in_span(self, start_ch: int, end_ch: int) -> List[BetterToken]:
        output = []
        for token in self.tokens:
            if (
                token.doc_character_span[0] >= start_ch
                and token.doc_character_span[1] <= end_ch
            ):
                output.append(token)
        return output

    def overlapping_tokens(self, start_ch: int, end_ch: int) -> List[BetterToken]:
        output = []
        for token in self.tokens:
            if (
                    start_ch <= token.doc_character_span[0] <= end_ch or
                    start_ch <= token.doc_character_span[1] <= end_ch
            ):
                output.append(token)
        return output


class BetterDocument:

    def __init__(
        self,
        doc_id: str,
        mitre_doc_id: str,
        task_type: str,
        sentences: List[BetterSentence],
        entities: List[BetterEntity],
        mitre_span_sets: List[BetterEntity],
        doc_text: DocumentText,
        granular_events: Optional[List[BetterEvent]],
        structural_elements: Optional[List[StructuralElement]],
        lang: Optional[str],
        properties: Dict[str, Any],
    ) -> None:
        self.doc_id: str = doc_id
        self.mitre_doc_id: str = mitre_doc_id
        self.task_type: str = task_type
        if self.task_type not in VALID_EVENT_TASKS:
            raise ValueError(f"Document task type {task_type} must be in {VALID_EVENT_TASKS}")
        self.doc_text: DocumentText = doc_text
        self.sentences: List[BetterSentence] = sentences
        self.sentences_by_id: Dict[int, BetterSentence] = {s.sent_id: s for s in self.sentences}
        self.entities: List[BetterEntity] = entities
        self.mitre_span_sets: List[BetterEntity] = mitre_span_sets
        self.structural_elements: List[StructuralElement] = structural_elements
        self.lang = lang # prefer ISO 639-1 codes
        self.properties = properties

        if self.structural_elements is None:
            self.structural_elements = []

        if granular_events is None:
            granular_events = []
        self.granular_events: List[BetterEvent] = granular_events

        sentence_spans = [sent.original_document_character_span
                          for sent in sentences]
        # test mutual exclusivity of spans
        for span1, span2 in combinations(sentence_spans, 2):
            if span1.contains(span2):
                raise ValueError(
                    f"Document has overlapping sentence spans {span1} "
                    f"and {span2}"
                    f"\nFull doc text: {doc_text}"
                )

    @classmethod
    def from_json(cls, json_dict: Dict):

        entities = [BetterEntity.from_json(x) for x in json_dict.get("entities", [])]
        mitre_span_sets = [BetterEntity.from_json(x)
                           for x in json_dict.get("mitre_span_sets", [])]

        # Backwards compatibility
        task_type = json_dict.get("task_type", None)
        if not task_type:
            for s in json_dict["sentences"]:
                if s.get("abstract_events", []):
                    task_type = ABSTRACT_TASK
                    break
                if s.get("basic_events", []):
                    task_type = BASIC_TASK
                    break
            for s in json_dict["sentences"]:
                s["task_type"] = task_type
        if not task_type:
            raise ValueError("No task type specified and none can be inferred")

        doc_text = DocumentText(json_dict.get("doc_text", ""))
        sentences = [BetterSentence.from_json(doc_text, x) for x in json_dict["sentences"]]
        basic_event_map = {}
        for s in sentences:
            for e in s.basic_events:
                basic_event_map[e.event_id] = e

        granular_events = [
            BetterEvent.from_json(doc_text, elt, basic_event_map)
            for elt in json_dict.get("granular_events", [])
        ]

        # noinspection PyArgumentList
        return cls(
            json_dict["doc_id"],
            json_dict["mitre_doc_id"],
            task_type,
            sentences,
            entities,
            mitre_span_sets,
            doc_text,
            granular_events,
            [
                StructuralElement.from_json(elt)
                for elt in json_dict.get("structural_elements", [])
            ],
            json_dict.get("lang", None),
            json_dict.get("properties", {})
        )

    @classmethod
    def from_mitre_abstract_events(
            cls, annotated_text: AnnotatedText, task_type: str, lang: Optional[str]
    ):

        doc_text = DocumentText(annotated_text.segment_text)
        if annotated_text.lang:
            lang = annotated_text.lang
        if not lang:
            raise ValueError("If lang is not provided in .bp.json, it must be "
                             "specified at ingest time.")


        mitre_span_sets: List[BetterEntity] = []
        combined_sents_char_range_start = 0
        combined_sents_char_range_end = (
                combined_sents_char_range_start +
                len(annotated_text.segment_text) - 1
        )
        sentence = \
            BetterSentence(
                doc_text=doc_text,
                sent_id=int(annotated_text.sent_id),
                task_type=ABSTRACT_TASK,
                tokens=[],
                pos_tags=[],
                dependencies=[],
                mentions=[],
                propositions=[],
                original_document_character_span=SimpleGroundedSpan(
                    doc_text=doc_text,
                    start_char=combined_sents_char_range_start,
                    end_char=combined_sents_char_range_end,
                    start_token=None,
                    end_token=None,
                ),
                sentence_type=STRUCTURAL_SENTENCE,
                abstract_events=[],
                basic_events=[],
                relations=[],
                original_sent_id=None,
                sent_split_index=None
            )

        mentions_for_sents, entities, _ = BetterDocument._ingest_mentions(
            [sentence], doc_text, annotated_text, task_type
        )

        events = []
        for e in annotated_text.abstract_event_set.events:
            event_id = e.event_id
            event_type = ABSTRACT_EVENT_TYPE
            properties = {
                HELPFUL_HARMFUL: e.helpful_harmful,
                MATERIAL_VERBAL: e.material_verbal,
            }
            anchors = cls.convert_mitre_span_set(
                doc_text,
                e.anchors,
                annotated_text.abstract_event_set,
                [sentence],
                mentions_for_sents[sentence.sent_id],
                annotated_text.sent_id,
            )

            arguments = []
            for span_id in e.agents:
                spanset = cls.convert_mitre_span_set(
                    doc_text,
                    span_id,
                    annotated_text.abstract_event_set,
                    [sentence],
                    mentions_for_sents[sentence.sent_id],
                    annotated_text.sent_id,
                )
                arguments.append(BetterArgument(
                    role=AGENT, span_set=spanset,
                    irrealis=None,
                    time_attachments=None,
                    mitre_coref_id=None))
            for span_id in e.patients:
                spanset = cls.convert_mitre_span_set(
                    doc_text,
                    span_id,
                    annotated_text.abstract_event_set,
                    [sentence],
                    mentions_for_sents[sentence.sent_id],
                    annotated_text.sent_id,
                )
                arguments.append(BetterArgument(
                    role=PATIENT, span_set=spanset,
                    irrealis=None,
                    time_attachments=None,
                    mitre_coref_id=None,
                ))

            events.append(
                BetterEvent(event_id, event_type, properties, anchors, arguments, [], None)
            )

        sentence.mentions = mentions_for_sents[sentence.sent_id]
        sentence.mentions_by_id = \
            {m.mention_id: m for m in mentions_for_sents[sentence.sent_id]}
        sentence.abstract_events = events

        return cls(
            annotated_text.entry_id, annotated_text.doc_id, ABSTRACT_TASK, [sentence],
            entities, mitre_span_sets, doc_text, None, annotated_text.segment_sections,
            lang, properties={}
        )

    @classmethod
    def from_mitre_basic_events(
            cls, annotated_text: AnnotatedText, task_type: str, lang: Optional[str]
    ):

        quiet = False
        if annotated_text.lang:
            lang = annotated_text.lang
        if not lang:
            raise ValueError("If lang is not provided in .bp.json, it must be "
                             "specified at ingest time.")

        assert annotated_text.segment_sections, \
            "No segment_sections provided for basic / granular events data!"
        doc_text = DocumentText(annotated_text.segment_text)
        sentences = []

        (
            sent_texts,
            sent_types,
            sent_ranges,
        ) = BetterDocument.initial_sentence_segmentation(
            annotated_text.segment_text, annotated_text.segment_sections
        )

        mitre_span_sets: List[BetterEntity] = []

        for sent_id, (sent_text, sent_type, sent_range) in enumerate(
            zip(sent_texts, sent_types, sent_ranges)
        ):
            sentences.append(
                BetterSentence(
                    doc_text=doc_text,
                    sent_id=sent_id,
                    task_type=task_type,
                    tokens=[],
                    pos_tags=[],
                    dependencies=[],
                    mentions=[],
                    propositions=[],
                    original_document_character_span=SimpleGroundedSpan(
                        doc_text=doc_text,
                        start_char=sent_range[0],
                        end_char=sent_range[1],
                        start_token=None,
                        end_token=None,
                    ),
                    sentence_type=sent_type,
                    abstract_events=[],
                    basic_events=[],
                    relations=[],
                    original_sent_id=None,
                    sent_split_index=None
                )
            )

        mentions_for_sents, entities, mention_id_counter = BetterDocument._ingest_mentions(
            sentences, doc_text, annotated_text, task_type
        )
        all_mentions = []
        for sent in sentences:
            all_mentions += mentions_for_sents[sent.sent_id]
            sent.mentions = mentions_for_sents[sent.sent_id]
            sent.mentions_by_id = {
                ment.mention_id: ment for ment in sent.mentions
            }

        for sent in sentences:
            sent.basic_events = []
            # instead of making
            temp_basic_events = []
            for e in annotated_text.basic_event_set.events:
                event_id = e.event_id
                event_type = e.event_type
                properties = {}
                anchors = cls.convert_mitre_span_set_sentence_restricted(
                    doc_text,
                    e.anchors,
                    annotated_text.basic_event_set,
                    sent,
                    mentions_for_sents[sent.sent_id],
                )
                # if there is no anchor for the given span-set that belongs i
                # n this sentence, continue
                if not anchors.spans:
                    continue

                arguments = []
                for span_id in e.agents:
                    spanset = cls.convert_mitre_span_set_sentence_restricted(
                        doc_text,
                        span_id,
                        annotated_text.basic_event_set,
                        sent,
                        mentions_for_sents[sent.sent_id],
                    )
                    if not spanset.spans:
                        spanset, mention_id = cls.convert_mitre_span_set_sentence_restricted_fallback(
                            doc_text,
                            span_id,
                            annotated_text.basic_event_set,
                            sent,
                            mentions_for_sents[sent.sent_id],
                            all_mentions,
                            entities,
                            mention_id_counter=mention_id_counter,
                        )
                        if not spanset.spans and not quiet:
                            print("Argument for event", span_id, " ",
                                  [hs.full_span
                                   for hs in
                                   annotated_text.basic_event_set.get_headed_strings_for_span_id(
                                       span_id)],
                                  "not found in trigger sentence for ",
                                  event_id)
                            for hs in annotated_text.basic_event_set.get_headed_strings_for_span_id(
                                    span_id):
                                print(hs.head, hs.grounding.head_start,
                                      hs.grounding.head_end)
                            print(sent.original_document_character_span)
                            print()

                    arguments.append(BetterArgument(
                        role=AGENT,
                        span_set=spanset,
                        irrealis=None,
                        time_attachments=None,
                        mitre_coref_id=None,
                    ))
                for span_id in e.patients:
                    spanset = cls.convert_mitre_span_set_sentence_restricted(
                        doc_text,
                        span_id,
                        annotated_text.basic_event_set,
                        sent,
                        mentions_for_sents[sent.sent_id],
                    )
                    if not spanset.spans:
                        spanset, mention_id = cls.convert_mitre_span_set_sentence_restricted_fallback(
                            doc_text,
                            span_id,
                            annotated_text.basic_event_set,
                            sent,
                            mentions_for_sents[sent.sent_id],
                            all_mentions,
                            entities,
                            mention_id_counter=mention_id_counter,
                        )
                        if not spanset.spans and not quiet:
                            print("Argument for event", span_id, " ",
                                  [hs.full_span
                                   for hs in annotated_text.basic_event_set.get_headed_strings_for_span_id(span_id)],
                                  "not found in trigger sentence for ",
                                  event_id)
                            for hs in annotated_text.basic_event_set.get_headed_strings_for_span_id(
                                    span_id):
                                print(hs.grounding.head_start,
                                      hs.grounding.head_end)
                            print(sent.original_document_character_span)
                            print()
                    arguments.append(BetterArgument(
                        role=PATIENT,
                        span_set=spanset,
                        irrealis=None,
                        time_attachments=None,
                        mitre_coref_id=None,
                    ))
                for span_id in e.money:
                    spanset = cls.convert_mitre_span_set_sentence_restricted(
                        doc_text,
                        span_id,
                        annotated_text.basic_event_set,
                        sent,
                        mentions_for_sents[sent.sent_id]
                    )
                    if not spanset.spans:
                        spanset, mention_id = cls.convert_mitre_span_set_sentence_restricted_fallback(
                            doc_text,
                            span_id,
                            annotated_text.basic_event_set,
                            sent,
                            mentions_for_sents[sent.sent_id],
                            all_mentions,
                            entities,
                            mention_id_counter=mention_id_counter,
                        )
                        if not spanset.spans and not quiet:
                            print("Argument for event", span_id, " ",
                                  [hs.full_span
                                   for hs in
                                   annotated_text.basic_event_set.get_headed_strings_for_span_id(
                                       span_id)],
                                  "not found in trigger sentence for ",
                                  event_id)
                            for hs in annotated_text.basic_event_set.get_headed_strings_for_span_id(
                                    span_id):
                                print(hs.grounding.head_start,
                                      hs.grounding.head_end)
                            print(sent.original_document_character_span)
                            print()
                    arguments.append(BetterArgument(
                        role=MONEY,
                        span_set=spanset,
                        irrealis=None,
                        time_attachments=None,
                        mitre_coref_id=None,
                    ))
                temp_basic_events.append(
                    BetterEvent(
                        event_id,
                        event_type,
                        properties,
                        anchors,
                        arguments,
                        [],
                        e.state_of_affairs,
                    )
                )
            # process ref events now that all  basic events for this
            # sentence are accounted for
            for temp_event in temp_basic_events:
                e = None
                for basic_event in annotated_text.basic_event_set.events:
                    if basic_event.event_id == temp_event.event_id:
                        e = basic_event
                ref_args = []
                if e.ref_events:
                    for ev_id in e.ref_events:
                        # find the event corresponding to this id
                        for temp_event2 in temp_basic_events:
                            if temp_event2.event_id == ev_id and \
                                    temp_event2.event_type not in \
                                    EXCLUDED_BASIC_EVENT_TYPES:
                                ref_args.append(
                                    BetterArgument(
                                        role=REF_EVENT,
                                        span_set=temp_event2.anchors,
                                        irrealis=None,
                                        time_attachments=None,
                                        mitre_coref_id=None
                                    )
                                )
                if temp_event.event_type in EXCLUDED_BASIC_EVENT_TYPES:
                    continue
                sent.basic_events.append(
                    BetterEvent(
                        temp_event.event_id,
                        temp_event.event_type,
                        temp_event.properties,
                        temp_event.anchors,
                        temp_event.arguments + ref_args,
                        [],
                        temp_event.state_of_affairs,
                    )
                )

        granular_events = cls.from_mitre_granular_events(doc_text, annotated_text, sentences,
                                                         mentions_for_sents, all_mentions,
                                                         entities, mention_id_counter)

        return cls(
            annotated_text.entry_id,
            annotated_text.doc_id,
            task_type,
            sentences,
            entities,
            mitre_span_sets,
            doc_text,
            granular_events,
            annotated_text.segment_sections,
            lang,
            properties={}
        )

    @classmethod
    def from_mitre_granular_events(cls,
                                   doc_text: DocumentText,
                                   annotated_text: AnnotatedText,
                                   sentences: List[BetterSentence],
                                   mentions_for_sents,
                                   all_mentions: List[BetterMention],
                                   entities: List[BetterEntity],
                                   mention_id_counter: int) \
            -> List[BetterEvent]:

        all_non_anchor_mentions = [m for m in all_mentions if m.mention_type != EVENT_ANCHOR]

        basic_event_map = {}
        event_to_sent_map = {}
        for s in sentences:
            for e in s.basic_events:
                basic_event_map[e.event_id] = e
                event_to_sent_map[e.event_id] = s

        # Process granular templates (which can have slots referring to Basic Events)
        granular_events: List[BetterEvent] = []
        for template in annotated_text.basic_event_set.granular_templates:
            properties = {}
            args = []
            event_args = []
            for slot_key, slot_val in template.slots.items():
                # most frequent case, all non-property arguments
                if isinstance(slot_val.args, list):
                    # slot_key is the argument role
                    # slot_val is the list of arguments filling that role
                    for slot_arg in slot_val.args:

                        # Convert time attachments
                        time_attachments = [cls.convert_mitre_span_set(
                            doc_text,
                            spanset_id,
                            annotated_text.basic_event_set,
                            sentences,
                            all_non_anchor_mentions,
                        ) for spanset_id in slot_arg.time_attachments]

                        if slot_arg.ref_id.startswith("event"):

                            # This is an event argument, so we need to find the basic event it points at
                            if slot_arg.ref_id not in basic_event_map:
                                b_event = annotated_text.basic_event_set.get_event_by_id(slot_arg.ref_id)
                                if b_event.event_type in EXCLUDED_BASIC_EVENT_TYPES:
                                    continue
                                raise(ValueError(f"No basic event with id {slot_arg.ref_id}"))

                            # Add that basic event, this is easy
                            event_args.append(
                                BetterEventArgument(
                                    role=slot_key,
                                    basic_event=basic_event_map[slot_arg.ref_id],
                                    irrealis=slot_arg.irrealis,
                                    time_attachments=time_attachments,
                                    score=None
                                )
                            )
                        else:

                            # This is a span argument, so we need to convert the entire span set,
                            # which can be spread across many sentences: this is OK for granular events!
                            span_set_id = slot_arg.ref_id
                            span_set = cls.convert_mitre_span_set(
                                        doc_text,
                                        span_set_id,
                                        annotated_text.basic_event_set,
                                        sentences,
                                        all_non_anchor_mentions,
                            )
                            # Make sure we successfully grounded at least one span before creating an argument
                            if span_set.spans:
                                args.append(BetterArgument(role=slot_key,
                                                           span_set=span_set,
                                                           irrealis=slot_arg.irrealis,
                                                           time_attachments=time_attachments,
                                                           mitre_coref_id=None))
                            else:
                                print(f"Unable to ground event span argument {span_set_id} in "
                                      f"document {annotated_text.doc_id}; must be a problem with "
                                      f"sentence boundaries?")
                else:
                    validate_granular_arg_property(None, None, slot_key)
                    properties[slot_key] = slot_val.args

            anchors = cls.convert_mitre_span_set(
                                        doc_text,
                                        template.anchor,
                                        annotated_text.basic_event_set,
                                        sentences,
                                        all_mentions,
                    )
            if not anchors.spans:
                continue
            resolved_args = [arg for arg in args if arg.span_set.spans]
            for arg in args:
                if arg not in resolved_args:
                    print(f"WARNING: argument {args[-1].role} {slot_arg.ref_id} "
                          f"occurred outside of anchor sentence")

            anchors = cls.convert_mitre_span_set(
                                        doc_text,
                                        template.anchor,
                                        annotated_text.basic_event_set,
                                        sentences,
                                        all_mentions,
            )

            granular_events.append(
                BetterEvent(
                    event_id=template.template_id,
                    event_type=template.template_type,
                    properties=properties,
                    anchors=anchors,
                    arguments=args,
                    event_arguments=event_args,
                    state_of_affairs=None
                )
            )

        return granular_events

    @classmethod
    def _ingest_mentions(
            cls, sentences: List[BetterSentence], doc_text: DocumentText,
            annotated_text: AnnotatedText, task_type: str) -> \
            Tuple[DefaultDict[int, BetterMention], List[BetterEntity], int]:
        """Create BetterMentions from provided span-sets with 'synclass'
           labels.
        """
        if task_type == "abstract":
            span_sets = annotated_text.abstract_event_set.span_set.spans
        elif task_type in ["basic", "granular"]:
            span_sets = annotated_text.basic_event_set.span_set.spans
        else:
            raise ValueError
        mentions_for_sents: DefaultDict[int, List[BetterMention]] = defaultdict(list)
        mention_id = 0
        entities: List[BetterEntity] = []
        for span_id, spans in span_sets.items():
            entity_list = []
            for span in spans:
                if span.grounding is None:
                    return mentions_for_sents, entities, mention_id
                # convert to [] char ranges
                sgs_full = SimpleGroundedSpan(
                    doc_text,
                    span.grounding.full_start,
                    span.grounding.full_end - 1,
                    start_token=None,
                    end_token=None,
                )
                sgs_head = SimpleGroundedSpan(
                    doc_text,
                    span.grounding.head_start,
                    span.grounding.head_end - 1,
                    start_token=None,
                    end_token=None,
                )
                # find the sentence that this mention belongs to
                sent_id = None
                for sent in sentences:
                    if (
                            span.grounding.full_start
                            >= sent.original_document_character_span.start_char
                            and span.grounding.full_end - 1
                            <= sent.original_document_character_span.end_char
                    ):
                        sent_id = sent.sent_id
                        break
                if sent_id is None:
                    print(f"WARNING: misplaced mention: {span.full_span}")
                    continue
                mentions_for_sents[sent_id].append(
                    BetterMention(
                        str(mention_id),
                        span.syn_class,
                        GroundedSpan(
                            sent_id, sgs_full, sgs_head, str(mention_id)
                        ),
                        "",
                        {}
                    )
                )
                entity_list.append(str(mention_id))
                mention_id += 1
            entities.append(
                BetterEntity(
                    "",
                    entity_list,
                    span_id.replace("ss", "e"),
                    [
                        ent_id.replace("ss", "e")
                        for key, val in
                        annotated_text.basic_event_set.includes_relations.items()
                        if key == span_id
                        for ent_id in val
                    ],
                )
            )
        return mentions_for_sents, entities, mention_id

    @classmethod
    def from_mitre(cls, annotated_text: AnnotatedText, task_type: str, lang: Optional[str]):
        segment_type = annotated_text.segment_type
        if segment_type == "sentence":
            return cls.from_mitre_abstract_events(
                annotated_text, task_type, lang
            )
        else:
            assert segment_type == "document"
            return cls.from_mitre_basic_events(
                annotated_text, task_type, lang
            )

    def to_mitre_abstract_events(self) -> Mapping:
        # The MITRE format stores all entries together as a corpus, regardless
        # of what document sentences come from. This method therefore just
        # returns an "entries dictionary that needs to get combined with
        # other such dictionaries at the corpus
        # level
        output = {}
        # get to_mitre from sentences
        for sent in self.sentences:
            sent_mitre = sent.to_mitre_abstract_events(self.entities)

            # There is a bit of variety in how doc/entry ids are constructed
            # in the different .bp.json datasets. We can't rely on the
            # original doc ids being unique, so we use the original entry-ids
            # to key our dictionaries of docs, and switch them back when
            # writing out in mitre format.
            entry = {
                "annotation-sets": sent_mitre,
                "doc-id": self.mitre_doc_id,
                "entry-id": self.doc_id,
                "segment-text": sent.original_document_character_span.text,
                "segment-type": "sentence",
                "sent-id": str(sent.sent_id),
                "lang": ISO_1_TO_ISO_2[self.lang], # back to 3-char
            }
            output[self.doc_id] = entry
        return output

    def to_mitre_basic_events(self):

        output: Dict[str, Any] = {"annotation-sets": {"basic-events": {"events": {}}}}
        all_mentions = {}
        # basic event things:
        for sent in self.sentences:
            for event_id, event in sent.to_mitre_basic_events(self).items():
                if event_id in output["annotation-sets"]["basic-events"]["events"]:
                    print("WARNING!! Non-unique event ID in ", self.doc_id,
                          "--", event_id, "-- please fix the event model!")
                output["annotation-sets"]["basic-events"]["events"][event_id] = event
            all_mentions.update(sent.mentions_by_id)

        # need to combine together the span sets from each sentence, since
        # sentences aren't an atomic unit of mitre fmt

        # Hack for HITL; uncomment if needed again
        # for elt in self.structural_elements:
        #     match = False
        #     for sent in self.sentences:
        #         if (sent.original_document_character_span.start_char == elt.start_char and
        #                 sent.original_document_character_span.end_char == elt.end_char):
        #             # found matching sentence
        #             elt.structural_type = sent.sentence_type
        #             match = True
        #     if not match:
        #         if elt.start_char == elt.end_char:
        #             print("No match for single-char sentence", elt.start_char, elt.end_char,
        #                   self.doc_id)
        #         else:
        #             print("No match for", elt.start_char, elt.end_char, self.doc_id)

        # NOTE: see the comment above in to_mitre_abstract_events
        #       re: mitre_doc_id / doc_id
        output.update(
            {
                "doc-id": self.mitre_doc_id,
                "entry-id": "{}".format(self.doc_id),
                "segment-sections": [
                    elt.to_mitre() for elt in self.structural_elements
                ],
                "segment-text": self.doc_text.text,
                "segment-type": "document",
                "lang": ISO_1_TO_ISO_2[self.lang], # back to 3-char
            }
        )

        output["annotation-sets"]["basic-events"].update(
            {
                "includes-relations": {
                    entity.entity_id:
                        entity.subset_coreferants
                    for entity in self.entities
                    if entity.subset_coreferants
                },
                "span-sets": {
                    entity.entity_id:
                        entity.to_mitre(all_mentions)
                    for entity in self.mitre_span_sets
                },
            }
        )
        return output

    def to_mitre_granular_events(self):
        # superset of returning basic events in mitre fmt
        output = self.to_mitre_basic_events()
        templates = {}  # template id -> list of dict entries for each slot

        for granular_event in self.granular_events:
            granular_to_mitre = {"template-id": granular_event.event_id,
                                 "template-anchor": granular_event.anchors.spans[0].grounded_span.mention_id,
                                 "template-type": granular_event.event_type}
            for ev_arg in granular_event.event_arguments:
                # artifact of ingested gold granular templates:
                if ev_arg.role in ["protest-event", "corrupt-event",
                                   "terror-event", "outbreak-event"]:
                    continue
                if is_guessed_event_argument_role(ev_arg.role):
                    continue
                if ev_arg.role not in granular_to_mitre:
                    granular_to_mitre[ev_arg.role] = []
                entry = {'event-id': ev_arg.basic_event.event_id}
                if ev_arg.irrealis:
                    entry['irrealis'] = ev_arg.irrealis
                granular_to_mitre[ev_arg.role].append(entry)

            # iterate over span args
            for span_arg in granular_event.arguments:
                if span_arg.role == "guessed":
                    continue
                if span_arg.role not in granular_to_mitre:
                    granular_to_mitre[span_arg.role] = []
                entry = {"ssid": span_arg.mitre_coref_id}
                if span_arg.irrealis:
                    entry["irrealis"] = span_arg.irrealis
                granular_to_mitre[span_arg.role].append(entry)

            # output any slots in properties dict
            for property_key, property_val in granular_event.properties.items():
                if property_key in VALID_GRANULAR_ARG_PROPERTIES:
                    granular_to_mitre[property_key] = property_val

            templates[granular_event.event_id] = granular_to_mitre

        output['annotation-sets']['basic-events']['granular-templates'] = templates

        return output

    @staticmethod
    def convert_mitre_span_set(
        doc_text: DocumentText,
        span_id: str,
        ann_set: AnnotationSet,
        sentences: List[BetterSentence],
        mentions: List[BetterMention],
        provided_sent_id: Optional[str] = None,
    ) -> BetterSpanSet:
        """Convenience method to convert a set of MITRE spans a BetterSpanSet"""
        better_spans = []
        for hs in ann_set.get_headed_strings_for_span_id(span_id):
            if hs.grounding:
                if provided_sent_id is not None:
                    sent_id = provided_sent_id
                else:
                    sent_id = None
                    for sent in sentences:
                        if (
                            hs.grounding.full_start
                            >= sent.original_document_character_span.start_char
                            and hs.grounding.full_end - 1
                            <= sent.original_document_character_span.end_char
                        ):
                            sent_id = sent.sent_id
                    if sent_id is None and provided_sent_id is None:
                        print("WARNING: Span does not fall in sentence boundaries! Trying head")
                        for sent in sentences:
                            if (
                                hs.grounding.head_start
                                >= sent.original_document_character_span.start_char
                                and hs.grounding.head_end - 1
                                <= sent.original_document_character_span.end_char
                            ):
                                sent_id = sent.sent_id
                        if sent_id is None:
                            print("WARNING: Span does not fall in sentence boundaries! Skipping.")
                            continue
                    # spans converted to inclusive []
                full_span = SimpleGroundedSpan(
                    doc_text,
                    start_char=hs.grounding.full_start,
                    end_char=hs.grounding.full_end - 1,
                    start_token=None,
                    end_token=None,
                )
                head_span = SimpleGroundedSpan(
                    doc_text,
                    start_char=hs.grounding.head_start,
                    end_char=hs.grounding.head_end - 1,
                    start_token=None,
                    end_token=None,
                )
                mention_id = None
                if mentions:
                    for mention in mentions:
                        if mention.grounded_span.full_span == full_span and \
                                mention.grounded_span.head_span == head_span:
                            mention_id = mention.mention_id
                            break
                grounding = GroundedSpan(
                    sent_id,
                    full_span=full_span,
                    head_span=head_span,
                    mention_id=mention_id,
                )
            else:
                grounding = None
            better_spans.append(
                ScoredBetterSpan(BetterSpan(hs.full_span, hs.head, grounding))
            )
        return BetterSpanSet(better_spans)

    @staticmethod
    def convert_mitre_span_set_sentence_restricted(
        doc_text: DocumentText,
        span_id: str,
        ann_set: AnnotationSet,
        sentence: BetterSentence,
        mentions: List[BetterMention]
    ):
        better_spans = []
        for hs in ann_set.get_headed_strings_for_span_id(span_id):
            # span set should only include spans local to sentence
            if (
                hs.grounding.full_end - 1
                < sentence.original_document_character_span.start_char
                or hs.grounding.full_start
                > sentence.original_document_character_span.end_char
            ):
                continue
            # find mention from previously processed list of BetterMentions
            # corresponding with this HeadedString
            corresponding_mention = None
            for mention in mentions:
                if (
                    mention.grounded_span.full_span.start_char
                    == hs.grounding.full_start
                    and mention.grounded_span.full_span.end_char
                    == hs.grounding.full_end - 1
                    # the span set specified for an event-anchor is NOT
                        # equivalent to the one specified for
                    # an event argument, even if they share a span
                    and mention.mention_type == hs.syn_class
                ):
                    corresponding_mention = mention
            if corresponding_mention is None:
                continue

            better_spans.append(
                ScoredBetterSpan(
                    BetterSpan(
                        hs.full_span, hs.head,
                        corresponding_mention.grounded_span
                    )
                )
            )

        return BetterSpanSet(better_spans)

    @staticmethod
    def convert_mitre_span_set_sentence_restricted_fallback(
        doc_text: DocumentText,
        span_id: str,
        ann_set: AnnotationSet,
        sentence: BetterSentence,
        mentions_for_sentence: List[BetterMention],
        mentions_for_document: List[BetterMention],
        entities: List[BetterEntity],
        mention_id_counter: int,
    ):
        """
        Fallback ingestion method to make up for non-exhaustive inclusion
        of every instance of an entity within a span-set.
        Attempts to find string matches from ann_set in sentence.
        Creates new mentions in mention list for matches as necessary.
        Adds new mention to appropriate entity.
        Args:
            doc_text: document text object
            span_id: span-set id to be resolved
            ann_set: annotations from which the span-id is drawn
            sentence: sentence to which annotations are to be grounded
            mentions_for_sentence: mentions for this sentence
            mentions_for_document: ''
            entities: entities (across whole doucment)
            mention_id_counter: counter for assigning mention ids

        Returns: BetterSpanSet, mention_id_counter

        """
        matched_strings = set()
        better_spans = []
        for hs in ann_set.get_headed_strings_for_span_id(span_id):
            # skip if string match not in sentence
            if hs.head in matched_strings or (
                hs.head not in sentence.original_document_character_span.text
            ):
                continue
            # find mention from previously processed list of BetterMentions
            # corresponding with this HeadedString
            corresponding_mention = None
            for mention in mentions_for_sentence:
                if (
                    hs.head in mention.full_span.text
                    and mention.mention_type == hs.syn_class
                ):
                    corresponding_mention = mention
                    break

            if corresponding_mention is None:
                # create new mention if text span occurs in existing mention
                for mention in mentions_for_document:
                    if (
                            hs.head in mention.full_span.text
                            and mention.mention_type == hs.syn_class
                    ):
                        from better_events.better_mapper import BetterMapper
                        head_span_offsets = BetterMapper.get_best_span(
                            sentence.original_document_character_span.text,
                            hs.head, []
                        )
                        head_span = SimpleGroundedSpan(
                            doc_text,
                            sentence.original_document_character_span.start_char +
                            head_span_offsets[0],
                            sentence.original_document_character_span.start_char +
                            head_span_offsets[1],
                            None,
                            None,
                        )
                        try:
                            full_span_offsets = BetterMapper.get_best_span(
                                sentence.original_document_character_span.text,
                                hs.full_span, []
                            )
                        except ValueError:
                            full_span_offsets = head_span_offsets
                        full_span = SimpleGroundedSpan(
                            doc_text,
                            sentence.original_document_character_span.start_char + full_span_offsets[0],
                            sentence.original_document_character_span.start_char + full_span_offsets[1],
                            None,
                            None,
                        )

                        new_mention = BetterMention(
                            str(mention_id_counter),
                            mention.mention_type,
                            GroundedSpan(
                                sentence.sent_id,
                                full_span,
                                head_span,
                                str(mention_id_counter)
                            ),
                            mention.entity_type,
                            mention.properties
                        )
                        # Debugging statement if one wants a paper trail of
                        # additions made to annotation:

                        # print(f"Created additional mention for {span_id}: '{hs.full_span}' "
                        #       "to avoid dropping argument")
                        sentence.mentions.append(new_mention)
                        sentence.mentions_by_id[str(mention_id_counter)] = new_mention
                        mention_id_counter += 1
                        # insert new mention into entity where associated
                        # mention was found
                        for entity in entities:
                            if mention.mention_id in entity.mentions:
                                entity.mentions.append(new_mention.mention_id)
                        corresponding_mention = new_mention
                        matched_strings.add(hs.head)
                        break
            if corresponding_mention is not None:
                better_spans.append(
                    ScoredBetterSpan(
                        BetterSpan(
                            corresponding_mention.grounded_span.full_span.text,
                            corresponding_mention.grounded_span.head_span.text,
                            corresponding_mention.grounded_span
                        )
                    )
                )
                # do NOT continue attempting to find matches from the HeadedStrings
                # associated with the span-set-id
                break

        return BetterSpanSet(better_spans), mention_id_counter

    def to_dict(self) -> Mapping:
        return {
            "doc_id": self.doc_id,
            "mitre_doc_id": self.mitre_doc_id,
            "task_type": self.task_type,
            "lang": self.lang,
            "properties": self.properties,
            "doc_text": self.doc_text.text,
            "sentences": [x.to_dict() for x in self.sentences],
            "entities": [x.to_dict() for x in self.entities],
            "mitre_span_sets": [x.to_dict() for x in self.mitre_span_sets],
            "granular_events": [x.to_dict() for x in self.granular_events],
            "structural_elements": [
                x.to_dict() for x in self.structural_elements
            ],
        }

    def call_method_on_event_sets(
            self,
            func: Callable[
                ['BetterDocument', Optional[BetterSentence], BetterEvent], BetterEvent]
    ) -> None:
        for sent in self.sentences:
            sent.relations = [func(self, sent, e) for e in sent.relations]
            if self.task_type == ABSTRACT_TASK:
                sent.abstract_events = [func(self, sent, e) for e in sent.abstract_events]
            if self.task_type in [BASIC_TASK, GRANULAR_TASK]:
                sent.basic_events = [func(self, sent, e) for e in sent.basic_events]
        if self.task_type == GRANULAR_TASK:
            self.granular_events = [func(self, None, e) for e in self.granular_events]

    # returns (sentence_texts, sentence_types, sentence_ranges)
    @staticmethod
    def initial_sentence_segmentation(
        full_text: str, structural_elements: [List[StructuralElement]]
    ) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:

        # first split according to structural elements, sections not containing
        # structural elements are then split
        # "orthographically" according to linebreak cues.
        text_splits: List[str] = []
        text_types: List[str] = []
        text_ranges: List[Tuple[int, int]] = []
        if structural_elements:
            # critical b/c the annotations are not ordered
            structural_elements.sort(key=lambda e: e.start_char)
            for i, structural_elt in enumerate(structural_elements):

                text_range = (
                    structural_elt.start_char, structural_elt.end_char
                )

                structural_elt_text = full_text[
                    structural_elt.start_char: structural_elt.end_char + 1
                ]
                if structural_elt_text == "\ufeff":
                    continue
                # don't output sentences that are only punctuation
                if all(unicodedata.category(ch).startswith('P')
                        for ch in structural_elt_text):
                    continue
                text_type = structural_elt.structural_type

                # check that this structural_elt is not already accounted for
                if text_ranges and text_range == text_ranges[-1]:
                    # if the already-placed text has type Sentence, and this
                    # type is more specific, replace it
                    if (
                        text_types[-1] == STRUCTURAL_SENTENCE
                        and text_type != STRUCTURAL_SENTENCE
                    ):
                        text_types[-1] = text_type
                    # skip ahead to avoid duplicate ranges
                    continue

                # check for sentence overlap (annotation error)
                skip_this_entry = False
                for range_idx, t_range in enumerate(text_ranges):
                    # start of current range contained within a previous range
                    if not text_range[0] >= t_range[1]:
                        # replace that range with whichever is longer
                        if text_range[1] - text_range[0] > \
                                t_range[1] - t_range[0]:
                            text_ranges[range_idx] = text_range
                            text_splits[range_idx] = structural_elt_text
                        skip_this_entry = True
                if skip_this_entry:
                    continue

                text_splits.append(structural_elt_text)
                text_types.append(text_type)
                text_ranges.append(text_range)

        # ensure that no sentence is only whitespace
        while any(text.isspace() or text == "\ufeff" for text in text_splits):
            for i in range(len(text_splits) - 1):
                if text_splits[i].isspace() or text_splits[i] == "\ufeff":
                    # take all preceding list elements, add this one to the one
                    # after
                    text_splits = (
                        text_splits[:i]
                        + [text_splits[i] + text_splits[i + 1]]
                        + text_splits[i + 2:]
                    )
                    # just delete the text type that corresponded with the
                    # sentence that was only a space character
                    text_types = (
                        text_types[:i] +
                        [text_types[i + 1]] + text_types[i + 2:]
                    )
                    text_ranges = (
                        text_ranges[:i] +
                        [text_ranges[i + 1]] + text_ranges[i + 2:]
                    )
                    break
            if text_splits[-1].isspace():
                text_splits = text_splits[:-2] + [text_splits[-2] + text_splits[-1]]
                text_types = text_types[:-2] + [text_types[-1]]
                text_ranges = text_ranges[:-2] + [text_ranges[-1]]

        assert len(text_splits) == len(text_types) == len(text_ranges)

        return text_splits, text_types, text_ranges

    def split_sentences(
        self, sentence_spans: Mapping[int, Iterable[Iterable[int]]]
    ) -> "BetterDocument":
        """Creates a new BetterDocument where the sentences, and performs
        splits as indicated in `sentence_spans`.
        -The key of sentence_spans is the sent_id. All sent_id's must be listed,
         event if they are not split.
        -The values are the indices of the first and last word of each output
        sentence.
        sentence_spans has the structure:
        { sent_id : [ [first_token_index, last_token_index], ...]

        """

        for sent in self.sentences:
            assert sent.sent_id in sentence_spans
        assert len(self.sentences) == len(sentence_spans.keys())

        new_doc: BetterDocument = BetterDocument(
            doc_id=self.doc_id,
            mitre_doc_id=self.mitre_doc_id,
            task_type=self.task_type,
            sentences=[],
            entities=self.entities,
            mitre_span_sets=self.mitre_span_sets,
            doc_text=self.doc_text,
            granular_events=self.granular_events,
            structural_elements=self.structural_elements,
            lang=self.lang,
            properties=self.properties,
        )

        split_offset = 0
        for top_sent_ind, sent in enumerate(self.sentences):
            split_sentences = []
            new_sent_ids = []
            event_and_span = []
            event_and_span.extend([(e, e.give_span(), ABSTRACT_TASK)
                                   for e in sent.abstract_events])
            event_and_span.extend([(e, e.give_span(), BASIC_TASK)
                                   for e in sent.basic_events])
            event_and_span.extend([(e, e.give_span(), "relations")
                                   for e in sent.relations])
            event_and_span = sorted(event_and_span, key=lambda x: (x[1][0], x[1][1]))
            mention_and_span = sorted(
                [(m, m.grounded_span) for m in sent.mentions],
                key=lambda x:
                (x[1].full_span.start_token, x[1].full_span.end_token),
            )
            event_ind = 0
            mention_ind = 0
            for split_ind, (start_ind, l) in enumerate(sentence_spans[sent.sent_id]):
                end_ind = -1 + start_ind + l
                new_sent_id = (
                    split_offset  # 1 + split_offset + top_sent_ind + split_ind - 1
                )
                new_sent_ids.append(new_sent_id)

                new_abstract_events = []
                new_basic_events = []
                new_relations = []

                new_mentions = []

                # I wasn't sure why this loop repeated itself: it seemed like a mistake? - cjenkins
                # for new_sent_id, (start_ind, l) in
                # zip(new_sent_ids, sentence_spans[sent.sent_id]):
                while (
                    event_ind < len(event_and_span)
                    and start_ind <= event_and_span[event_ind][1][0] < start_ind + l
                ):
                    event, (e_start_word, e_end_word), task = event_and_span[event_ind]
                    if e_end_word > start_ind + l:
                        print(
                            f"ERROR: Skipping event {event.event_id} in sentence {sent.sent_id} "
                            f"because the event span {(e_start_word, e_end_word)} "
                            f"does not fit in the current sentence span: "
                            f"{(start_ind, start_ind + l - 1)}"
                        )
                    else:
                        new_event = event.offset_variant(-start_ind, new_sent_id)
                        if task == ABSTRACT_TASK:
                            new_abstract_events.append(new_event)
                        elif task == BASIC_TASK:
                            new_basic_events.append(new_event)
                        elif task == "relations":
                            new_relations.append(new_event)
                    event_ind += 1

                while (
                    mention_ind < len(mention_and_span)
                    and start_ind <= mention_and_span[mention_ind][1].full_span.start_token
                        < start_ind + l
                ):
                    if (
                        mention_and_span[mention_ind][1].full_span.end_token
                        > start_ind + l
                    ):
                        if (mention_and_span[mention_ind][1].head_span.end_token
                                > start_ind + 1):
                            print(
                                f"ERROR: Skipping mention {mention_and_span[mention_ind][0]}"
                                f" As it did not fit in the new split sentence span"
                            )
                            # remove missing mention from entities list
                            mention_id = mention_and_span[mention_ind][0].mention_id
                            for entity in new_doc.entities:
                                if mention_id in entity.mentions:
                                    entity.mentions.remove(mention_id)
                            # remove any entities with no mentions
                            new_doc.entities = [ent for ent in new_doc.entities if len(ent.mentions)]
                        else:
                            print(f"WARNING: mention {mention_and_span[mention_ind][0]}"
                                  f" truncated to its head span to fit in new sentence split")
                            new_mentions.append(
                                mention_and_span[mention_ind][0].with_new_grounded_span(
                                    # replace full span with head span
                                    mention_and_span[mention_ind][1].with_new_full(
                                        mention_and_span[mention_ind][1].head_span
                                    )
                                ).offset_variant(
                                    -start_ind, new_sent_id
                                )
                            )
                    else:
                        new_mentions.append(
                            mention_and_span[mention_ind][0].offset_variant(
                                -start_ind, new_sent_id
                            )
                        )
                    mention_ind += 1

                split_offset += 1  # -1 + len(sentence_spans[sent.sent_id])

                new_sent = BetterSentence.derive_subsentence(
                    sent,
                    self.doc_text,
                    start_ind,
                    end_ind,
                    new_sent_id,
                    split_ind,
                    new_abstract_events,
                    new_basic_events,
                    new_relations,
                    new_mentions,
                )
                split_sentences.append(new_sent)
                new_doc.sentences.append(new_sent)

            if (
                split_sentences[0].original_document_character_span.start_char
                != sent.original_document_character_span.start_char
                or split_sentences[-1].original_document_character_span.end_char
                != sent.original_document_character_span.end_char
            ):
                raise Exception("Corrupted text!")
            if event_ind != len(event_and_span):
                print(f"ERROR: Missed some events.")
            if mention_ind != len(mention_and_span):
                print(f"ERROR: missed some mentions")

        new_doc.sentences_by_id = {s.sent_id: s for s in new_doc.sentences}
        all_basic_ev_ids = [ev.event_id for sent in new_doc.sentences for ev in sent.basic_events]
        new_granular_ev = []
        for granular_ev in self.granular_events:
            new_ev_args = []
            # if granular event arg refers to a basic event that has been removed: remove it
            for ev_arg in granular_ev.event_arguments:
                if ev_arg.basic_event.event_id not in all_basic_ev_ids:
                    print(f"WARNING: granular event argument pointing to event_id: "
                          f"{ev_arg.basic_event.event_id} removed due to sentence splitting.")
                    continue
                new_ev_args.append(ev_arg)

            new_granular_ev.append(granular_ev.with_new_event_arguments(new_ev_args))

        new_doc.granular_events = new_granular_ev

        return new_doc

    def has_sentences_to_recombine(self):
        for s in self.sentences:
            if s.original_sent_id is not None:
                return True
        return False

    def recombine_sentences(self) -> "BetterDocument":
        if not self.has_sentences_to_recombine():
            return self

        new_doc: BetterDocument = self.__class__(
            doc_id=self.doc_id,
            mitre_doc_id=self.mitre_doc_id,
            task_type=self.task_type,
            sentences=[],
            entities=self.entities,
            granular_events=self.granular_events,
            mitre_span_sets=self.mitre_span_sets,
            doc_text=self.doc_text,
            structural_elements=self.structural_elements,
            lang=self.lang,
            properties=self.properties,
        )

        by_original_id = defaultdict(list)
        for sent in sorted(self.sentences, key=lambda x: x.sent_id):
            by_original_id[sent.original_sent_id].append(sent)

        for orig_sent_id in sorted(by_original_id.keys()):
            # this call handles the aggregation of events / mentions
            # from the split sentences
            out_sent = \
                BetterSentence.recombine_sentences(by_original_id[orig_sent_id])
            new_doc.sentences.append(out_sent)

        new_doc.sentences_by_id = {
            sent.sent_id: sent for sent in new_doc.sentences
        }

        return new_doc

    @staticmethod
    def span_to_sent_id(
            span: Tuple[int, int], sentences: List[BetterSentence]
    ) -> int:
        for sent in sentences:
            if (
                span[0] >= sent.original_document_character_span.start_char
                and span[1] <= sent.original_document_character_span.end_char
            ):
                return sent.sent_id

        raise ValueError(f"Requested sent id for invalid char range {span}")

    def doc_span_to_sent_id(self, span: Tuple[int, int]) -> int:
        return BetterDocument.span_to_sent_id(span, self.sentences)

    def is_doc_span_structural(self, span: Tuple[int, int]) -> bool:
        for structural_elt in self.structural_elements:
            if (
                span[0] >= structural_elt.start_char
                and span[1] <= structural_elt.end_char
            ):
                return True
        return False

    def give_abstract_event_spans(self, sent_id: int) -> Iterable[Iterable[int]]:
        return [
            event.give_span()
            for event in self.sentences_by_id[sent_id].abstract_events
        ]

    def give_basic_event_spans(self, sent_id: int) -> Iterable[Iterable[int]]:
        out_spans = []
        for event in self.sentences_by_id[sent_id].basic_events:
            sent_ids = set(
                [
                    span.grounded_span.sent_id
                    for span in event.anchors.spans
                    if span.grounded_span is not None
                ]
            )
            if len(sent_ids) == 1 and sent_id in sent_ids:
                out_spans.append(event.give_span())
        return out_spans

    def ground_document_level_features(self):
        from better_events.better_mapper import BetterMapper
        self.call_method_on_event_sets(BetterMapper.ground_event_to_tokens)
