"""Generally speaking, definitions in this file are used to ingest from BP to BETTER formats"""
from collections import defaultdict
from typing import List, Dict, Mapping, Iterator, Tuple, Optional, Union

from better_events.better_validation import (
    validate_granular_template_type,
    validate_structural_type
)

ISO_2_TO_ISO_1 = {
    "aar": "aa", "abk": "ab", "afr": "af", "aka": "ak", "sqi": "sq", "amh": "am", "ara": "ar",
    "arg": "an", "hye": "hy", "asm": "as", "ava": "av", "ave": "ae", "aym": "ay", "aze": "az",
    "bak": "ba", "bam": "bm", "eus": "eu", "bel": "be", "ben": "bn", "bih": "bh", "bis": "bi",
    "bod": "bo", "bos": "bs", "bre": "br", "bul": "bg", "mya": "my", "cat": "ca", "ces": "cs",
    "cha": "ch", "che": "ce", "zho": "zh", "chu": "cu", "chv": "cv", "cor": "kw", "cos": "co",
    "cre": "cr", "cym": "cy", "dan": "da", "deu": "de", "div": "dv", "nld": "nl", "dzo": "dz",
    "ell": "el", "eng": "en", "epo": "eo", "est": "et", "ewe": "ee", "fao": "fo", "fas": "fa",
    "fij": "fj", "fin": "fi", "fra": "fr", "fry": "fy", "ful": "ff", "kat": "ka", "gla": "gd",
    "gle": "ga", "glg": "gl", "glv": "gv", "grn": "gn", "guj": "gu", "hat": "ht", "hau": "ha",
    "heb": "he", "her": "hz", "hin": "hi", "hmo": "ho", "hrv": "hr", "hun": "hu", "ibo": "ig",
    "isl": "is", "ido": "io", "iii": "ii", "iku": "iu", "ile": "ie", "ina": "ia", "ind": "id",
    "ipk": "ik", "ita": "it", "jav": "jv", "jpn": "ja", "kal": "kl", "kan": "kn", "kas": "ks",
    "kau": "kr", "kaz": "kk", "khm": "km", "kik": "ki", "kin": "rw", "kir": "ky", "kom": "kv",
    "kon": "kg", "kor": "ko", "kua": "kj", "kur": "ku", "lao": "lo", "lat": "la", "lav": "lv",
    "lim": "li", "lin": "ln", "lit": "lt", "ltz": "lb", "lub": "lu", "lug": "lg", "mah": "mh",
    "mal": "ml", "mri": "mi", "mar": "mr", "msa": "ms", "mkd": "mk", "mlg": "mg", "mlt": "mt",
    "mon": "mn", "nau": "na", "nav": "nv", "nbl": "nr", "nde": "nd", "ndo": "ng", "nep": "ne",
    "nno": "nn", "nob": "nb", "nor": "no", "nya": "ny", "oci": "oc", "oji": "oj", "ori": "or",
    "orm": "om", "oss": "os", "pan": "pa", "pli": "pi", "pol": "pl", "por": "pt", "pus": "ps",
    "que": "qu", "roh": "rm", "ron": "ro", "run": "rn", "rus": "ru", "sag": "sg", "san": "sa",
    "sin": "si", "slk": "sk", "slv": "sl", "sme": "se", "smo": "sm", "sna": "sn", "snd": "sd",
    "som": "so", "sot": "st", "spa": "es", "srd": "sc", "srp": "sr", "ssw": "ss", "sun": "su",
    "swa": "sw", "swe": "sv", "tah": "ty", "tam": "ta", "tat": "tt", "tel": "te", "tgk": "tg",
    "tgl": "tl", "tha": "th", "tir": "ti", "ton": "to", "tsn": "tn", "tso": "ts", "tuk": "tk",
    "tur": "tr", "twi": "tw", "uig": "ug", "ukr": "uk", "urd": "ur", "uzb": "uz", "ven": "ve",
    "vie": "vi", "vol": "vo", "wln": "wa", "wol": "wo", "xho": "xh", "yid": "yi", "yor": "yo",
    "zha": "za", "zul": "zu",
}
ISO_1_TO_ISO_2 = {v: k for k, v in ISO_2_TO_ISO_1.items()}

# not actually correct ISO 2 but present in a sample file:
ISO_2_TO_ISO_1["chn"] = "zh"


class HeadedStringGrounding:
    def __init__(self, hstart: str, hend: str, span_start: str, span_end: str):
        self.head_start = int(hstart)
        self.head_end = int(hend)
        self.full_start = int(span_start)
        self.full_end = int(span_end)


class HeadedString:
    def __init__(
        self,
        span: str,
        head: str,
        syn_class: str,
        grounding: Optional[HeadedStringGrounding] = None,
    ):
        self.full_span = span
        self.head = head
        self.grounding = grounding
        # used analogously to BetterMention mention_type (name, pronoun, etc) but is also
        # used to indicate that a span is an anchor
        self.syn_class = syn_class


class SpanSet:
    """This provides a way to map from a span ID to a set of HeadedStrings"""

    def __init__(self, span_set_json):
        self.spans: Dict[str, List[HeadedString]] = SpanSet.span_converter(
            span_set_json
        )

    @staticmethod
    def span_converter(
        d: Dict[str, Dict[str, List[Dict[str, str]]]]
    ) -> Dict[str, List[str]]:
        new_dict = {}
        for key in d:
            ls = []
            for elt in d[key]["spans"]:
                # if "hstring" not in elt:
                #     print("Missing hstring", elt)
                # Default to the full string as the head where we need to
                if "hstart" in elt:
                    grounding = HeadedStringGrounding(
                        elt["hstart"], elt["hend"], elt["start"], elt["end"]
                    )
                # if span provides grounding for full span but not head span,
                # copy full span into head field
                elif "start" in elt:
                    grounding = HeadedStringGrounding(
                        elt["start"], elt["end"], elt["start"], elt["end"]
                    )
                # it is possible for legacy annotations to have no grounding info at all
                else:
                    grounding = None
                ls.append(
                    HeadedString(
                        elt["string"],
                        elt.get("hstring", elt["string"]),
                        elt.get("synclass", ""),
                        grounding,
                    )
                )
            new_dict[key] = ls
        return new_dict


class AbstractEvent:

    # Not currently used, just for the record
    HELPFUL_HARMFUL = ["helpful", "harmful", "neutral"]
    MATERIAL_VERBAL = ["material", "verbal", "both", "unk"]

    def __init__(self, event) -> None:
        # Note that the strings here are span IDs, not text strings
        self.agents: List[str] = event["agents"]
        self.patients: List[str] = event["patients"]
        self.anchors: str = event["anchors"]

        self.event_id: str = event["eventid"]
        self.helpful_harmful: str = event["helpful-harmful"]
        self.material_verbal: str = event["material-verbal"]

    def __str__(self) -> str:
        return (
            f"{self.event_id}\n\tAgents: {self.agents}\n\tPatients: {self.patients}\n\t"
            f"Anchors: {self.anchors}\n\t{self.helpful_harmful}\n\t{self.material_verbal}"
        )


class BasicEvent:
    def __init__(self, event):
        self.agents: List[str] = event["agents"]
        self.patients: List[str] = event["patients"]
        self.money: List[str] = event.get("money", [])
        self.anchors: List[str] = event["anchors"]

        self.event_id: str = event["eventid"]
        self.event_type: str = event["event-type"]
        self.ref_events: List[str] = event["ref-events"]
        self.state_of_affairs: bool = event["state-of-affairs"]

    def __str__(self) -> str:
        return (
            f"{self.event_id}\n\tAgents: {self.agents}\n\tPatients: {self.patients}\n\tMoney: {self.money}\n\t"
            f"Anchors: {self.anchors}\n\tType: {self.event_type}\n\tRef-events: {self.ref_events}"
        )


class TemplateArg:
    def __init__(self, arg):
        ref_id = arg.get('ssid', None)
        if ref_id is None:
            ref_id = arg.get('event-id', None)
        self.ref_id = ref_id
        self.irrealis = arg.get('irrealis', None)
        self.time_attachments = arg.get('time-attachments', [])


class TemplateSlot:
    def __init__(self, name, args):
        self.name: str = name
        if isinstance(args, bool):
            self.args = args
        elif name == 'completion':
            self.args = args
        elif name == "type":
            self.args = args
        else:
            self.args = [TemplateArg(val) for val in args]


class GranularTemplate:
    def __init__(self, template):
        self.anchor: str = template["template-anchor"]
        self.template_id: str = template["template-id"]
        self.template_type: str = template["template-type"]
        validate_granular_template_type(None, None, self.template_type)
        self.slots: Dict[str, TemplateSlot] = {}
        for key in template:
            if (
                key == "template-anchor"
                or key == "template-id"
                or key == "template-type"
            ):
                continue
            self.slots[key] = TemplateSlot(key, template[key])


class AnnotationSet:
    """An annotation set links together a set of events and the spans they use."""

    def __init__(self, ann_type, ann_set):
        if ann_type == "abstract-events":
            events = [AbstractEvent(val) for val in ann_set["events"].values()]
        elif ann_type == "basic-events":
            events = [BasicEvent(val) for val in ann_set["events"].values()]
        else:
            raise ValueError(
                "Attempted to create AnnotationSet with type other "
                "than 'abstract-events' | 'basic-events'"
            )
        self.events: Union[List[AbstractEvent], List[BasicEvent]] = events
        self.granular_templates: List[GranularTemplate] = (
            [GranularTemplate(val) for val in ann_set["granular-templates"].values()]
            if "granular-templates" in ann_set
            else []
        )
        self.template_filler_coref_events = ann_set.get("template-filler-coref-events", [])
        self.span_set = SpanSet(ann_set["span-sets"])
        self.includes_relations = {}
        includes = ann_set.get("includes-relations", {})
        for key, span_list in includes.items():
            self.includes_relations[key] = span_list

    def get_event_by_id(self, event_id: str
                        ) -> Optional[Union[AbstractEvent, BasicEvent]]:
        for ev in self.events:
            if ev.event_id == event_id:
                return ev
        return None

    def __str__(self):
        return (
            "Events:\n"
            + "\n".join(str(e) for e in self.events)
            + "\n"
            + str(self.span_set)
        )

    def get_headed_strings_for_span_id(self, span_id: str) -> List[HeadedString]:
        return self.span_set.spans[span_id]


class StructuralElement:
    """Introduced for Basic Events - Document-level annotation marking
    segments of text as structural in nature (headlines, datelines, etc)
    and as of 9/1/2020 also which sections *are* sentences"""

    def __init__(
        self,
        start_char: int,
        end_char: int,
        structural_type: str,
    ):
        self.start_char: int = start_char
        self.end_char: int = end_char
        self.structural_type = structural_type
        validate_structural_type(None, None, structural_type)

        if start_char < 0 or end_char < 0 or end_char < start_char:
            raise ValueError(
                f"Character indices specifying StructuralElement are incoherent"
            )

    @classmethod
    def from_mitre(cls, entry: Dict):
        # is this the appropriate place to convert to [] indices?)
        return cls(entry["start"], entry["end"] - 1, entry["structural-element"])

    def to_mitre(self):
        return {
            "start": self.start_char,
            "end": self.end_char + 1,
            "structural-element": self.structural_type,
        }

    @classmethod
    def from_json(cls, json_dict: Dict):
        return cls(
            json_dict["start_char"], json_dict["end_char"], json_dict["structural_type"]
        )

    def __str__(self):
        return "{}: [{}:{}]".format(
            self.structural_type, self.start_char, self.end_char
        )

    def to_dict(self):
        return {
            "structural_type": self.structural_type,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


class AnnotatedText:
    """An annotated text holds its raw text and annotation sets.
    Its 'segment_type' indicates whether it is an individual sentence or a whole document

    One of those annotation sets of for abstract events.
    """

    def __init__(self, entry):
        # According to the v10 format only "entry-id" and "doc-id" fields are
        # required, and all others are optional.
        self.doc_id: str = entry["doc-id"]  # document this sentence came from
        self.entry_id: str = entry["entry-id"]
        self.lang: Optional[str] = entry.get("lang", "")
        if self.lang:
            self.lang = ISO_2_TO_ISO_1[self.lang]
        self.segment_text: str = entry.get("segment-text", "")
        self.segment_type: str = entry.get("segment-type", "")
        self.segment_sections: List[StructuralElement] = [
            StructuralElement.from_mitre(elt)
            for elt in entry.get("segment-sections", {})
        ]
        self.sent_id: str = entry.get("sent-id", "")
        self.annotation_sets = {
            key: AnnotationSet(
                key, val
            )  # the key indicates whether the annotations are for abstract events or basic events
            for key, val in entry.get("annotation-sets", {}).items()
        }

        # Convenient aliases
        self.abstract_event_set = self.annotation_sets.get(
            "abstract-events",
            AnnotationSet("abstract-events", {"events": {}, "span-sets": {}}),
        )
        self.basic_event_set = self.annotation_sets.get(
            "basic-events",
            AnnotationSet("basic-events", {"events": {}, "span-sets": {}}),
        )

    def __str__(self):
        return (
            f"doc-id: {self.doc_id}, entry-id: {self.entry_id},\nText: {self.segment_text}"
            + f"\ntype: {self.segment_type}, sent_id: {self.sent_id}\n"
            + "\n".join(str(e) for e in self.annotation_sets)
        )


class AnnotatedCorpus:
    """A corpus is a list of annotated texts, whether these be sentences or whole documents"""

    def __init__(self, json_data, target_segment_type):
        self.texts: List[AnnotatedText] = [
            AnnotatedText(val) for val in json_data["entries"].values()
            if val['segment-type'] == target_segment_type
        ]

    def return_texts_as_docs(self) -> Mapping[str, AnnotatedText]:
        result = defaultdict(list)
        # The entry-id from the .bp.json file should be unique
        # the doc-id there *might* be unique, but it is not a guarantee.
        for s in self.texts:
            result[s.entry_id] = s
        return dict(result)

    #
    # Methods for corpus analysis
    #

    def all_abstract_events(self) -> Iterator[AbstractEvent]:
        for sent in self.texts:
            for event in sent.abstract_event_set.events:
                yield event

    def num_sentences(self) -> int:
        return len(self.texts)

    def num_words(self) -> int:
        return sum(len(s.segment_text.split()) for s in self.texts)

    def num_abstract_events(self) -> int:
        return sum(len(s.abstract_event_set.events) for s in self.texts)

    def quad_value_counts(self) -> Mapping[str, int]:
        result = defaultdict(int)
        for e in self.all_abstract_events():
            quad = "{}:{}".format(e.helpful_harmful, e.material_verbal)
            result[quad] += 1
        return result

    def helpful_harmful_counts(self) -> Mapping[str, int]:
        result = defaultdict(int)
        for e in self.all_abstract_events():
            result[e.helpful_harmful] += 1
        return result

    def material_verbal_counts(self) -> Mapping[str, int]:
        result = defaultdict(int)
        for e in self.all_abstract_events():
            result[e.material_verbal] += 1
        return result

    def agent_patient_statuses(self) -> List[Tuple[bool, bool]]:
        result = []
        for e in self.all_abstract_events():
            status = [bool(e.agents), bool(e.patients)]
            result.append(status)
        return result

    def event_counts_by_sentence(self) -> List[int]:
        result = []
        for s in self.texts:
            result.append(len(s.abstract_event_set.events))
        return result
