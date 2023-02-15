from typing import List, Optional, Dict, Mapping

from attr import attrib, attrs, asdict


@attrs(frozen=True)
class SimpleDependencyEdge:
    source_label: str = attrib()
    label: str = attrib()
    target_label: str = attrib()

    @classmethod
    def from_json(cls, json_dict: Dict):
        # noinspection PyArgumentList
        return cls(
            json_dict["source_label"],
            json_dict["label"],
            json_dict["target_label"]
        )

    def to_dict(self) -> Mapping:
        return asdict(self)


class PropositionArgument:
    def __init__(self,
                 role: str,
                 prop: 'Proposition',
                 edge: SimpleDependencyEdge,
                 inferred: bool) -> None:
        self.role = role
        self.prop: 'Proposition' = prop
        # This will be a flexnlp DependencyEdge during building,
        # but it is finalized to a SimpleDependencyEdge at the end of the building stage
        self.original_edge: SimpleDependencyEdge = edge
        self.inferred: bool = inferred

    @classmethod
    def from_json(cls, json_dict: Dict):
        # noinspection PyArgumentList
        return cls(
            json_dict["role"],
            json_dict["prop_id"],  # NOTE: This must be eventually resolved!
            SimpleDependencyEdge.from_json(json_dict["original_edge"]),
            json_dict["inferred"]
        )

    def to_dict(self) -> Mapping:
        return {
            'role': self.role,
            'prop_id': self.prop.prop_id,
            'original_edge': self.original_edge.to_dict(),
            'inferred': self.inferred
        }

    SUBJECT = 'subj'
    DIRECT_OBJECT = 'dobj'
    INDIRECT_OBJECT = 'iobj'
    COMPLEMENT = 'comp'
    POSSESSIVE = 'poss'
    MODIFIER = 'mod'
    COREF = 'coref'
    TEMPORAL = 'temporal'
    LOCATION = 'location'
    PARENT = 'parent'
    CHILD = 'child'
    UNDETERMINED = 'undetermined'

    def as_inferred_subject(self):
        return PropositionArgument(PropositionArgument.SUBJECT, self.prop, self.original_edge, True)

    def with_new_role(self, role: str):
        return PropositionArgument(role, self.prop, self.original_edge, self.inferred)

    def with_new_edge(self, edge: SimpleDependencyEdge):
        return PropositionArgument(self.role, self.prop, edge, self.inferred)

    # Given a mapping of ids to propositions, replace a proposition ID with the real proposition
    def with_instantiated_prop(self, prop_mapping: Mapping[str, 'Proposition']):
        # noinspection PyTypeChecker
        real_prop = prop_mapping[self.prop]
        return PropositionArgument(self.role, real_prop, self.original_edge, self.inferred)


@attrs(frozen=True)
class PropositionModality:
    modality: str = attrib()
    text: str = attrib()
    _post_processed: bool = attrib(default=False)

    #FUTURE = 'future'
    POSSIBLE = 'possible'
    MUST = 'must'
    SHOULD = 'should'
    WOULD = 'would'
    COULD = 'could'
    CONDITIONAL = 'conditional'


    UNCONFIRMED = 'unconfirmed'
    HYPOTHETICAL = 'hypothetical'
    COUNTERFACTUAL = 'counterfactual'
    NEGATION = 'negation'
    DICENDI = 'dicendi'
    VOLITION = 'volition'
    FUTURE = 'future'
    EPISTEMIC = 'epistemic'
    DEONTIC = 'deontic'

    COMPUTED_MODALITIES = {UNCONFIRMED, HYPOTHETICAL, COUNTERFACTUAL, NEGATION, EPISTEMIC,
                           DICENDI, VOLITION, FUTURE, DEONTIC}
    VALID_MODALITIES = {POSSIBLE, MUST, SHOULD, WOULD, COULD, CONDITIONAL} | COMPUTED_MODALITIES



    @classmethod
    def from_json(cls, json_dict: Dict):
        return cls(
            json_dict['modality'],
            json_dict['text']
        )

    def to_dict(self) -> Mapping:
        return asdict(self, filter=lambda attr, value: not attr.name.startswith("_"))

    @modality.validator
    def validate_modality(self, attribute, value: str) -> None:
        if value not in PropositionModality.VALID_MODALITIES:
            print(f"WARNING: invalid modality: {value}")

    def is_postprocessed(self):
        return self._post_processed == True

@attrs(frozen=True)
class PropositionTense:
    tense: str = attrib()
    text: str = attrib()

    FUTURE = 'future'
    PAST = 'past'
    PRESENT = 'present'
    PERFECT = 'perfect'
    INFINTIVE = 'infinitive'

    VALID_TENSES = {FUTURE, PAST, PRESENT, PERFECT, INFINTIVE}

    @tense.validator
    def validate_modality(self, attribute, value: str) -> None:
        if value not in PropositionTense.VALID_TENSES:
            print(f"WARNING: invalid tense: {value}")

@attrs(frozen=True)
class PropositionPolarity:
    polarity: str = attrib()
    text: str = attrib()

    POSITIVE = 'positive'
    NEGATIVE = 'negative'

    VALID_POLARITIES = {POSITIVE, NEGATIVE}

    @polarity.validator
    def validate_modality(self, attribute, value: str) -> None:
        if value not in PropositionPolarity.VALID_POLARITIES:
            print(f"WARNING: invalid polarity: {value}")

    def is_negative(self):
        return self.polarity == PropositionPolarity.NEGATIVE



def convert_mention_type(mention_type: str) -> str:
    if mention_type in PropositionMention.VALID_MENTION_TYPES:
        return mention_type
    if mention_type in ['desc', 'descriptor']:
        return PropositionMention.NOMINAL
    return mention_type


@attrs(frozen=True)
class PropositionMention:
    # Note: start_token/end_token are SENTENCE-LEVEL; be careful of this during building
    start_token: int = attrib()
    end_token: int = attrib()
    mention_type: str = attrib(converter=convert_mention_type)
    entity_type: str = attrib()

    NAME = 'name'
    NOMINAL = 'nominal'
    PRONOUN = 'pronoun'
    VALID_MENTION_TYPES = [NAME, NOMINAL, PRONOUN]

    @classmethod
    def from_json(cls, json_dict: Dict):
        # noinspection PyArgumentList
        return cls(
            json_dict['start_token'],
            json_dict['end_token'],
            json_dict['mention_type'],
            json_dict['entity_type']
        )

    def to_dict(self) -> Mapping:
        return asdict(self)

    # noinspection PyUnusedLocal
    @mention_type.validator
    def validate_mention_type(self, attribute, value: str) -> None:
        if value and value not in PropositionMention.VALID_MENTION_TYPES:
            print(f"WARNING: invalid mention type: {value}")

    PER = 'PER'
    ORG = 'ORG'
    GPE = 'GPE'
    LOC = 'LOC'
    FAC = 'FAC'
    VEH = 'VEH'
    WEA = 'WEA'
    DISEASE = 'DISEASE'
    VALID_ENTITY_TYPES = [PER, ORG, GPE, LOC, FAC, VEH, WEA, DISEASE]

    # noinspection PyUnusedLocal
    @entity_type.validator
    def validate_entity_type(self, attribute, value: str) -> None:
        if value and value not in PropositionMention.VALID_ENTITY_TYPES:
            print(f"WARNING: invalid entity type: {value}")


class Proposition:
    def __init__(self, prop_id: str, prop_type: str, token_text: str,
                 token_index: int, pos_tag: str,
                 arguments: List[PropositionArgument], particles: List[str],
                 modalities: List[PropositionModality],
                 mention: Optional[PropositionMention]) -> None:
        self.prop_id: str = prop_id
        self.prop_type: str = prop_type
        self.token_text: str = token_text
        # Note: token_index is a document-level index during building, and then it is finalized to
        # sentence-level at the end of the building process
        self.token_index: int = token_index
        self.pos_tag: str = pos_tag
        self.arguments: List[PropositionArgument] = arguments
        self.particles: List[str] = particles
        self.modalities: List[PropositionModality] = modalities
        self.tense: List[PropositionTense] = []
        self.polarity: PropositionPolarity = None
        self.mention: Optional[PropositionMention] = mention

    VERB = 'verb'
    NOUN = 'noun'
    MODIFIER = 'mod'
    PARTITIVE = 'partitive'
    CONJUNCTION = 'conj'
    PART_WHOLE = 'part_whole'

    @classmethod
    def from_json(cls, json_dict: Dict):
        # noinspection PyArgumentList
        return cls(
            json_dict['prop_id'],
            json_dict['prop_type'],
            json_dict['token_text'],
            json_dict['token_index'],
            json_dict['pos_tag'],
            [PropositionArgument.from_json(x) for x in json_dict['arguments']],
            json_dict['particles'],
            [PropositionModality.from_json(x) for x in json_dict['modalities']],
            PropositionMention.from_json(json_dict['mention']) if 'mention' in json_dict else None
        )

    def to_dict(self) -> Mapping:
        result = {
            'prop_id': self.prop_id,
            'prop_type': self.prop_type,
            'token_text': self.token_text,
            'token_index': self.token_index,
            'pos_tag': self.pos_tag,
            'arguments': [a.to_dict() for a in self.arguments],
            'particles': self.particles,
            'modalities': [m.to_dict() for m in self.modalities],
        }
        if self.mention:
            result['mention'] = self.mention.to_dict()
        return result

    def add_negation(self, word: str):
        self.add_polarity(PropositionPolarity.NEGATIVE, word)

    def add_modal(self, modal_type: str, word: str):
        self.modalities.append(PropositionModality(modal_type, word))

    def postprocess_modal(self, modal_type: str, word: str):
        self.modalities.append(PropositionModality(modal_type, word, post_processed=True))

    def remove_preprocessed_modalities(self):
        self.modalities = [modality for modality in self.modalities if modality.is_postprocessed()]

    def add_tense(self, tense_type: str, word: str):
        self.tense.append(PropositionTense(tense_type, word))

    def add_polarity(self, polarity_type: str, word: str):
        self.polarity = PropositionPolarity(polarity_type, word)

    # Convenience method
    def get_subject_args(self) -> List[PropositionArgument]:
        return [a for a in self.arguments if a.role == PropositionArgument.SUBJECT]

    def pprint(self, indent=0) -> str:
        indent_str = " " * indent
        particle_str = ""
        if self.particles:
            particle_str = " [" + " ".join(self.particles) + "]"
        mention_str = ""
        if self.mention:
            mention_str = f" [{self.mention.entity_type}.{self.mention.mention_type}]"

        prop_text = self.token_text
        is_name = False
        if self.mention and self.mention.mention_type == PropositionMention.NAME:
            is_name = True
            compound_toks = [(a.prop.token_index, a.prop.token_text)
                             for a in self.arguments
                             if a.original_edge.label == 'compound'
                             and a.prop.mention
                             and a.prop.mention.mention_type == PropositionMention.NAME]
            compound_toks.append((self.token_index, self.token_text))
            prop_text = " ".join([x[1] for x in sorted(compound_toks)])

        result = f"{prop_text} ({self.prop_type}){particle_str}{mention_str}\n"
        for m in self.modalities:
            result += f"  {indent_str}modality-{m.modality}: {m.text}\n"
        args_to_print = []
        for a in self.arguments:
            if is_name and a.original_edge.label == 'compound' \
                    and a.prop.mention \
                    and a.prop.mention.mention_type == PropositionMention.NAME:
                continue
            inferred_str = " (INFERRED)" if a.inferred else ""
            arg_to_print = f"  {indent_str}{a.role}{inferred_str}: "
            arg_to_print += a.prop.pprint(indent + 2 + len(a.role))
            args_to_print.append(arg_to_print)
        # Make print order more deterministic
        for a in sorted(args_to_print):
            result += a
        return result
