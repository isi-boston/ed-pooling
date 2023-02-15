from typing import List, Set


# General purpose helper
def _validate_string(value: str, valid_strings: List[str], category_name: str) -> None:
    if value not in valid_strings:
        print(f"WARNING: invalid {category_name}: {value}")


#
# IMPORTANT: If you add a new validation list to this file, you must
# add it to the VALIDATION_LISTS at the bottom, so we ensure it
# never has duplicate entries.
#
# The alternative is to use OrderedDict or something, but for now this
# is what we are doing, since lists are easier for other components to use.
#
# Validators are in general intended for use with attrs so must take
# instance, attribute, value as parameters. instance/attribute are
# typically unused here.
#


VALID_STRUCTURAL_TYPES = [
    "Headline",
    "Dateline",
    "Byline",
    "Story-Lead",
    # underscore instead of hyphen in Section_Header is intentional
    "Section_Header",
    "Sentence",
    "Ignore"
]
STRUCTURAL_SENTENCE = "Sentence"


# noinspection PyUnusedLocal
def validate_structural_type(instance, attribute, value: str) -> None:
    _validate_string(value, VALID_STRUCTURAL_TYPES, "structural element type")


VALID_ABSTRACT_EVENT_TYPES = ['abstract']


# noinspection PyUnusedLocal
def validate_abstract_event_type(instance, attribute, value: str) -> None:
    _validate_string(value, VALID_ABSTRACT_EVENT_TYPES, "abstract event type")


VALID_ABSTRACT_ARG_ROLES = ['agent', 'patient']


# noinspection PyUnusedLocal
def validate_abstract_arg_role(instance, attribute, value: str) -> None:
    _validate_string(value, VALID_ABSTRACT_ARG_ROLES, "abstract argument role")


VALID_BASIC_ARG_ROLES = ['agent', 'patient', 'money', 'ref-event']


# noinspection PyUnusedLocal
def validate_basic_arg_role(instance, attribute, value: str) -> None:
    _validate_string(value, VALID_BASIC_ARG_ROLES, "basic argument role")


VALID_ACE_EVENT_TYPES = [
    "Business.Declare-Bankruptcy",
    "Business.Merge-Org",
    "Business.Start-Org",
    "Business.End-Org",
    "Conflict.Attack",
    "Conflict.Demonstrate",
    "Contact.Phone-Write",
    "Contact.Meet",
    "Justice.Charge-Indict",
    "Justice.Release-Parole",
    "Justice.Appeal",
    "Justice.Pardon",
    "Justice.Extradite",
    "Justice.Acquit",
    "Justice.Arrest-Jail",
    "Justice.Trial-Hearing",
    "Justice.Sentence",
    "Justice.Execute",
    "Justice.Fine",
    "Justice.Convict",
    "Justice.Sue",
    "Life.Injure",
    "Life.Marry",
    "Life.Divorce",
    "Life.Be-Born",
    "Life.Die",
    "Movement.Transport",
    "Personnel.Start-Position",
    "Personnel.End-Position",
    "Personnel.Nominate",
    "Personnel.Elect",
    "Transaction.Transfer-Ownership",
    "Transaction.Transfer-Money",
]


# noinspection PyUnusedLocal
def validate_ace_event_type(instance, attribute, value: str) -> None:
    _validate_string(value, VALID_ACE_EVENT_TYPES, "ace event type")


VALID_ACE_ENTITY_TYPES = [
    "PER",
    "ORG",
    "GPE",
    "LOC",
    "FAC",
    "VEH",
    "WEA",
    "VAL",
    "TME"
]


def validate_ace_entity_type(instance, attribute, value: str) -> None:
    _validate_string(value, VALID_ACE_ENTITY_TYPES, "ace entity type")

VALID_BASIC_EVENT_TYPES_PHASE_1 = [
    "Communicate-Event",
    "Coordinated-Comm",
    "Conduct-Protest",
    "Conduct-Violent-Protest",
    "Organize-Protest",
    "Suppress-Communication",
    "Suppress-or-Breakup-Protest",
    "Corruption",
    "Bribery",
    "Extortion",
    "Financial-Crime",
    "Violence",
    "Conspiracy",
    "Coup",
    "Suppression-of-Free-Speech",
    "Persecution",
    "Illegal-Entry",
    "Other-Crime",
    "Violence-Attack",
    "Violence-Bombing",
    "Violence-Set-Fire",
    "Violence-Kill",
    "Violence-Wound",
    "Violence-Damage",
    "Violence-Other",
    "Change-of-Govt",
    "Military-Attack",
    "Military-Other",
    "Political-Election-Event",
    "Political-Other",
    "Fiscal-or-Monetary-Action",
    "Other-Government-Action",
    "Law-Enforcement-Investigate",
    "Law-Enforcement-Arrest",
    "Law-Enforcement-Extradite",
    "Law-Enforcement-Other",
    "Judicial-Indict",
    "Judicial-Prosecute",
    "Judicial-Convict",
    "Judicial-Sentence",
    "Judicial-Acquit",
    "Judicial-Seize",
    "Judicial-Plead",
    "Judicial-Other",
    "Conduct-Meeting",
    "Leave-Job",
    "Economic-Event-or-SoA",
    "Environmental-Event-or-SoA",
    "Business-Event-or-SoA",
    "Political-Event-or-SoA",
    "War-Event-or-SoA",
    "Declare-Emergency",
    "Monitor-Disease",
    "Restrict-Travel",
    "Impose-Quarantine",
    "Close-Schools",
    "Restrict-Business",
    "Cull-Livestock",
    "Apply-NPI",
    "Hospitalize",
    "Vaccinate",
    "Test-Patient",
    "Disease-Outbreak",
    "Disease-Infects",
    "Disease-Exposes",
    "Disease-Kills",
    "Disease-Recovery",
    "Kidnapping",
    "Require-PPE",
]

BASIC_EVENT_TYPES_PHASE_2_ONLY = [
    "Conduct-Diplomatic-Talks",
    "Natural-Phenomenon-Event-or-SoA",
    "Lift-Quarantine",
    "Loosen-Business-Restrictions",
    "Loosen-Travel-Restrictions",
    "Open-Schools",
    "Treat-Patient",
    "Conduct-Medical-Research",
    "Aid-Needs",
    "Evacuate",
    "Expel",
    "Provide-Aid",
    "Repair",
    "Rescue",
    "Migrant-Detain",
    "Migrant-Relocation",
    "Migrant-Smuggling",
    "Migration-Blocked",
    "Refugee-Movement",
    "Death-from-Crisis-Event",
    "Missing-from-Crisis-Event",
    "Wounding-from-Crisis-Event",
    "Weather-or-Environmental-Damage",
]

VALID_BASIC_EVENT_TYPES_PHASE_2 = \
    VALID_BASIC_EVENT_TYPES_PHASE_1 + BASIC_EVENT_TYPES_PHASE_2_ONLY

BASIC_EVENT_TYPES_PHASE_3_ONLY = [
    "Establish-Project",
    "Propose-Project",
    "Sign-Agreement",
    "Award-Contract",
    "Fund-Project",
    "Make-Repayment",
    "Change-Repayment",
    "Employ-Workers",
    "Dismiss-Workers",
    "Construct-Project",
    "Interrupt-Construction",
    "Infrastructure-Operation",

    "Cyber-Crime-Attack",
    "Information-Theft",
    "Information-Release",
    "Interrupt-Operations",
    "Cyber-Crime-Other",

    "Financial-Loss",
    "Pay-Ransom",
    "Cybersecurity-Measure",
    "Identify-Vulnerability",

    "Legislative-Action",
]

VALID_BASIC_EVENT_TYPES_PHASE_3 = \
    VALID_BASIC_EVENT_TYPES_PHASE_2 + BASIC_EVENT_TYPES_PHASE_3_ONLY

VALID_BASIC_EVENT_TYPES = VALID_BASIC_EVENT_TYPES_PHASE_3

EXCLUDED_BASIC_EVENT_TYPES = [
    "Suppress-Meeting", ## ignored via scorer update v1.4.5
    "Military-Declare-War", # ignored via "only event types present
    # in the training data (ALL Training data released so far) will be scored."
    "Famine-Event-or-SoA", ## ignored via scorer update v1.4.5
    "Migration-Impeded-Failed", # ignored via "only event types present
    # in the training data (ALL Training data released so far) will be scored."
    "Weather-Event-or-SoA", ## ignored via scorer update v1.4.5
    "Distribute-PPE", ## ignored via scorer update v1.4.5
    "Lift-PPE-Requirements", ## ignored via scorer update v1.4.5
    "Natural-Disaster-Event-or-SoA", ## ignored via scorer update v1.4.5
]


# noinspection PyUnusedLocal
def validate_basic_event_type(instance, attribute, value: str) -> None:
    _validate_string(value, VALID_BASIC_EVENT_TYPES, "basic event type")


VALID_GRANULAR_TEMPLATE_TYPES = [
    "Protestplate",
    "Corruplate",
    "Terrorplate",
    "Epidemiplate",
    "Disasterplate",
    "Displacementplate",
]


# noinspection PyUnusedLocal
def validate_granular_template_type(instance, attribute, value: str) -> None:
    _validate_string(value, VALID_GRANULAR_TEMPLATE_TYPES, "granular template type")


VALID_GRANULAR_ARG_PROPERTIES = [
    "over-time",
    "completion",
    "coordinated",
    "type"
]


# noinspection PyUnusedLocal
def validate_granular_arg_property(instance, attribute, value: str) -> None:
    _validate_string(value, VALID_GRANULAR_ARG_PROPERTIES, "granular argument property")


GUESSED_ANCHOR_ROLE = "guessed_anchor"
GUESSED_SUPERSET_ROLE = "guessed_superset"
GUESSED_SHARED_ARG_ROLE = "guessed_shared_arg"
GUESSED_CORE_ROLE = "guessed_core"
SUPPLEMENTARY_ROLE = 'supplementary'

def is_guessed_event_argument_role(value: str) -> bool:
    return value.startswith('guessed')

VALID_GRANULAR_ARG_ROLES = [

    # ISI only
    GUESSED_ANCHOR_ROLE,
    GUESSED_SUPERSET_ROLE,
    GUESSED_SHARED_ARG_ROLE,
    GUESSED_CORE_ROLE,
    SUPPLEMENTARY_ROLE,

    # shared
    "who",
    "where",
    "when",
    "outcome-occurred",
    "outcome-averted",
    "outcome-hypothetical",
    "wounded",
    "killed",
    "injured-count",
    "killed-count",
    "missing-count",
    "outcome",

    # protestplate
    "organizer",
    "occupy",
    "arrested",
    "imprisoned",
    "protest-for",
    "protest-against",
    "protest-event",

    # corruplate
    "job",
    "charged-with",
    "judicial-actions",
    "prison-term",
    "fine",
    "corrupt-event",

    # terrorplate
    "named-perp",
    "named-perp-org",
    "named-organizer",
    "claimed-by",
    "blamed-by",
    "target-physical",
    "target-human",
    "weapon",
    "kidnapped",
    "perp-captured",
    "perp-wounded",
    "perp-killed",
    "perp-objective",
    "terror-event",

    # epidemiplate
    "disease",
    "NPI-Events",
    "infected-individuals",
    "infected-count",
    "infected-cumulative",
    "killed-individuals",
    # "killed-count", (also used in other templates)
    "killed-cumulative",
    "exposed-individuals",
    "exposed-count",
    "exposed-cumulative",
    "tested-individuals",
    "tested-count",
    "tested-cumulative",
    "vaccinated-individuals",
    "vaccinated-count",
    "vaccinated-cumulative",
    "recovered-individuals",
    "recovered-count",
    "recovered-cumulative",
    "outbreak-event",
    "hospitalized-count",
    # This has to be fixed! MITRE is inconsistent
    "hospitalized-individual",
    "hospitalized-individuals",
    "hospitalized-cumulative",

    # Disasterplate
    "affected-cumulative-count",
    "announce-disaster-warnings",
    "assistance-needed",
    "assistance-provided",
    "damage",
    "declare-emergency",
    "disease-outbreak-events",
    "human-displacement-events",
    "individuals-affected",
    "major-disaster-event",
    "rescue-events",
    "rescued-count",
    "related-natural-phenomena",
    "repair",
    "responders",

    # Displacementplate
    "Assistance-provided",
    "Assistance-needed",
    "blocked-migration-count",
    "current-location",
    "destination",
    "detained-count",
    "event-or-SoA-at-origin",
    "group-identity",
    "human-displacement-event",
    "origin",
    "settlement-status-event-or-SoA",
    "total-displaced-count",
    "Transitory-events",
    "transiting-location",
]

VALID_ACE_ARG_ROLES = [
    "Adjudicator",
    "Agent",
    "Artifact",
    "Attacker",
    "Beneficiary",
    "Buyer",
    "Crime",
    "Defendant",
    "Destination",
    "Entity",
    "Giver",
    "Instrument",
    "Money",
    "Org",
    "Origin",
    "Person",
    "Place",
    "Plaintiff",
    "Position",
    "Price",
    "Prosecutor",
    "Recipient",
    "Seller",
    "Sentence",
    "Target",
    "Time-After",
    "Time-At-Beginning",
    "Time-At-End",
    "Time-Before",
    "Time-Ending",
    "Time-Holds",
    "Time-Starting",
    "Time-Within",
    "Vehicle",
    "Victim",
]

ACE_RELATION_ARG1_ROLE = "Arg-1"
ACE_RELATION_ARG2_ROLE = "Arg-2"
VALID_ACE_RELATION_ARG_ROLES = [
    ACE_RELATION_ARG1_ROLE,
    ACE_RELATION_ARG2_ROLE
]

ACE_RELATION_ART = "ART"
ACE_RELATION_GEN_AFF = "GEN-AFF"
ACE_RELATION_ORG_AFF = "ORG-AFF"
ACE_RELATION_PART_WHOLE = "PART-WHOLE"
ACE_RELATION_PER_SOC = "PER-SOC"
ACE_RELATION_PHYS = "PHYS"
VALID_ACE_RELATION_TYPES = [
    ACE_RELATION_ART,
    ACE_RELATION_GEN_AFF,
    ACE_RELATION_ORG_AFF,
    ACE_RELATION_PART_WHOLE,
    ACE_RELATION_PER_SOC,
    ACE_RELATION_PHYS
]

# noinspection PyUnusedLocal
def validate_ace_arg_role(instance, attribute, value: str) -> None:
    _validate_string(value, VALID_ACE_ARG_ROLES, "ACE argument role")


# noinspection PyUnusedLocal
def validate_granular_arg_role(instance, attribute, value: str) -> None:
    _validate_string(value, VALID_GRANULAR_ARG_ROLES, "granular argument role")


VALID_GRANULAR_COMPONENT_EVENT_ARGUMENT_ROLES = {
    'Corruplate': 'corrupt-event',
    'Epidemiplate': 'outbreak-event',
    'Protestplate': 'protest-event',
    'Terrorplate': 'terror-event',
    'Disasterplate': 'major-disaster-event',
    'Displacementplate': 'human-displacement-event',
}


def get_core_granular_role_for_template_type(granular_event_type: str):
    if granular_event_type not in VALID_GRANULAR_COMPONENT_EVENT_ARGUMENT_ROLES:
        ValueError(f"Unknown granular event type: {granular_event_type}")
    return VALID_GRANULAR_COMPONENT_EVENT_ARGUMENT_ROLES[granular_event_type]


for v in VALID_GRANULAR_COMPONENT_EVENT_ARGUMENT_ROLES.values():
    if v not in VALID_GRANULAR_ARG_ROLES:
        raise ValueError("Mistake in VALID_GRANULAR_COMPONENT_EVENT_ARGUMENT_ROLES:", v)


def is_granular_component_event_arg(value: str) -> bool:
    return value in VALID_GRANULAR_COMPONENT_EVENT_ARGUMENT_ROLES.values()


def get_granular_component_event_types(granular_event_type: str) -> Set[str]:
    if granular_event_type == "Protestplate":
        return {'Conduct-Protest', 'Conduct-Violent-Protest',
                'Organize-Protest', 'Suppress-or-Breakup-Protest',
                'Violence-Kill', 'Violence-Wound', 'Law-Enforcement-Arrest'}
    elif granular_event_type == "Corruplate":
        return {'Corruption', 'Bribery', 'Extortion', 'Financial-Crime', 'Other-Crime'}
    elif granular_event_type == "Terrorplate":
        return {'Violence-Bombing', 'Violence-Set-Fire', 'Violence-Kill', 'Violence-Attack',
                'Violence-Wound', 'Violence-Damage', 'Violence-Other', 'Kidnapping',
                'Violence'}
    elif granular_event_type == "Epidemiplate":
        return {'Disease-Outbreak', 'Disease-Infects', 'Disease-Exposes',
                'Disease-Kills', 'Disease-Recovery', 'Test-Patient', 'Vaccinate',
                'Hospitalize', 'Apply-NPI', 'Cull-Livestock', 'Require-PPE',
                'Impose-Quarantine', 'Hospitalize', 'Restrict-Travel',
                'Monitor-Disease',
                'Restrict-Business', 'Close-Schools'}
    elif granular_event_type == "Disasterplate":
        print("Need to specify granular component event types for", granular_event_type)
        return {'Natural-Phenomenon-Event-or-SoA'}
    elif granular_event_type == "Displacementplate":
        print("Need to specify granular component event types for", granular_event_type)
        return {'Refugee-Movement', 'Migrant-Detain', 'Migrant-Relocation', 'Migrant-Smuggling'}
    else:
        raise ValueError(f"Unknown granular event type: {granular_event_type}")


def get_granular_core_event_types(granular_event_type: str) -> Set[str]:
    if granular_event_type == "Protestplate":
        return {'Conduct-Protest', 'Conduct-Violent-Protest'}
    elif granular_event_type == "Corruplate":
        return {'Corruption', 'Bribery', 'Extortion', 'Financial-Crime'}
    elif granular_event_type == "Terrorplate":
        return {'Violence-Bombing', 'Violence-Set-Fire', 'Violence-Kill', 'Violence-Attack',
                'Violence-Wound', 'Violence-Damage', 'Violence-Other', 'Kidnapping',
                'Violence'}
    elif granular_event_type == "Epidemiplate":
        return {'Disease-Outbreak', 'Disease-Infects', 'Disease-Kills'}
    elif granular_event_type == "Disasterplate":
        return {'Natural-Phenomenon-Event-or-SoA'}
    elif granular_event_type == "Displacementplate":
        return {'Refugee-Movement'}
    else:
        raise ValueError(f"Unknown granular event type: {granular_event_type}")


VALID_TERROR_SUBTYPES = [
    "arson",
    "assault",
    "bombing",
    "kidnapping",
    "murder",
    "unspecified",
]


# noinspection PyUnusedLocal
def validate_terror_subtype(instance, attribute, value: str) -> None:
    _validate_string(value, VALID_TERROR_SUBTYPES, "terror subtype")


VALID_IRREALIS = [
    "counterfactual",
    "hypothetical",
    "future",
    "unconfirmed",
    "unspecified",
    "non-occurrence",
]


# noinspection PyUnusedLocal
def validate_irrealis(instance, attribute, value: str) -> None:
    if value:
        _validate_string(value, VALID_IRREALIS, "irrealis")


NAME = 'name'
NOMINAL = 'nominal'
PRONOUN = 'pronoun'
EVENT_ANCHOR = 'event-anchor'
TIME_MENTION = 'time-mention'
DURATION_MENTION = 'duration-mention'
COREF_MENTION = 'coref-mention'
LIST = 'list'
TEMPLATE_ANCHOR = 'template-anchor'
# value mention only occurs in ACE
VALUE_MENTION = 'value-mention'
VALID_MENTION_TYPES = [NAME, NOMINAL, PRONOUN, EVENT_ANCHOR, TIME_MENTION,
                       DURATION_MENTION, LIST, TEMPLATE_ANCHOR,
                       VALUE_MENTION, COREF_MENTION]


# noinspection PyUnusedLocal
def validate_mention_type(instance, attribute, value: str) -> None:
    # We allow null mentions types, at least for now
    if value:
        _validate_string(value, VALID_MENTION_TYPES, "mention type")


VALID_EVENT_TYPES = VALID_ABSTRACT_EVENT_TYPES + \
                    VALID_BASIC_EVENT_TYPES + \
                    VALID_GRANULAR_TEMPLATE_TYPES + \
                    VALID_ACE_EVENT_TYPES + \
                    VALID_ACE_RELATION_TYPES


# noinspection PyUnusedLocal
def validate_event_type(instance, attribute, value: str) -> None:
    _validate_string(value, VALID_EVENT_TYPES, "event type")


VALID_ARGUMENT_ROLES = VALID_ABSTRACT_ARG_ROLES + \
                       VALID_BASIC_ARG_ROLES + \
                       VALID_GRANULAR_ARG_ROLES + \
                       VALID_ACE_ARG_ROLES + \
                       VALID_ACE_RELATION_ARG_ROLES


# noinspection PyUnusedLocal
def validate_argument_role(instance, attribute, value: str) -> None:
    _validate_string(value, VALID_ARGUMENT_ROLES, "argument role")


# Validate the validators (to avoid inadvertent repetition)
VALIDATION_LISTS = [
    VALID_ABSTRACT_ARG_ROLES,
    VALID_ABSTRACT_EVENT_TYPES,
    VALID_ACE_EVENT_TYPES,
    VALID_BASIC_ARG_ROLES,
    VALID_BASIC_EVENT_TYPES,
    VALID_GRANULAR_ARG_PROPERTIES,
    VALID_GRANULAR_ARG_ROLES,
    VALID_GRANULAR_TEMPLATE_TYPES,
    VALID_IRREALIS,
    VALID_MENTION_TYPES,
    VALID_STRUCTURAL_TYPES,
    VALID_TERROR_SUBTYPES,
    VALID_ACE_ENTITY_TYPES
]

for vl in VALIDATION_LISTS:
    if len(vl) != len(set(vl)):
        repeated_elts = set()
        for elt in set(vl):
            if vl.count(elt) > 1:
                repeated_elts.add(elt)
        raise ValueError("Validation list has repetition: {}".format(repeated_elts))


# GEONAMES
GEONAMES_ID = 'geonames_id'
COUNTRY_CODE = 'country_code'
IS_COUNTRY = 'is_country'
