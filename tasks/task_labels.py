from typing import Dict, List

from better_events.better_validation import (
    VALID_BASIC_EVENT_TYPES_PHASE_1,
    BASIC_EVENT_TYPES_PHASE_2_ONLY,
    BASIC_EVENT_TYPES_PHASE_3_ONLY,
    VALID_ACE_EVENT_TYPES
)
from better_events.better_core import ABSTRACT_EVENT_TYPE


def convert_ids_to_labels(
        label_map: Dict[int, str],
        ids: List[List[int]]
) -> List[List[str]]:

    return [[label_map[idx] for idx in l] for l in ids]


def get_bio_labels(labels: List[str]) -> List[str]:
    bio_labels = []
    for l in labels:
        bio_labels.append(f'B-{l}')
        bio_labels.append(f'I-{l}')
    bio_labels.append('O')
    return bio_labels


BetterBasicTriggerLabels = VALID_BASIC_EVENT_TYPES_PHASE_1 \
                           + BASIC_EVENT_TYPES_PHASE_2_ONLY \
                           + BASIC_EVENT_TYPES_PHASE_3_ONLY

BetterAbstractTriggerLabelsWithQuad = [
    'harmful+verbal',
    'harmful+material',
    'harmful+both',
    'harmful+unk',
    'helpful+verbal',
    'helpful+material',
    'helpful+both',
    'helpful+unk',
    'neutral+verbal',
    'neutral+material',
    'neutral+both',
    'neutral+unk'
]

BetterAbstractTriggerLabels = [
    ABSTRACT_EVENT_TYPE,
]

AceTriggerLabels = VALID_ACE_EVENT_TYPES

MinionLabels = [
    'Business:START-ORG',
    'Conflict:Attack',
    'Conflict:Demonstrate',
    'Contact:Meet',
    'Contact:Phone-Write',
    'Justice:Arrest-Jail',
    'Life:Be-Born',
    'Life:Die',
    'Life:Divorce',
    'Life:Injure',
    'Life:Marry',
    'Movement:Transport',
    'Personnel:End-Position',
    'Personnel:Start-Position',
    'Transaction:Transfer-Money',
    'Transaction:Transfer-Ownership'
]
