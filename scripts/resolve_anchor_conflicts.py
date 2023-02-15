import argparse
import json
import unicodedata
from collections import defaultdict
from typing import List

from better_events.better_core import (
    BetterEvent,
    BetterSpanSet,
    GroundedSpan,
    SimpleGroundedSpan,
    BetterSpan,
    ScoredBetterSpan,
    BetterDocument,
    BetterSentence
)
from better_events.better_mapper import BetterMapper
from better_events.better_utilities import load_documents


# From P1 and P2 and P3 training data, using count_event_types_from_mitre.py
EVENT_TYPE_COUNTS_BETTER = {
    "Aid-Needs": 50,
    "Apply-NPI": 189,
    "Award-Contract": 25,
    "Bribery": 40,
    "Business-Event-or-SoA": 255,
    "Change-Repayment": 6,
    "Change-of-Govt": 95,
    "Close-Schools": 7,
    "Communicate-Event": 3391,
    "Conduct-Diplomatic-Talks": 36,
    "Conduct-Medical-Research": 64,
    "Conduct-Meeting": 137,
    "Conduct-Protest": 350,
    "Conduct-Violent-Protest": 28,
    "Conspiracy": 9,
    "Construct-Project": 260,
    "Coordinated-Comm": 13,
    "Corruption": 89,
    "Coup": 34,
    "Cull-Livestock": 2,
    "Cyber-Crime-Attack": 532,
    "Cyber-Crime-Other": 25,
    "Cybersecurity-Measure": 177,
    "Death-from-Crisis-Event": 137,
    "Declare-Emergency": 17,
    "Disease-Exposes": 10,
    "Disease-Infects": 290,
    "Disease-Kills": 169,
    "Disease-Outbreak": 263,
    "Disease-Recovery": 35,
    "Dismiss-Workers": 3,
    "Distribute-PPE": 5,
    "Economic-Event-or-SoA": 371,
    "Employ-Workers": 25,
    "Environmental-Event-or-SoA": 44,
    "Evacuate": 40,
    "Expel": 36,
    "Extortion": 33,
    "Famine-Event-or-SoA": 1,
    "Financial-Crime": 132,
    "Financial-Loss": 14,
    "Fiscal-or-Monetary-Action": 120,
    "Fund-Project": 84,
    "Hospitalize": 45,
    "Identify-Vulnerability": 21,
    "Illegal-Entry": 43,
    "Impose-Quarantine": 44,
    "Information-Release": 27,
    "Information-Theft": 103,
    "Infrastructure-Operation": 31,
    "Interrupt-Construction": 21,
    "Interrupt-Operations": 34,
    "Judicial-Acquit": 9,
    "Judicial-Convict": 35,
    "Judicial-Indict": 116,
    "Judicial-Other": 95,
    "Judicial-Plead": 15,
    "Judicial-Prosecute": 81,
    "Judicial-Seize": 12,
    "Judicial-Sentence": 95,
    "Kidnapping": 31,
    "Law-Enforcement-Arrest": 107,
    "Law-Enforcement-Extradite": 5,
    "Law-Enforcement-Investigate": 122,
    "Law-Enforcement-Other": 117,
    "Leave-Job": 27,
    "Legislative-Action": 13,
    "Lift-Quarantine": 5,
    "Loosen-Business-Restrictions": 7,
    "Loosen-Travel-Restrictions": 14,
    "Make-Repayment": 5,
    "Migrant-Detain": 54,
    "Migrant-Relocation": 74,
    "Migrant-Smuggling": 24,
    "Migration-Blocked": 23,
    "Migration-Impeded-Failed": 9,
    "Military-Attack": 82,
    "Military-Other": 96,
    "Missing-from-Crisis-Event": 27,
    "Monitor-Disease": 78,
    "Natural-Phenomenon-Event-or-SoA": 509,
    "Open-Schools": 5,
    "Organize-Protest": 26,
    "Other-Crime": 173,
    "Other-Government-Action": 633,
    "Pay-Ransom": 7,
    "Persecution": 16,
    "Political-Election-Event": 73,
    "Political-Event-or-SoA": 258,
    "Political-Other": 67,
    "Provide-Aid": 175,
    "Refugee-Movement": 316,
    "Repair": 12,
    "Require-PPE": 8,
    "Rescue": 39,
    "Restrict-Business": 20,
    "Restrict-Travel": 39,
    "Sign-Agreement": 52,
    "Suppress-Communication": 12,
    "Suppress-or-Breakup-Protest": 29,
    "Suppression-of-Free-Speech": 5,
    "Test-Patient": 121,
    "Treat-Patient": 73,
    "Vaccinate": 43,
    "Violence": 66,
    "Violence-Attack": 162,
    "Violence-Bombing": 75,
    "Violence-Damage": 17,
    "Violence-Kill": 169,
    "Violence-Other": 65,
    "Violence-Set-Fire": 6,
    "Violence-Wound": 50,
    "War-Event-or-SoA": 55,
    "Weather-or-Environmental-Damage": 165,
    "Wounding-from-Crisis-Event": 36
}

EVENT_TYPE_COUNTS_MINION = {
    "Business:START-ORG": 294,
    "Conflict:Attack": 2196,
    "Conflict:Demonstrate": 140,
    "Contact:Meet": 317,
    "Contact:Phone-Write": 243,
    "Justice:Arrest-Jail": 167,
    "Life:Be-Born": 3124,
    "Life:Die": 1300,
    "Life:Divorce": 40,
    "Life:Injure": 419,
    "Life:Marry": 409,
    "Movement:Transport": 2272,
    "Personnel:End-Position": 912,
    "Personnel:Start-Position": 1244,
    "Transaction:Transfer-Money": 407,
    "Transaction:Transfer-Ownership": 705
}

EVENT_TYPE_COUNTS_PER_TASK = {
    'better': EVENT_TYPE_COUNTS_BETTER, 'minion': EVENT_TYPE_COUNTS_MINION
}


def is_breaking_char(c):
    return c.isspace() or unicodedata.category(c).startswith("P")


def make_new_span_set(doc: BetterDocument, sent: BetterSentence, score: float, start: int, end: int) -> BetterSpanSet:
    new_sgs = SimpleGroundedSpan(doc.doc_text, start, end, None, None)
    new_gs = GroundedSpan(sent.sent_id, new_sgs, new_sgs, None)
    new_bs = BetterSpan(new_sgs.text, new_sgs.text, new_gs)
    new_sbs = ScoredBetterSpan(new_bs, score)
    return BetterSpanSet([new_sbs])


# Identify the "best" event of several options
# Currently we take the one with the event type that occurs most frequently in the training set
# (as hard-coded in the EVENT_TYPE_COUNTS map). Tiebreaker is alphabetical.
def get_best_event(events: List[BetterEvent], task) -> BetterEvent:

    # Just return for trivial case
    if len(events) == 1:
        return events[0]

    # Grab event counts, use -1 to enforce reverse sorting for count but not for name
    event_types = sorted([(-1*EVENT_TYPE_COUNTS_PER_TASK[task].get(e.event_type, 0), e.event_type) for e in events])
    best_event_type = event_types[0][1]

    for e in events:
        if e.event_type == best_event_type:
            # if len(set(event_types)) > 1:
            #     print("Choosing", best_event_type, "from", event_types)
            return e

    # This will never happen, but let's just have it anyway
    return events[0]


def events_overlap(e1: BetterEvent, e2: BetterEvent) -> bool:
    # After I wrote this, I decided probably we'd always have one anchor per event, but better safe than sorry
    for a in e1.anchors:
        for b in e2.anchors:
            if a.grounded_span.full_span.overlaps(b.grounded_span.full_span):
                return True
    return False


def merge_overlapping_clusters(
        doc: BetterDocument, sent: BetterSentence,
        event_list: List[BetterEvent],
        task: str
    ) -> List[BetterEvent]:
    overlapping_clusters: List[List[BetterEvent]] = []
    for event in event_list:
        found_overlap = False
        for oc in overlapping_clusters:
            for other_event in oc:
                if events_overlap(event, other_event):
                    oc.append(event)
                    found_overlap = True
                    break
            if found_overlap:
                break
        if not found_overlap:
            overlapping_clusters.append([event])

    results = []
    for oc in overlapping_clusters:
        if len(oc) == 1:
            results.append(oc[0])
            continue

        # print("Found overlapping events to merge:")
        # for event in oc:
        #     print(event.anchors[0].grounded_span.full_span)

        starts = []
        ends = []
        for event in oc:
            starts.extend([anchor.grounded_span.full_span.start_char for anchor in event.anchors])
            ends.extend([anchor.grounded_span.full_span.end_char for anchor in event.anchors])
        start = min(starts)
        end = max(ends)

        # Take the score from the "best event"? Not really sure what to do here but that seems reasonable.
        best_event = get_best_event(oc, task=task)
        new_bss = make_new_span_set(doc, sent, best_event.anchors[0].score, start, end)
        results.append(best_event.with_new_anchors(new_bss))

    return results


def expand_event_list(
        doc: BetterDocument,
        sent: BetterSentence,
        events: List[BetterEvent],
        task: str
) -> List[BetterEvent]:
    start_sent_char = sent.original_document_character_span.start_char
    end_sent_char = sent.original_document_character_span.end_char

    expanded_events = defaultdict(list)

    for e in events:
        if len(e.anchors) > 1:
            raise ValueError("Cannot resolve anchor conflicts for events with more than one anchor")
        anchor = e.anchors[0]

        # Note: offsets are inclusive

        start_char = anchor.grounded_span.full_span.start_char
        if is_breaking_char(doc.doc_text.text[start_char]):
            # Edge case when the model finds punctuation as part of an anchor
            final_start_char = start_char
        else:
            prev_char = start_char - 1
            while prev_char >= start_sent_char and not is_breaking_char(doc.doc_text.text[prev_char]):
                prev_char -= 1
            final_start_char = prev_char + 1

        end_char = anchor.grounded_span.full_span.end_char
        if is_breaking_char(doc.doc_text.text[end_char]):
            # Edge case when the model finds punctuation as part of an anchor
            final_end_char = end_char
        else:
            next_char = end_char + 1
            while next_char <= end_sent_char and not is_breaking_char(doc.doc_text.text[next_char]):
                next_char += 1
            final_end_char = next_char - 1

        if start_char != final_start_char or end_char != final_end_char:
            # print("Expanded to nearest breaking chars:")
            # print(doc.doc_text.text[start_char:end_char + 1])
            # print(doc.doc_text.text[final_start_char:final_end_char + 1])
            # print()

            new_bss = make_new_span_set(doc, sent, anchor.score, final_start_char, final_end_char)
            new_event = e.with_new_anchors(new_bss)
        else:
            # print("Did not expand:", doc.doc_text.text[start_char:end_char + 1])
            new_event = e

        key = (final_start_char, final_end_char)
        expanded_events[key].append(new_event)

    semifinal_events = []
    for key, conflicting_events in expanded_events.items():
        semifinal_events.append(get_best_event(conflicting_events, task=task))

    # Remove events that are strict substrings
    # Again we assume all events here have exactly one anchor
    final_events = []
    for event in semifinal_events:
        event_anchor_full_span = event.anchors[0].grounded_span.full_span
        is_substring = False
        for other_event in semifinal_events:
            if event.event_id == other_event.event_id:
                continue
            # There should be no exact matches, from the earlier process, so it's safe to just check contains
            if other_event.anchors[0].grounded_span.full_span.contains(event_anchor_full_span):
                is_substring = True
                break
        if not is_substring:
            final_events.append(event)

    # Theoretically, there could be complex situations that require us to merge this process more than once,
    # so we'll just do so. Efficiency hit is extremely small since there are usually only a handful of events
    # per sentence and this merging happens rarely anyway.
    # Example of situation that could require multiple passes:
    # ABC DEF BCD --> ABC BCDEF
    # ABC BCDEF --> ABCDEF
    # (I think sorting would fix this, but whatever, it's already done.)
    while True:
        reduced_events = merge_overlapping_clusters(doc, sent, final_events, task=task)
        if len(reduced_events) == len(final_events):
            break
        final_events = reduced_events

    return reduced_events


def resolve_anchor_conflicts(doc: BetterDocument, task: str) -> BetterDocument:

    for sent in doc.sentences:
        sent.abstract_events = expand_event_list(doc, sent, sent.abstract_events, task=task)
        sent.basic_events = expand_event_list(doc, sent, sent.basic_events, task=task)

    # Fill in the token indices for our new anchors etc.
    BetterMapper.ground_events_to_tokens(doc)

    return doc


def main():
    parser = argparse.ArgumentParser(description="Resolve trigger conflicts")
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('task', type=str)
    args = parser.parse_args()

    docs = load_documents(args.input)
    results = {}
    for doc_id, doc in docs.items():
        if doc.lang in ['zh', "zh-cn", "zh-tw", "ja", "my", "lo", "th"]:
            results[doc_id] = doc.to_dict()
        else:
            results[doc_id] = resolve_anchor_conflicts(doc, task=args.task).to_dict()

    with open(args.output, "w", encoding="utf8") as outfile:
        outfile.write(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
