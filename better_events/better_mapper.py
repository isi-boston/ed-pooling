import re
import string
import sys
from typing import List, Tuple, Optional, AbstractSet, Union

from better_events.better_core import (
    BetterSentence,
    ScoredBetterSpan,
    BetterSpan,
    BetterDocument,
    BetterSpanSet,
    BetterEvent,
    SimpleGroundedSpan,
    GroundedSpan,
    REF_EVENT)
from better_events.better_validation import COREF_MENTION


def find_new_sentence_for_span(doc: BetterDocument,
                               span: Union[BetterSpan, ScoredBetterSpan]) -> Optional[int]:
    # We could just get the sentence ID from the grounded span, but
    # what if we have split sentences and that is unreliable? Let's actually check
    # the offsets
    hs = span.grounded_span.head_span
    for sent in doc.sentences:
        if sent.original_document_character_span.contains(hs):
            return sent.sent_id
    return None




class BetterMapper:

    def new_span_with_mention(
            self,
            doc: BetterDocument,
            sent: BetterSentence,
            span: ScoredBetterSpan,
            valid_mention_types: AbstractSet
    ) -> ScoredBetterSpan:

        if not span.grounded_span:
            return span

        if not sent:
            sent = doc.sentences_by_id[span.grounded_span.sent_id]

        choices = []
        for m in sent.mentions:
            if m.mention_type in {"list", "event-anchor"}:
                continue

            if valid_mention_types and m.mention_type not in valid_mention_types:
                continue

            if m.grounded_span.full_span == span.grounded_span.full_span:
                # same span
                choices.append((10, m))
            elif m.grounded_span.head_span == span.grounded_span.head_span:
                # same head (probably won't occur with system output since AA now produces only full spans?)
                choices.append((9, m))
            else:
                if doc.lang == 'en' or \
                        ("MT_lang" in doc.properties and doc.properties["MT_lang"] == "en"):
                    if m.grounded_span.head_span.end_token == span.grounded_span.head_span.end_token:
                        # heads end in the same place
                        choices.append((8, m))
                elif doc.lang in {'ar', 'fa', 'zh', 'ru', 'ko'}:
                    if m.grounded_span.head_span.start_token == span.grounded_span.head_span.start_token:
                        # heads start in the same place
                        choices.append((8, m))
                else:
                    print("Unknown language in BetterMapper", doc.lang)

            # TODO: What are we missing? Lots of things in English, especially.
            # We probably need to look into this more, but we may need to pass in more than
            # the span-- for instance, dates probably shouldn't be resolved, but places should

        mention_id = None
        if choices:
            # Get only the choices with the highest scores
            best_score = max([c[0] for c in choices])
            valid_choices = [c for c in choices if c[0] == best_score]

            # Select the smallest possible remaining mention
            # We include the actual span in the sort key as a tiebreaker just in case for determinism?
            # (Actual spans sort on character offsets)
            sorted_valid_choices = sorted(valid_choices,
                                          key=lambda c: (len(c[1].grounded_span.head_span.text),
                                                         len(c[1].grounded_span.full_span.text),
                                                         c[1]))

            mention_id = sorted_valid_choices[0][1].mention_id
        else:
            # For future debugging
            pass
            # print("No match for span", span.grounded_span.full_span.text, " / ", span.grounded_span.head_span.text)
            # for m in sent.mentions:
            #     if m.mention_type in {"list", "event-anchor"}:
            #         continue
            #
            #     if valid_mention_types and m.mention_type not in valid_mention_types:
            #         continue
            # 
            #     # Print overlapping choices
            #     if m.full_span.overlaps(span.grounded_span.full_span):
            #         print(" * ", m.full_span.text, " / ", m.head_span.text)
            # print()

        return ScoredBetterSpan(
            BetterSpan(
                span.text,
                span.head_text,
                GroundedSpan(
                    span.grounded_span.sent_id,
                    span.grounded_span.full_span,
                    span.grounded_span.head_span,
                    mention_id,
                ),
            ),
            span.score
        )

    def map_arguments_to_any_mentions_for_event(self,
                                                doc: BetterDocument,
                                                sent: BetterSentence,
                                                e: BetterEvent) -> BetterEvent:
        return self.map_arguments_to_mentions_for_event(doc, sent, e, set())

    def map_arguments_to_coref_mentions_for_event(self,
                                                  doc: BetterDocument,
                                                  sent: BetterSentence,
                                                  e: BetterEvent) -> BetterEvent:
        return self.map_arguments_to_mentions_for_event(doc, sent, e, {COREF_MENTION})

    def map_arguments_to_mentions_for_event(self,
                                            doc: BetterDocument,
                                            sent: BetterSentence,
                                            e: BetterEvent,
                                            valid_mention_types: AbstractSet) -> BetterEvent:
        new_arguments = []
        for arg in e.arguments:

            # We only map non-event arguments to mentions
            if arg.role == REF_EVENT:
                new_arguments.append(arg)
                continue

            new_spans = [
                self.new_span_with_mention(doc, sent, s, valid_mention_types)
                for s in arg.span_set
            ]

            new_arguments.append(arg.with_new_spans(new_spans))
        new_event_arguments = []
        for ev_arg in e.event_arguments:
            new_anchor_spans = [
                self.new_span_with_mention(doc, sent, s, valid_mention_types)
                for s in ev_arg.basic_event.anchors.spans
            ]
            new_args = []
            for arg in ev_arg.basic_event.arguments:
                new_spans = [
                    self.new_span_with_mention(doc, sent, s, valid_mention_types) for s in arg.span_set
                ]
                new_args.append(arg.with_new_spans(new_spans))
            new_time_attachments = []
            new_event_arguments.append(
                ev_arg.with_new_spans(new_anchor_spans, new_args, new_time_attachments)
            )

        return BetterEvent(
            e.event_id,
            e.event_type,
            e.properties,
            e.anchors,
            new_arguments,
            new_event_arguments,
            e.state_of_affairs
        )

    def map_arguments_to_mentions(self, doc: BetterDocument, coref_mentions_only: bool = False) -> None:
        if coref_mentions_only:
            doc.call_method_on_event_sets(self.map_arguments_to_coref_mentions_for_event)
        else:
            doc.call_method_on_event_sets(self.map_arguments_to_any_mentions_for_event)

    # Extra arguments included so we can call this generically via call_method_on_event_sets
    # noinspection PyUnusedLocal
    @staticmethod
    def remove_existing_mentions_from_event(doc: BetterDocument,
                                            sent: BetterSentence,
                                            e: BetterEvent) -> BetterEvent:
        new_arguments = []
        for arg in e.arguments:
            new_spans = [s.with_new_mention_id(None) for s in arg.span_set]
            new_arguments.append(arg.with_new_spans(new_spans))
        new_anchors = BetterSpanSet([s.with_new_mention_id(None) for s in e.anchors])
        return BetterEvent(
            e.event_id,
            e.event_type,
            e.properties,
            new_anchors,
            new_arguments,
            e.event_arguments,
            e.state_of_affairs
        )

    @staticmethod
    def remove_existing_mentions(doc: BetterDocument) -> None:
        for sent in doc.sentences:
            sent.mentions = []
            sent.mentions_by_id = {}
        doc.call_method_on_event_sets(BetterMapper.remove_existing_mentions_from_event)

    @staticmethod
    def get_distance_from_other_spans(
        span: Tuple[int, int], other_spans: List[Tuple[int, int]],
    ) -> Tuple[int, int, int]:
        """Get the minimum distance (in characters) of this span from any of
        the other spans provided. All spans are represented as a tuple of
        start/end offsets.

        Also returns the span start/end so we can trivially use this for
        deterministic sorting.
        """
        if not other_spans:
            return 0, span[0], span[1]

        distances = []
        for a in other_spans:
            if span[1] <= a[0]:
                distances.append(a[0] - span[1])
            elif span[0] >= a[1]:
                distances.append(span[0] - a[1])
            else:
                # span and anchor overlap
                distances.append(0)

        # Including span start/end to avoid ties
        return min(distances), span[0], span[1]

    @staticmethod
    def get_best_span(
        text_to_search: str, text_to_find: str, anchors: List[Tuple[int, int]],
        subset_of: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, int]:
        """Find all instances of a string.

        Instances are required to align with word boundaries
        (whitespace and punctuation).
        Only if no such match can be found do we fall back to
        non-word-boundary-delimited spans. This is usually an indication of
        a typo, so far.

        This returns inclusive offsets.
        """

        # Strip out sentence-final character if it's there; otherwise re picks 
        # it up as part of the match, but it's not part of the token
        text_to_search = text_to_search.rstrip("\u202c")

        text = text_to_find

        # We really prefer spans without these ending contractions, 
        # they screw everything up
        text = re.sub("'s$", "", text)
        text = re.sub("'m?$", "", text)
        text = re.sub("'ve?$", "", text)
        text = re.sub("'re?$", "", text)

        # Strip leading and trailing punctuation (and whitespace) from target text
        text = text.strip(string.punctuation + string.whitespace)

        # The above command only deals with ASCII
        text = re.sub(r"\W*$", "", text)
        text = re.sub(r"^\W*", "", text)

        # Something went wrong
        if not text:
            text = text_to_find

        # Theoretically all this stripping could reduce our ability to disambiguate
        # between two identical spans, but at least for the abstract event task,
        # which uses only a single sentence, this seems unlikely

        # Find and return all instances of text that fall along word boundaries
        # Python regex \b seems to incorporate punctuation as well as whitespace
        results = []
        so_far = 0
        while True:
            match = re.search(r"\b{}\b".format(re.escape(text)), text_to_search)
            if not match:
                break
            results.append((so_far + match.start(), so_far + match.end() - 1))
            text_to_search = text_to_search[match.end():]
            so_far += match.end()

        # Failure! We will have to abandon the word boundary restriction
        # This is very rare and usually a typo
        # We could also consider abandoning these instances
        so_far = 0
        if not results:
            sys.stderr.write(
                "WARNING: Unable to find word-boundary-delimited "
                "span '{}' in '{}'\n\n".format(text, text_to_search)
            )
            while True:
                match = re.search(re.escape(text), text_to_search)
                if not match:
                    break
                results.append((so_far + match.start(), so_far + match.end() - 1))
                text_to_search = text_to_search[match.end():]
                so_far += match.end()

        # Try to figure out which one we think is the intended span
        if len(results) == 0:
            # We hope this should never happen
            raise ValueError(
                "Unable to find text for span '{}' in '{}'".format(
                    text_to_find, text_to_search
                )
            )
        elif len(results) == 1:
            # Easy case
            return results[0]
        else:
            # Select the target span that's closest to an anchor
            sorted_possible_spans = sorted(
                results, key=lambda x: BetterMapper.get_distance_from_other_spans(x, anchors)
            )

            # Debugging info to look at span disambiguation
            # anchors = [(doc.text.span_text(a), a.start, a.end) for a in anchors]
            # print("Multiple spans for '{}' in '{}': {}... picked {} due to proximity to "
            #       "anchors {}".format(text, better_sent.text, possible_spans,
            # target_span, anchors))
            # print()

            # if we have a full span that this span (a head span) should be
            # contained within:
            if subset_of:
                for result in sorted_possible_spans:
                    if result[0] >= subset_of[0] and result[1] <= subset_of[1]:
                        return result
            return sorted_possible_spans[0]

    @staticmethod
    def convert_to_sgs(
        sent: BetterSentence,
        old_better_span: Optional[SimpleGroundedSpan],
        span: Tuple[int, int],
        text: str,
    ) -> SimpleGroundedSpan:
        # Input is inclusive offsets
        if old_better_span:
            text = old_better_span.text

        # Overlapping tokens are possible here, so we need to be careful

        included_tokens = set()
        for i, tok in enumerate(sent.tokens):
            # This one precedes our span entirely
            if tok.doc_character_span[1] < span[0]:
                continue

            # This one follows our span entirely
            if tok.doc_character_span[0] > span[1]:
                continue

            # This one must therefore overlap it
            included_tokens.add(i)

        if not included_tokens:
            raise ValueError(f"Span {span} does not overlap ANY tokens in sentence {sent.sent_id}! What should we do?")

        start = min(included_tokens)
        end = max(included_tokens)

        return SimpleGroundedSpan(
            sent.doc_text,
            start_char=sent.tokens[start].doc_character_span[0],
            end_char=sent.tokens[end].doc_character_span[1],
            start_token=start,
            end_token=end,
        )

    @staticmethod
    def _get_text_from_tokens(
            sent: BetterSentence, start: int, end: int
    ) -> str:
        """inserts spaces if there is any gap between tokens"""
        # only consider the subrange of tokens specified
        sent_start_offset = sent.original_document_character_span.start_char
        return sent.original_document_character_span.text[
            sent.tokens[start].doc_character_span[0] - sent_start_offset:
            sent.tokens[end].doc_character_span[1] - sent_start_offset + 1
        ]

    @staticmethod
    def ground_to_tokens(
        doc: BetterDocument,
        sent: Optional[BetterSentence],
        s: ScoredBetterSpan,
        anchors: List[Tuple[int, int]],
    ) -> ScoredBetterSpan:
        # map event spans to tokens, but keep character indices referring to document text

        if not doc and not sent:
            raise ValueError("At least one of doc/sent must be specified in ground_to_tokens")

        if not sent:
            # Don't rely on existing sentence ID; make sure it's right
            sent_id = find_new_sentence_for_span(doc, s)
            sent = doc.sentences_by_id[sent_id]

        # if span is already grounded in document character indices:
        if s.grounded_span:
            target_full_span = (
                s.grounded_span.full_span.start_char,
                s.grounded_span.full_span.end_char,
            )
            target_head_span = (
                s.grounded_span.head_span.start_char,
                s.grounded_span.head_span.end_char,
            )
            full_better_span = s.grounded_span.full_span
            head_better_span = s.grounded_span.head_span
        else:

            target_full_span = BetterMapper.get_best_span(
                sent.original_document_character_span.text, s.text, anchors
            )

            # Now look inside the target full span for the head
            target_full_span_text = sent.original_document_character_span.text[
                                    target_full_span[0]: target_full_span[1] + 1
                                    ]
            target_head_span = BetterMapper.get_best_span(
                sent.original_document_character_span.text, s.head_text,
                anchors, target_full_span
            )
            target_head_span_text = sent.original_document_character_span.text[
                                    target_head_span[0]: target_head_span[1] + 1
                                    ]

            # Re-adjust the offsets to be with respect to the document
            target_full_span = (
                target_full_span[
                    0] + sent.original_document_character_span.start_char,
                target_full_span[
                    1] + sent.original_document_character_span.start_char,
            )
            target_head_span = (
                target_head_span[
                    0] + sent.original_document_character_span.start_char,
                target_head_span[
                    1] + sent.original_document_character_span.start_char,
            )
            # target_head_span = (target_head_span[0] + target_full_span[0],
            #                     target_head_span[1] + target_full_span[0])
            full_better_span = SimpleGroundedSpan(
                sent.doc_text,
                target_full_span[0], target_full_span[1],
                None, None
            )
            head_better_span = SimpleGroundedSpan(
                sent.doc_text,
                target_head_span[0], target_head_span[1],
                None, None
            )

        sgs_full = BetterMapper.convert_to_sgs(
            sent, full_better_span, target_full_span,
            full_better_span.text
        )
        sgs_head = BetterMapper.convert_to_sgs(
            sent, head_better_span,
            target_head_span,
            head_better_span.text
        )

        if s.grounded_span:
            # copy mention id
            grounded_span = GroundedSpan(
                sent.sent_id, sgs_full, sgs_head, s.grounded_span.mention_id
            )
        else:
            grounded_span = GroundedSpan(sent.sent_id, sgs_full, sgs_head, None)

        return ScoredBetterSpan(
            BetterSpan(sgs_full.text, sgs_head.text, grounded_span), s.score)

    @staticmethod
    def ground_to_tokens_sgs(sent: BetterSentence,
                             sgs: SimpleGroundedSpan) -> SimpleGroundedSpan:
        target_span = (
            sgs.start_char,
            sgs.end_char,
        )
        return BetterMapper.convert_to_sgs(
            sent, sgs, target_span, sgs.text
        )

    @staticmethod
    def get_offset_range_for_tokens(
        sent: BetterSentence, sgs: SimpleGroundedSpan
    ) -> Tuple[int, int]:
        return (
            sent.get_offset_in_sentence(sent.tokens[sgs.start_token])[0],
            sent.get_offset_in_sentence(sent.tokens[sgs.end_token])[1],
        )

    # Extra arguments included so we can call this generically via call_method_on_event_sets
    @staticmethod
    # noinspection PyUnusedLocal
    def ground_event_to_tokens(doc: Optional[BetterDocument],
                               sent: Optional[BetterSentence],
                               e: BetterEvent) -> BetterEvent:

        if not doc and not sent:
            raise ValueError("At least one of doc/sent must be specified in ground_event_to_tokens")

        new_anchors = []
        anchor_offsets = []
        for a in e.anchors:
            if not a.text:
                continue
            local_sent = sent
            if not local_sent:
                # Don't rely on existing sentence ID; make sure it's right
                sent_id = find_new_sentence_for_span(doc, a)
                local_sent = doc.sentences_by_id[sent_id]
            new_anchor = BetterMapper.ground_to_tokens(doc, local_sent, a, [])
            new_anchors.append(new_anchor)
            anchor_offsets.append(BetterMapper.get_offset_range_for_tokens(
                local_sent, new_anchor.grounded_span.head_span))

        new_arguments = []
        for arg in e.arguments:
            # ensure that all spans being grounded belong to current sent
            new_spans = []
            for span in arg.span_set:
                local_sent = sent
                if not local_sent and span.grounded_span is not None:
                    try:
                        sent_id = doc.doc_span_to_sent_id(
                            (span.grounded_span.full_span.start_char,
                             span.grounded_span.full_span.end_char)
                        )
                    except ValueError:
                        continue
                    local_sent = doc.sentences_by_id[sent_id]
                if sent and span.grounded_span and not sent.original_document_character_span.contains(
                        span.grounded_span.full_span):
                    continue
                if span.text:
                    new_spans.append(
                        BetterMapper.ground_to_tokens(
                            doc, local_sent, span, anchor_offsets
                        )
                    )
            new_arguments.append(arg.with_new_spans(new_spans))
        return BetterEvent(
            e.event_id,
            e.event_type,
            e.properties,
            BetterSpanSet(new_anchors),
            new_arguments,
            e.event_arguments,
            e.state_of_affairs,
        )

    @staticmethod
    def ground_events_to_tokens(doc: BetterDocument) -> None:
        doc.call_method_on_event_sets(BetterMapper.ground_event_to_tokens)
