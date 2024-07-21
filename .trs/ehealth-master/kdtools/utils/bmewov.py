import re
from functools import reduce
import operator

def add(n, t):
    return [n+i for i in t]

class BMEWOV:
    @staticmethod
    def _discontinuous_match_regex1(match):
        entities_spans = []

        start, end = match.span()
        string = match.string[start:end]

        reg_v = r"V+"
        reg_me = r"M*E"

        match_v = re.search(reg_v, string)

        for match_me in re.finditer(reg_me, string):
            entities_spans.append(add(start, list(range(*match_v.span())) + list(range(*match_me.span()))))

        return entities_spans

    @staticmethod
    def _discontinuous_match_regex2(match):
        entities_spans = []

        start, end = match.span()
        string = match.string[start:end]

        reg_b = r"B"
        reg_v = r"V+"

        match_v = re.search(reg_v, string)

        for match_b in re.finditer(reg_b, string):
            entities_spans.append(add(start, list(range(*match_b.span())) + list(range(*match_v.span()))))

        return entities_spans

    @staticmethod
    def _discontinuous_entities(sequence: list):
        sequence_str = "".join(sequence)

        cont_matches = []
        entities_spans = []

        reg1 = r"V+(M*EO*)+M*E"
        reg2 = r"(BO)+BV+"

        for match in re.finditer(reg1, sequence_str):
            entities_spans.extend(BMEWOV._discontinuous_match_regex1(match))
            for i in range(*match.span()):
                sequence[i] = "O"

        for match in re.finditer(reg2, sequence_str):
            entities_spans.extend(BMEWOV._discontinuous_match_regex2(match))
            for i in range(*match.span()):
                sequence[i] = "O"

        return entities_spans

    @staticmethod
    def _continuous_entities(sequence: list):
        partial = []
        prev_state = "O"
        entities_spans = []

        def emit(entity: list):
            nonlocal entities_spans
            entities_spans.append(entity)

        def flush():
            nonlocal partial
            for entity in partial:
                emit(entity)
            partial = []

        def add(index):
            partial.append([index])

        def extend(index):
            nonlocal partial
            for entity in partial:
                entity.append(index)

        for i, tag in enumerate(sequence):
            if tag == "O":
                flush()
            if prev_state in ["O","W","E"]:
                if tag in ["B","M","V"]:
                    add(i)
                    if tag == "V":
                        add(i)
                if tag in ["W","E"]:
                    emit([i])
            if prev_state == "V":
                if tag=="V":
                    extend(i)
                if tag=="B":
                    flush()
                    add(i)
                if tag in ["M","E"]:
                    emit(partial.pop())
                    extend(i)
                    if tag=="E":
                        flush()
                if tag=="W":
                    flush()
                    emit([i])
            if prev_state in ["B","M"]:
                if tag=="V":
                    extend(i)
                    add(i)
                if tag=="B":
                    flush()
                    add(i)
                if tag=="M":
                    extend(i)
                if tag=="E":
                    extend(i)
                    flush()
                if tag=="W":
                    flush()
                    emit([i])
            prev_state = tag
        return entities_spans

    @staticmethod
    def decode(sequence: list):
        return BMEWOV._discontinuous_entities(sequence) + BMEWOV._continuous_entities(sequence)

    @staticmethod
    def encode(sentence_spans: list, entities_spans: list):
        ret_tags = []
        for span in sentence_spans:
            tag = None
            ent_with_span = [entity for entity in entities_spans if span in entity]
            if len(ent_with_span) == 0:
                tag = "O"
            elif len(ent_with_span)>1:
                tag = "V"
            else:
                entity = ent_with_span[0]
                if len(entity) == 1:
                    tag = "W"
                elif span == entity[0]:
                    tag = "B"
                elif span == entity[-1]:
                    tag = "E"
                else:
                    tag = "M"
            ret_tags.append(tag)
        return ret_tags