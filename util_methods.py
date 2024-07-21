from scripts.anntools import Collection, ENTITIES
import spacy

tokenize = spacy.load("es_core_news_md")
def get_dataset_from_collection(c: Collection):
    total = 0
    added_s_e = 0
    retrieved = 0
    dataset = []
    for sentence in c.sentences:
        new_dct = {}
        tokens = tokenize(sentence.text)
        new_dct['tokenized_text'] = list(tokens)
        new_dct['ner'] = []
        for keyphr in sentence.keyphrases:
            span1 = keyphr.spans[0]
            span2 = keyphr.spans[-1]
        
            FULL = tokens.char_span(span1[0],span2[1])
            if FULL:
                start = FULL.start
                end = FULL.end - 1
                new_dct['ner'].append([start, end, keyphr.label])
                added_s_e = added_s_e +1
            else:
                found_first = False
                found_last = False
                for token in tokens:
                    if token.idx + len(token) > span1[0] and not found_first:
                        found_first = True
                        first = token.idx
                        start = token.i
                    if token.idx + len(token) > span2[1]:
                        found_last = True
                        last = token.idx + len(token)
                        end = token.i
                    if found_first and found_last:
                        new_dct['ner'].append([start, end, keyphr.label])
                        retrieved = retrieved + 1
                        break
            total = total +1
        dataset.append(new_dct)
    return dataset