import spacy
from kdtools.utils.latin import CORPUS_CHARS as latin_chars, UNITS as units, CURRENCY as currencies
import re
from kdtools.utils.model_helpers import Tree
from string import ascii_lowercase
from functools import lru_cache

import torch
from torch.nn.functional import one_hot
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import numpy as np
from numpy.random import permutation

class SpacyComponent:
    nlp = None
    def __init__(self):
        self.nlp = SpacyComponent.nlp if SpacyComponent.nlp else spacy.load("es_core_news_md")
        SpacyComponent.nlp = self.nlp

    @lru_cache(maxsize=200)
    def nlp_wrapper(self, sentence):
        return self.nlp(sentence)

class TokenizerComponent(SpacyComponent):
    def __init__(self):
        SpacyComponent.__init__(self)

    def get_spans(self, sentence: str):
        return [(token.idx, token.idx+len(token)) for token in self.nlp_wrapper(sentence)]

class SpacyVectorsComponent(SpacyComponent):

    @property
    def word_vector_size(self):
        return len(self.get_spacy_vector("hola"))

    def get_spacy_vector(self, word: str):
        return self.nlp.vocab.get_vector(word)


class DependencyTreeComponent(SpacyComponent):

    def get_dependency_tree(self, sentence: str):
        tokens = list(self.nlp_wrapper(sentence))
        nodes = [Tree(token.i, token.text+" "+token.dep_) for token in tokens]

        for node in nodes:
            for child in tokens[node.idx].children:
                node.add_child(nodes[child.i])

        root = list(filter(lambda x: x.dep_ == "ROOT", tokens))[0]
        return nodes[root.i]


class CharEmbeddingComponent:
    def __init__(self):
        self.abc = ['<pad>','<unk>']  +  list(" \"%(),-./012345679:;ABCDEFGHIKLMNOPRSTUVXYZabcdefghijklmnopqrstuvwxyzÉáéíñóöú")
        self.int2char = dict(enumerate(self.abc))
        self.char2int = {char: index for index,char in self.int2char.items()}

    def encode_chars_indices(self, word, max_word_len, sent_len):
        solve = [self.char2int[char] if char in self.char2int else self.char2int['<unk>']\
            for char in word]
        len_solve = len(solve)
        for i in range(max_word_len - len_solve):
            solve += [self.char2int['<pad>']]

        return solve

class EmbeddingComponent:
    def __init__(self, wv):
        self.wv = wv
        self.vocab = wv.vocab

    @property
    def word_vector_size(self):
        return len(self.wv.vectors[0])

    def map_word(self, word: str):
        tokens = ['<padding>', '<unseen>', '<notlatin>', '<unit>', '<number>']

        if word in tokens:
            return word
        if re.findall(r"[0-9]", word):
            return "<number>"
        if re.fullmatch(units,word):
            return "<unit>"
        if re.fullmatch(currencies, word):
            return "<currency>"
        if len(re.findall(latin_chars, word)) != len(word):
            return "<notlatin>"
        return word

    def get_word_index(self, word):
        word = self.map_word(word)
        return self.vocab[word].index if word in self.vocab else self.vocab["<unseen>"].index

class PostagComponent(SpacyComponent):
    def __init__(self):
        super(SpacyComponent).__init__()

        self.postags = [
            "ADJ",
            "ADP",
            "ADV",
            "AUX",
            "CONJ",
            "CCONJ",
            "DET",
            "INTJ",
            "NOUN",
            "NUM",
            "PART",
            "PRON",
            "PROPN",
            "PUNCT",
            "SCONJ",
            "SYM",
            "VERB",
            "X",
            "SPACE"
        ]
        self.postag2index = {postag: idx for (idx, postag) in enumerate(self.postags)}

    def get_sentence_postags(self, sentence: str):
        tokens = list(self.nlp_wrapper(sentence))

        return [self.postag2index[token.pos_] for token in tokens]


class RelationComponent(SpacyComponent):
    def __init__(self, include_none = False):
        super().__init__()

        self.relations = [
            "subject",
            "target",
            "in-place",
            "in-time",
            "in-context",
            "arg",
            "domain",
            "has-property",
            "part-of",
            "is-a",
            "same-as",
            "causes",
            "entails"
        ]
        self.relations += ["none"] if include_none else []
        self.relation2index = {relation: self.relations.index(relation) for relation in self.relations}

class DependencyComponent(SpacyComponent):
    def __init__(self):
        super(SpacyComponent).__init__()

        self.dependencies = [
            "acl",
            "advcl",
            "advmod",
            "amod",
            "appos",
            "aux",
            "case",
            "cc",
            "ccomp",
            "clf",
            "compound",
            "conj",
            "cop",
            "csubj",
            "dep",
            "det",
            "discourse",
            "dislocated",
            "expl",
            "fixed",
            "flat",
            "goeswith",
            "iobj",
            "list",
            "mark",
            "nmod",
            "nsubj",
            "nummod",
            "obj",
            "obl",
            "orphan",
            "parataxis",
            "punct",
            "reparandum",
            "ROOT",
            "vocative",
            "xcomp",
            "other",
            'expl:pass'
        ]
        self.dep2index = {dep: idx for (idx, dep) in enumerate(self.dependencies)}

    def get_sentence_dependencies(self, sentence: str):
        tokens = list(self.nlp_wrapper(sentence))
        return [self.dep2index[token.dep_ if token.dep_ != '' else 'other'] for token in tokens]

class PositionComponent:
    def __init__(self, max_sent_len):
        self.max_sent_len = max_sent_len
        self.no_positions = 2*max_sent_len+1

    def get_position_encoding(self, no_words, index):
        return [i - index + self.max_sent_len for i in range(no_words)]

class EntityTypesComponent:
    def __init__(self):

        self.entity_types = [
            "Concept",
            "Action",
            "Reference",
            "Predicate",
            "<None>"
        ]

        self.type2index = {type: idx for (idx, type) in enumerate(self.entity_types)}

    def get_type_encoding(self, sequence):
        return [self.type2index[type] for type in sequence]


class ShufflerComponent:

    def get_shuffled_data(self):
        perm = permutation(len(self))
        for idx in perm:
            yield self[idx]

class BMEWOVTagsComponent:
    def __init__(self):
        self.entity_tags = ["B", "M", "E", "W", "O", "V"]
        self.tag2index = {tag: idx for (idx, tag) in enumerate(self.entity_tags)}


class BERTComponent:
    tokenizer = None
    bert_model = None

    def __init__(self, model):
        self.bert_vector_size = 9216
        self.sent_vector_size = 768
        self.model = model
        self.tokenizer = BERTComponent.tokenizer if BERTComponent.tokenizer else BertTokenizer.from_pretrained(model)
        BERTComponent.tokenizer = self.tokenizer
        self.bert_model = BERTComponent.bert_model if BERTComponent.bert_model else BertModel.from_pretrained(model)
        BERTComponent.bert_model = self.bert_model
        self.bert_model.eval()


    def get_bert_spans(self, words, bert_tokens):
        if self.model == 'bert-base-multilingual-uncased':
            words = [self._flat_word(word) for word in words]

        i = 0
        j = 1
        idx = 0

        bert_words_indexes = []
        bert_words = []
        while i < len(words):
            word = words[i]

            bert_word = bert_tokens[j]
            bert_word = bert_word[2:] if bert_word.startswith("##") else bert_word
            bert_word = bert_word[idx:]

            #Spacing control
            if word in [" ", "  ", "   "]:
                bert_words.append([word])
                bert_words_indexes.append([-1])

            #When the current word is [UNK] for bert
            elif bert_word == "[UNK]":
                bert_words.append(["[UNK]"])
                bert_words_indexes.append([j])
                j += 1
                idx = 0

            #When the current word is contained in bert token. Very weird
            elif len(word) < len(bert_word) and bert_word.find(word) >= 0:
                bert_words.append([bert_word])
                bert_words_indexes.append([j])

                idx = bert_word.find(word) + len(word)
                if idx == len(bert_word):
                    j += 1
                    idx = 0

            #Otherwise
            else:
                k = 0
                span = []
                span_indexes = []

                while k < len(word):
                    if word.find(bert_word, k) == k:
                        span.append(bert_word)
                        span_indexes.append(j)
                        k += len(bert_word)
                        j += 1
                        idx = 0
                        bert_word = bert_tokens[j]
                        bert_word = bert_word[2:] if bert_word.startswith("##") else bert_word
                    else:
                        print("Error")
                        return bert_words, bert_words_indexes

                bert_words.append(span)
                bert_words_indexes.append(span_indexes)

            i += 1

        assert len(bert_words_indexes) == len(words)

        return bert_words, bert_words_indexes

    def _flat_word(self, word):
        word = word.lower()
        word = word.replace("ñ", "n")
        word = word.replace("á", "a")
        word = word.replace("é", "e")
        word = word.replace("í", "i")
        word = word.replace("ó", "o")
        word = word.replace("ú", "u")
        word = word.replace("ä", "a")
        word = word.replace("ü", "u")
        word = word.replace("ö", "o")
        word = word.replace("ū", "u")
        word = word.replace("ā", "a")
        word = word.replace("ī", "i")
        word = word.replace("ș", "s")
        word = word.replace("ã", "a")
        word = word.replace("ô", "o")

        return word

    def _sum_merge(self, vectors):
        return torch.sum(torch.stack(vectors), dim=0)

    def _mean_merge(self, vectors):
        return torch.mean(torch.stack(vectors), dim=0)

    def _last_merge(self, vectors):
        return vectors[-1]

    def _get_merge_tensors(self, token_vec_sums, words_indexes):
        pad_tensor = torch.zeros(self.bert_vector_size)
        real_vec = []
        for word_indexes in words_indexes:
            vectors = [(token_vec_sums[idx] if idx != -1 else pad_tensor) for idx in word_indexes]
            real_vec.append(self._mean_merge(vectors))

        return real_vec

    def get_bert_embeddings(self, sentence, spans):
        tokenized_sentence = self.tokenizer.tokenize(sentence)
        tokenized_sentence = ['[CLS]'] + tokenized_sentence + ['[SEP]']
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        segments_ids = [1] * len(tokenized_sentence)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            encoded_layers, _ = self.bert_model(tokens_tensor, segments_tensors)

        token_embeddings = torch.stack(encoded_layers, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)

        token_vec_sums = []
        for token in token_embeddings:
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4], token[-5], token[-6], token[-7], token[-8], token[-9], token[-10], token[-11], token[-12]), dim=-1)
            token_vec_sums.append(cat_vec)

        words = [sentence[beg:end] for (beg, end) in spans]
        bert_words, bert_words_indexes = self.get_bert_spans(words, tokenized_sentence)

        bert_embeddings = self._get_merge_tensors(token_vec_sums, bert_words_indexes)

        return bert_embeddings




