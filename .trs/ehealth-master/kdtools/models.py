import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from kdtools.layers import *
from kdtools.utils.model_helpers import Tree

#this is a recycled code, it doesn't make use of our BiLSTM, nor CRF layer
class BiLSTM_CRF(nn.Module):

    def __init__(self, input_dim, tagset_size, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size+2
        self.START_TAG = tagset_size
        self.STOP_TAG = tagset_size+1

        self.lstm = nn.LSTM(input_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.START_TAG, :] = -10000
        self.transitions.data[:, self.STOP_TAG] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.START_TAG] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.STOP_TAG]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        sentence = sentence.view(-1, 1, self.input_dim)
        lstm_out, self.hidden = self.lstm(sentence, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.START_TAG], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.STOP_TAG, tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.START_TAG] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.STOP_TAG]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.START_TAG  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

class EmbeddingBiLSTM_CRF(nn.Module):
    def __init__(self, tagset_size, hidden_dim, wv):
        super().__init__()
        embed_size = len(wv.vectors[0])
        self.embedding = PretrainedEmbedding(wv)
        self.bislstmcrf = BiLSTM_CRF(embed_size, tagset_size, hidden_dim)

    def neg_log_likelihood(self, X, y):
        X = self.embedding(X)
        return self.bislstmcrf.neg_log_likelihood(X, y)

    def forward(self, X):
        return self.bislstmcrf(self.embedding(X))

class EmbeddingAttentionBiLSTM_CRF(nn.Module):
    def __init__(self, tagset_size, hidden_dim, no_heads, wv):
        super().__init__()
        embed_size = len(wv.vectors[0])
        self.wv = wv
        self.embedding = PretrainedEmbedding(wv)
        self.attention = MultiheadAttention(embed_size, no_heads)
        self.bislstmcrf = BiLSTM_CRF(embed_size, tagset_size, hidden_dim)

    def neg_log_likelihood(self, X, y):
        X = self.attention(self.embedding(X))
        return self.bislstmcrf.neg_log_likelihood(X, y)

    def forward(self, X):
        return self.bislstmcrf(self.attention(self.embedding(X)))

class BiLSTMDoubleDenseOracleParser(nn.Module):
    def __init__(self,
        input_size,
        lstm_hidden_size,
        dropout_ratio,
        hidden_dense_size,
        wv,
        actions_no,
        relations_no
    ):
        super().__init__()

        self.wv = wv

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(wv.vectors))

        self.bilstmencoder_sent = BiLSTM(input_size, lstm_hidden_size, batch_first=True)
        self.bilstmencoder_stack = BiLSTM(input_size, lstm_hidden_size, batch_first=True)

        self.dropout_sent = nn.Dropout(p = dropout_ratio)
        self.dropout_stack = nn.Dropout(p = dropout_ratio)

        self.dense_sent = nn.Linear(self.bilstmencoder_sent.hidden_size, hidden_dense_size)
        self.dense_stack = nn.Linear(self.bilstmencoder_stack.hidden_size, hidden_dense_size)

        dense_input_size = 2*hidden_dense_size

        self.action_dense = nn.Linear(dense_input_size, actions_no)
        self.relation_dense = nn.Linear(dense_input_size, relations_no)

    def forward(self, x):
        x0 = self.embedding(x[0])
        x1 = self.embedding(x[1])

        stack_encoded, _ = self.bilstmencoder_stack(x0)
        sent_encoded, _ = self.bilstmencoder_sent(x1)

        stack_encoded = self.dropout_stack(stack_encoded)
        sent_encoded = self.dropout_sent(sent_encoded)

        stack_encoded = torch.tanh(self.dense_stack(stack_encoded))
        sent_encoded = torch.tanh(self.dense_sent(sent_encoded))

        encoded = torch.cat([stack_encoded, sent_encoded], 1)

        action_out = F.softmax(self.action_dense(encoded), 1)
        relation_out = F.softmax(self.relation_dense(encoded), 1)

        return [action_out, relation_out]

class BiLSTMSelectiveRelationClassifier(nn.Module):
    def __init__(self, sent_hidden_size, entities_hidden_size, dense_hidden_size, no_relations, wv, dropout_ratio = 0.2):
        super().__init__()

        embed_size = len(wv.vectors[0])
        self.wv = wv
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(wv.vectors))

        self.sent_encoder = BiLSTM(embed_size, sent_hidden_size, return_sequence=True, batch_first=True)

        self.origin_encoder = BiLSTM(2*sent_hidden_size, entities_hidden_size, batch_first = True)
        self.destination_encoder = BiLSTM(2*sent_hidden_size, entities_hidden_size, batch_first = True)

        self.origin_dense_hidden = nn.Linear(2*entities_hidden_size, dense_hidden_size)
        self.destination_dense_hidden = nn.Linear(2 * entities_hidden_size, dense_hidden_size)

        self.origin_dropout = nn.Dropout(p=dropout_ratio)
        self.destination_dropout = nn.Dropout(p=dropout_ratio)

        self.dense_output = nn.Linear(2*dense_hidden_size, no_relations)

    def forward(self, X, mask_origin, mask_destination):
        X = self.embedding(X)
        sentence_encoded, _ = self.sent_encoder(X)

        origin_encoded, _ = self.origin_encoder(sentence_encoded * mask_origin)
        destination_encoded, _ = self.destination_encoder(sentence_encoded * mask_destination)

        origin_encoded = torch.tanh(self.origin_dropout(self.origin_dense_hidden(origin_encoded)))
        destination_encoded = torch.tanh(self.destination_dropout(self.destination_dense_hidden(destination_encoded)))
        return F.softmax(self.dense_output(torch.cat((origin_encoded, destination_encoded), dim = 1)), dim = 1)

class BERT_TreeLSTM_BiLSTM_CNN_JointModel(nn.Module):

    def __init__(
        self,
        embedding_size,
        wv,
        bert_size,
        no_postags,
        postag_size,
        no_dependencies,
        dependency_size,
        no_positions,
        position_size,
        no_chars,
        charencoding_size,
        tree_lstm_hidden_size,
        bilstm_hidden_size,
        local_cnn_channels,
        local_cnn_window_size,
        global_cnn_channels,
        global_cnn_window_size,
        dropout_chance,
        no_entity_types,
        no_entity_tags,
        no_relations
        ):

        super().__init__()

        #INPUT PROCESSING

        #Word Embedding layer
        self.word_embedding = PretrainedEmbedding(wv)

        #Char Embedding layer
        self.char_embedding = CharCNN(1, no_chars, charencoding_size)

        #POS-tag Embedding layer
        self.postag_embedding = nn.Embedding(no_postags, postag_size)

        #Position Embedding layer
        self.position_embedding = nn.Embedding(no_positions, position_size)

        #Dependency Embedding layer
        self.dependency_embedding = nn.Embedding(no_dependencies, dependency_size)


        #ENCODING (SHARED PARAMETERS)
        bert_size = 0
        word_rep_size = embedding_size + bert_size + postag_size + dependency_size + position_size + charencoding_size

        #Word-encoding BiLSTM
        self.word_bilstm = BiLSTM(word_rep_size, bilstm_hidden_size//2, return_sequence=True)

        #Word-encoding CNN
        self.word_cnn = nn.Conv1d(word_rep_size, local_cnn_channels, local_cnn_window_size, padding=1)

        #DependencyTree-enconding TreeLSTM
        self.tree_lstm = ChildSumTreeLSTM(word_rep_size, tree_lstm_hidden_size)

        #Global CNN
        self.sentence_cnn = nn.Conv1d(word_rep_size, global_cnn_channels, global_cnn_window_size, padding=1)

        #OUTPUT
        self.dropout = nn.Dropout(dropout_chance)

        tokens_features_size = bilstm_hidden_size + local_cnn_channels + tree_lstm_hidden_size
        sentence_features_size = 2 * tokens_features_size + global_cnn_channels

        #Entity type
        self.entity_type_decoder = nn.Linear(tokens_features_size + global_cnn_channels, no_entity_types)

        #Entites
        self.entities_crf_decoder = CRF(sentence_features_size, no_entity_tags)

        #Relations
        self.relations_decoder = nn.Linear(sentence_features_size, no_relations)

    def forward(self, X):
        (
            # bert_embeddings,
            word_inputs,
            char_inputs,
            postag_inputs,
            dependency_inputs,
            position_inputs,
            trees,
            pointed_token_idx
        ) = X

        # bert_embeddings, word_inputs, char_embeddings, postag_inputs, position_inputs, trees, pointed_token_idx = X
        sent_len = len(trees)

        #obtaining embeddings vectors
        word_embeddings = self.word_embedding(word_inputs)
        char_embeddings = self.char_embedding(char_inputs)
        postag_embeddings = self.postag_embedding(postag_inputs)
        position_embeddings = self.position_embedding(position_inputs)
        dependency_embeddings = self.position_embedding(dependency_inputs)

        # print(
        #     # "bert_embeddings: ", bert_embeddings.shape, "\n",
        #     "word_embeddings: ", word_embeddings.shape, "\n",
        #     "char_embeddings: ", char_embeddings.shape, "\n",
        #     "postag_embeddings: ", postag_embeddings.shape, "\n",
        #     "position_embeddings: ", position_embeddings.shape, "\n",
        #     "dependency_embeddings: ", dependency_embeddings.shape, "\n"
        # )

        inputs = torch.cat(
            (
                # bert_embeddings,
                word_embeddings,
                char_embeddings,
                postag_embeddings,
                position_embeddings,
                dependency_embeddings
            ), dim=-1)

        # print(
        #     "inputs: ", inputs.shape, "\n"
        # )

        #encoding those inputs
        local_bilstm_encoding, _ = self.word_bilstm(inputs)
        local_cnn_encoding = self.word_cnn(inputs.permute(0,2,1)).permute(0,2,1)
        local_deptree_encoding = torch.cat([self.tree_lstm(tree, inputs.squeeze(0))[1] for tree in trees], dim=0).unsqueeze(0)
        global_cnn_encoding = F.max_pool1d(self.sentence_cnn(inputs.permute(0,2,1)), sent_len).permute(0,2,1)

        # print(
        #     "local_bilstm_encoding: ", local_bilstm_encoding.shape, "\n",
        #     "local_cnn_encoding: ", local_cnn_encoding.shape, "\n",
        #     "local_deptree_encoding: ", local_deptree_encoding.shape, "\n",
        #     "global_cnn_encoding: ", global_cnn_encoding.shape, "\n"
        # )

        #and putting all of them together
        tokens_info = torch.cat(
            (
                local_bilstm_encoding,
                local_cnn_encoding,
                local_deptree_encoding
            ), dim=-1)

        #vector associated to the highlighted token
        pointed_token_info = tokens_info[:, pointed_token_idx,:].expand(sent_len, -1).unsqueeze(0)

        #expanding global info
        global_info = global_cnn_encoding.expand(-1, sent_len, -1)

        # print(
        #     "tokens_info: ", tokens_info.shape, "\n",
        #     "pointed_token_info: ", pointed_token_info.shape, "\n",
        #     "global_info: ", global_info.shape, "\n"
        # )

        #finals inputs are a concatenation of token's info, highlighted token's info and global info
        sentence_encoding = torch.cat(
            (
                tokens_info,
                pointed_token_info,
                global_info
            ), dim=-1)
        sentence_encoding = self.dropout(sentence_encoding)

        # print(
        #     "sentence_encoding: ", sentence_encoding.shape, "\n"
        # )

        #output entity type
        sentence_one_vector = torch.cat([global_cnn_encoding.squeeze(0), tokens_info[:, pointed_token_idx,:]], dim = -1)
        # print(
        #     "sentence_one_vector: ", sentence_one_vector.shape, "\n"
        # )
        entitytype_output = F.softmax(self.entity_type_decoder(sentence_one_vector), dim = -1)

        #output entities
        _, entities_output = self.entities_crf_decoder(sentence_encoding)

        #output relations
        relations_output = torch.sigmoid(self.relations_decoder(sentence_encoding))

        # print(
        #     "entitytype_output: ", entitytype_output.shape, "\n",
        #     "entities_output: ", len(entities_output), "\n",
        #     "relations_output: ", relations_output.shape, "\n"
        # )

        return sentence_encoding, entitytype_output, entities_output, relations_output

class DependencyJointModel(nn.Module):

    def __init__(
        self,
        embedding_size,
        wv,
        no_chars,
        charencoding_size,
        no_dependencies,
        dependency_size,
        entity_type_size,
        entity_tag_size,
        tree_lstm_hidden_size,
        bilstm_hidden_size,
        dropout_chance,
        no_entity_types,
        no_entity_tags,
        no_relations
        ):

        super().__init__()

        self.wv = wv

        #INPUT PROCESSING

        #Word Embedding layer
        self.word_embedding = PretrainedEmbedding(wv)

        #Char Embedding layer
        self.char_embedding = CharCNN(1, no_chars, charencoding_size)

        #Word-encoding BiLSTM
        self.word_bilstm = BiLSTM(embedding_size + charencoding_size, bilstm_hidden_size//2, return_sequence=True)

        #OUTPUT
        self.dropout = nn.Dropout(dropout_chance)

        #Entity type
        self.entities_types_crf_decoder = CRF(bilstm_hidden_size, no_entity_types)

        #Entites
        self.entities_tags_crf_decoder = CRF(bilstm_hidden_size, no_entity_tags)

        #Relations
        self.entity_type_embedding = nn.Embedding(no_entity_types, entity_type_size)
        self.entity_tag_embedding = nn.Embedding(no_entity_tags, entity_tag_size)
        self.dependency_embedding = nn.Embedding(no_dependencies, dependency_size)

        tree_lstm_input_size = entity_type_size + entity_tag_size + bilstm_hidden_size + dependency_size
        self.tree_lstm = ChildSumTreeLSTM(tree_lstm_input_size, tree_lstm_hidden_size)

        self.relations_decoder = nn.Linear(2*tree_lstm_hidden_size, no_relations)

    def forward(self, X, relation=False):
        return self.relation_forward(X) if relation else self.entities_forward(X)

    def entities_forward(self, X):
        (
            word_inputs,
            char_inputs
        ) = X

        #obtaining embeddings vectors
        word_embeddings = self.word_embedding(word_inputs)
        char_embeddings = self.char_embedding(char_inputs)

        bilstm_inputs = torch.cat(
            (
                word_embeddings,
                char_embeddings,
            ), dim=-1)

        #encoding those inputs
        bilstm_encoding, _ = self.word_bilstm(bilstm_inputs)
        bilstm_encoding = self.dropout(bilstm_encoding)

        #OUTPUTS

        #entity-types
        _, entities_types_output = self.entities_types_crf_decoder(bilstm_encoding)

        #entity-tags
        _, entities_tags_output = self.entities_tags_crf_decoder(bilstm_encoding)

        return bilstm_encoding, entities_types_output, entities_tags_output

    def relation_forward(self, X):
        (
            bi_lstm_encoding,
            entities_types_output,
            entities_tags_output,
            dependency_inputs,
            trees,
            origin,
            destination
        ) = X

        entities_types_inputs = torch.tensor(entities_types_output, dtype = torch.long).unsqueeze(0)
        entities_tags_inputs = torch.tensor(entities_tags_output, dtype = torch.long).unsqueeze(0)

        entities_types_embeddings = self.entity_type_embedding(entities_types_inputs)
        entities_tags_embeddings = self.entity_tag_embedding(entities_tags_inputs)
        dependency_embeddings = self.dependency_embedding(dependency_inputs)

        tree_lstm_inputs = torch.cat([bi_lstm_encoding, entities_types_embeddings, entities_tags_embeddings, dependency_embeddings], dim = -1)

        origin_tree_encoding = self.tree_lstm(trees[origin], tree_lstm_inputs.squeeze(0))[1]
        destination_tree_encoding = self.tree_lstm(trees[destination], tree_lstm_inputs.squeeze(0))[1]

        relation_pair = torch.cat([origin_tree_encoding, destination_tree_encoding], dim = -1)

        return F.softmax(self.relations_decoder(relation_pair), dim = -1)


class ShortestDependencyPathJointModel(nn.Module):

    def __init__(
        self,
        embedding_size,
        wv,
        no_chars,
        charencoding_size,
        no_postags,
        postag_size,
        no_dependencies,
        dependency_size,
        entity_type_size,
        entity_tag_size,
        bilstm_shared_hidden_size,
        tree_lstm_hidden_size,
        bilstm_rels_hidden_size,
        relations_dense_size,
        shared_dropout_chance,
        relations_dropout_chance,
        no_entity_types,
        no_entity_tags,
        no_relations
        ):

        super().__init__()

        self.wv = wv

        #INPUT PROCESSING

        #Word Embedding layer
        self.word_embedding = PretrainedEmbedding(wv)

        #Char Embedding layer
        self.char_embedding = CharCNN(1, no_chars, charencoding_size)

        #POSTtag Embedding layer
        self.postag_embedding = nn.Embedding(no_postags, postag_size)

        #Word-encoding BiLSTMs
        self.word_bilstm1 = BiLSTM(embedding_size + charencoding_size + postag_size, bilstm_shared_hidden_size // 2, return_sequence=True)
        self.word_bilstm2 = BiLSTM(bilstm_shared_hidden_size, bilstm_shared_hidden_size//2, return_sequence=True)

        #OUTPUT
        self.dropout_shared = nn.Dropout(shared_dropout_chance)

        #Entity type
        self.entities_types_crf_decoder = CRF(bilstm_shared_hidden_size, no_entity_types)

        #Entites
        self.entities_tags_crf_decoder = CRF(bilstm_shared_hidden_size, no_entity_tags)

        #Relations
        self.entity_type_embedding = nn.Embedding(no_entity_types, entity_type_size)
        self.entity_tag_embedding = nn.Embedding(no_entity_tags, entity_tag_size)
        self.dependency_embedding = nn.Embedding(no_dependencies, dependency_size)

        tree_lstm_input_size = entity_type_size + entity_tag_size + bilstm_shared_hidden_size + dependency_size
        self.tree_lstm = ChildSumTreeLSTM(tree_lstm_input_size, tree_lstm_hidden_size)

        bilstm_rels_input_size = bilstm_shared_hidden_size + dependency_size + entity_type_size + entity_tag_size
        self.dep_path_bilstm = BiLSTM(bilstm_rels_input_size, bilstm_rels_hidden_size // 2, batch_first=True)

        relations_dense_input_size = 2*tree_lstm_hidden_size + bilstm_rels_hidden_size
        self.relations_dense = nn.Linear(relations_dense_input_size, relations_dense_size)

        self.dropout_relations = nn.Dropout(relations_dropout_chance)

        self.relations_decoder = nn.Linear(relations_dense_size, no_relations)

    def forward(self, X, relation=False):
        return self.relation_forward(X) if relation else self.entities_forward(X)

    def entities_forward(self, X):
        (
            word_inputs,
            char_inputs,
            postag_inputs
        ) = X

        #obtaining embeddings vectors
        word_embeddings = self.word_embedding(word_inputs)
        char_embeddings = self.char_embedding(char_inputs)
        postag_embeddings = self.postag_embedding(postag_inputs)

        bilstm_inputs = torch.cat(
            (
                word_embeddings,
                char_embeddings,
                postag_embeddings
            ), dim=-1)

        #encoding those inputs
        bilstm_encoding, _ = self.word_bilstm1(bilstm_inputs)
        bilstm_encoding, _ = self.word_bilstm2(bilstm_encoding)
        bilstm_encoding = self.dropout_shared(bilstm_encoding)

        #OUTPUTS

        #entity-types
        _, entities_types_output = self.entities_types_crf_decoder(bilstm_encoding)

        #entity-tags
        _, entities_tags_output = self.entities_tags_crf_decoder(bilstm_encoding)

        return bilstm_encoding, entities_types_output, entities_tags_output

    def relation_forward(self, X):
        (
            bi_lstm_encoding,
            entities_types_output,
            entities_tags_output,
            dependency_inputs,
            trees,
            origin,
            destination
        ) = X

        print(origin, destination)


        entities_types_inputs = torch.tensor(entities_types_output, dtype = torch.long).unsqueeze(0)
        entities_tags_inputs = torch.tensor(entities_tags_output, dtype = torch.long).unsqueeze(0)

        entities_types_embeddings = self.entity_type_embedding(entities_types_inputs)
        entities_tags_embeddings = self.entity_tag_embedding(entities_tags_inputs)
        dependency_embeddings = self.dependency_embedding(dependency_inputs)

        # print(
        #     "entities_types_embeddings", entities_types_embeddings.shape, "\n",
        #     "entities_tags_embeddings", entities_tags_embeddings.shape, "\n",
        #     "dependency_embeddings", dependency_embeddings.shape, "\n"
        # )

        tree_lstm_inputs = torch.cat([bi_lstm_encoding, entities_types_embeddings, entities_tags_embeddings, dependency_embeddings], dim = -1)
        origin_tree_encoding = self.tree_lstm(trees[origin], tree_lstm_inputs.squeeze(0))[1]
        destination_tree_encoding = self.tree_lstm(trees[destination], tree_lstm_inputs.squeeze(0))[1]

        bilstm_inputs = torch.cat([bi_lstm_encoding, entities_types_embeddings, entities_tags_embeddings, dependency_embeddings], dim=-1)
        dep_path = Tree.path(trees[origin], trees[destination])
        bilstm_inputs_in_path = torch.cat([bilstm_inputs[:, node.idx,:].unsqueeze(0) for node in dep_path], dim=1)

        # print(len(dep_path))
        # print(
        #     "bilstm_inputs_in_path", bilstm_inputs_in_path.shape, "\n",
        # )

        bilstm_out, _ = self.dep_path_bilstm(bilstm_inputs_in_path)

        # print(
        #     "bilstm_out", bilstm_out.shape, "\n",
        #     "origin_tree_encoding", origin_tree_encoding.shape, "\n",
        #     "destination_tree_encoding", destination_tree_encoding.shape
        # )

        relations_dense_input = torch.cat([origin_tree_encoding, bilstm_out, destination_tree_encoding], dim=-1)
        dense_encoding = self.dropout_relations(torch.tanh(self.relations_dense(relations_dense_input)))

        return F.softmax(self.relations_decoder(dense_encoding), dim = -1)


class BERTStackedBiLSTMCRFModel(nn.Module):
    def __init__(
        self,
        embedding_size,
        bert_embedding_size,
        wv,
        no_chars,
        charencoding_size,
        no_postags,
        postag_size,
        bilstm_hidden_size,
        dropout_chance,
        no_entity_types,
        no_entity_tags,
        ablation = {
            "bert_embedding": True,
            "word_embedding": False,
            "chars_info": True,
            "postag": True,
            "dependency": False,
        }
        ):

        super().__init__()

        self.wv = wv
        self.ablation = ablation
        self.bert_size = bert_embedding_size

        #INPUT PROCESSING

        #Word Embedding layer
        if ablation["word_embedding"]:
            self.word_embedding = PretrainedEmbedding(wv)

        #Char Embedding layer
        self.char_embedding = CharCNN(1, no_chars, charencoding_size)
        #POSTtag Embedding layer
        self.postag_embedding = nn.Embedding(no_postags, postag_size)

        bilstm_input_size = (
           (embedding_size if ablation["word_embedding"] else 0) +
           (bert_embedding_size if ablation["bert_embedding"] else 0) +
           (charencoding_size if ablation["chars_info"] else 0) +
           (postag_size if ablation["postag"] else 0)
           #(dependency_size if ablation["dependency"] else 0)
        )
        print(bilstm_input_size)
        #Word-encoding BiLSTMs
        self.word_bilstm1 = BiLSTM(bilstm_input_size, bilstm_hidden_size // 2, return_sequence=True)
        self.word_bilstm2 = BiLSTM(bilstm_hidden_size, bilstm_hidden_size//2, return_sequence=True)

        #OUTPUT
        self.dropout_in = nn.Dropout2d(p=0.5)
        self.dropout_rnn_in = nn.Dropout(p=0.5)
        self.dropout = nn.Dropout(dropout_chance)

        #Entity type
        self.entities_types_crf_decoder = CRF(bilstm_hidden_size, no_entity_types)

        #Entites
        self.entities_tags_crf_decoder = CRF(bilstm_hidden_size, no_entity_tags)

    def forward(self, X):
        (
            word_inputs,
            char_inputs,
            bert_embeddings,
            postag_inputs
        ) = X

        #obtaining embeddings vectors
        word_embeddings = self.word_embedding(word_inputs) if self.ablation["word_embedding"] else None

        char_embeddings = self.char_embedding(char_inputs) if self.ablation["chars_info"] else None
        char_embeddings = self.dropout_in(char_embeddings) if self.ablation["chars_info"] else None

        postag_embeddings = self.postag_embedding(postag_inputs) if self.ablation["postag"] else None
        postag_embeddings = self.dropout_in(postag_embeddings) if self.ablation["postag"] else None


        bert_embeddings = bert_embeddings if self.ablation["bert_embedding"] else None
        bert_embeddings = bert_embeddings[:,:,:self.bert_size]

        bilstm_inputs = torch.cat([x for x in [
                word_embeddings,
                char_embeddings,
                bert_embeddings,
                postag_embeddings
            ] if x is not None], dim = -1)

        #encoding those inputs
        bilstm_inputs = self.dropout_rnn_in(bilstm_inputs)
        bilstm_encoding, _ = self.word_bilstm1(bilstm_inputs)
        bilstm_encoding = self.dropout(bilstm_encoding)
        bilstm_encoding, _ = self.word_bilstm2(bilstm_encoding)
        bilstm_encoding = self.dropout(bilstm_encoding)


        #OUTPUTS

        #entity-types
        _, entities_types_output = self.entities_types_crf_decoder(bilstm_encoding)

        #entity-tags
        _, entities_tags_output = self.entities_tags_crf_decoder(bilstm_encoding)

        return bilstm_encoding, entities_types_output, entities_tags_output

class StackedBiLSTMCRFModel(nn.Module):
    def __init__(
        self,
        embedding_size,
        wv,
        no_chars,
        charencoding_size,
        no_postags,
        postag_size,
        bilstm_hidden_size,
        dropout_chance,
        no_entity_types,
        no_entity_tags,
        ):

        super().__init__()

        self.wv = wv

        #INPUT PROCESSING

        #Word Embedding layer
        self.word_embedding = PretrainedEmbedding(wv)

        #Char Embedding layer
        self.char_embedding = CharCNN(1, no_chars, charencoding_size)

        #POSTtag Embedding layer
        self.postag_embedding = nn.Embedding(no_postags, postag_size)

        #Word-encoding BiLSTMs
        self.word_bilstm1 = BiLSTM(embedding_size + charencoding_size + postag_size, bilstm_hidden_size // 2, return_sequence=True)
        self.word_bilstm2 = BiLSTM(bilstm_hidden_size, bilstm_hidden_size//2, return_sequence=True)

        #OUTPUT
        self.dropout = nn.Dropout(dropout_chance)

        #Entity type
        self.entities_types_crf_decoder = CRF(bilstm_hidden_size, no_entity_types)

        #Entites
        self.entities_tags_crf_decoder = CRF(bilstm_hidden_size, no_entity_tags)

    def forward(self, X):
        (
            word_inputs,
            char_inputs,
            postag_inputs
        ) = X

        word_embeddings = self.word_embedding(word_inputs)
        char_embeddings = self.char_embedding(char_inputs)
        postag_embeddings = self.postag_embedding(postag_inputs)

        bilstm_inputs = torch.cat(
            (
                word_embeddings,
                char_embeddings,
                postag_embeddings
            ), dim=-1)

        #encoding those inputs
        bilstm_encoding, _ = self.word_bilstm1(bilstm_inputs)
        bilstm_encoding, _ = self.word_bilstm2(bilstm_encoding)
        bilstm_encoding = self.dropout(bilstm_encoding)

        #OUTPUTS

        #entity-types
        _, entities_types_output = self.entities_types_crf_decoder(bilstm_encoding)

        #entity-tags
        _, entities_tags_output = self.entities_tags_crf_decoder(bilstm_encoding)

        return bilstm_encoding, entities_types_output, entities_tags_output

class DependencyRelationsModel(nn.Module):
    def __init__(
        self,
        word_encoding_size,
        no_dependencies,
        dependency_size,
        entity_type_size,
        entity_tag_size,
        bilstm_hidden_size,
        tree_lstm_hidden_size,
        relations_dense_size,
        dropout_chance,
        no_entity_types,
        no_entity_tags,
        no_relations
        ):

        super().__init__()

        #Relations
        self.entity_type_embedding = nn.Embedding(no_entity_types, entity_type_size)
        self.entity_tag_embedding = nn.Embedding(no_entity_tags, entity_tag_size)
        self.dependency_embedding = nn.Embedding(no_dependencies, dependency_size)

        inputs_total_size = word_encoding_size + entity_type_size + entity_tag_size + dependency_size

        self.tree_lstm = ChildSumTreeLSTM(inputs_total_size, tree_lstm_hidden_size)
        self.dep_path_bilstm = BiLSTM(inputs_total_size, bilstm_hidden_size // 2, batch_first=True)

        relations_dense_input_size = 2*tree_lstm_hidden_size + bilstm_hidden_size
        self.relations_dense = nn.Linear(relations_dense_input_size, relations_dense_size)

        self.dropout = nn.Dropout(dropout_chance)

        self.relations_decoder = nn.Linear(relations_dense_size, no_relations)

    def forward(self, X):
        (
            words_encoding,
            entities_types_output,
            entities_tags_output,
            dependency_inputs,
            trees,
            origin,
            destination
        ) = X

        entities_types_inputs = torch.tensor(entities_types_output, dtype = torch.long).unsqueeze(0)
        entities_tags_inputs = torch.tensor(entities_tags_output, dtype = torch.long).unsqueeze(0)

        entities_types_embeddings = self.entity_type_embedding(entities_types_inputs)
        entities_tags_embeddings = self.entity_tag_embedding(entities_tags_inputs)
        dependency_embeddings = self.dependency_embedding(dependency_inputs)

        inputs = torch.cat([words_encoding, entities_types_embeddings, entities_tags_embeddings, dependency_embeddings], dim = -1)

        origin_tree_encoding = self.tree_lstm(trees[origin], inputs.squeeze(0))[1]
        destination_tree_encoding = self.tree_lstm(trees[destination], inputs.squeeze(0))[1]

        dep_path = Tree.path(trees[origin], trees[destination])
        inputs_in_path = torch.cat([inputs[:, node.idx,:].unsqueeze(0) for node in dep_path], dim=1)
        bilstm_out, _ = self.dep_path_bilstm(inputs_in_path)

        relations_dense_input = torch.cat([origin_tree_encoding, bilstm_out, destination_tree_encoding], dim=-1)
        dense_encoding = self.dropout(torch.tanh(self.relations_dense(relations_dense_input)))

        return F.softmax(self.relations_decoder(dense_encoding), dim=-1)

class ShortestDependencyPathRelationsModel(nn.Module):

    def __init__(
        self,
        embedding_size,
        wv,
        no_chars,
        charencoding_size,
        no_postags,
        postag_size,
        no_dependencies,
        dependency_size,
        entity_type_size,
        entity_tag_size,
        bilstm_words_hidden_size,
        bilstm_path_hidden_size,
        dropout_chance,
        no_entity_types,
        no_entity_tags,
        no_relations
        ):

        super().__init__()

        self.wv = wv


        #INPUT PROCESSING

        #Word Embedding layer
        self.word_embedding = PretrainedEmbedding(wv)

        #Char Embedding layer
        self.char_embedding = CharCNN(1, no_chars, charencoding_size)

        #POSTtag Embedding layer
        self.postag_embedding = nn.Embedding(no_postags, postag_size)

        #Word-encoding BiLSTMs
        self.word_bilstm1 = BiLSTM(embedding_size + charencoding_size + postag_size, bilstm_words_hidden_size // 2, return_sequence=True)
        self.word_bilstm2 = BiLSTM(bilstm_words_hidden_size, bilstm_words_hidden_size//2, return_sequence=True)

        self.dropout = nn.Dropout(dropout_chance)

        self.entity_type_embedding = nn.Embedding(no_entity_types, entity_type_size)
        self.entity_tag_embedding = nn.Embedding(no_entity_tags, entity_tag_size)
        self.dependency_embedding = nn.Embedding(no_dependencies, dependency_size)

        bilstm_path_input_size = bilstm_words_hidden_size + dependency_size + entity_type_size + entity_tag_size
        self.dep_path_bilstm = nn.LSTM(bilstm_path_input_size, bilstm_path_hidden_size, batch_first=True)

        self.binary = no_relations == 1
        self.relations_decoder = nn.Linear(bilstm_path_hidden_size, no_relations)

    def forward(self, X):

        (
            word_inputs,
            char_inputs,
            postag_inputs,
            entities_types_output,
            entities_tags_output,
            dependency_inputs,
            trees,
            origin,
            destination
        ) = X

        #obtaining embeddings vectors
        word_embeddings = self.word_embedding(word_inputs)
        char_embeddings = self.char_embedding(char_inputs)
        postag_embeddings = self.postag_embedding(postag_inputs)

        bilstm_inputs = torch.cat(
            (
                word_embeddings,
                char_embeddings,
                postag_embeddings
            ), dim=-1)

        #encoding those inputs
        bilstm_encoding, _ = self.word_bilstm1(bilstm_inputs)
        bilstm_encoding, _ = self.word_bilstm2(bilstm_encoding)
        bilstm_encoding = self.dropout(bilstm_encoding)

        entities_types_inputs = torch.tensor(entities_types_output, dtype = torch.long).unsqueeze(0)
        entities_tags_inputs = torch.tensor(entities_tags_output, dtype = torch.long).unsqueeze(0)

        entities_types_embeddings = self.entity_type_embedding(entities_types_inputs)
        entities_tags_embeddings = self.entity_tag_embedding(entities_tags_inputs)
        dependency_embeddings = self.dependency_embedding(dependency_inputs)

        bilstm_inputs = torch.cat([bilstm_encoding, entities_types_embeddings, entities_tags_embeddings, dependency_embeddings], dim=-1)
        dep_path = Tree.path(trees[origin], trees[destination])
        bilstm_inputs_in_path = torch.cat([bilstm_inputs[:, node.idx,:].unsqueeze(0) for node in dep_path], dim=1)

        bilstm_out, _ = self.dep_path_bilstm(bilstm_inputs_in_path)
        bilstm_out = bilstm_out[:, -1, :]

        if self.binary:
            return torch.sigmoid(self.relations_decoder(bilstm_out))
        return F.softmax(self.relations_decoder(bilstm_out), dim=-1)

class OracleParserModel(nn.Module):
    def __init__(self,
        word_vector_size,
        wv,
        no_chars,
        char_embedding_size,
        lstm_hidden_size,
        dropout_ratio,
        hidden_dense_size,
        actions_no,
    ):
        super().__init__()

        self.wv = wv

        self.word_embedding = PretrainedEmbedding(wv)

        self.char_embedding = CharCNN(1, no_chars, char_embedding_size)

        lstm_input_size = word_vector_size + char_embedding_size
        self.lstmencoder_sent = nn.LSTM(lstm_input_size, lstm_hidden_size, batch_first=True)
        self.lstmencoder_stack = nn.LSTM(lstm_input_size, lstm_hidden_size, batch_first=True)

        self.dropout_sent = nn.Dropout(p = dropout_ratio)
        self.dropout_stack = nn.Dropout(p = dropout_ratio)

        self.dense_sent = nn.Linear(self.lstmencoder_sent.hidden_size, hidden_dense_size)
        self.dense_stack = nn.Linear(self.lstmencoder_stack.hidden_size, hidden_dense_size)

        dense_input_size = 2*hidden_dense_size

        self.action_dense = nn.Linear(dense_input_size, actions_no)

    def forward(self, X):
        stack_word_inputs, stack_char_inputs, sent_word_inputs, sent_char_inputs = X

        word_embeddings_stack = self.word_embedding(stack_word_inputs)
        word_embeddings_sent = self.word_embedding(sent_word_inputs)

        char_embeddings_stack = self.char_embedding(stack_char_inputs)
        char_embeddings_sent = self.char_embedding(sent_char_inputs)

        stack_lstm_inputs = torch.cat(
            (
                word_embeddings_stack,
                char_embeddings_stack,
            ), dim=-1)

        sent_lstm_inputs = torch.cat(
            (
                word_embeddings_sent,
                char_embeddings_sent,
            ), dim=-1)

        stack_encoded, _ = self.lstmencoder_stack(stack_lstm_inputs)
        sent_encoded, _ = self.lstmencoder_sent(sent_lstm_inputs)

        stack_encoded = stack_encoded[:, -1, :]
        sent_encoded = sent_encoded[:, -1, :]

        stack_encoded = self.dropout_stack(stack_encoded)
        sent_encoded = self.dropout_sent(sent_encoded)

        stack_encoded = torch.tanh(self.dense_stack(stack_encoded))
        sent_encoded = torch.tanh(self.dense_sent(sent_encoded))

        encoded = torch.cat([stack_encoded, sent_encoded], 1)

        action_out = F.softmax(self.action_dense(encoded), 1)

        return action_out


class TreeBiLSTMPathModel(nn.Module):
    def __init__(
        self,
        embedding_size,
        wv,
        no_chars,
        charencoding_size,
        no_postags,
        postag_size,
        no_dependencies,
        dependency_size,
        no_entity_types,
        entity_type_size,
        no_entity_tags,
        entity_tag_size,
        bilstm_path_hidden_size,
        lstm_path_hidden_size,
        tree_lstm_hidden_size,
        bilstm_dropout_chance,
        lstm_dropout_chance,
        tree_lstm_dropout_chance,
        no_relations,
        ablation = {
            "chars_info": True,
            "postag": True,
            "dependency": True,
            "entity_type": True,
            "entity_tag": True
        }
    ):
        super().__init__()

        self.ablation = ablation

        self.word_embedding = PretrainedEmbedding(wv)
        self.char_embedding = CharCNN(1, no_chars, charencoding_size)
        self.postag_embedding = nn.Embedding(no_postags, postag_size)
        self.dependency_embedding = nn.Embedding(no_dependencies, dependency_size)
        self.entity_type_embedding = nn.Embedding(no_entity_types, entity_type_size)
        self.entity_tag_embedding = nn.Embedding(no_entity_tags, entity_tag_size)

        bilstm_input_size = (
            embedding_size +
            (charencoding_size if ablation["chars_info"] else 0) +
            (postag_size if ablation["postag"] else 0) +
            (dependency_size if ablation["dependency"] else 0) +
            (entity_type_size if ablation["entity_type"] else 0) +
            (entity_tag_size if ablation["entity_tag"] else 0)
        )

        self.bilstm1 = BiLSTM(bilstm_input_size, bilstm_path_hidden_size//2, batch_first = True, return_sequence=True)
        self.dropout1 = nn.Dropout(bilstm_dropout_chance)
        self.bilstm2 = nn.LSTM(bilstm_path_hidden_size, lstm_path_hidden_size, batch_first = True)
        self.dropout2 = nn.Dropout(lstm_dropout_chance)

        self.tree_lstm = ChildSumTreeLSTM(bilstm_input_size, tree_lstm_hidden_size)
        self.dropout3 = nn.Dropout(tree_lstm_dropout_chance)

        dense_input_size = lstm_path_hidden_size + 2*tree_lstm_hidden_size
        self.dense = nn.Linear(dense_input_size, no_relations)

        self.binary = no_relations == 1


    def forward(self, X):
        (
            word_inputs,
            char_inputs,
            postag_inputs,
            dependency_inputs,
            entity_type_inputs,
            entity_tag_inputs,
            trees,
            origin,
            destination
        ) = X

        word_embeddings = self.word_embedding(word_inputs)
        char_embeddings = self.char_embedding(char_inputs) if self.ablation["chars_info"] else None
        postag_embeddings = self.postag_embedding(postag_inputs) if self.ablation["postag"] else None
        dependency_embeddings = self.dependency_embedding(dependency_inputs) if self.ablation["dependency"] else None
        type_embeddings = self.entity_type_embedding(entity_type_inputs) if self.ablation["entity_type"] else None
        tag_embeddings = self.entity_tag_embedding(entity_tag_inputs) if self.ablation["entity_tag"] else None

        inputs = torch.cat([x for x in [
            word_embeddings,
            char_embeddings,
            postag_embeddings,
            dependency_embeddings,
            type_embeddings,
            tag_embeddings
        ] if x is not None], dim = -1)

        dep_path = Tree.path(trees[origin], trees[destination])
        bilstm_inputs_in_path = torch.cat([inputs[:, node.idx,:].unsqueeze(0) for node in dep_path], dim=1)

        path_encoded, _ = self.bilstm1(bilstm_inputs_in_path)
        path_encoded = self.dropout1(path_encoded)
        path_encoded, _ = self.bilstm2(path_encoded)
        path_encoded = path_encoded[:,-1,:]
        path_encoded = self.dropout2(path_encoded)

        origin_tree_encoding = self.tree_lstm(trees[origin], inputs.squeeze(0))[1]
        destination_tree_encoding = self.tree_lstm(trees[destination], inputs.squeeze(0))[1]

        encoding = torch.cat([origin_tree_encoding, destination_tree_encoding, path_encoded], dim = -1)
        encoding = self.dropout3(encoding)

        if self.binary:
            return torch.sigmoid(self.dense(encoding))
        return F.softmax(self.dense(encoding), dim=-1)


class BERTTreeBiLSTMPathModel(nn.Module):
    def __init__(
        self,
        embedding_size,
        wv,
        bert_size,
        no_chars,
        charencoding_size,
        no_postags,
        postag_size,
        no_dependencies,
        dependency_size,
        no_entity_types,
        entity_type_size,
        no_entity_tags,
        entity_tag_size,
        bilstm_path_hidden_size,
        lstm_path_hidden_size,
        tree_lstm_hidden_size,
        bilstm_dropout_chance,
        lstm_dropout_chance,
        tree_lstm_dropout_chance,
        no_relations,
        ablation = {
            "bert_embedding": True,
            "word_embedding": True,
            "chars_info": True,
            "postag": True,
            "dependency": True,
            "entity_type": True,
            "entity_tag": True
        }
    ):
        super().__init__()

        self.ablation = ablation
        self.bert_size = bert_size

        if ablation["word_embedding"]:
            self.word_embedding = PretrainedEmbedding(wv)
        self.char_embedding = CharCNN(1, no_chars, charencoding_size)
        self.postag_embedding = nn.Embedding(no_postags, postag_size)
        self.dependency_embedding = nn.Embedding(no_dependencies, dependency_size)
        self.entity_type_embedding = nn.Embedding(no_entity_types, entity_type_size)
        self.entity_tag_embedding = nn.Embedding(no_entity_tags, entity_tag_size)

        bilstm_input_size = (
            (bert_size if ablation["bert_embedding"] else 0) +
            (embedding_size if ablation["word_embedding"] else 0) +
            (charencoding_size if ablation["chars_info"] else 0) +
            (postag_size if ablation["postag"] else 0) +
            (dependency_size if ablation["dependency"] else 0) +
            (entity_type_size if ablation["entity_type"] else 0) +
            (entity_tag_size if ablation["entity_tag"] else 0)
        )

        self.bilstm1 = BiLSTM(bilstm_input_size, bilstm_path_hidden_size//2, batch_first = True, return_sequence=True)
        self.dropout1 = nn.Dropout(bilstm_dropout_chance)
        self.bilstm2 = nn.LSTM(bilstm_path_hidden_size, lstm_path_hidden_size, batch_first = True)
        self.dropout2 = nn.Dropout(lstm_dropout_chance)

        self.tree_lstm = ChildSumTreeLSTM(bilstm_input_size, tree_lstm_hidden_size)
        self.dropout3 = nn.Dropout(tree_lstm_dropout_chance)

        dense_input_size = lstm_path_hidden_size + 2*tree_lstm_hidden_size
        self.dense = nn.Linear(dense_input_size, no_relations)


    def forward(self, X):
        (
            word_inputs,
            char_inputs,
            bert_embeddings,
            postag_inputs,
            dependency_inputs,
            entity_type_inputs,
            entity_tag_inputs,
            trees,
            origin,
            destination
        ) = X

        bert_embeddings = bert_embeddings[:,:,:self.bert_size] if self.ablation["bert_embedding"] else None
        word_embeddings = self.word_embedding(word_inputs) if self.ablation["word_embedding"] else None
        char_embeddings = self.char_embedding(char_inputs) if self.ablation["chars_info"] else None
        postag_embeddings = self.postag_embedding(postag_inputs) if self.ablation["postag"] else None
        dependency_embeddings = self.dependency_embedding(dependency_inputs) if self.ablation["dependency"] else None
        type_embeddings = self.entity_type_embedding(entity_type_inputs) if self.ablation["entity_type"] else None
        tag_embeddings = self.entity_tag_embedding(entity_tag_inputs) if self.ablation["entity_tag"] else None

        inputs = torch.cat([x for x in [
            bert_embeddings,
            word_embeddings,
            char_embeddings,
            postag_embeddings,
            dependency_embeddings,
            type_embeddings,
            tag_embeddings
        ] if x is not None], dim = -1)

        dep_path = Tree.path(trees[origin], trees[destination])
        bilstm_inputs_in_path = torch.cat([inputs[:, node.idx,:].unsqueeze(0) for node in dep_path], dim=1)

        path_encoded, _ = self.bilstm1(bilstm_inputs_in_path)
        path_encoded = self.dropout1(path_encoded)
        path_encoded, _ = self.bilstm2(path_encoded)
        path_encoded = path_encoded[:,-1,:]
        path_encoded = self.dropout2(path_encoded)

        origin_tree_encoding = self.dropout3(self.tree_lstm(trees[origin], inputs.squeeze(0))[1])
        destination_tree_encoding = self.dropout3(self.tree_lstm(trees[destination], inputs.squeeze(0))[1])

        encoding = torch.cat([origin_tree_encoding, destination_tree_encoding, path_encoded], dim = -1)

        return torch.sigmoid(self.dense(encoding))


class BERTBiLSTMRelationModel(nn.Module):

    def __init__(
            self,
            embedding_size,
            wv,
            bert_size,
            no_chars,
            charencoding_size,
            no_postags,
            postag_size,
            no_dependencies,
            dependency_size,
            no_entity_types,
            entity_type_size,
            no_entity_tags,
            entity_tag_size,
            bilstm_entities_size,
            dropout_entities_chance,
            lstm_sentence_input_size,
            lstm_sentence_hidden_size,
            dropout_sentence1_chance,
            dropout_sentence2_chance,
            no_relations,
            ablation = {
                "bert_embedding": True,
                "word_embedding": True,
                "chars_info": True,
                "postag": True,
                "dependency": True,
                "entity_type": True,
                "entity_tag": True
            }
        ):

        super().__init__()

        self.ablation = ablation
        self.bert_size = bert_size

        if ablation["word_embedding"]:
            self.word_embedding = PretrainedEmbedding(wv)
        self.char_embedding = CharCNN(1, no_chars, charencoding_size)
        self.postag_embedding = nn.Embedding(no_postags, postag_size)
        self.dependency_embedding = nn.Embedding(no_dependencies, dependency_size)
        self.entity_type_embedding = nn.Embedding(no_entity_types, entity_type_size)
        self.entity_tag_embedding = nn.Embedding(no_entity_tags, entity_tag_size)

        bilstm_input_size = (
            (bert_size if ablation["bert_embedding"] else 0) +
            (embedding_size if ablation["word_embedding"] else 0) +
            (charencoding_size if ablation["chars_info"] else 0) +
            (postag_size if ablation["postag"] else 0) +
            (dependency_size if ablation["dependency"] else 0) +
            (entity_type_size if ablation["entity_type"] else 0) +
            (entity_tag_size if ablation["entity_tag"] else 0)
        )

        self.bilstm_origin = BiLSTM(bilstm_input_size, bilstm_entities_size//2, batch_first = True, return_sequence=False)
        self.bilstm_destination = BiLSTM(bilstm_input_size, bilstm_entities_size//2, batch_first = True, return_sequence=False)
        self.bilstm_sentence = BiLSTM(bilstm_input_size, lstm_sentence_input_size//2, batch_first = True, return_sequence=True)
        self.lstm_sentence = nn.LSTM(lstm_sentence_input_size, lstm_sentence_hidden_size, batch_first = True)

        self.dropout_entities = nn.Dropout(dropout_entities_chance)
        self.dropout_sentence1 = nn.Dropout(dropout_sentence1_chance)
        self.dropout_sentence2 = nn.Dropout(dropout_sentence2_chance)

        dense_input_size = lstm_sentence_hidden_size + 2*bilstm_entities_size
        self.dense = nn.Linear(dense_input_size, no_relations)

    def forward(self, X):
        (
            word_inputs,
            char_inputs,
            bert_embeddings,
            postag_inputs,
            dependency_inputs,
            entity_type_inputs,
            entity_tag_inputs,
            origin_tokens,
            destination_tokens
        ) = X

        bert_embeddings = bert_embeddings[:,:,:self.bert_size] if self.ablation["bert_embedding"] else None
        word_embeddings = self.word_embedding(word_inputs) if self.ablation["word_embedding"] else None
        char_embeddings = self.char_embedding(char_inputs) if self.ablation["chars_info"] else None
        postag_embeddings = self.postag_embedding(postag_inputs) if self.ablation["postag"] else None
        dependency_embeddings = self.dependency_embedding(dependency_inputs) if self.ablation["dependency"] else None
        type_embeddings = self.entity_type_embedding(entity_type_inputs) if self.ablation["entity_type"] else None
        tag_embeddings = self.entity_tag_embedding(entity_tag_inputs) if self.ablation["entity_tag"] else None

        inputs = torch.cat([x for x in [
            bert_embeddings,
            word_embeddings,
            char_embeddings,
            postag_embeddings,
            dependency_embeddings,
            type_embeddings,
            tag_embeddings
        ] if x is not None], dim = -1)

        bilstm_origin_inputs = torch.cat([inputs[:, idx,:].unsqueeze(0) for idx in origin_tokens], dim=1)
        origin_encoded = self.dropout_entities(self.bilstm_origin(bilstm_origin_inputs)[0])

        bilstm_destination_inputs = torch.cat([inputs[:, idx,:].unsqueeze(0) for idx in destination_tokens], dim=1)
        destination_encoded = self.dropout_entities(self.bilstm_destination(bilstm_destination_inputs)[0])

        sentence_encoded = self.dropout_sentence1(self.bilstm_sentence(inputs)[0])
        sentence_encoded = self.dropout_sentence2(self.lstm_sentence(sentence_encoded)[0][:,-1,:])

        encoding = torch.cat([origin_encoded, destination_encoded, sentence_encoded], dim = -1)

        return torch.sigmoid(self.dense(encoding))