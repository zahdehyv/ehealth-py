import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss, BCELoss
from tqdm import tqdm
from itertools import product

from scripts.submit import Algorithm
from scripts.utils import Collection, Keyphrase, Relation, Sentence
from python_json_config import ConfigBuilder

from numpy.random import permutation, shuffle
from sortedcollections import ValueSortedDict

from pathlib import Path

from kdtools.datasets import (
    SimpleWordIndexDataset,
    RelationsDependencyParseActionsDataset,
    RelationsEmbeddingDataset,
    SentenceEmbeddingDataset,
    JointModelDataset,
    DependencyJointModelDataset,
    DependencyBERTJointModelDataset,
    RelationsOracleDataset
)
from kdtools.models import (
    BiLSTMDoubleDenseOracleParser,
    EmbeddingAttentionBiLSTM_CRF,
    EmbeddingBiLSTM_CRF,
    BERT_TreeLSTM_BiLSTM_CNN_JointModel,
    DependencyJointModel,
    ShortestDependencyPathJointModel,
    StackedBiLSTMCRFModel,
    BERTStackedBiLSTMCRFModel,
    DependencyRelationsModel,
    ShortestDependencyPathRelationsModel,
    OracleParserModel,
    TreeBiLSTMPathModel
)
from kdtools.utils.bmewov import BMEWOV

from gensim.models.word2vec import Word2VecKeyedVectors

from numpy.random import random


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print(device)

class BiLSTMCRF_RelationsParsing(Algorithm):
    def __init__(self):
        self.models_taskA = {}
        self.model_taskB = None

    def run(self, collection: Collection, *args, taskA: bool, taskB: bool, **kargs):
        if taskA:
            self.run_taskA(collection)

        if taskB:
            self.run_taskB(collection)

    def run_taskA(self, collection: Collection):
        print("Running taskA")

        idx = 0
        for entity_type, model in self.models_taskA.items():
            print(entity_type)
            dataset = SentenceEmbeddingDataset(collection, model.wv)

            model.eval()
            for spans, sentence, X in tqdm(dataset.evaluation):
                X = X.view(1,-1)
                output = [dataset.labels[idx] for idx in model(X)[1]]
                kps = [[spans[idx] for idx in span_list] for span_list in BMEWOV.decode(output)]
                for kp_spans in kps:
                    idx += 1
                    sentence.keyphrases.append(Keyphrase(sentence, entity_type, idx, kp_spans))

    def run_taskB(self, collection: Collection):
        print("Running taskB")
        model = self.model_taskB
        dataset = RelationsEmbeddingDataset(collection, model.wv)

        model.eval()
        it = 0
        for spans, sentence, state in tqdm(dataset.evaluation):
            it += 1
            words = [sentence.text[start:end] for (start, end) in spans]
            while state[1]:
                o, t, h, d = state

                X = (
                    *dataset.encode_word_sequence(["<padding>"]+[words[i - 1] for i in o]).view(1,-1),
                    *dataset.encode_word_sequence([words[i - 1] for i in t]).view(1,-1)
                )
                output_act, output_rel = model(X)
                action = dataset.actions[torch.argmax(output_act)]
                relation = dataset.relations[torch.argmax(output_rel)]

                try:
                    if action in ["LEFT", "RIGHT"]:
                        if action == "LEFT":
                            origidx = t[-1]
                            destidx = o[-1]
                        else:
                            origidx = o[-1]
                            destidx = t[-1]

                        origins = [kp.id for kp in sentence.keyphrases if spans[origidx-1] in kp.spans]
                        destinations = [kp.id for kp in sentence.keyphrases if spans[destidx-1] in kp.spans]

                        for origin, destination in product(origins, destinations):
                            sentence.relations.append(Relation(sentence, origin, destination, relation))

                    dataset.actions_funcs[action]["do"](state, relation)
                except Exception as e:
                    # print(e)
                    print(it)
                    state[1].clear()


    def train(self, collection: Collection):
        builder = ConfigBuilder()

        model_taskB_config = builder.parse_config('./configs/config_BiLSTM-Double-Dense-Oracle-Parser.json')
        model_taskA_config = builder.parse_config('./configs/config_BiLSTM-CRF.json')
        train_taskB_config = builder.parse_config('./configs/config_Train_TaskB.json')
        train_taskA_config = builder.parse_config('./configs/config_Train_TaskA.json')

        self.models_taskA = {
            entity_type: self.train_taskA(
                collection,
                model_taskA_config,
                train_taskA_config.__getattr__(entity_type),
                entity_type
            )
            for entity_type in ["Concept", "Action", "Reference", "Predicate"]
        }
        # self.model_taskB = self.train_taskB(collection, model_taskB_config, train_taskB_config)

    def train_taskA(self, collection, model_config, train_config, entity_type):
        print(f"Training taskA-{entity_type} model.")  #this should be a log

        wv = Word2VecKeyedVectors.load(model_config.embedding_path)
        dataset = SentenceEmbeddingDataset(collection, wv, lambda x: x.label == entity_type)

        model = EmbeddingBiLSTM_CRF(
            len(dataset.labels),
            model_config.hidden_dim,
            wv
        )

        optimizer = optim.SGD(
            model.parameters(),
            lr=train_config.optimizer.lr,
            weight_decay=train_config.optimizer.weight_decay
        )

        for epoch in range(train_config.epochs):
            running_loss = 0
            for data in tqdm(dataset):
                X, y = data
                X = X.view(1,-1)
                model.zero_grad()
                loss = model.neg_log_likelihood(X, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            correct = 0
            total = 0
            with torch.no_grad():
                for data in tqdm(dataset):
                    X, y = data
                    X = X.view(1,-1)
                    _, predicted = model(X)
                    correct += sum(torch.tensor(predicted) == y).item()
                    total += len(predicted)

            print(f"[{epoch + 1}] loss: {running_loss / len(dataset) :0.3}")
            print(f"[{epoch + 1}] accuracy: {correct / total}")

        return model

    def train_taskB(self, collection: Collection, model_config, train_config):
        wv = Word2VecKeyedVectors.load(model_config.embedding_path)
        dataset = RelationsEmbeddingDataset(collection, wv)

        model = BiLSTMDoubleDenseOracleParser(
            dataset.word_vector_size,
            model_config.lstm_hidden_size,
            model_config.dropout_ratio,
            model_config.hidden_dense_size,
            wv,
            len(dataset.actions),
            len(dataset.relations)
        )

        optimizer = optim.Adam(
            model.parameters()
        )

        criterion_act = CrossEntropyLoss(weight = dataset.get_actions_weights())
        criterion_rel = CrossEntropyLoss(weight = dataset.get_relations_weights())

        for epoch in range(train_config.epochs):
            correct = 0
            correct_act = 0
            correct_rel = 0
            total_act = 0
            total_rel = 0
            total = 0
            running_loss_act = 0.0
            running_loss_rel = 0.0

            for data in tqdm(dataset):
                *X, y_act, y_rel = data
                X = [x.view(1, -1) for x in X]

                optimizer.zero_grad()

                # forward + backward + optimize
                model.train()
                output_act, output_rel = model(X)

                if y_rel is not None:
                    loss_rel = criterion_rel(output_rel, y_rel)
                    loss_rel.backward(retain_graph=True)
                    running_loss_rel += loss_rel.item()

                loss_act = criterion_act(output_act, y_act)
                loss_act.backward()
                running_loss_act += loss_act.item()

                optimizer.step()

                #collecting data for metrics
                model.eval()
                output_act, output_rel = model(X)
                predicted_act = torch.argmax(output_act, -1)
                predicted_rel = torch.argmax(output_rel, -1)

                total_act += 1
                total_rel += int(y_rel is not None)
                total += 1
                correct_act += int(predicted_act == y_act)
                correct_rel += int(predicted_rel == y_rel) if y_rel is not None else 0
                correct += int(predicted_act == y_act and (predicted_rel == y_rel if y_rel is not None else True))

            print(f"[{epoch + 1}] loss_act: {running_loss_act / total_act}")
            print(f"[{epoch + 1}] loss_rel: {running_loss_rel / total_rel}")
            print(f"[{epoch + 1}] accuracy_act: {correct_act / total_act}")
            print(f"[{epoch + 1}] accuracy_rel: {correct_rel / total_rel}")
            print(f"[{epoch + 1}] accuracy: {correct / total}")

        return model


class JointModel(Algorithm):

    def __init__(self):
        self.model = None

    def run(self, collection: Collection):
        pass

    def train(self, collection: Collection):
        builder = ConfigBuilder()
        model_config = builder.parse_config('./configs/config_JointModel.json')
        train_config = builder.parse_config('./configs/config_Train_JointModel.json')

        wv = Word2VecKeyedVectors.load(model_config.embedding_path)
        dataset = JointModelDataset(collection, wv)

        self.model = BERT_TreeLSTM_BiLSTM_CNN_JointModel(
            dataset.embedding_size,
            wv,
            0,
            # dataset.bert_size,
            dataset.no_postags,
            model_config.postag_size,
            dataset.no_dependencies,
            model_config.dependency_size,
            dataset.no_positions,
            model_config.position_size,
            dataset.no_chars,
            model_config.charencoding_size,
            model_config.tree_lstm_hidden_size,
            model_config.bilstm_hidden_size,
            model_config.local_cnn_channels,
            model_config.local_cnn_window_size,
            model_config.global_cnn_channels,
            model_config.global_cnn_window_size,
            model_config.dropout_chance,
            dataset.no_entity_types,
            dataset.no_entity_tags,
            dataset.no_relations
        )

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=train_config.optimizer.lr,
        )

        ent_type_criterion = CrossEntropyLoss()
        rels_criterion = MSELoss()

        for epoch in range(train_config.epochs):
            #log variables
            running_loss_ent_type = 0
            running_loss_ent_tag = 0
            running_loss_rels = 0
            correct_ent_type = 0
            correct_ent_tags = 0
            total_tags = 0
            true_positive_rels = 0
            true_negative_rels = 0
            false_positive_rels = 0

            self.model.train()
            for data in tqdm(dataset):
                * X, y_ent_type, y_ent_tag, y_rels = data
                y_rels = y_rels.unsqueeze(0)

                (
                    word_inputs,
                    char_inputs,
                    postag_inputs,
                    dependency_inputs,
                    position_inputs,
                    trees,
                    pointed_token_idx
                ) = X

                X = (
                    word_inputs.unsqueeze(0),
                    char_inputs.unsqueeze(0),
                    postag_inputs.unsqueeze(0),
                    dependency_inputs.unsqueeze(0),
                    position_inputs.unsqueeze(0),
                    trees,
                    pointed_token_idx
                )

                optimizer.zero_grad()

                sentence_features, out_ent_type, out_ent_tag, out_rels = self.model(X)

                loss_ent_type = ent_type_criterion(out_ent_type, y_ent_type)
                running_loss_ent_type += loss_ent_type.item()
                loss_ent_type.backward(retain_graph=True)

                loss_ent_tag = self.model.entities_crf_decoder.neg_log_likelihood(sentence_features, y_ent_tag)
                running_loss_ent_tag += loss_ent_tag.item()
                loss_ent_tag.backward(retain_graph=True)

                loss_rels = rels_criterion(out_rels, y_rels)
                running_loss_rels += loss_rels.item()
                loss_rels.backward()

                optimizer.step()

            self.model.eval()
            for data in tqdm(dataset):
                * X, y_ent_type, y_ent_tag, y_rels = data
                y_rels = y_rels.unsqueeze(0)

                (
                    word_inputs,
                    char_inputs,
                    postag_inputs,
                    dependency_inputs,
                    position_inputs,
                    trees,
                    pointed_token_idx
                ) = X

                X = (
                    word_inputs.unsqueeze(0),
                    char_inputs.unsqueeze(0),
                    postag_inputs.unsqueeze(0),
                    dependency_inputs.unsqueeze(0),
                    position_inputs.unsqueeze(0),
                    trees,
                    pointed_token_idx
                )

                sentence_features, out_ent_type, out_ent_tag, out_rels = self.model(X)

                #entity type
                predicted_entity_type = torch.argmax(out_ent_type, -1)
                correct_ent_type += int(predicted_entity_type == y_ent_type)

                #entity tags
                correct_ent_tags += sum(torch.tensor(out_ent_tag) == y_ent_tag).item()
                total_tags += len(out_ent_tag)

                #relations
                #[1,sent_len, rels]
                predicted_rels = (out_rels.squeeze() > model_config.relations_threshold).type(dtype=torch.long)
                for predicted, gold in zip(predicted_rels.flatten(), y_rels.flatten()):
                    true_positive_rels += int(gold == predicted == 1)
                    true_negative_rels += int(gold == 1 and predicted == 0)
                    false_positive_rels += int(gold == 0 and predicted == 1)


            print(f"[{epoch + 1}] ent_type_loss: {running_loss_ent_type / len(dataset) :0.3}")
            print(f"[{epoch + 1}] ent_tag_loss: {running_loss_ent_tag / total_tags :0.3}")
            print(f"[{epoch + 1}] rels_loss: {running_loss_rels / len(dataset) :0.3}")
            print(f"[{epoch + 1}] ent_type_acc: {correct_ent_type / len(dataset) :0.3}")
            print(f"[{epoch + 1}] ent_tag_acc: {correct_ent_tags / total_tags :0.3}")
            if true_positive_rels > 0:
                print(f"[{epoch + 1}] rels_recovery: {true_positive_rels / (true_positive_rels+true_negative_rels) :0.3}")
                print(f"[{epoch + 1}] rels_precision: {true_positive_rels / (true_positive_rels+false_positive_rels) :0.3}")
            else:
                print("No positive relations")

class DependencyJointAlgorithm(Algorithm):

    def __init__(self):
        self.model = None

    def run(self, collection: Collection, *args, taskA: bool, taskB: bool, **kargs):

        if taskA:
            self.run_taskA(collection)

        if taskB:
            self.run_taskB(collection)

    def run_taskA(self, collection: Collection):
        print("Running task A...")
        dataset = DependencyJointModelDataset(collection, self.model.wv)

        entity_id = 0

        print("Running...")
        for data in tqdm(dataset.evaluation):
            (
                sentence,
                sentence_spans,
                head_words,
                word_inputs,
                char_inputs,
                postag_inputs,
                dependency_inputs,
                trees
            ) = data

            #ENTITIES
            X = (
                word_inputs.unsqueeze(0),
                char_inputs.unsqueeze(0),
                postag_inputs.unsqueeze(0)
            )

            sentence_features, out_ent_type, out_ent_tag = self.model(X)
            predicted_entities_types = [dataset.entity_types[idx] for idx in out_ent_type]
            predicted_entities_tags = [dataset.entity_tags[idx] for idx in out_ent_tag]

            kps = [[sentence_spans[idx] for idx in span_list] for span_list in BMEWOV.decode(predicted_entities_tags)]
            for kp_spans in kps:
                count = ValueSortedDict([(type,0) for type in dataset.entity_types])
                for span in kp_spans:
                    span_index = sentence_spans.index(span)
                    span_type = predicted_entities_types[span_index]
                    count[span_type] -= 1
                entity_type = list(count.items())[0][0]

                entity_id += 1
                sentence.keyphrases.append(Keyphrase(sentence, entity_type, entity_id, kp_spans))


    def run_taskB(self, collection: Collection):
        print("Running task B...")

        dataset = DependencyJointModelDataset(collection, self.model.wv)

        print("Running...")
        for data in tqdm(dataset.evaluation):
            (
                sentence,
                sentence_spans,
                head_words,
                word_inputs,
                char_inputs,
                postag_inputs,
                dependency_inputs,
                trees
            ) = data

            X = (
                word_inputs.unsqueeze(0),
                char_inputs.unsqueeze(0),
                postag_inputs.unsqueeze(0)
            )

            sentence_features, out_ent_type, out_ent_tag = self.model(X)

            head_entities = [(idx, kp) for (idx, entities) in enumerate(head_words) for kp in entities]
            for origin_pair, destination_pair in product(head_entities, head_entities):
                origin, kp_origin = origin_pair
                destination, kp_destination = destination_pair

                #positive direction
                X = (
                    sentence_features,
                    out_ent_type,
                    out_ent_tag,
                    dependency_inputs.unsqueeze(0),
                    trees,
                    origin,
                    destination
                )

                out_rel = self.model(X, relation=True)
                relation = dataset.relations[torch.argmax(out_rel)]
                if relation != "none":
                    sentence.relations.append(Relation(sentence, kp_origin.id, kp_destination.id, relation))


    def evaluate(self, model, dataset):
        model.eval()

        correct_true_relations = 0
        total_true_relations = 0
        correct_false_relations = 0
        total_false_relations = 0
        false_positive_relations = 0
        correct_ent_types = 0
        correct_ent_tags = 0
        total_words = 0

        for data in tqdm(dataset):
            * X, y_ent_type, y_ent_tag, relations = data

            (
                word_inputs,
                char_inputs,
                postag_inputs,
                dependency_inputs,
                trees
            ) = X

            X = (
                word_inputs.unsqueeze(0),
                char_inputs.unsqueeze(0),
                postag_inputs.unsqueeze(0)
            )

            sentence_features, out_ent_type, out_ent_tag = model(X)

            positive_rels = relations["pos"]
            for origin, destination, y_rel in positive_rels:
                #positive direction
                X = (
                    sentence_features,
                    out_ent_type,
                    out_ent_tag,
                    dependency_inputs.unsqueeze(0),
                    trees,
                    origin,
                    destination
                )

                out_rel = model(X, relation=True)
                correct_true_relations += int(torch.argmax(out_rel) == y_rel)
                total_true_relations += 1

            negative_rels = relations["neg"]
            for origin, destination, y_rel in negative_rels:
                #positive direction
                X = (
                    sentence_features,
                    out_ent_type,
                    out_ent_tag,
                    dependency_inputs.unsqueeze(0),
                    trees,
                    origin,
                    destination
                )

                out_rel = model(X, relation=True)
                correct_false_relations += int(torch.argmax(out_rel) == y_rel)
                total_false_relations += 1

            #entity type
            correct_ent_types += sum(torch.tensor(out_ent_type) == y_ent_type).item()

            #entity tags
            correct_ent_tags += sum(torch.tensor(out_ent_tag) == y_ent_tag).item()

            total_words += len(out_ent_tag)

        return {
            "entities":{
                "entity_types_accuracy": correct_ent_types/total_words,
                "entity_tags_accuracy": correct_ent_tags / total_words
            },
            "relations":{
                "true_relations_accuracy": correct_true_relations / total_true_relations,
                "false_relations_accuracy": correct_false_relations / total_false_relations
            }
        }

    def train(self, train_collection: Collection, validation_collection: Collection, save_path = None, mode = "joint"):
        builder = ConfigBuilder()
        model_config = builder.parse_config('./configs/config_DependencyJointModel.json')

        wv = Word2VecKeyedVectors.load(model_config.embedding_path)
        dataset = DependencyJointModelDataset(train_collection, wv)
        val_data = DependencyJointModelDataset(validation_collection, wv)

        self.model = ShortestDependencyPathJointModel(
            dataset.embedding_size,
            wv,
            dataset.no_chars,
            model_config.charencoding_size,
            dataset.no_postags,
            model_config.postag_size,
            dataset.no_dependencies,
            model_config.dependency_size,
            model_config.entity_type_size,
            model_config.entity_tag_size,
            model_config.bilstm_shared_hidden_size,
            model_config.tree_lstm_hidden_size,
            model_config.bilstm_relations_hidden_size,
            model_config.relations_dense_size,
            model_config.shared_dropout_chance,
            model_config.relations_dropout_chance,
            dataset.no_entity_types,
            dataset.no_entity_tags,
            dataset.no_relations
        )

        if mode == "joint":
            print("Training jointly")
            train_config = builder.parse_config('./configs/config_Train_DependencyJointModel.json').joint
            self.train_joint(dataset, val_data, train_config)
            if save_path is not None:
                torch.save(self.model.state_dict(), save_path + "joint_model.ptdict")
        elif mode == "separated":
            # print("Training taskA")
            # train_configA = builder.parse_config('./configs/config_Train_DependencyJointModel.json').taskA
            # self.train_taskA(dataset, val_data, train_configA)
            # if save_path is not None:
            #     print("Saving taskA weights...")
            #     torch.save(self.model.state_dict(), save_path + "joint_modelA.ptdict")

            print("Training taskB")
            train_configB = builder.parse_config('./configs/config_Train_DependencyJointModel.json').taskB
            self.train_taskB(dataset, val_data, train_configB)
            if save_path is not None:
                print("Saving taskB weights...")
                torch.save(self.model.state_dict(), save_path + "joint_modelB.ptdict")

    def train_joint(self, dataset, val_data, train_config):

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=train_config.optimizer.lr,
        )

        relations_criterion = CrossEntropyLoss()

        for epoch in range(train_config.epochs):
            #log variables
            running_loss_ent_type = 0
            running_loss_ent_tag = 0
            running_loss_relations = 0
            train_total_words = -1 #avoid divide by zero
            train_total_relations = -1 #avoid divide by zero

            train_data = list(dataset.get_shuffled_data())

            self.model.train()
            print("Optimizing ", "relations..." if epoch % 5 != 0 else "entities...")
            for data in tqdm(train_data):
                * X, y_ent_type, y_ent_tag, relations = data

                (
                    word_inputs,
                    char_inputs,
                    dependency_inputs,
                    trees,
                ) = X

                X = (
                    word_inputs.unsqueeze(0),
                    char_inputs.unsqueeze(0)
                )

                optimizer.zero_grad()

                sentence_features, out_ent_type, out_ent_tag = self.model(X)

                train_total_words += len(trees)

                if epoch % 5 != 0:
                    train_total_relations += 1

                    positive_rels = relations["pos"]
                    # if epoch == 3:
                    #     shuffle(relations["neg"])
                    # negative_rels = relations["neg"][:len(positive_rels)]
                    # relations = permutation(positive_rels + negative_rels)
                    rels_loss = 0
                    for origin, destination, y_rel in positive_rels:
                        origin = int(origin)
                        destination = int(destination)
                        y_rel = torch.LongTensor([y_rel])

                        #positive direction
                        X = (
                            sentence_features,
                            out_ent_type,
                            out_ent_tag,
                            dependency_inputs.unsqueeze(0),
                            trees,
                            origin,
                            destination
                        )

                        out_rel = self.model(X, relation=True)
                        rels_loss += relations_criterion(out_rel, y_rel)
                        train_total_relations += 1

                    rels_loss.backward()
                    running_loss_relations += rels_loss.item()

                else:
                    train_total_words += 1
                    train_total_words += len(trees)

                    loss_ent_type = self.model.entities_types_crf_decoder.neg_log_likelihood(sentence_features, y_ent_type)
                    running_loss_ent_type += loss_ent_type.item()
                    loss_ent_type.backward(retain_graph=True)

                    loss_ent_tag = self.model.entities_tags_crf_decoder.neg_log_likelihood(sentence_features, y_ent_tag)
                    running_loss_ent_tag += loss_ent_tag.item()
                    loss_ent_tag.backward()

                optimizer.step()

            print("Evaluating on training data...")
            train_diagnostics = self.evaluate(self.model, train_data)

            print("Evaluating on validation data...")
            val_diagnostics = self.evaluate(self.model, val_data)

            print(f"[{epoch + 1}] relations_loss: {running_loss_relations / train_total_relations :0.3}")
            print(f"[{epoch + 1}] ent_type_loss: {running_loss_ent_type / train_total_words :0.3}")
            print(f"[{epoch + 1}] ent_tag_loss: {running_loss_ent_tag / train_total_words :0.3}")

            for key, value in train_diagnostics["entities"].items():
                print(f"[{epoch + 1}] train_{key}: {value :0.3}")
            for key, value in train_diagnostics["relations"].items():
                print(f"[{epoch + 1}] train_{key}: {value :0.3}")
            for key, value in val_diagnostics["entities"].items():
                print(f"[{epoch + 1}] val_{key}: {value :0.3}")
            for key, value in val_diagnostics["relations"].items():
                print(f"[{epoch + 1}] val_{key}: {value :0.3}")

    def train_taskA(self, dataset, val_data, train_config):

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=train_config.optimizer.lr,
        )

        for epoch in range(train_config.epochs):
            #log variables
            running_loss_ent_type = 0
            running_loss_ent_tag = 0
            train_total_words = 0

            train_data = list(dataset.get_shuffled_data())

            self.model.train()
            print("Optimizing...")
            for data in tqdm(train_data):
                * X, y_ent_type, y_ent_tag, _ = data

                (
                    word_inputs,
                    char_inputs,
                    postag_inputs,
                    dependency_inputs,
                    trees,
                ) = X

                X = (
                    word_inputs.unsqueeze(0),
                    char_inputs.unsqueeze(0),
                    postag_inputs.unsqueeze(0)
                )

                optimizer.zero_grad()

                sentence_features, out_ent_type, out_ent_tag = self.model(X)

                train_total_words += len(trees)

                loss_ent_type = self.model.entities_types_crf_decoder.neg_log_likelihood(sentence_features, y_ent_type)
                running_loss_ent_type += loss_ent_type.item()
                loss_ent_type.backward(retain_graph=True)

                loss_ent_tag = self.model.entities_tags_crf_decoder.neg_log_likelihood(sentence_features, y_ent_tag)
                running_loss_ent_tag += loss_ent_tag.item()
                loss_ent_tag.backward()

                optimizer.step()

            print("Evaluating on training data...")
            train_diagnostics = self.evaluate(self.model, train_data)

            print("Evaluating on validation data...")
            val_diagnostics = self.evaluate(self.model, val_data)

            print(f"[{epoch + 1}] ent_type_loss: {running_loss_ent_type / train_total_words :0.3}")
            print(f"[{epoch + 1}] ent_tag_loss: {running_loss_ent_tag / train_total_words :0.3}")

            for key, value in train_diagnostics["entities"].items():
                print(f"[{epoch + 1}] train_{key}: {value :0.3}")
            for key, value in val_diagnostics["entities"].items():
                print(f"[{epoch + 1}] val_{key}: {value :0.3}")

    def train_taskB(self, dataset, val_data, train_config):

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=train_config.optimizer.lr,
        )

        relations_criterion = CrossEntropyLoss()

        for epoch in range(train_config.epochs):
            #log variables
            running_loss_relations = 0
            train_total_relations = 0

            train_data = list(dataset.get_shuffled_data())

            self.model.train()
            print("Optimizing...")
            for data in tqdm(train_data):
                * X, y_ent_type, y_ent_tag, relations = data

                (
                    word_inputs,
                    char_inputs,
                    postag_inputs,
                    dependency_inputs,
                    trees
                ) = X

                X = (
                    word_inputs.unsqueeze(0),
                    char_inputs.unsqueeze(0),
                    postag_inputs.unsqueeze(0)
                )

                optimizer.zero_grad()

                sentence_features, out_ent_type, out_ent_tag = self.model(X)

                positive_rels = relations["pos"]
                # negative_rels = relations["neg"]
                if epoch == 0:
                    shuffle(relations["neg"])
                negative_rels = relations["neg"][:1]
                # relations = permutation(positive_rels + negative_rels)
                rels_loss = 0
                for origin, destination, y_rel in positive_rels + negative_rels:
                    # origin = int(origin)
                    # destination = int(destination)
                    # y_rel = torch.LongTensor([y_rel])

                    X = (
                        sentence_features,
                        out_ent_type,
                        out_ent_tag,
                        dependency_inputs.unsqueeze(0),
                        trees,
                        origin,
                        destination
                    )

                    out_rel = self.model(X, relation=True)
                    rels_loss += relations_criterion(out_rel, y_rel)
                    train_total_relations += 1

                rels_loss.backward()
                running_loss_relations += rels_loss.item()

                optimizer.step()

            print("Evaluating on training data...")
            train_diagnostics = self.evaluate(self.model, train_data)

            print("Evaluating on validation data...")
            val_diagnostics = self.evaluate(self.model, val_data)

            print(f"[{epoch + 1}] relations_loss: {running_loss_relations / train_total_relations :0.3}")

            for key, value in train_diagnostics["relations"].items():
                print(f"[{epoch + 1}] train_{key}: {value :0.3}")
            for key, value in val_diagnostics["relations"].items():
                print(f"[{epoch + 1}] val_{key}: {value :0.3}")


class TransferAlgorithm(Algorithm):

    def __init__(self):
        self.taskA_model = None
        self.taskB_model = None
        self.taskB_recog_model_path = None
        self.taskB_recog_model_oracle = None
        self.taskB_class_model = None

        builder = ConfigBuilder()
        model_configA = builder.parse_config('./configs/transfer_models/config_StackedBiLSMTCRF.json')
        self.wv = Word2VecKeyedVectors.load(model_configA.embedding_path)

        self.fake_dependency_dataset = DependencyJointModelDataset(Collection(), self.wv)
        self.fake_oracle_dataset = RelationsOracleDataset(Collection([Sentence("Fake news")]), self.wv)

    def load_taskA_model(self, load_path = None):
        builder = ConfigBuilder()
        model_config = builder.parse_config('./configs/transfer_models/config_StackedBiLSMTCRF.json')

        self.taskA_model = BERTStackedBiLSTMCRFModel(
            self.fake_dependency_dataset.embedding_size,
            self.fake_dependency_dataset.wv,
            self.fake_dependency_dataset.no_chars,
            model_config.charencoding_size,
            self.fake_dependency_dataset.no_postags,
            model_config.postag_size,
            model_config.bilstm_hidden_size,
            model_config.dropout_chance,
            self.fake_dependency_dataset.no_entity_types,
            self.fake_dependency_dataset.no_entity_tags,
        )
        if load_path is not None:
            print("Loading taskA_model weights..")
            self.taskA_model.load_state_dict(torch.load(load_path + "transfer_modelA.ptdict"))

    def load_taskB_model(self, load_path = None):
        builder = ConfigBuilder()
        model_config = builder.parse_config('./configs/transfer_models/config_ShortestDependencyPathRelationsModel.json')

        self.taskB_model = ShortestDependencyPathRelationsModel(
            self.fake_dependency_dataset.embedding_size,
            self.fake_dependency_dataset.wv,
            self.fake_dependency_dataset.no_chars,
            model_config.charencoding_size,
            self.fake_dependency_dataset.no_postags,
            model_config.postag_size,
            self.fake_dependency_dataset.no_dependencies,
            model_config.dependency_size,
            model_config.entity_type_size,
            model_config.entity_tag_size,
            model_config.bilstm_words_hidden_size,
            model_config.bilstm_path_hidden_size,
            model_config.dropout_chance,
            self.fake_dependency_dataset.no_entity_types,
            self.fake_dependency_dataset.no_entity_tags,
            self.fake_dependency_dataset.no_relations
        )
        if load_path is not None:
            print("Loading taskB_model weights..")
            self.taskB_model.load_state_dict(torch.load(load_path + "transfer_modelB.ptdict"))

    def load_taskB_recog_model_oracle(self, load_path = None):
        builder = ConfigBuilder()
        model_config = builder.parse_config('./configs/transfer_models/config_OracleParserModel.json')

        self.taskB_recog_model_oracle = OracleParserModel(
            self.fake_oracle_dataset.word_vector_size,
            self.fake_oracle_dataset.wv,
            self.fake_oracle_dataset.no_chars,
            model_config.charencoding_size,
            model_config.lstm_hidden_size,
            model_config.dropout_chance,
            model_config.dense_hidden_size,
            self.fake_oracle_dataset.no_actions,
        )
        if load_path is not None:
            print("Loading taskB_recog_model_oracle weights..")
            self.taskB_recog_model_oracle.load_state_dict(torch.load(load_path + "transfer_modelB_recog_oracle.ptdict"))

    def load_taskB_recog_model_path(self, load_path = None):
        builder = ConfigBuilder()
        model_config = builder.parse_config('./configs/transfer_models/config_ShortestDependencyPathRelationsModel.json')

        self.taskB_recog_model_path = ShortestDependencyPathRelationsModel(
            self.fake_dependency_dataset.embedding_size,
            self.fake_dependency_dataset.wv,
            self.fake_dependency_dataset.no_chars,
            model_config.charencoding_size,
            self.fake_dependency_dataset.no_postags,
            model_config.postag_size,
            self.fake_dependency_dataset.no_dependencies,
            model_config.dependency_size,
            model_config.entity_type_size,
            model_config.entity_tag_size,
            model_config.bilstm_words_hidden_size,
            model_config.bilstm_path_hidden_size,
            model_config.dropout_chance,
            self.fake_dependency_dataset.no_entity_types,
            self.fake_dependency_dataset.no_entity_tags,
            1
        )
        if load_path is not None:
            print("Loading taskB_recog_model_path weights..")
            self.taskB_recog_model_path.load_state_dict(torch.load(load_path + "transfer_modelB_recog_path.ptdict"))

    def load_taskB_class_model(self, load_path = None):
        builder = ConfigBuilder()
        model_config = builder.parse_config('./configs/transfer_models/config_ShortestDependencyPathRelationsModel.json')

        self.taskB_class_model = ShortestDependencyPathRelationsModel(
            self.fake_dependency_dataset.embedding_size,
            self.fake_dependency_dataset.wv,
            self.fake_dependency_dataset.no_chars,
            model_config.charencoding_size,
            self.fake_dependency_dataset.no_postags,
            model_config.postag_size,
            self.fake_dependency_dataset.no_dependencies,
            model_config.dependency_size,
            model_config.entity_type_size,
            model_config.entity_tag_size,
            model_config.bilstm_words_hidden_size,
            model_config.bilstm_path_hidden_size,
            model_config.dropout_chance,
            self.fake_dependency_dataset.no_entity_types,
            self.fake_dependency_dataset.no_entity_tags,
            self.fake_dependency_dataset.no_relations
        )
        if load_path is not None:
            print("Loading taskB_class_model weights..")
            self.taskB_class_model.load_state_dict(torch.load(load_path + "transfer_modelB_class.ptdict"))


    def run(self, collection: Collection, *args, taskA: bool, taskB: bool, load_path = None, taskB_mode = "whole", **kargs):

        if taskA:
            self.run_taskA(collection, load_path)

        if taskB_mode == "whole":
            self.run_taskB_whole(collection, load_path)
        elif taskB_mode in ["oracle", "path"]:
            self.run_taskB_separate(collection, load_path, taskB_mode)

    def run_taskA(self, collection: Collection, load_path = None):
        print("Running task A...")
        dataset = DependencyBERTJointModelDataset(collection, self.wv)

        if load_path is not None:
            print("Loading weights...")
            self.load_taskA_model(load_path)

        self.taskA_model.eval()

        entity_id = 0

        print("Running...")
        for data in tqdm(dataset.evaluation):
            (
                sentence,
                sentence_spans,
                head_words,
                word_inputs,
                char_inputs,
                bert_embeddings,
                #sent_embedding,
                postag_inputs,
                trees
            ) = data

            #ENTITIES
            X = (
                word_inputs.unsqueeze(0),
                char_inputs.unsqueeze(0),
                bert_embeddings.unsqueeze(0),
                #sent_embedding.unsqueeze(0),
                postag_inputs.unsqueeze(0)
            )

            sentence_features, out_ent_type, out_ent_tag = self.taskA_model(X)
            predicted_entities_types = [dataset.entity_types[idx] for idx in out_ent_type]
            predicted_entities_tags = [dataset.entity_tags[idx] for idx in out_ent_tag]

            kps = [[sentence_spans[idx] for idx in span_list] for span_list in BMEWOV.decode(predicted_entities_tags)]
            for kp_spans in kps:
                count = ValueSortedDict([(type,0) for type in dataset.entity_types])
                for span in kp_spans:
                    span_index = sentence_spans.index(span)
                    span_type = predicted_entities_types[span_index]
                    count[span_type] -= 1
                entity_type = list(count.items())[0][0]

                entity_id += 1
                sentence.keyphrases.append(Keyphrase(sentence, entity_type, entity_id, kp_spans))

    def run_taskB_whole(self, collection: Collection, load_path = None):
        print("Running task B...")

        dataset = DependencyJointModelDataset(collection, self.wv)

        if load_path is not None:
            print("Loading weights...")

            if self.taskA_model is None:
                self.load_taskA_model(load_path)

            self.load_taskB_model(load_path)

        self.taskA_model.eval()
        self.taskB_model.eval()

        print("Running...")
        for data in tqdm(dataset.evaluation):
            (
                sentence,
                sentence_spans,
                head_words,
                word_inputs,
                char_inputs,
                postag_inputs,
                dependency_inputs,
                trees
            ) = data

            X = (
                word_inputs.unsqueeze(0),
                char_inputs.unsqueeze(0),
                postag_inputs.unsqueeze(0)
            )

            sentence_features, out_ent_type, out_ent_tag = self.taskA_model(X)

            head_entities = [(idx, kp) for (idx, entities) in enumerate(head_words) for kp in entities]
            for origin_pair, destination_pair in product(head_entities, head_entities):
                origin, kp_origin = origin_pair
                destination, kp_destination = destination_pair

                #positive direction
                X = (
                    word_inputs.unsqueeze(0),
                    char_inputs.unsqueeze(0),
                    postag_inputs.unsqueeze(0),
                    out_ent_type,
                    out_ent_tag,
                    dependency_inputs.unsqueeze(0),
                    trees,
                    origin,
                    destination
                )

                out_rel = self.taskB_model(X)
                relation = dataset.relations[torch.argmax(out_rel)]
                if relation != "none":
                    sentence.relations.append(Relation(sentence, kp_origin.id, kp_destination.id, relation))

    def run_taskB_recog_path(self, collection: Collection, load_path = None):
        print("Running task B recognition...")

        dataset = DependencyJointModelDataset(collection, self.model.wv)

        if load_path is not None:
            print("Loading weights...")

            self.load_taskB_recog_model_path(load_path)

        self.taskB_recog_model_path.eval()

        print("Running...")
        for data in tqdm(dataset.evaluation):
            (
                sentence,
                sentence_spans,
                head_words,
                word_inputs,
                char_inputs,
                postag_inputs,
                dependency_inputs,
                trees
            ) = data

            X = (
                word_inputs.unsqueeze(0),
                char_inputs.unsqueeze(0),
                postag_inputs.unsqueeze(0)
            )

            _, out_ent_type, out_ent_tag = self.taskA_model(X)

            head_entities = [(idx, kp) for (idx, entities) in enumerate(head_words) for kp in entities]
            for origin_pair, destination_pair in product(head_entities, head_entities):
                origin, kp_origin = origin_pair
                destination, kp_destination = destination_pair

                X = (
                    word_inputs.unsqueeze(0),
                    char_inputs.unsqueeze(0),
                    postag_inputs.unsqueeze(0),
                    out_ent_type,
                    out_ent_tag,
                    dependency_inputs.unsqueeze(0),
                    trees,
                    origin,
                    destination
                )

                out_rel = self.taskB_recog_model_path(X).squeeze(0)

                if torch.argmax(out_rel) > 0.5:
                    sentence.relations.append(Relation(sentence, kp_origin.id, kp_destination.id, 'none'))

    def run_taskB_recog_oracle(self, collection: Collection, load_path = None):
        print("Running task B recognition...")

        dataset = RelationsOracleDataset(collection, self.wv)

        if load_path is not None:
            print("Loading weights...")

            self.load_taskB_recog_model_oracle(load_path)

        self.taskB_recog_model_oracle.eval()

        print("Running...")
        it = 0
        for spans, sentence, state in tqdm(dataset.evaluation):
            it += 1
            words = [sentence.text[start:end] for (start, end) in spans]
            while state[1]:
                o, t, h, d = state

                X = (
                    *dataset.encode_word_sequence(["<padding>"]+[words[i - 1] for i in o]),
                    *dataset.encode_word_sequence([words[i - 1] for i in t])
                )
                X = [x.unsqueeze(0) for x in X]

                output_act = self.taskB_recog_model_oracle(X)
                action = dataset.actions[torch.argmax(output_act)]
                relation = 'none'

                try:
                    if action in ["LEFT", "RIGHT"]:
                        if action == "LEFT":
                            origidx = t[-1]
                            destidx = o[-1]
                        else:
                            origidx = o[-1]
                            destidx = t[-1]

                        origins = [kp.id for kp in sentence.keyphrases if spans[origidx-1] in kp.spans]
                        destinations = [kp.id for kp in sentence.keyphrases if spans[destidx-1] in kp.spans]

                        for origin, destination in product(origins, destinations):
                            sentence.relations.append(Relation(sentence, origin, destination, relation))

                    dataset.actions_funcs[action]["do"](state, relation)
                except Exception as e:
                    # print(e)
                    print(it)
                    state[1].clear()

    def run_taskB_class(self, collection: Collection, load_path = None):
        print("Running task B classification...")

        dataset = DependencyJointModelDataset(collection, self.wv)

        if load_path is not None:
            print("Loading weights...")

            if self.taskA_model is None:
                self.load_taskA_model(load_path)

            self.load_taskB_class_model(load_path)

        self.taskA_model.eval()
        self.taskB_class_model.eval()

        print("Running...")
        for data in tqdm(dataset.evaluation):
            (
                sentence,
                sentence_spans,
                head_words,
                word_inputs,
                char_inputs,
                postag_inputs,
                dependency_inputs,
                trees
            ) = data

            X = (
                word_inputs.unsqueeze(0),
                char_inputs.unsqueeze(0),
                postag_inputs.unsqueeze(0)
            )

            _, out_ent_type, out_ent_tag = self.taskA_model(X)

            head_entities = [(idx, kp) for (idx, entities) in enumerate(head_words) for kp in entities]
            for origin_pair, destination_pair in product(head_entities, head_entities):
                origin, kp_origin = origin_pair
                destination, kp_destination = destination_pair

                for relation in sentence.relations:
                    if relation.origin == kp_origin.id and relation.destination == kp_destination.id:
                        X = (
                            word_inputs.unsqueeze(0),
                            char_inputs.unsqueeze(0),
                            postag_inputs.unsqueeze(0),
                            out_ent_type,
                            out_ent_tag,
                            dependency_inputs.unsqueeze(0),
                            trees,
                            origin,
                            destination
                        )

                        out_rel = self.taskB_class_model(X)

                        relation.label = dataset.relations[torch.argmax(out_rel)]

    def run_taskB_separate(self, collection: Collection, load_path = None, rel_recog_model = "path"):
        print("Running taskB...")

        if rel_recog_model == "path":
            self.run_taskB_recog_path(collection, load_path)
        elif rel_recog_model == "oracle":
            self.run_taskB_recog_oracle(collection, load_path)

        self.run_taskB_class(collection, load_path)


    def evaluate_taskA(self, dataset):
        self.taskA_model.eval()

        correct_ent_types = 0
        correct_ent_tags = 0
        total_words = 0

        for data in tqdm(dataset):
            * X, y_ent_type, y_ent_tag, relations = data

            (
                word_inputs,
                char_inputs,
                bert_embeddings,
                #sent_embedding,
                postag_inputs,
                trees
            ) = X

            X = (
                word_inputs.unsqueeze(0),
                char_inputs.unsqueeze(0),
                bert_embeddings.unsqueeze(0),
                #sent_embedding.unsqueeze(0),
                postag_inputs.unsqueeze(0)
            )

            sentence_features, out_ent_type, out_ent_tag = self.taskA_model(X)

            #entity type
            correct_ent_types += sum(torch.tensor(out_ent_type) == y_ent_type).item()

            #entity tags
            correct_ent_tags += sum(torch.tensor(out_ent_tag) == y_ent_tag).item()

            total_words += len(out_ent_tag)

        return {
            "entity_types_accuracy": correct_ent_types/total_words,
            "entity_tags_accuracy": correct_ent_tags / total_words
        }

    def evaluate_taskB(self, dataset):
        self.taskA_model.eval()
        self.taskB_model.eval()

        correct_true_relations = 0
        total_true_relations = 0
        correct_false_relations = 0
        total_false_relations = 0

        for data in tqdm(dataset):
            * X, y_ent_type, y_ent_tag, relations = data

            (
                word_inputs,
                char_inputs,
                postag_inputs,
                dependency_inputs,
                trees
            ) = X

            X = (
                word_inputs.unsqueeze(0),
                char_inputs.unsqueeze(0),
                postag_inputs.unsqueeze(0)
            )

            _, out_ent_type, out_ent_tag = self.taskA_model(X)

            positive_rels = relations["pos"]
            for origin, destination, y_rel in positive_rels:
                #positive direction
                X = (
                    word_inputs.unsqueeze(0),
                    char_inputs.unsqueeze(0),
                    postag_inputs.unsqueeze(0),
                    out_ent_type,
                    out_ent_tag,
                    dependency_inputs.unsqueeze(0),
                    trees,
                    origin,
                    destination
                )

                out_rel = self.taskB_model(X)
                correct_true_relations += int(torch.argmax(out_rel) == y_rel)
                total_true_relations += 1

            negative_rels = relations["neg"]
            for origin, destination, y_rel in negative_rels:
                #positive direction
                X = (
                    word_inputs.unsqueeze(0),
                    char_inputs.unsqueeze(0),
                    postag_inputs.unsqueeze(0),
                    out_ent_type,
                    out_ent_tag,
                    dependency_inputs.unsqueeze(0),
                    trees,
                    origin,
                    destination
                )

                out_rel = self.taskB_model(X)
                correct_false_relations += int(torch.argmax(out_rel) == y_rel)
                total_false_relations += 1

        return {
            "true_relations_accuracy": correct_true_relations / total_true_relations,
            "false_relations_accuracy": correct_false_relations / total_false_relations
        }

    def evaluate_taskB_class(self, dataset):
        self.taskA_model.eval()
        self.taskB_class_model.eval()

        correct_true_relations = 0
        total_true_relations = 0

        for data in tqdm(dataset):
            * X, y_ent_type, y_ent_tag, relations = data

            (
                word_inputs,
                char_inputs,
                postag_inputs,
                dependency_inputs,
                trees
            ) = X

            X = (
                word_inputs.unsqueeze(0),
                char_inputs.unsqueeze(0),
                postag_inputs.unsqueeze(0)
            )

            _, out_ent_type, out_ent_tag = self.taskA_model(X)

            positive_rels = relations["pos"]
            for origin, destination, y_rel in positive_rels:
                #positive direction
                X = (
                    word_inputs.unsqueeze(0),
                    char_inputs.unsqueeze(0),
                    postag_inputs.unsqueeze(0),
                    out_ent_type,
                    out_ent_tag,
                    dependency_inputs.unsqueeze(0),
                    trees,
                    origin,
                    destination
                )

                out_rel = self.taskB_class_model(X)
                correct_true_relations += int(torch.argmax(out_rel) == y_rel)
                total_true_relations += 1

        return {
            "true_relations_accuracy": correct_true_relations / total_true_relations,
        }

    def evaluate_taskB_recog_path(self, dataset):
        self.taskA_model.eval()
        self.taskB_recog_model_path.eval()

        true_positive = 0
        true_negative = 0
        false_positive = 0

        for data in tqdm(dataset):
            * X, y_ent_type, y_ent_tag, relations = data

            (
                word_inputs,
                char_inputs,
                postag_inputs,
                dependency_inputs,
                trees
            ) = X

            X = (
                word_inputs.unsqueeze(0),
                char_inputs.unsqueeze(0),
                postag_inputs.unsqueeze(0)
            )

            _, out_ent_type, out_ent_tag = self.taskA_model(X)

            positive_rels = relations["pos"]
            for origin, destination, y_rel in positive_rels:
                #positive direction
                X = (
                    word_inputs.unsqueeze(0),
                    char_inputs.unsqueeze(0),
                    postag_inputs.unsqueeze(0),
                    out_ent_type,
                    out_ent_tag,
                    dependency_inputs.unsqueeze(0),
                    trees,
                    origin,
                    destination
                )

                out_rel = self.taskB_recog_model_path(X)
                true_positive += int(out_rel > 0.5)
                false_positive += int(out_rel <= 0.5)

            negative_rels = relations["neg"]
            for origin, destination, y_rel in negative_rels:
                #positive direction
                X = (
                    word_inputs.unsqueeze(0),
                    char_inputs.unsqueeze(0),
                    postag_inputs.unsqueeze(0),
                    out_ent_type,
                    out_ent_tag,
                    dependency_inputs.unsqueeze(0),
                    trees,
                    origin,
                    destination
                )

                out_rel = self.taskB_recog_model_path(X)
                true_negative += int(out_rel > 0.5)

        if true_positive + true_negative == 0:
            true_negative = -1

        precision = true_positive / (true_positive + true_negative)
        recovery = true_positive / (true_positive + false_positive)
        f1 = 2*(precision * recovery) / (precision + recovery) if precision + recovery > 0 else 0.

        return {
            "precision": precision,
            "recovery": recovery,
            "f1": f1
        }

    def evaluate_taskB_recog_oracle(self, dataset):
        correct_leftright = 0
        total_leftright = 0
        correct_notleftright = 0
        total_notleftright = 0
        correct_act = 0
        total_act = 0

        self.taskB_recog_model_oracle.eval()
        for data in tqdm(dataset):
            *X, y_act, _ = data
            X = [x.unsqueeze(0) for x in X]

            output_act = self.taskB_recog_model_oracle(X)
            predicted_act = torch.argmax(output_act, -1)

            equals = predicted_act == y_act
            leftright = dataset.actions[y_act] in ["LEFT", "RIGHT"]

            total_act += 1
            correct_act += int(equals)

            correct_leftright += int(leftright and equals)
            total_leftright += int(leftright)

            correct_notleftright += int((not leftright) and equals)
            total_notleftright += int(not leftright)

        return {
            "actions_accuracy": correct_act / total_act,
            "leftright_actions_accuracy": correct_leftright / total_leftright,
            "not_leftright_actions_accuracy": correct_notleftright / total_notleftright
        }

    def train(self, train_collection: Collection, validation_collection: Collection, save_path = None):
        builder = ConfigBuilder()
        model_configA = builder.parse_config('./configs/transfer_models/config_StackedBiLSMTCRF.json')
        train_configA = builder.parse_config('./configs/transfer_models/config_Train_DependencyJointModel.json').taskA
        model_configB_recog = builder.parse_config('./configs/transfer_models/config_OracleParserModel.json')
        train_configB_recog = builder.parse_config('./configs/transfer_models/config_Train_DependencyJointModel.json').taskB_recog
        model_configB_class = builder.parse_config('./configs/transfer_models/config_ShortestDependencyPathRelationsModel.json')
        train_configB_class = builder.parse_config('./configs/transfer_models/config_Train_DependencyJointModel.json').taskB_class

        wv = Word2VecKeyedVectors.load(model_configA.embedding_path)
        #dataset = DependencyJointModelDataset(train_collection, wv)
        #val_data = DependencyJointModelDataset(validation_collection, wv)

        print("Training taskA")
        self.train_taskA(train_collection, validation_collection)
        if save_path is not None:
            print("Saving taskA weights...")
            torch.save(self.taskA_model.state_dict(), save_path + "transfer_modelA.ptdict")

        # print("Training taskB classification")
        # self.train_taskB_class(dataset, val_data, model_configB_class, train_configB_class)
        # if save_path is not None:
        #     print("Saving taskB weights...")
        #     torch.save(self.taskB_class_model.state_dict(), save_path + "transfer_modelB_class.ptdict")

        print("Training taskB binary")
        self.train_taskB_binary(dataset, val_data, model_configB_class, train_configB_class)
        if save_path is not None:
            print("Saving taskB weights...")
            torch.save(self.taskB_class_model.state_dict(), save_path + "transfer_modelB_binary.ptdict")

        # dataset = RelationsOracleDataset(train_collection, wv)
        # val_data = RelationsOracleDataset(validation_collection, wv)

        # print("Training taskB recognition")
        # self.train_taskB_recog(dataset, val_data, model_configB_recog, train_configB_recog)
        # if save_path is not None:
        #     print("Saving taskB weights...")
        #     torch.save(self.taskB_recog_model.state_dict(), save_path + "transfer_modelB_recog.ptdict")


    def train_taskA(self, train_collection, validation_collection, save_path = None):
        builder = ConfigBuilder()
        model_config = builder.parse_config('./configs/transfer_models/config_StackedBiLSMTCRF.json')
        train_config = builder.parse_config('./configs/transfer_models/config_Train_DependencyJointModel.json').taskA

        dataset = DependencyBERTJointModelDataset(train_collection, self.wv)
        val_data = DependencyBERTJointModelDataset(validation_collection, self.wv)

        self.taskA_model = BERTStackedBiLSTMCRFModel(
            dataset.embedding_size,
            dataset.bert_vector_size,
            #dataset.sent_vector_size,
            dataset.wv,
            dataset.no_chars,
            model_config.charencoding_size,
            dataset.no_postags,
            model_config.postag_size,
            model_config.bilstm_hidden_size,
            model_config.dropout_chance,
            dataset.no_entity_types,
            dataset.no_entity_tags,
        )

        optimizer = optim.Adam(
            self.taskA_model.parameters(),
            lr=train_config.optimizer.lr,
        )

        for epoch in range(train_config.epochs):
            #log variables
            running_loss_ent_type = 0
            running_loss_ent_tag = 0
            train_total_words = 0

            train_data = list(dataset.get_shuffled_data())

            self.taskA_model.train()
            print("Optimizing...")
            for data in tqdm(train_data):
                * X, y_ent_type, y_ent_tag, _ = data

                (
                    word_inputs,
                    char_inputs,
                    bert_embeddings,
                    #sent_embedding,
                    postag_inputs,
                    trees,
                ) = X

                X = (
                    word_inputs.unsqueeze(0),
                    char_inputs.unsqueeze(0),
                    bert_embeddings.unsqueeze(0),
                    #sent_embedding.unsqueeze(0),
                    postag_inputs.unsqueeze(0)
                )

                optimizer.zero_grad()

                sentence_features, out_ent_type, out_ent_tag = self.taskA_model(X)

                train_total_words += len(trees)

                loss_ent_type = self.taskA_model.entities_types_crf_decoder.neg_log_likelihood(sentence_features, y_ent_type)
                running_loss_ent_type += loss_ent_type.item()
                loss_ent_type.backward(retain_graph=True)

                loss_ent_tag = self.taskA_model.entities_tags_crf_decoder.neg_log_likelihood(sentence_features, y_ent_tag)
                running_loss_ent_tag += loss_ent_tag.item()
                loss_ent_tag.backward()

                optimizer.step()

            print("Evaluating on training data...")
            train_diagnostics = self.evaluate_taskA(train_data)

            print("Evaluating on validation data...")
            val_diagnostics = self.evaluate_taskA(val_data)

            print(f"[{epoch + 1}] ent_type_loss: {running_loss_ent_type / train_total_words :0.3}")
            print(f"[{epoch + 1}] ent_tag_loss: {running_loss_ent_tag / train_total_words :0.3}")

            for key, value in train_diagnostics.items():
                print(f"[{epoch + 1}] train_{key}: {value :0.3}")
            for key, value in val_diagnostics.items():
                print(f"[{epoch + 1}] val_{key}: {value :0.3}")

        if save_path is not None:
            print("Saving weights...")
            torch.save(self.taskA_model.state_dict(), save_path + "transfer_modelA.ptdict")

    def train_taskB(self, train_collection, validation_collection, save_path = None):
        builder = ConfigBuilder()
        model_config = builder.parse_config('./configs/transfer_models/config_ShortestDependencyPathRelationsModel.json')
        train_config = builder.parse_config('./configs/transfer_models/config_Train_DependencyJointModel.json').taskB

        dataset = DependencyJointModelDataset(train_collection, self.wv)
        val_data = DependencyJointModelDataset(validation_collection, self.wv)

        self.taskB_model = ShortestDependencyPathRelationsModel(
            dataset.embedding_size,
            dataset.wv,
            dataset.no_chars,
            model_config.charencoding_size,
            dataset.no_postags,
            model_config.postag_size,
            dataset.no_dependencies,
            model_config.dependency_size,
            model_config.entity_type_size,
            model_config.entity_tag_size,
            model_config.bilstm_words_hidden_size,
            model_config.bilstm_path_hidden_size,
            model_config.dropout_chance,
            dataset.no_entity_types,
            dataset.no_entity_tags,
            dataset.no_relations
        )

        optimizer = optim.Adam(
            self.taskB_model.parameters(),
            lr=train_config.optimizer.lr,
        )

        relations_criterion = CrossEntropyLoss()

        for epoch in range(train_config.epochs):
            #log variables
            running_loss_relations = 0
            train_total_relations = 0

            train_data = list(dataset.get_shuffled_data())

            self.taskB_model.train()
            print("Optimizing...")
            for data in tqdm(train_data):
                * X, y_ent_type, y_ent_tag, relations = data

                (
                    word_inputs,
                    char_inputs,
                    postag_inputs,
                    dependency_inputs,
                    trees
                ) = X

                X = (
                    word_inputs.unsqueeze(0),
                    char_inputs.unsqueeze(0),
                    postag_inputs.unsqueeze(0)
                )

                optimizer.zero_grad()

                self.taskA_model.eval()
                _, out_ent_type, out_ent_tag = self.taskA_model(X)

                positive_rels = relations["pos"]
                if epoch == 0:
                    shuffle(relations["neg"])
                negative_rels = relations["neg"][:len(positive_rels)]
                premuted_rels = permutation(positive_rels + negative_rels)
                rels_loss = 0
                for origin, destination, y_rel in premuted_rels:
                    y_rel = torch.tensor([y_rel], dtype = torch.long)

                    X = (
                        word_inputs.unsqueeze(0),
                        char_inputs.unsqueeze(0),
                        postag_inputs.unsqueeze(0),
                        out_ent_type,
                        out_ent_tag,
                        dependency_inputs.unsqueeze(0),
                        trees,
                        origin,
                        destination
                    )

                    out_rel = self.taskB_model(X)
                    rels_loss += relations_criterion(out_rel, y_rel)
                    train_total_relations += 1

                rels_loss.backward()
                running_loss_relations += rels_loss.item()

                optimizer.step()

            print("Evaluating on training data...")
            train_diagnostics = self.evaluate_taskB(train_data)

            print("Evaluating on validation data...")
            val_diagnostics = self.evaluate_taskB(val_data)

            print(f"[{epoch + 1}] relations_loss: {running_loss_relations / train_total_relations :0.3}")

            for key, value in train_diagnostics.items():
                print(f"[{epoch + 1}] train_{key}: {value :0.3}")
            for key, value in val_diagnostics.items():
                print(f"[{epoch + 1}] val_{key}: {value :0.3}")

        if save_path is not None:
            print("Saving weights...")
            torch.save(self.taskB_model.state_dict(), save_path + "transfer_modelB.ptdict")

    def train_taskB_class(self, train_collection, validation_collection, save_path = None):
        builder = ConfigBuilder()
        model_config = builder.parse_config('./configs/transfer_models/config_ShortestDependencyPathRelationsModel.json')
        train_config = builder.parse_config('./configs/transfer_models/config_Train_DependencyJointModel.json').taskB_class

        dataset = DependencyJointModelDataset(train_collection, self.wv)
        val_data = DependencyJointModelDataset(validation_collection, self.wv)

        self.taskB_class_model = ShortestDependencyPathRelationsModel(
            dataset.embedding_size,
            dataset.wv,
            dataset.no_chars,
            model_config.charencoding_size,
            dataset.no_postags,
            model_config.postag_size,
            dataset.no_dependencies,
            model_config.dependency_size,
            model_config.entity_type_size,
            model_config.entity_tag_size,
            model_config.bilstm_words_hidden_size,
            model_config.bilstm_path_hidden_size,
            model_config.dropout_chance,
            dataset.no_entity_types,
            dataset.no_entity_tags,
            dataset.no_relations
        )

        optimizer = optim.Adam(
            self.taskB_class_model.parameters(),
            lr=train_config.optimizer.lr,
        )

        relations_criterion = CrossEntropyLoss()

        for epoch in range(train_config.epochs):
            #log variables
            running_loss_relations = 0
            train_total_relations = 0

            train_data = list(dataset.get_shuffled_data())

            self.taskB_class_model.train()
            print("Optimizing...")
            for data in tqdm(train_data):
                * X, y_ent_type, y_ent_tag, relations = data

                (
                    word_inputs,
                    char_inputs,
                    postag_inputs,
                    dependency_inputs,
                    trees
                ) = X

                X = (
                    word_inputs.unsqueeze(0),
                    char_inputs.unsqueeze(0),
                    postag_inputs.unsqueeze(0)
                )

                optimizer.zero_grad()

                self.taskA_model.eval()
                _, out_ent_type, out_ent_tag = self.taskA_model(X)

                positive_rels = relations["pos"]
                rels_loss = 0
                for origin, destination, y_rel in positive_rels:

                    X = (
                        word_inputs.unsqueeze(0),
                        char_inputs.unsqueeze(0),
                        postag_inputs.unsqueeze(0),
                        out_ent_type,
                        out_ent_tag,
                        dependency_inputs.unsqueeze(0),
                        trees,
                        origin,
                        destination
                    )

                    out_rel = self.taskB_class_model(X)
                    rels_loss += relations_criterion(out_rel, y_rel)
                    train_total_relations += 1

                rels_loss.backward()
                running_loss_relations += rels_loss.item()

                optimizer.step()

            print("Evaluating on training data...")
            train_diagnostics = self.evaluate_taskB_class(train_data)

            print("Evaluating on validation data...")
            val_diagnostics = self.evaluate_taskB_class(val_data)

            print(f"[{epoch + 1}] relations_loss: {running_loss_relations / train_total_relations :0.3}")

            for key, value in train_diagnostics.items():
                print(f"[{epoch + 1}] train_{key}: {value :0.3}")
            for key, value in val_diagnostics.items():
                print(f"[{epoch + 1}] val_{key}: {value :0.3}")

        if save_path is not None:
            print("Saving weights...")
            torch.save(self.taskB_class_model.state_dict(), save_path + "transfer_modelB_class.ptdict")

    def train_taskB_recog_oracle(self, train_collection, validation_collection, save_path = None):
        builder = ConfigBuilder()
        model_config = builder.parse_config('./configs/transfer_models/config_OracleParserModel.json')
        train_config = builder.parse_config('./configs/transfer_models/config_Train_DependencyJointModel.json').taskB_recog_oracle

        dataset = RelationsOracleDataset(train_collection, self.wv)
        val_data = RelationsOracleDataset(validation_collection, self.wv)

        self.taskB_recog_model_oracle = OracleParserModel(
            dataset.word_vector_size,
            dataset.wv,
            dataset.no_chars,
            model_config.charencoding_size,
            model_config.lstm_hidden_size,
            model_config.dropout_chance,
            model_config.dense_hidden_size,
            dataset.no_actions,
        )

        optimizer = optim.Adam(
            self.taskB_recog_model_oracle.parameters(),
            lr = train_config.optimizer.lr
        )

        criterion_act = CrossEntropyLoss(weight=dataset.get_actions_weights())

        for epoch in range(train_config.epochs):
            total_act = 0
            running_loss_act = 0.0

            # train_data = list(dataset.get_shuffled_data())

            print("Optimizing...")
            for data in tqdm(dataset):
                *X, y_act, _ = data
                X = [x.unsqueeze(0) for x in X]

                optimizer.zero_grad()

                # forward + backward + optimize
                self.taskB_recog_model_oracle.train()
                output_act = self.taskB_recog_model_oracle(X)

                loss_act = criterion_act(output_act, y_act)
                loss_act.backward()
                running_loss_act += loss_act.item()
                total_act += 1

                optimizer.step()

            print("Evaluating on training data...")
            train_diagnostics = self.evaluate_taskB_recog_oracle(dataset)

            print("Evaluating on validation data...")
            val_diagnostics = self.evaluate_taskB_recog_oracle(val_data)

            print(f"[{epoch + 1}] loss_act: {running_loss_act / total_act}")

            for key, value in train_diagnostics.items():
                print(f"[{epoch + 1}] train_{key}: {value :0.3}")
            for key, value in val_diagnostics.items():
                print(f"[{epoch + 1}] val_{key}: {value :0.3}")

        if save_path is not None:
            print("Saving weights...")
            torch.save(self.taskB_recog_model_oracle.state_dict(), save_path + "transfer_modelB_recog_oracle.ptdict")

    def train_taskB_recog_path(self, train_collection, validation_collection, save_path = None):
        builder = ConfigBuilder()
        model_config = builder.parse_config('./configs/transfer_models/config_ShortestDependencyPathRelationsModel.json')
        train_config = builder.parse_config('./configs/transfer_models/config_Train_DependencyJointModel.json').taskB_recog_path

        dataset = DependencyJointModelDataset(train_collection, self.wv)
        val_data = DependencyJointModelDataset(validation_collection, self.wv)

        self.taskB_recog_model_path = ShortestDependencyPathRelationsModel(
            dataset.embedding_size,
            dataset.wv,
            dataset.no_chars,
            model_config.charencoding_size,
            dataset.no_postags,
            model_config.postag_size,
            dataset.no_dependencies,
            model_config.dependency_size,
            model_config.entity_type_size,
            model_config.entity_tag_size,
            model_config.bilstm_words_hidden_size,
            model_config.bilstm_path_hidden_size,
            model_config.dropout_chance,
            dataset.no_entity_types,
            dataset.no_entity_tags,
            1
        )

        optimizer = optim.Adam(
            self.taskB_recog_model_path.parameters(),
            lr=train_config.optimizer.lr,
        )

        relations_criterion = BCELoss()

        cv_best_f1 = 0

        for epoch in range(train_config.epochs):
            #log variables
            running_loss_relations = 0
            train_total_relations = 0

            train_data = list(dataset.get_shuffled_data())

            self.taskB_recog_model_path.train()
            print("Optimizing...")
            for data in tqdm(train_data):
                * X, y_ent_type, y_ent_tag, relations = data

                (
                    word_inputs,
                    char_inputs,
                    postag_inputs,
                    dependency_inputs,
                    trees
                ) = X

                X = (
                    word_inputs.unsqueeze(0),
                    char_inputs.unsqueeze(0),
                    postag_inputs.unsqueeze(0)
                )

                optimizer.zero_grad()

                self.taskA_model.eval()
                _, out_ent_type, out_ent_tag = self.taskA_model(X)

                positive_rels = relations["pos"]
                # if epoch == 0:
                #     shuffle(relations["neg"])
                negative_rels = relations["neg"]
                premuted_rels = permutation(positive_rels + negative_rels)
                rels_loss = 0
                for origin, destination, y_rel in premuted_rels:

                    X = (
                        word_inputs.unsqueeze(0),
                        char_inputs.unsqueeze(0),
                        postag_inputs.unsqueeze(0),
                        out_ent_type,
                        out_ent_tag,
                        dependency_inputs.unsqueeze(0),
                        trees,
                        origin,
                        destination
                    )

                    if dataset.relations[y_rel] != "none":
                        y_rel = torch.tensor([1], dtype = torch.float32)
                    else:
                        y_rel = torch.tensor([0], dtype = torch.float32)

                    out_rel = self.taskB_recog_model_path(X).squeeze(0)

                    rels_loss += relations_criterion(out_rel, y_rel)
                    train_total_relations += 1

                rels_loss.backward()
                running_loss_relations += rels_loss.item()

                optimizer.step()

            print("Evaluating on training data...")
            train_diagnostics = self.evaluate_taskB_recog_path(train_data)

            print("Evaluating on validation data...")
            val_diagnostics = self.evaluate_taskB_recog_path(val_data)

            print(f"[{epoch + 1}] relations_loss: {running_loss_relations / train_total_relations :0.3}")

            for key, value in train_diagnostics.items():
                print(f"[{epoch + 1}] train_{key}: {value :0.3}")
            for key, value in val_diagnostics.items():
                print(f"[{epoch + 1}] val_{key}: {value :0.3}")

            if save_path is not None and val_diagnostics["f1"] > cv_best_f1:
                cv_best_f1 = val_diagnostics["f1"]
                print("Saving taskB weights...")
                torch.save(self.taskB_recog_model_path.state_dict(), save_path + "transfer_modelB_recog_path_cv.ptdict")

        if save_path is not None:
            print("Saving taskB weights...")
            torch.save(self.taskB_recog_model_path.state_dict(), save_path + "transfer_modelB_recog_path.ptdict")


class BiLSTMCRFDepPathAlgorithm(Algorithm):
    def __init__(self, ablation = {
            "chars_info": True,
            "postag": True,
            "dependency": True,
            "entity_type": True,
            "entity_tag": True
        }):
        self.taskA_model = None
        self.taskB_model_recog = None
        self.taskB_model_class = None

        self.wv = Word2VecKeyedVectors.load("./trained/embeddings/wiki_classic/wiki_classic.wv")

        self.fake_dependency_dataset = DependencyJointModelDataset(Collection(), self.wv)

        self.ablation = ablation

    def load_taskA_model(self, load_path = None):
        self.taskA_model = StackedBiLSTMCRFModel(
            self.fake_dependency_dataset.embedding_size,
            self.fake_dependency_dataset.wv,
            self.fake_dependency_dataset.no_chars,
            50,
            self.fake_dependency_dataset.no_postags,
            50,
            300,
            0.2,
            self.fake_dependency_dataset.no_entity_types,
            self.fake_dependency_dataset.no_entity_tags,
        )

        if load_path is not None:
            print("Loading taskA_model weights..")
            self.taskA_model.load_state_dict(torch.load(load_path + "modelA.ptdict"))

        if use_cuda:
            self.taskA_model.cuda(device)

    def load_taskB_recog_model(self, load_path = None):
        self.taskB_model_recog = TreeBiLSTMPathModel(
            self.fake_dependency_dataset.embedding_size,
            self.fake_dependency_dataset.wv,
            self.fake_dependency_dataset.no_chars,
            50,
            self.fake_dependency_dataset.no_postags,
            50,
            self.fake_dependency_dataset.no_dependencies,
            50,
            self.fake_dependency_dataset.no_entity_types,
            25,
            self.fake_dependency_dataset.no_entity_tags,
            25,
            200,
            100,
            50,
            0.4,
            0.2,
            0.2,
            1,
            ablation = self.ablation
        )
        if load_path is not None:
            print("Loading taskB_recog_model weights..")
            self.taskB_model_recog.load_state_dict(torch.load(load_path + "modelB_recog.ptdict"))

        if use_cuda:
            self.taskB_model_recog.cuda(device)

    def load_taskB_class_model(self, load_path = None):
        self.taskB_model_class = TreeBiLSTMPathModel(
            self.fake_dependency_dataset.embedding_size,
            self.fake_dependency_dataset.wv,
            self.fake_dependency_dataset.no_chars,
            50,
            self.fake_dependency_dataset.no_postags,
            50,
            self.fake_dependency_dataset.no_dependencies,
            50,
            self.fake_dependency_dataset.no_entity_types,
            25,
            self.fake_dependency_dataset.no_entity_tags,
            25,
            200,
            100,
            50,
            0.4,
            0.2,
            0.2,
            self.fake_dependency_dataset.no_relations - 1,
            ablation = self.ablation
        )
        if load_path is not None:
            print("Loading taskB_class_model weights..")
            self.taskB_model_class.load_state_dict(torch.load(load_path + "modelB_class.ptdict"))

        if use_cuda:
            self.taskB_model_class.cuda(device)


    def evaluate_taskA(self, dataset):
        self.taskA_model.eval()

        correct_ent_types = 0
        correct_ent_tags = 0
        total_words = 0

        running_loss_ent_type = 0
        running_loss_ent_tag = 0

        for data in tqdm(dataset):
            * X, y_ent_type, y_ent_tag, relations = data

            y_ent_type = y_ent_type.to(device)
            y_ent_tag = y_ent_tag.to(device)

            (
                word_inputs,
                char_inputs,
                postag_inputs,
                dependency_inputs,
                trees
            ) = X

            X = (
                word_inputs.unsqueeze(0).to(device),
                char_inputs.unsqueeze(0).to(device),
                postag_inputs.unsqueeze(0).to(device)
            )

            sentence_features, out_ent_type, out_ent_tag = self.taskA_model(X)

            loss_ent_type = self.taskA_model.entities_types_crf_decoder.neg_log_likelihood(sentence_features, y_ent_type)
            running_loss_ent_type += loss_ent_type.item()

            loss_ent_tag = self.taskA_model.entities_tags_crf_decoder.neg_log_likelihood(sentence_features, y_ent_tag)
            running_loss_ent_tag += loss_ent_tag.item()

            #entity type
            correct_ent_types += sum(torch.tensor(out_ent_type) == y_ent_type).item()

            #entity tags
            correct_ent_tags += sum(torch.tensor(out_ent_tag) == y_ent_tag).item()

            total_words += len(out_ent_tag)

        return {
            "loss": (running_loss_ent_type + running_loss_ent_tag) / total_words,
            "loss_entity_type": running_loss_ent_type / total_words,
            "loss_entity_tag": running_loss_ent_tag / total_words,
            "entity_types_accuracy": correct_ent_types / total_words,
            "entity_tags_accuracy": correct_ent_tags / total_words,
            "accuracy": (correct_ent_types + correct_ent_tags) / (2*total_words)
        }

    def evaluate_taskB_recog(self, dataset):
        self.taskB_model_recog.eval()

        true_positive = 0
        true_negative = 0
        false_positive = 0
        total_loss = 0
        total_rels = 0

        criterion = BCELoss()
        if use_cuda:
            criterion.cuda(device)

        for data in tqdm(dataset):
            * X, y_ent_type, y_ent_tag, relations = data

            (
                word_inputs,
                char_inputs,
                postag_inputs,
                dependency_inputs,
                trees
            ) = X

            for origin, destination, y_rel in relations["pos"]:
                X = (
                    word_inputs.unsqueeze(0).to(device),
                    char_inputs.unsqueeze(0).to(device),
                    postag_inputs.unsqueeze(0).to(device),
                    dependency_inputs.unsqueeze(0).to(device),
                    y_ent_type.unsqueeze(0).to(device),
                    y_ent_tag.unsqueeze(0).to(device),
                    trees,
                    origin,
                    destination
                )

                y_rel = torch.tensor([1], dtype = torch.float32).to(device)
                out_rel = self.taskB_model_recog(X).squeeze(0)

                total_loss += criterion(out_rel, y_rel).item()
                total_rels += 1
                true_positive += int(out_rel > 0.5)
                false_positive += int(out_rel <= 0.5)

            for origin, destination, y_rel in relations["neg"]:
                X = (
                    word_inputs.unsqueeze(0).to(device),
                    char_inputs.unsqueeze(0).to(device),
                    postag_inputs.unsqueeze(0).to(device),
                    dependency_inputs.unsqueeze(0).to(device),
                    y_ent_type.unsqueeze(0).to(device),
                    y_ent_tag.unsqueeze(0).to(device),
                    trees,
                    origin,
                    destination
                )

                y_rel = torch.tensor([0], dtype = torch.float32).to(device)
                out_rel = self.taskB_model_recog(X).squeeze(0)

                total_loss += criterion(out_rel, y_rel).item()
                total_rels += 1
                true_negative += int(out_rel > 0.5)

        if true_positive + true_negative == 0:
                true_negative = -1

        precision = true_positive / (true_positive + true_negative)
        recovery = true_positive / (true_positive + false_positive)
        f1 = 2*(precision * recovery) / (precision + recovery) if precision + recovery > 0 else 0.

        return {
            "loss": total_loss / total_rels,
            "precision": precision,
            "recovery": recovery,
            "f1": f1
        }

    def evaluate_taskB_class(self, dataset):
        self.taskB_model_class.eval()

        correct_true_relations = 0
        total_true_relations = 0
        correct_false_relations = 0
        total_false_relations = 0
        total_loss = 0

        criterion = CrossEntropyLoss()
        if use_cuda:
            criterion.cuda(device)

        for data in tqdm(dataset):
            * X, y_ent_type, y_ent_tag, relations = data

            (
                word_inputs,
                char_inputs,
                postag_inputs,
                dependency_inputs,
                trees
            ) = X

            for origin, destination, y_rel in relations["pos"]:
                X = (
                    word_inputs.unsqueeze(0).to(device),
                    char_inputs.unsqueeze(0).to(device),
                    postag_inputs.unsqueeze(0).to(device),
                    dependency_inputs.unsqueeze(0).to(device),
                    y_ent_type.unsqueeze(0).to(device),
                    y_ent_tag.unsqueeze(0).to(device),
                    trees,
                    origin,
                    destination
                )

                out_rel = self.taskB_model_class(X)
                y_rel = y_rel.to(device)

                total_loss += criterion(out_rel, y_rel).item()
                correct_true_relations += int(torch.argmax(out_rel) == y_rel)
                total_true_relations += 1

        return {
            "loss": total_loss / total_true_relations,
            "accuracy": correct_true_relations / total_true_relations
        }


    def train_taskA(self, train_collection, validation_collection, save_path = None):

        self.load_taskA_model()

        dataset = DependencyJointModelDataset(train_collection, self.wv)
        val_data = DependencyJointModelDataset(validation_collection, self.wv)

        optimizer = optim.Adam(
            self.taskA_model.parameters(),
            lr=0.001,
        )

        cv_best_acc = 0
        history = {
            "train": [],
            "validation": []
        }

        for epoch in range(1):
            train_data = list(dataset.get_shuffled_data())

            self.taskA_model.train()
            print("Optimizing...")
            for data in tqdm(train_data):
                * X, y_ent_type, y_ent_tag, _ = data

                y_ent_type = y_ent_type.to(device)
                y_ent_tag = y_ent_tag.to(device)

                (
                    word_inputs,
                    char_inputs,
                    postag_inputs,
                    dependency_inputs,
                    trees,
                ) = X

                X = (
                    word_inputs.unsqueeze(0).to(device),
                    char_inputs.unsqueeze(0).to(device),
                    postag_inputs.unsqueeze(0).to(device)
                )

                optimizer.zero_grad()

                sentence_features, out_ent_type, out_ent_tag = self.taskA_model(X)

                loss_ent_type = self.taskA_model.entities_types_crf_decoder.neg_log_likelihood(sentence_features, y_ent_type)
                loss_ent_type.backward(retain_graph=True)

                loss_ent_tag = self.taskA_model.entities_tags_crf_decoder.neg_log_likelihood(sentence_features, y_ent_tag)
                loss_ent_tag.backward()

                optimizer.step()

            print("Evaluating on training data...")
            train_diagnostics = self.evaluate_taskA(train_data)
            history["train"].append(train_diagnostics)

            print("Evaluating on validation data...")
            val_diagnostics = self.evaluate_taskA(val_data)
            history["validation"].append(val_diagnostics)

            for key, value in train_diagnostics.items():
                print(f"[{epoch + 1}] train_{key}: {value :0.3}")
            for key, value in val_diagnostics.items():
                print(f"[{epoch + 1}] val_{key}: {value :0.3}")

            if save_path is not None and val_diagnostics["accuracy"] > cv_best_acc:
                print("Saving weights...")
                torch.save(self.taskA_model.state_dict(), save_path + "modelA.ptdict")

        torch.save(history, save_path + "modelA.hst")
        return history

    def train_taskB_recog(self, train_collection, validation_collection, save_path=None):
        self.load_taskB_recog_model()

        dataset = DependencyJointModelDataset(train_collection, self.wv)
        val_data = DependencyJointModelDataset(validation_collection, self.wv)

        optimizer = optim.Adam(
            self.taskB_model_recog.parameters(),
            lr = 0.001
        )

        criterion = BCELoss()
        if use_cuda:
            criterion.cuda(device)

        best_cv_f1 = 0
        history = {
            "train": [],
            "validation": []
        }

        for epoch in range(10):
            self.taskB_model_recog.train()
            print("Optimizing...")
            for data in tqdm(dataset):
                * X, y_ent_type, y_ent_tag, relations = data

                (
                    word_inputs,
                    char_inputs,
                    postag_inputs,
                    dependency_inputs,
                    trees
                ) = X

                optimizer.zero_grad()

                positive_rels = relations["pos"]
                negative_rels = relations["neg"]
                premuted_rels = permutation(positive_rels + negative_rels)
                rels_loss = 0
                for origin, destination, y_rel in premuted_rels:
                    X = (
                        word_inputs.unsqueeze(0).to(device),
                        char_inputs.unsqueeze(0).to(device),
                        postag_inputs.unsqueeze(0).to(device),
                        dependency_inputs.unsqueeze(0).to(device),
                        y_ent_type.unsqueeze(0).to(device),
                        y_ent_tag.unsqueeze(0).to(device),
                        trees,
                        origin,
                        destination
                    )

                    if dataset.relations[y_rel] != "none":
                        y_rel = torch.tensor([1], dtype = torch.float32).to(device)
                    else:
                        y_rel = torch.tensor([0], dtype = torch.float32).to(device)

                    out_rel = self.taskB_model_recog(X).squeeze(0)

                    rels_loss += criterion(out_rel, y_rel)

                rels_loss.backward()
                optimizer.step()


            print("Evaluating on training data...")
            results_train = self.evaluate_taskB_recog(dataset)
            history["train"].append(results_train)

            print("Evaluating on validation data...")
            results_val = self.evaluate_taskB_recog(val_data)
            history["validation"].append(results_val)

            for key, value in results_train.items():
                print(f"[{epoch+1}] train_{key}: {value}")

            for key, value in results_val.items():
                print(f"[{epoch+1}] val_{key}: {value}")

            if save_path is not None and results_val["f1"] > best_cv_f1:
                best_cv_f1 = results_val["f1"]
                print("Saving taskB recog weights...")
                torch.save(self.taskB_model_recog.state_dict(), save_path + "modelB_recog.ptdict")

        if save_path is not None:
            torch.save(history, save_path + "modelB_recog.hst")
        return history

    def train_taskB_class(self, train_collection, validation_collection, save_path = None):
        self.load_taskB_class_model()

        dataset = DependencyJointModelDataset(train_collection, self.wv)
        val_data = DependencyJointModelDataset(validation_collection, self.wv)

        optimizer = optim.Adam(
            self.taskB_model_class.parameters(),
            lr = 0.001
        )

        criterion = CrossEntropyLoss()
        if use_cuda:
            criterion.cuda(device)

        best_cv_acc = 0
        history = {
            "train": [],
            "validation": []
        }

        for epoch in range(40):
            self.taskB_model_class.train()
            print("Optimizing...")
            for data in tqdm(dataset):
                * X, y_ent_type, y_ent_tag, relations = data

                (
                    word_inputs,
                    char_inputs,
                    postag_inputs,
                    dependency_inputs,
                    trees
                ) = X

                optimizer.zero_grad()

                rels_loss = 0
                for origin, destination, y_rel in relations["pos"]:
                    X = (
                        word_inputs.unsqueeze(0),
                        char_inputs.unsqueeze(0),
                        postag_inputs.unsqueeze(0),
                        dependency_inputs.unsqueeze(0),
                        y_ent_type.unsqueeze(0),
                        y_ent_tag.unsqueeze(0),
                        trees,
                        origin,
                        destination
                    )

                    out_rel = self.taskB_model_class(X)
                    y_rel = y_rel.to(device)

                    loss_positive = criterion(out_rel, y_rel)
                    rels_loss += loss_positive

                rels_loss.backward()
                optimizer.step()


            print("Evaluating on training data...")
            results_train = self.evaluate_taskB_class(dataset)
            history["train"].append(results_train)

            print("Evaluating on validation data...")
            results_val = self.evaluate_taskB_class(val_data)
            history["validation"].append(results_val)

            for key, value in results_train.items():
                print(f"[{epoch+1}] train_{key}: {value}")

            for key, value in results_val.items():
                print(f"[{epoch+1}] val_{key}: {value}")

            if save_path is not None and results_val["accuracy"] > best_cv_acc:
                best_cv_acc = results_val["accuracy"]
                print("Saving taskB class weights...")
                torch.save(self.taskB_model_class.state_dict(), save_path + "modelB_class.ptdict")

        if save_path is not None:
            torch.save(history, save_path + "modelB_class.hst")
        return history


    def run_taskA(self, collection: Collection, load_path = None):
        print("Running task A...")
        dataset = DependencyJointModelDataset(collection, self.wv)

        if load_path is not None:
            print("Loading weights...")
            self.load_taskA_model(load_path)

        self.taskA_model.eval()

        entity_id = 0

        print("Running...")
        for data in tqdm(dataset.evaluation):
            (
                sentence,
                sentence_spans,
                head_words,
                word_inputs,
                char_inputs,
                postag_inputs,
                dependency_inputs,
                trees
            ) = data

            #ENTITIES
            X = (
                word_inputs.unsqueeze(0).to(device),
                char_inputs.unsqueeze(0).to(device),
                postag_inputs.unsqueeze(0).to(device)
            )

            sentence_features, out_ent_type, out_ent_tag = self.taskA_model(X)
            predicted_entities_types = [dataset.entity_types[idx] for idx in out_ent_type]
            predicted_entities_tags = [dataset.entity_tags[idx] for idx in out_ent_tag]

            kps = [[sentence_spans[idx] for idx in span_list] for span_list in BMEWOV.decode(predicted_entities_tags)]
            for kp_spans in kps:
                count = ValueSortedDict([(type,0) for type in dataset.entity_types])
                for span in kp_spans:
                    span_index = sentence_spans.index(span)
                    span_type = predicted_entities_types[span_index]
                    count[span_type] -= 1
                entity_type = list(count.items())[0][0]

                entity_id += 1
                sentence.keyphrases.append(Keyphrase(sentence, entity_type, entity_id, kp_spans))

    def run_taskB_recog(self, collection: Collection, load_path = None):
        print("Running task B recognition...")
        dataset = DependencyJointModelDataset(collection, self.wv)

        if load_path is not None:
            print("Loading weights...")
            self.load_taskB_recog_model(load_path)

        self.taskB_model_recog.eval()

        for eval_data, data in tqdm(zip(dataset.evaluation, dataset)):
            (
                sentence,
                _,
                head_words,
                word_inputs,
                char_inputs,
                postag_inputs,
                dependency_inputs,
                trees
            ) = eval_data

            *p, y_ent_type, y_ent_tag, _ = data

            head_entities = [(idx, kp) for (idx, entities) in enumerate(head_words) for kp in entities]
            for origin_pair, destination_pair in product(head_entities, head_entities):
                origin, kp_origin = origin_pair
                destination, kp_destination = destination_pair

                X = (
                    word_inputs.unsqueeze(0).to(device),
                    char_inputs.unsqueeze(0).to(device),
                    postag_inputs.unsqueeze(0).to(device),
                    dependency_inputs.unsqueeze(0).to(device),
                    y_ent_type.unsqueeze(0).to(device),
                    y_ent_tag.unsqueeze(0).to(device),
                    trees,
                    origin,
                    destination
                )

                out_rel = self.taskB_model_recog(X).squeeze(0)

                if out_rel > 0.5:
                    sentence.relations.append(Relation(sentence, kp_origin.id, kp_destination.id, 'none'))

    def run_taskB_class(self, collection: Collection, load_path = None):
        print("Running task B classification...")
        dataset = DependencyJointModelDataset(collection, self.wv)

        if load_path is not None:
            print("Loading weights...")
            self.load_taskB_class_model(load_path)

        self.taskB_model_class.eval()

        for eval_data, data in tqdm(zip(dataset.evaluation, dataset)):
            (
                sentence,
                _,
                head_words,
                word_inputs,
                char_inputs,
                postag_inputs,
                dependency_inputs,
                trees
            ) = eval_data

            *p, y_ent_type, y_ent_tag, _ = data

            head_entities = [(idx, kp) for (idx, entities) in enumerate(head_words) for kp in entities]
            for origin_pair, destination_pair in product(head_entities, head_entities):
                origin, kp_origin = origin_pair
                destination, kp_destination = destination_pair

                for relation in sentence.relations:
                    if relation.origin == kp_origin.id and relation.destination == kp_destination.id:
                        X = (
                            word_inputs.unsqueeze(0).to(device),
                            char_inputs.unsqueeze(0).to(device),
                            postag_inputs.unsqueeze(0).to(device),
                            dependency_inputs.unsqueeze(0).to(device),
                            y_ent_type.unsqueeze(0).to(device),
                            y_ent_tag.unsqueeze(0).to(device),
                            trees,
                            origin,
                            destination
                        )

                        out_rel = self.taskB_model_class(X)

                        relation.label = dataset.relations[torch.argmax(out_rel)]

    def run(self, collection: Collection, *args, taskA: bool, taskB: bool, load_path = None, **kargs):
        if taskA:
            self.run_taskA(collection, load_path)


if __name__ == "__main__":
    from pathlib import Path

    algorithm = UHMajaModel()

    training = Collection().load(Path("data/training/scenario.txt"))
    algorithm.train(training)
