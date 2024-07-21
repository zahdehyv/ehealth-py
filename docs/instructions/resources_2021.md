---
title: Resources
permalink: /2021/resources
layout: "2021"
---

# Linguistic resources

For the eHealth-KD 2021 edition, systems will be evaluated in a **cross-domain** and **multi-language corpus**. Both the training, development, and test collections will contain sentences extracted from MedlinePlus, Wikinews, and the CORD-19 corpus, all of them related with health topics, but showing a significant variety in terms of format and structure. Furthermore, the majority of sentences are in Spanish, but a small set of English sentences is also included, to evaluate generalization across domains.

In the training collection, each sentence will be labeled with its corresponding domain and language (Spanish or English), so that participants can potentially fine-tune different models for each domain/language and learn to identify them in the text. In the development collection, sentences will be mixed from all different sources and no labelling will be provided, so that participants can evaluate their systems in a similar environment to the testing set.

Participants can opt-out of the cross-domain / multi-language challenge at evaluation time, in which case only the metrics from the main domain (MedlinePlus) and language (Spanish) will be used.

As in the previous edition, the corpus for eHealth-KD 2021 will use a corpus extracted from MedlinePlus sources, plus additional resources. First, the same corpus used in the 2020 edition will be provided for training and development, while a new set of previously unlabelled sentences will be manually annotated and used for the test collection. Additionally, health-related news sourced from Wikinews will be also provided for training and development. Finally, a small set of sentences from scientific papers in the CORD-19 corpus (in English language) will be selected, annotated, and distributed in the development, and testing collections.

> **NOTE:** All the sentences are manually annotated using Brat by a group of annotators. The resulting documents and output files are distributed along with the Task. There is no need for participants to download extra data from any of the sources, since all the input is already distributed.

All resources will be distributed in the [eHealthKD corpora repository](https://github.com/ehealthkd/corpora).

## Corpus data

The corpus will be divided into three sections. Training and development sets are published along with baseline implementations, for participants to train and fine-tune their systems. These files consist of both plain text input and the expected outputs for both subtasks.

In no case, participants will be able to access the correct output files for the test set before the challenge ends. Afterward, the full corpus, including Brat-annotated files will be freely available under a suitable license for the research community.

## Testing data

The testing data (for evaluation) is available in the `eval/testing` folder. Additionally, fully annotated testing data, separated by source, is now available in the `ref/testing` folder.

The folder `scenario1-main` contains `3000` sentences. The first `2700` are in Spanish and the last `300` are in English. Of these, only `50` in each language will be used for evaluation (the rest are provided to discourage manual annotation), but you must output annotations for all of the sentences, since the reference sentences are shuffled.

Teams that prefer to participate only in one language can ignore the remaining sentences and not output annotations for them. **However**, make sure to respect character positions to maintain the alignment with the reference annotations. For example, if you are ignoring the Spanish sentences, then your first entity annotation should start around char 195,700 which is roughly where English sentences begin.

Folders `scenario2-taskA` and `scenario3-taskB` contains `50` Spanish sentences and `50` English sentences in that order.

In respect of the ethics of the competition, we kindly ask participants **not** to manually review the testing output, beyond the minimum necessary to guarantee there are no implementation errors. Especially, please do not manually annotate any of the test-set sentences, evaluate your predicted annotations, or make any design decision based on perceived performance of the test set. Doing so would incur in overfitting the testing data and diminish the value of the challenge for all participants.

### Download links:

- [🏋️ Training data](https://github.com/ehealthkd/corpora/tree/master/2021/ref/training)
- [🏋️ Develop data](https://github.com/ehealthkd/corpora/tree/master/2021/ref/develop)
- [🏋️ Testing data (for evaluation)](https://github.com/ehealthkd/corpora/tree/master/2021/eval/testing)
- [🏋️ Testing data (reference)](https://github.com/ehealthkd/corpora/tree/master/2021/ref/testing)

## Evaluation and utility scripts

Evaluation scripts will be provided so that participants can test their systems offline with respect to the same metrics used in the challenge. Since participants will not have access to the test gold annotations, their offline performance will need to be evaluated in the development set. This metric will not be exactly the same as the one obtained in the test set, but it should serve for participants to compare different strategies and perform hyper-parameter tunning.

A [utility script](https://github.com/ehealthkd/corpora/tree/master/scripts) to load, manipulate, and save ANN files is provided. Instructions for its use are available at that location.

### **Download links**:

- [🔧 Tools for loading and manipulating ANN files](https://github.com/ehealthkd/corpora/tree/master/scripts/anntools.py)

## Baselines

A simple baseline is released along with the corpus. The baseline source code is freely available as well. The baselines performance on the development and the test set are now published on [a training server](https://competitions.codalab.org/competitions/30333).

Check out [the submission section](/2021/submission) for details on running the baseline and scoring scripts.

### **Download links**:

- [🔧 Baseline implementation](https://github.com/ehealthkd/corpora/tree/master/scripts/baseline.py)
- [🔧 Scoring script](https://github.com/ehealthkd/corpora/tree/master/scripts/score.py)

# Additional resources

Participants may freely use any additional resources they consider necessary to improve their systems, from other corpora (annotated or not), to external knowledge either explicitly (i.e., using knowledge bases) or implicitly (i.e., captured in word embeddings). For the purpose of sharing the results we expect participants to fully disclose everything they use.
However, participants may not manually annotate the test set, since doing so would be in violation of the ethics of the competition. Furthermore, we expect participants to perform all the fine tuning using only the training and development, and then perform one single run in the test set for submission, so that no accidental overfitting occurs in the test set.