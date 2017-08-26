# -*- coding: utf-8 -*-
import pickle
import logging
from collections import defaultdict
from morphological_classifier import constants, utils
import numpy as np
import matplotlib.pyplot as plt
from itertools import product as cartesian_product
from datetime import datetime
import pprint


def parse_word_tag(string_element):
    '''Parses an element of the form Word_tag1+tag2...|extra_info
    into a (word, tags) tuple.'''
    # TODO: CONSIDER USING THE NORMALIZATION CLASS METHOD INSTEAD
    word, tags_str = string_element.split('_')
    return word.lower(), tags_str


def parse_sentence(sentence):
    '''Gets "Word1_tag1 word2_tag2 word3_tag3..."
        returns [("word1", "tag1"), ("word2", "tag2"), ...]'''
    parsed_sentence = []
    for word_tags in sentence.split():
        parsed_sentence.append(parse_word_tag(word_tags))
    return parsed_sentence


class MorphologicalClassifier:
    def __init__(self, tagger, save_path=None):
        self.tagger = tagger
        self.tags = set()
        self.isTrained = False

    def predict(self, phrase):
        return self.tagger.tag(phrase.split())

    def save(self, filepath):
        self.tagger.save(filepath)

    def load(self, filepath):
        self.tagger.load(filepath)
        self.isTrained = True
        self.tags = self.tagger.tags

    def train(self, filepath):
        if self.isTrained:
            print('Classifier already trained')
            return

        with open(filepath, 'r', encoding=constants.ENCODING) as f:
            sentences = f.readlines()
        parsed_sentences = [parse_sentence(sentence) for sentence in sentences]
        self.tagger.train(parsed_sentences)
        self.isTrained = True
        self.tags = self.tagger.tags

    def test(self, filepath):
        if not self.isTrained:
            print('Tagger not yet trained')
            return

        print('Starting testing phase...')
        with open(filepath, 'r', encoding=constants.ENCODING) as f:
            sentences = f.readlines()
        parsed_sentences = [
            parse_sentence(s) for s in sentences
            ]
        # Metrics stuff
        num_sentences = len(parsed_sentences)
        metrics = PerformanceMetrics(self.tags, num_sentences)

        for sent_num, sentence in enumerate(parsed_sentences):
            utils.update_progress((sent_num + 1)/num_sentences)

            metrics.init_sentence_score(len(sentence))

            words, true_tags = zip(*sentence)
            test_phrase = str.join(' ', words)
            wordtag_guess = self.predict(test_phrase)

            for index, (word, guess_tag) in enumerate(wordtag_guess):
                true_tag = true_tags[index]
                metrics.update_predicted(true_tag, guess_tag)
                if guess_tag == true_tag:
                    metrics.update_sentence_score(index)
            metrics.checkout_sentence_score()
        metrics.log()

        plotter = StatsPlotter()
        plotter.plot_confusion_matrix(metrics, metrics.tags)


class PerformanceMetrics:
    '''Measures performance of the classifier on a test file'''
    def __init__(self, tags, num_sentences=0):
        self.tags = tags
        actual_vs_predicted = cartesian_product(tags, tags)
        self.confusion_dict = {each: 0 for each in actual_vs_predicted}
        self.correct_sentences = 0
        self.total_sentences = num_sentences
        self.sentence_score = []
        self.isNormalized = False

    def update_predicted(self, actual, predicted):
        if (actual, predicted) in self.confusion_dict:
            self.confusion_dict[actual, predicted] += 1

    def init_sentence_score(self, sentence_len):
        '''Counts how many correct words are in the sentence'''
        self.sentence_score = [False] * sentence_len

    def update_sentence_score(self, i):
        self.sentence_score[i] = True

    def checkout_sentence_score(self):
        if all(self.sentence_score):
            self.correct_sentences += 1

    def sentences_accuracy(self):
        '''Gets the % of sentences that were perfectly predicted.'''
        return utils.safe_division(self.correct_sentences, self.total_sentences)

    def _total_count(self, tag):
        count = 0
        for actual, predicted in self.confusion_dict:
            if actual == tag:
                count += self.confusion_dict[actual, predicted]
        return count

    def tag_accuracies(self):

        def _tag_hit_rate(tag):
            tag_hits = self.confusion_dict[tag, tag]
            tag_total_count = self._total_count(tag)
            return utils.safe_division(tag_hits, tag_total_count)

        tag_hit_rates = {}
        for tag in self.tags:
            tag_hit_rates[tag] = _tag_hit_rate(tag)
        return tag_hit_rates

    def normalize(self):
        normalized_conf_dict = {}
        for actual, pred in self.confusion_dict:
            normalized_conf_dict[actual, pred] = self.confusion_dict[actual, pred] / self._total_count(actual)
        self.confusion_dict = normalized_conf_dict
        self.isNormalized = True

    def get_confusion_matrix(self):
        N = len(self.tags)
        dtype = 'uint8' if not self.isNormalized else 'float32'
        confusion_matrix = np.empty([N, N], dtype=dtype)
        index_to_tag = {index: tag for index, tag in enumerate(sorted(self.tags))}
        for i, j in cartesian_product(range(N), range(N)):
            confusion_matrix[i, j] = self.confusion_dict[index_to_tag[i], index_to_tag[j]]
        return confusion_matrix

    def log(self):
        logging.basicConfig(
            filename='performance_statistics.log',
            format='%(asctime)s [%(levelname)s]:  %(message)s',
            datefmt='%d-%m-%Y %H:%M:%S',
            level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info('Fraction of correct sentences: ' + str(self.sentences_accuracy()))
        logger.info('Tag accuracies: ' + pprint.pformat(self.tag_accuracies()))

    def __getitem__(self, key):
        return self.confusion_dict[key]


class StatsPlotter:
    def __init__(self):
        pass

    def plot_confusion_matrix(self, confusion_dict, classes,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
        confusion_dict.normalize()
        cm = confusion_dict.get_confusion_matrix()
        classes = sorted(classes)
        n_classes = len(classes)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(n_classes)
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2
        fmt = '.2f'
        for i, j in cartesian_product(range(n_classes), range(n_classes)):
            plt.text(j, i, format(cm[i, j], fmt).rstrip('0').rstrip('.'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=6)

        plt.tight_layout()
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        fig = plt.figure(num=1)
        plt.draw()
        fig.savefig('confusion_matrix.png', dpi=fig.dpi)
        plt.show()
