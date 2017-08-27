# -*- coding: utf-8 -*-
import logging
import numpy as np
import sklearn
from morphological_classifier import constants, utils
from time import time


class PerformanceMetrics:
    '''Measures performance of the classifier on a test file'''
    def __init__(self, num_sentences=0):
        self.correct_sentences = 0
        self.total_sentences = num_sentences
        self.sentence_score = []
        self.timestamp = 0
        self.test_time = ''

        # Sklearn compliant
        self.tags = constants.TAGS
        self.confusion_matrix = np.empty([len(self.tags), len(self.tags)], dtype='uint8')
        self.y_true = []
        self.y_pred = []

    def update_predicted(self, actual, predicted):
        self.y_true.append(actual)
        self.y_pred.append(predicted)

    def build_confusion_matrix(self):
        self.confusion_matrix = sklearn.metrics.confusion_matrix(
                self.y_true, self.y_pred,
                labels=self.tags)

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

    def checkin_time(self):
        self.timestamp = time()

    def checkout_time(self):
        time_elapsed = time() - self.timestamp
        hours, remainder = divmod(time_elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.test_time = '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(minutes),seconds)

    def log(self):
        logging.basicConfig(
            filename='performance_statistics.log',
            format='%(asctime)s [%(levelname)s]:  %(message)s',
            datefmt='%d-%m-%Y %H:%M:%S',
            level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info('ORIGINAL CODE')
        logger.info('Testing time: ' + self.train_time)
        logger.info('Fraction of correct sentences: ' + str(self.sentences_accuracy()))
        logger.info('\n' + sklearn.metrics.classification_report(self.y_true,
                                                                 self.y_pred,
                                                                 target_names=self.tags))
