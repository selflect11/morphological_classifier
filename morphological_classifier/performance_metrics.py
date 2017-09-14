# -*- coding: utf-8 -*-
import logging
import numpy as np
import sklearn
from morphological_classifier import constants, utils
from time import time
from statistics import *
import pprint


class PerformanceMetrics:
    '''Measures performance of the classifier on a test file'''
    def __init__(self, train_size, log_msgs):
        self.timestamp = 0
        self.train_size = train_size if train_size else constants.TRAIN_SIZE
        self.test_size = 1 - self.train_size
        self.tags = constants.TAGS

        self.confusion_matrix = np.empty([len(self.tags), len(self.tags)], dtype='uint8')
        self.y_true = []
        self.y_pred = []
        self.classification_reports = []
        self.testing_times = []
        self.accuracy_scores = []

        self.f1_scores = {tag: [] for tag in self.tags + ['avg/total']}
        self.precision_scores = {tag: [] for tag in self.tags + ['avg/total']}
        self.recall_scores = {tag: [] for tag in self.tags + ['avg/total']}

        self.log_msgs = log_msgs

    def update_predicted(self, actual, predicted):
        self.y_true.append(actual)
        self.y_pred.append(predicted)

    def build_confusion_matrix(self):
        self.confusion_matrix = sklearn.metrics.confusion_matrix(
                self.y_true, self.y_pred,
                labels=self.tags)

    def checkin(self):
        self.timestamp = time()

    def checkout(self):
        self.testing_times.append(time() - self.timestamp)
        self.classification_reports.append(
            self.get_classification_report())
        self.accuracy_scores.append(self.get_accuracy_score())

    def get_accuracy_score(self):
        return sklearn.metrics.accuracy_score(
                self.y_true, self.y_pred)

    def get_classification_report(self):
        return sklearn.metrics.classification_report(
                self.y_true, self.y_pred, target_names=self.tags)

    def get_f1_precision_recall_scores(self):
        for cr in self.classification_reports:
            for line in cr.split('\n'):
                tag, (prec, rec, f1) = utils.get_scores_from_text(line)
                if tag in self.tags:
                    self.precision_scores[tag].append(prec)
                    self.recall_scores[tag].append(rec)
                    self.f1_scores[tag].append(f1)
                total_prec, total_rec, total_f1 = utils.get_total_scores_from_text(line)
                if any([total_prec, total_rec, total_f1]):
                    self.precision_scores['avg/total'].append(total_prec)
                    self.recall_scores['avg/total'].append(total_rec)
                    self.f1_scores['avg/total'].append(total_f1)

    def get_averaged_classification_report(self):
        self.get_f1_precision_recall_scores()
        avg_class_rep = {tag: None for tag in self.tags + ['avg/total']}
        for tag in avg_class_rep:
            ps = self.precision_scores[tag]
            rs = self.recall_scores[tag]
            f1 = self.f1_scores[tag]
            avg_class_rep[tag] = {
                'precision': '{:.4} +/- {:.4}'.format(
                    *utils.get_mean_stdev(ps)),
                'recall': '{:.4} +/- {:.4}'.format(
                    *utils.get_mean_stdev(rs)),
                'f1': '{:.4} +/- {:.4}'.format(
                    *utils.get_mean_stdev(f1))}
        return avg_class_rep

    def format_class_report(self, cr):
        # TODO
        headers = ['tag', 'precision', 'recall', 'f1']
        header_fmt = '{:^15}'*len(headers)
        header = header_fmt.format(*headers)

        line_fmt = str.join(spaces,  ['{:^20' + h + '}' for h in headers])
        return

    def format_time(self, t):
        hours, remainder = divmod(t, 3600)
        minutes, seconds = divmod(remainder, 60)
        test_time = '{:0>2}:{:0>2}:{:05.2f}'.format(
            int(hours),int(minutes),seconds)
        return test_time

    def log(self):
        logging.basicConfig(
            filename='performance_statistics.log',
            format='%(asctime)s [%(levelname)s]: %(message)s',
            datefmt='%d-%m-%Y %H:%M:%S',
            level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info('IMPLEMENTED CROSS VALIDATION')
        if self.log_msgs:
            for msg in self.log_msgs:
                logger.info(msg)
        logger.info('REMOVED THE word_i_pref_1 FEATURE')
        logger.info('Train/Test sizes: {:.2}/{:.2}'.format(
            self.train_size, self.test_size))

        avg_time = mean(self.testing_times)
        total_time = sum(self.testing_times)
        logger.info('Total time elapsed during testing {}'.format(
            self.format_time(total_time)))
        logger.info('Average time elapsed during testing {}'.format(
            self.format_time(avg_time)))

        logger.info('Average accuracy scores: {:.4} +/- {:.4}'.format(
            mean(self.accuracy_scores), stdev(self.accuracy_scores)))
        logger.info(pprint.pformat(self.get_averaged_classification_report()))
