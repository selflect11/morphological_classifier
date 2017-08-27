# -*- coding: utf-8 -*-
import random
from collections import defaultdict
import pickle
from morphological_classifier import constants, utils


class AveragedPerceptron:
    def __init__(self):
        # Each feature gets its own weight vector, so weights is a dict-of-dicts
        # self.weights[feature] := {tag1: w1, tag2: w2, ...}
        self.weights = defaultdict(utils.defaultdict_float)
        self.tags = constants.TAGS
        # The accumulated values, for the averaging. These will be keyed by
        # feature/tag tuples
        self._totals = defaultdict(int)
        # The last time the feature was changed, for the averaging. Also
        # keyed by feature/tag tuples
        self._tstamps = defaultdict(int)
        # Number of instances seen
        self.i = 0

    def predict_tag(self, features):
        '''Dot-product the features and current weights and return the best label.'''
        scores = defaultdict(float)
        for feat, value in features.items():
            if feat not in self.weights or value == 0:
                continue
            tag_weight_dict = self.weights[feat]
            for tag, weight in tag_weight_dict.items():
                scores[tag] += value * weight
        return max(self.tags, key=lambda tag: scores[tag])

    def update(self, features, true_tag, guess):
        '''Update the feature weights.'''
        def upd_feat(feature, tag, val):
            param = (feature, tag)
            # If the weight doesn't exist, it's initialized as 0.0
            # Courtesy of defaultdict
            curr_weight = self.weights[feature][tag]
            self._totals[param] += (self.i - self._tstamps[param]) * curr_weight
            self._tstamps[param] = self.i
            self.weights[feature][tag] += val
        self.i += 1
        if true_tag == guess:
            return
        else:
            for feature in features:
                upd_feat(feature, true_tag, 1.0)
                upd_feat(feature, guess, -1.0)

    def average_weights(self):
        '''Average weights from all iterations.'''
        for feat, weights in self.weights.items():
            new_feat_weights = {}
            for tag, weight in weights.items():
                param = (feat, tag)
                total = self._totals[param]
                total += (self.i - self._tstamps[param]) * weight
                averaged = round(total / self.i, 3)
                if averaged:
                    new_feat_weights[tag] = averaged
            self.weights[feat] = new_feat_weights

    def erase_useless_data(self):
        # Used after training, for pickling
        self._sentences = None
        self._totals = None
        self._tstamps = None
        self.i = None
