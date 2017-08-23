# -*- coding: utf-8 -*-
import random
from collections import defaultdict
import pickle
from morphological_classifier import utils

def defaultdict_float():                   #only needed for dumbass serialization
    return defaultdict(float)


class AveragedPerceptron:
    def __init__(self):
        # Each feature gets its own weight vector, so weights is a dict-of-dicts
        # self.weights[feature] := {tag1: w1, tag2: w2, ...}
        self.weights = defaultdict(defaultdict_float)
        self.tags = set()
        # The accumulated values, for the averaging. These will be keyed by
        # feature/tag tuples
        self._totals = defaultdict(int)
        # The last time the feature was changed, for the averaging. Also
        # keyed by feature/tag tuples
        # (tstamps is short for timestamps)
        self._tstamps = defaultdict(int)
        # Number of instances seen
        self.i = 0

    def predict(self, features):
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
            # If weight doesn't exist, it's initialized as 0.0
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
        # for pickling
        self._totals = self._tstamps = self.i = None

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f, -1)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.__dict__ = pickle.load(f)


class PerceptronTagger:
    '''Greedy Averaged Perceptron tagger'''

    # Tags used for padding, since the context uses
    # two words before and after the current word
    START = ['-START-', '-START2-']
    END = ['-END-', '-END2-']

    def __init__(self):
        self.model = AveragedPerceptron()
        self.tagdict = {}
        self.tags = set()

    def tag(self, words):
        prev, prev2 = self.START
        output = []
        context = self.START + [self.normalize(w) for w in words] + self.END
        for i, word in enumerate(words):
            tag = self.tagdict.get(word)
            if not tag:
                features = self._get_features(i, word, context, prev, prev2)
                tag = self.model.predict(features)
            output.append((word, tag))
            prev2 = prev
            prev = tag
        return output

    def train(self, sentences, nr_iter=5):
        ''':param sentences: A list or iterator of sentences, where each sentence
            is a list of (words, tags) tuples.
            :param nr_iter: Number of training iterations.'''

        self._sentences = list()            # to be populated by self._make_tagdict...
        self._make_tagdict(sentences)
        self.model.tags = self.tags
        for iter_ in range(nr_iter):
            num_sentences = len(self._sentences)
            for curr_sentence, sentence in enumerate(self._sentences):
                if not sentence:
                    continue
                utils.update_progress((curr_sentence + 1)/num_sentences)

                words, tags = zip(*sentence)
                prev, prev2 = self.START
                context = self.START + [self.normalize(w) for w in words] + self.END
                for i, word in enumerate(words):
                    guess = self.tagdict.get(word)
                    if not guess:
                        feats = self._get_features(i, word, context, prev, prev2)
                        guess = self.model.predict(feats)
                        self.model.update(feats, tags[i], guess)
                    prev2 = prev
                    prev = guess
            random.shuffle(self._sentences)
        # We don't need the training sentences anymore, and we don't want to
        # waste space on them when we pickle the trained tagger.
        self._sentences = None
        self.model.average_weights()

    def normalize(self, word):
        '''Normalization used in pre-processing.
        - All words are lower cased
        - All numeric words are returned as !DIGITS'''
        if word.isdigit():
                return '!DIGITS'
        else:
                return word.lower()

    def _get_features(self, i, word, context, prev, prev2):
        '''Map words into a feature representation.'''
        def add(name, *args):
            features[str.join(' ', (name,) + tuple(args))] += 1

        i += len(self.START)
        features = defaultdict(int)
        add('bias')
        add('i suffix', word[-3:])
        add('i pref1', word[0])
        add('i-1 tag', prev)
        add('i-2 tag', prev2)
        add('i tag+i-2 tag', prev, prev2)
        add('i word', context[i])
        add('i-1 tag+i word', prev, context[i])
        add('i-1 word', context[i-1])
        add('i-1 suffix', context[i-1][-3:])
        add('i-2 word', context[i-2])
        add('i+1 word', context[i+1])
        add('i+1 suffix', context[i+1][-3:])
        add('i+2 word', context[i+2])
        return features

    def _make_tagdict(self, sentences):
        '''Make a tag dictionary for single-tag words.
        :param sentences: A list of list of (word, tag) tuples.'''
        counts = defaultdict(lambda: defaultdict(int))
        for sentence in sentences:
            self._sentences.append(sentence)
            for word, tag in sentence:
                counts[word][tag] += 1
                self.tags.add(tag)
        freq_thresh = 20
        ambiguity_thresh = 0.97
        for word, tag_freqs in counts.items():
            tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
            n = sum(tag_freqs.values())
            # Don't add rare words to the tag dictionary
            # Only add quite unambiguous words
            if n >= freq_thresh and (mode / n) >= ambiguity_thresh:
                self.tagdict[word] = tag

    def erase_useless_data(self):
        self.model.erase_useless_data()

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model.load(filepath)
