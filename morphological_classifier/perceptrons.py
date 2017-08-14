# -*- coding: utf-8 -*-
import random
from collections import defaultdict


class AveragedPerceptron(object):
    def __init__(self):
        # Each feature gets its own weight vector, so weights is a dict-of-dicts
        self.weights = {}
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
            weights = self.weights[feat]
            for label, weight in weights.items():
                scores[label] += value * weight
        # Do a secondary alphabetic sort, for stability
        return max(self.tags, key=lambda label: (scores[label], label))

    def update(self, true_value, guess, features):
        '''Update the feature weights.'''
        def upd_feat(guess, feature, w, v):
            param = (feature, guess)
            self._totals[param] += (self.i - self._tstamps[param]) * w
            self._tstamps[param] = self.i
            self.weights[feature][guess] = w + v

        self.i += 1
        if true_value == guess:
            return
        else:
            for feature in features:
                weights = self.weights.setdefault(feature, {})
                upd_feat(true_value, feature, weights.get(true_value, 0.0), 1.0)
                upd_feat(guess, feature, weights.get(guess, 0.0), -1.0)

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


class PerceptronTagger:
    '''
    Greedy Averaged Perceptron tagger

    >>> from nltk.tag.perceptron import PerceptronTagger
    Train the model

    >>> tagger = PerceptronTagger(load=False)

    >>> tagger.train([[('today','NN'),('is','VBZ'),('good','JJ'),('day','NN')],
    ... [('yes','NNS'),('it','PRP'),('beautiful','JJ')]])

    >>> tagger.tag(['today','is','a','beautiful','day'])
    [('today', 'NN'), ('is', 'PRP'), ('a', 'PRP'), ('beautiful', 'JJ'), ('day', 'NN')]
    '''

    START = ['-START-', '-START2-']
    END = ['-END-', '-END2-']

    def __init__(self):
        self.model = AveragedPerceptron()
        self.tagdict = {}
        self.tags = set()

    def tag(self, tokens):
        prev, prev2 = self.START
        output = []

        context = self.START + [self.normalize(w) for w in tokens] + self.END
        for i, word in enumerate(tokens):
            tag = self.tagdict.get(word)
            if not tag:
                features = self._get_features(i, word, context, prev, prev2)
                tag = self.model.predict(features)
            output.append((word, tag))
            prev2 = prev
            prev = tag
        return output

    def train(self, sentences, nr_iter=5):
        '''Train a model from sentences. nr_iter controls the number of Perceptron
        training iterations.
        :param sentences: A list or iterator of sentences, where each sentence
            is a list of (words, tags) tuples.
        :param nr_iter: Number of training iterations.
        '''
        # We'd like to allow sentences to be either a list or an iterator,
        # the latter being especially important for a large training dataset.
        # Because self._make_tagdict(sentences) runs regardless, we make
        # it populate self._sentences (a list) with all the sentences.
        # This saves the overheard of just iterating through sentences to
        # get the list by sentences = list(sentences).

        self._sentences = list()  # to be populated by self._make_tagdict...
        self._make_tagdict(sentences)
        self.model.tags = self.tags
        for iter_ in range(nr_iter):
            for sentence in self._sentences:
                if not sentence:
                    continue
                words, tags = zip(*sentence)
                prev, prev2 = self.START
                context = self.START + [self.normalize(w) for w in words] + self.END
                for i, word in enumerate(words):
                    guess = self.tagdict.get(word)
                    if not guess:
                        feats = self._get_features(i, word, context, prev, prev2)
                        guess = self.model.predict(feats)
                        self.model.update(tags[i], guess, feats)
                    prev2 = prev
                    prev = guess
            random.shuffle(self._sentences)
        # We don't need the training sentences anymore, and we don't want to
        # waste space on them when we pickle the trained tagger.
        self._sentences = None
        self.model.average_weights()

    def normalize(self, word):
        '''
        Normalization used in pre-processing.
        - All words are lower cased
        - Groups of digits of length 4 are represented as !YEAR;
        - Other digits are represented as !DIGITS
        :rtype: str
        '''
        if '-' in word and word[0] != '-':
            return '!HYPHEN'
        elif word.isdigit() and len(word) == 4:
            return '!YEAR'
        elif word[0].isdigit():
            return '!DIGITS'
        else:
            return word.lower()

    def _get_features(self, i, word, context, prev, prev2):
        '''Map tokens into a feature representation, implemented as a
        {hashable: int} dict. If the features change, a new model must be
        trained.
        '''
        def add(name, *args):
            features[' '.join((name,) + tuple(args))] += 1

        i += len(self.START)
        features = defaultdict(int)
        # It's useful to have a constant feature, which acts sort of like a prior
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
        '''
        Make a tag dictionary for single-tag words.
        :param sentences: A list of list of (word, tag) tuples.
        '''
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
