# -*- coding: utf-8 -*-
from morphological_classifier.perceptron import AveragedPerceptron
from morphological_classifier.performance_metrics import PerformanceMetrics
from morphological_classifier.stats_plot import StatsPlotter
from morphological_classifier import constants, utils
import numpy as np
from collections import defaultdict
from sklearn import model_selection
import pickle
import random


class MorphologicalClassifier:
    # Tags used for padding, since the _get_features method uses
    # two words before and after the current word
    START = ['__START__', '__START2__']
    END = ['__END__', '__END2__']
    def __init__(self, metrics, plotter, save_path, data_path, logging, n_splits):
        self.model = AveragedPerceptron()
        self.metrics = metrics
        self.plotter = plotter
        self.tags = constants.TAGS
        self.tag_dict = dict()
        self.save_path = save_path
        self.data_path = data_path
        self.n_splits = n_splits
        self.logging = logging
        self.isTrained = False

    def predict(self, phrase):
        ''':type phrase: str
           :rtype: list(tuple(str, str))'''
        output = []
        tags = []
        words = phrase.split()
        for i, word in enumerate(words):
            tag = self.tag_dict.get(word)
            if not tag:
                features = self._get_features(words, tags, i)
                tag = self.model.predict_tag(features)
            output.append((word, tag))
            tags.append(tag)
        return output

    def _get_features(self, words, tags, i):
        '''
        Map words into a feature representation.

        :type words: list(str)
        :type tags: list(str)
        :type i: int
        '''
        features = defaultdict(int)
        starts_capitalized = words[i][0].isupper()
        # Padding the tags, words and index
        words = self.START + [self.normalize(w) for w in words] + self.END
        tags = self.START + tags
        i += len(self.START)

        def add_feature(feat_id, *values):
            features[str.join(' ', (feat_id,) + tuple(values))] += 1

        add_feature('bias')
        #add_feature('word_i_pref_1', words[i][0])
        add_feature('tag_(i-1)', tags[i-1])
        add_feature('tag_(i-2)', tags[i-2])
        add_feature('tag_(i-1) tag_(i-2)', tags[i-1], tags[i-2])
        add_feature('word_i_suffix', utils.get_suffix(words[i]))
        add_feature('word_i', words[i])
        add_feature('tag_(i-1) word_i', tags[i-1], words[i])
        add_feature('word_(i-1)', words[i-1])
        add_feature('word_(i-1)_suffix', utils.get_suffix(words[i-1]))
        add_feature('word_(i-2)', words[i-2])
        add_feature('word_(i+1)', words[i+1])
        add_feature('word_(i+1)_suffix', utils.get_suffix(words[i+1]))
        add_feature('word_(i+2)', words[i+2])
        #add_feature('word_i_starts_capitalized', str(starts_capitalized))
        return features

    def _make_tag_dict(self, sentences):
        '''Make a tag dictionary for single-tag words.
        :param sentences: A list of list of (word, tag) tuples.'''
        counts = defaultdict(lambda: defaultdict(int))
        for sentence in sentences:
            for word, tag in sentence:
                counts[word][tag] += 1
        freq_thresh = 20
        ambiguity_thresh = 0.97
        for word, tag_freqs in counts.items():
            tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
            n = sum(tag_freqs.values())
            # Don't add rare words to the tag dictionary
            # Only add quite unambiguous words
            if n >= freq_thresh and (mode / n) >= ambiguity_thresh:
                self.tag_dict[word] = tag

    def parse_sentence(self, sentence):
        '''Gets "Word1_tag1 word2_tag2 word3_tag3..."
            returns [("word1", "tag1"), ("word2", "tag2"), ...]'''

        def parse_word_tag(string_element):
            '''Parses an element of the form Word_tag1+tag2...|extra_info
            into a (word, tags) tuple.'''
            word, tags_str = string_element.split('_')
            return self.normalize(word), tags_str

        parsed_sentence = []
        for word_tags in sentence.split():
            parsed_sentence.append(parse_word_tag(word_tags))
        return parsed_sentence

    def normalize(self, word):
        '''Normalization used in pre-processing.
        - All words are lower cased
        - All numeric words are returned as !DIGITS'''
        if word.isdigit():
            return '!DIGITS'
        else:
            return word.lower()

    def train_test(self):
        with open(self.data_path, 'r', encoding=constants.ENCODING) as f:
            sentences = f.readlines()
        parsed_sentences = np.array([self.parse_sentence(s) for s in sentences])
        kf = model_selection.KFold(n_splits=self.n_splits)
        for i, (train, test) in enumerate(kf.split(parsed_sentences)):
            print('\nStarting train/test {} of {}'.format(i+1, self.n_splits))
            self.train(train_sentences=parsed_sentences[train])
            self.test(test_sentences=parsed_sentences[test], metrics=self.metrics)
            self.reset()
        if self.logging:
            self.metrics.log()
        self.plotter.plot_confusion_matrix(self.metrics.confusion_matrix, normalize=True)
        #self.save()

    def train(self, train_sentences, nr_iter=5):
        if self.isTrained:
            print('Classifier already trained')
            return

        print('Starting training phase...')
        self._make_tag_dict(train_sentences)
        num_sentences = len(train_sentences)

        for iter_ in range(nr_iter):
            # Padding
            sent_padd = num_sentences * iter_
            for sent_num, sentence in enumerate(train_sentences):
                if not sentence:
                    continue
                utils.update_progress((sent_num + sent_padd + 1)/(nr_iter * num_sentences))

                words, true_tags = zip(*sentence)
                guess_tags = []
                for i, word in enumerate(words):
                    guess = self.tag_dict.get(word)
                    if not guess:
                        feats = self._get_features(words, guess_tags, i)
                        guess = self.model.predict_tag(feats)
                        self.model.update(feats, true_tags[i], guess)
                    guess_tags.append(guess)
            random.shuffle(train_sentences)
        self.model.average_weights()

        self.erase_useless_data()
        self.isTrained = True

    def test(self, test_sentences, metrics):
        if not self.isTrained:
            print('Model not yet trained')
            return

        print('Starting testing phase...')
        # Metrics stuff
        num_sentences = len(test_sentences)
        metrics.checkin()

        for sent_num, sentence in enumerate(test_sentences):
            utils.update_progress((sent_num + 1)/num_sentences)

            words, true_tags = zip(*sentence)
            test_phrase = str.join(' ', words)
            wordtag_guess = self.predict(test_phrase)

            for index, (word, guess_tag) in enumerate(wordtag_guess):
                true_tag = true_tags[index]
                metrics.update_predicted(true_tag, guess_tag)

        metrics.checkout()
        metrics.build_confusion_matrix()

    def save(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.__dict__, f, -1)

    def load(self):
        with open(self.save_path, 'rb') as f:
            self.__dict__ = pickle.load(f)
        self.isTrained = True

    def erase_useless_data(self):
        self.model.erase_useless_data()

    def reset(self):
        self.model = AveragedPerceptron()
        self.tag_dict = dict()
        self.isTrained = False

    def __getitem__(self, key):
        return self.confusion_matrix[key]
