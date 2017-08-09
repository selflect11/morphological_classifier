from .lexicon import WordTags
from .tools import pairwise
import itertools


class MorphologicalClassifier:
    def __init__(self, filepath):
        self.word_tags = WordTags(filepath)

    def predict(self, phrase):
        words_list = phrase.split()
        probable_tags = self.get_most_likely_tags(phrase)
        return zip(words_list, probable_tags)

    def get_most_likely_tags(self, phrase):
        tags_phrase = self.get_tags_phrase(phrase)
        all_paths = self.get_all_paths(tags_phrase)
        probabilities = []
        for path in all_paths:
            first, *rest = path
            init_prob = self.word_tags.get_initial_probability(first)
            trans_probs = 1
            for tag_i, tag_j in pairwise(rest):
                trans_probs *= self.word_tags.get_transition_probability((tag_i, tag_j))
            probabilities.append(init_prob * trans_probs)
        greatest_prob_index = probabilities.index(max(probabilities))
        return all_paths[greatest_prob_index]

    def get_all_paths(self, tags_phrase):
        # in: [{tag1}, {tag21, tag22}, {tag3}, {tag41, tag42, ... }, ...]
        # out: (tag1, tag21, tag3, tag41, ...),
        #      (tag1, tag22, tag3, tag41, ...),
        #      (tag1, tag21, tag3, tag42, ...),
        # ...
        return list(itertools.product(*tags_phrase))

    def get_tags_phrase(self, phrase):
        # in: "Word1 word2 word3..."
        # out: [{tag1}, {tag21, tag22}, {tag3}, ...]
        tags_phrase = []
        for word in phrase.split():
            if word in self.word_tags:
                tags_phrase.append(self.word_tags[word])
            else:
                # Eventually, do processing to find tag.
                # Right now we're guessing N for all unrecognized tags.
                tags_phrase.append({'N'})
        return tags_phrase


class PerformanceEvaluator:
    def __init__(self):
        ''' Loads one big file as input for tags and
            tests on another and outputs performance.'''
        pass
