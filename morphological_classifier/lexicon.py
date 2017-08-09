# -*- coding: iso-8859-1 -*-
from . import constants
from .tools import update_progress, pairwise
from .markov_chain import TransitionProbabilities, InitialProbabilities
from collections import defaultdict
import itertools
import re

def parse_word_tag(string_element):
    ''' Parses an element of the form Word_tag1+tag2...|extra_info
        into a (word, [tag1, tag2,...]) tuple. '''
    word, tags_str = string_element.split('_')
    # Gets rid of extra information elements after the - character
    tags = [re.sub('-.*', '', tag) for tag in tags_str.split('+')]
    return word.lower(), tags

def tag_is_valid(tag):
    if tag in constants.TARGET_TAGS:
        return True
    return False


class WordTags:
    def __init__(self, filepath):
        ''' Stores a dictionary of words
            with their respective set of tags,
            and all possible tag transition
            probabilities. '''
        # { word1 : set(tag1, tag2, ...), ... }
        self.word_tag_dict = defaultdict(set)
        self.trans_prob = TransitionProbabilities()
        self.init_prob = InitialProbabilities()
        self.setup(filepath)

    def setup(self, source_filepath):
        self.load_word_tags_from_file(source_filepath)
        self.trans_prob.calculate_probabilities()
        self.init_prob.calculate_probabilities()

    def load_word_tags_from_file(self, filepath):
        with open(filepath, 'r', encoding=constants.ENCODING) as f:
            lines = f.readlines()
            num_lines = len(lines)
            for line_num, line in enumerate(lines):
                self.add_line(line)
                percent_done = (line_num + 1)/num_lines
                update_progress(percent_done)

    def add_line(self, line):
        line_tags = []
        initial_tag = None
        for wordtag in line.split():
            word, tags_list = parse_word_tag(wordtag)
            for tag in tags_list:
                if tag_is_valid(tag):
                    line_tags.append(tag)
                    if initial_tag is None:
                        initial_tag = tag
                    self.word_tag_dict[word].update([tag])
        self.trans_prob.add_transitions(line_tags)
        self.init_prob.add_tag(initial_tag)

    def get_transition_probability(self, transition):
        return self.trans_prob[transition]

    def get_initial_probability(self, tag):
        return self.init_prob[tag]

    def __getitem__(self, tag):
        return self.word_tag_dict[tag]

    def __contains__(self, tag):
        return (tag in self.word_tag_dict)
