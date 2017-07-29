# -*- coding: iso-8859-1 -*-
from . import constants
from .tools import update_progress, pairwise
from .markov_chain import TransitionProbabilities, InitialProbabilities
import itertools
import re

def parse_word_tag(string_element):
    ''' Parses an element of the form Word_tag1+tag2...|extra_info
        into a (word, [tag1, tag2,...]) tuple. '''
    # gets rid of extra information elements after the | character
    string_element = re.sub('\|.*', '', string_element)
    word, tags_str = string_element.split('_')
    tags = tags_str.split('+')
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
        self.word_tag_dict = {}
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
                percent_done = line_num/(num_lines - 1)
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
                    if word in self.word_tag_dict:
                        self.word_tag_dict[word].update([tag])
                    else:
                        self.word_tag_dict[word] = set([tag])
        self.trans_prob.add_transitions(line_tags)
        self.init_prob.add_tag(initial_tag)

    def __getitem__(self, tag):
        return self.word_tag_dict[tag]
