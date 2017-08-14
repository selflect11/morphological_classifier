# -*- coding: utf-8 -*-
import re
from morphological_classifier import constants

def parse_word_tag(string_element):
    ''' Parses an element of the form Word_tag1+tag2...|extra_info
    into a (word, [tag1, tag2,...]) tuple. '''
    word, tags_str = string_element.split('_')
    # Gets rid of extra information elements after the - character
    tags = [re.sub('-.*', '', tag) for tag in tags_str.split('+')]
    # Returns the first tag because the current classifier cant handle more than one tag per word
    return word.lower(), tags[0]

def parse_sentence(sentence):
    parsed_sentence = []
    for word_tags in sentence.split():
        parsed_sentence.append(parse_word_tag(word_tags))
    return parsed_sentence

class MorphologicalClassifier:
    def __init__(self, tagger):
        self.tagger = tagger

    def classify(self, phrase):
        return tuple(map(self.tagger.tag, phrase.split()))

    def train(self, filepath):
        with open(filepath, 'r', encoding=constants.ENCODING) as f:
            sentences = f.readlines()
        # prepares the sentences
        parsed_sentences = [parse_sentence(sentence) for sentence in sentences]
        self.tagger.train(parsed_sentences)
