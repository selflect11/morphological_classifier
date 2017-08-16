# -*- coding: utf-8 -*-
import re
import pickle
from morphological_classifier import constants, utils

def parse_word_tag(string_element):
    ''' Parses an element of the form Word_tag1+tag2...|extra_info
    into a (word, [tag1, tag2,...]) tuple. '''
    word, tags_str = string_element.split('_')
    # Gets rid of extra information elements after the - character
    tags = [re.sub('-.*', '', tag) for tag in tags_str.split('+')]
    # Returns the first tag because the current classifier cant handle more than one tag per word
    return word.lower(), tags[0]

def parse_sentence(sentence):
    '''Gets "Word1_tag1 word2_tag2 word3_tag3..."
        returns [("word1", "tag1"), ("word2", "tag2"), ...]
    '''
    parsed_sentence = []
    for word_tags in sentence.split():
        parsed_sentence.append(parse_word_tag(word_tags))
    return parsed_sentence


class MorphologicalClassifier:
    def __init__(self, tagger, save_path=None):
        self.tagger = tagger
        self.isTrained = False

    def predict(self, phrase):
        return self.tagger.tag(phrase.split())

    def get_tags(self):
        return self.tagger.get_tags()

    def save(self, filepath):
        self.erase_useless()
        self.tagger.save(filepath)

    def load(self, filepath):
        self.tagger.load(filepath)
        self.isTrained = True

    def erase_useless(self):
        self.tagger.erase_useless()

    def train(self, filepath):
        if self.isTrained:
            print('Classifier already trained')
            return

        with open(filepath, 'r', encoding=constants.ENCODING) as f:
            sentences = f.readlines()
        parsed_sentences = [parse_sentence(sentence) for sentence in sentences]
        self.tagger.train(parsed_sentences)
        self.isTrained = True

    def test(self, filepath):
        if not self.isTrained:
            print('Tagger not yet trained')
            return

        print('Starting testing phase...')
        with open(filepath, 'r', encoding=constants.ENCODING) as f:
            sentences = f.readlines()
        parsed_sentences = [
            parse_sentence(sentence) for sentence in sentences
            ]
        # Metric variables
        total_accuracy = 0
        num_sentences = len(parsed_sentences)
        tag_hit_count = {
            tag: {'right': 0, 'total': 0} for tag in self.get_tags()
            }

        for sent_num, sentence in enumerate(parsed_sentences):
            utils.update_progress((sent_num + 1)/num_sentences)
            # measures how many right words in the sentence
            sentence_score = [False for word in sentence]

            words, true_tags = zip(*sentence)
            test_phrase = str.join(' ', words)
            wordtag_guess = self.predict(test_phrase)

            for index, guess in enumerate(wordtag_guess):
                true_tag = true_tags[index]
                guess_tag = guess[1]
                if guess_tag == true_tag:
                    tag_hit_count[true_tag]['right'] += 1
                    sentence_score[index] = True
                tag_hit_count[true_tag]['total'] += 1
            if all(sentence_score):
                total_accuracy += 1

        tag_accuracy = {
            tag: utils.safe_division(
                tag_hit_count[tag]['right'], tag_hit_count[tag]['total'])
            for tag
            in tag_hit_count
            }
        total_accuracy /= num_sentences

        print('Total accuracy {}'.format(total_accuracy))
        print('Tag accuracy {}'.format(tag_accuracy))
