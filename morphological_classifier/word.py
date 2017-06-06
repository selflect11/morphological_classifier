from . import constants
from . import word_parser
from . import data_formatter
from .tools import list_to_float
from nltk import stem
import numpy as np
import pickle
import string
from collections import OrderedDict

class Text:
    def __init__(self, words):
        self.words = []
    def add_line(self, words_list):
        for w in words_list:
            self.words.append(Word(w))
    def read_file(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                words = line.split(' ')
                self.add_line(words)
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)
    def write_to_file(self):
        # not sure if necessary
        pass

class Word:
    def __init__(self, word_plus_tags):
        self.word, tags_str = self.word_tag_separate(word_plus_tags)
        self.array = WordArray(self.word)
        self.tags_str = TagSet(tags_str)
    def word_tag_separate(self, word_plus_tags):
        word, tags_str = word_plus_tags.split('_')
        tags = TagSet(tags_str)
        separated_word = self.separate_word(word)
        return separated_word, tags
    def separate_word(self, word_str):
        # RSLP algorithm
        separator = constants.SEPARATOR
        word_str = word_str.lower()
        stemmer = stem.rslp.RSLPStemmer()
        radical = stemmer.stem(word_str)
        rest = word_str.split(radical, 1)[1]
        if not rest:
            print('Couldnt separate word {}'.format(word_str))
            return word_str
        return radical + separator + rest
    
class WordArray:
    def __init__(self, separated_word):
        self.letter_dic = self.string_to_dic(separated_word)
        self.array = self.dic_to_array(self.letter_dic)
    def string_to_dic(self, separated_word):
        SEPARATOR = constants.SEPARATOR
        binary_mask = [0 for each in range(len(separated_word))]
        max_binary = list_to_float([1 for each in binary_mask])
        letter_dic = OrderedDict()
        # ascii_lowercase = 'abcdef...xyz'
        for letter in string.ascii_lowercase:
            letter_dic[letter] = list(binary_mask)
        letter_dic[SEPARATOR] = list(binary_mask)
        # populates letter_dic
        for index, letter in enumerate(separated_word):
            letter_dic[letter][index] = 1 
        # converts letter_dic to numeric form, also normalizing it
        letter_dic = {letter : list_to_float(vec)/max_binary for letter, vec in letter_dic.items()}
        return letter_dic
    def dic_to_array(self, dic):
        return np.array(list(dic.values()), dtype=constants.D_TYPE)
    def __getitem__(self, key):
        if (self.letter_dic is None) or (key not in self.letter_dic):
            raise KeyError('Key {} not found'.format(key))
        return self.letter_dic[key]

class TagSet:
    def __init__(self, tags_string = ''):
        self.tag_set = self.str_to_tags(tags_string)
        self.tag_class = self.get_hybrid_class()
    def tag_separator(self, tags_string):
        tags_list = tags_string.split('+')
        return tags_list
    def str_to_tags(self, tags_list):
        tag_set = []
        for t in tags_list:
            new_tag = Tag(t)
            if new_tag:
                tag_set.append(new_tag)
        return tag_set
    def get_hybrid_class(self):
        return sum(tag.tag_class for tag in self.tag_set)
    def __bool__(self):
        if self.tag_set:
            return True
        return False
    def __getitem__(self, index):
        if index > len(self.tag_set):
            raise IndexError('{} index too big'.format(index))
        return self.tag_set[index]
    def __eq__(self, other):
        return (set(self.tag_set) == set(other.tag_set))

class Tag:
    def __init__(self, tag_string):
        self.tag = self.tag_strip(tag_string)
        self.tag_class = self.tag_to_class(self.tag)
    def tag_strip(self, tag_string):
        target_tags = constants.TARGET_TAGS
        for tt in target_tags:
            # If the tag is composite, strip it
            # e.g. ADV-KS -> ADV
            if tt == tag_string.split('-')[0]:
                return tt
        return None
    def tag_to_class(self, tag):
        tags_classes = constants.TAGS_CLASSES
        if tag in tags_classes:
            return tags_classes[tag]
        return None
    def __bool__(self):
        if self.tag:
            return True
        return False
    def __eq__(self, other):
        if self.tag == other.tag:
            return True
        return False
    def __hash__(self):
        return hash(self.tag)
