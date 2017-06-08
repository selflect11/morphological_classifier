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
    def __init__(self):
        self.words_list = []
    def add_line(self, line):
        for w in line.split(' '):
            new_word = Word(w)
            if new_word:
                self.words_list.append(Word(w))
    def read_file(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.add_line(line)
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
    def __getitem__(self, index):
        return self.words_list[index]

class Word:
    def __init__(self, word_plus_tags):
        self.word, self.tag_set = self.word_tag_separate(word_plus_tags)
        if self.tag_set:
            self.array = WordArray(self.word)
        else:
            self.array = None
    def word_tag_separate(self, word_plus_tags):
        word, tags_str = word_plus_tags.split('_')
        tags = TagSet(tags_str)
        separated_word = self.separate_word_from_radical(word)
        return separated_word, tags
    def separate_word_from_radical(self, word_str):
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
    def __eq__(self, other):
        if isinstance(other, Word):
            return (self.word == other.word)
        else:
            return (self.word == other)
    def __bool__(self):
        return bool(self.tag_set)
    
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
        if key not in self.letter_dic:
            raise KeyError('Key {} not found'.format(key))
        elif self.letter_dic is None:
            raise Exception('Dict not initialized')
        return self.letter_dic[key]

class TagSet:
    def __init__(self, tags_string = ''):
        self.tags_list = self.str_to_tags_list(tags_string)
        self.tag_class = self.get_hybrid_class()
    def str_to_tags_list(self, tags_string):
        str_list = tags_string.split('+')
        tags_list = []
        for t in str_list:
            new_tag = Tag(t)
            if new_tag:
                tags_list.append(new_tag)
        return tags_list
    def get_hybrid_class(self):
        if bool(self):
            return sum(tag.tag_class for tag in self.tags_list)
        return np.zeros(4)
    def __bool__(self):
        if self.tags_list:
            return True
        return False
    def __getitem__(self, index):
        if index > len(self.tags_list):
            raise IndexError('{} index too big'.format(index))
        return self.tags_list[index]
    def __eq__(self, other):
        return (set(self.tags_list) == set(other.tags_list))
    def __str__(self):
        return ",".join(str(tag) for tag in self)

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
    def __str__(self):
        return self.tag
