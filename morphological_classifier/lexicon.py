# -*- coding: iso-8859-1 -*-
from . import constants
from .tools import list_to_float, separate_word_from_radical, update_progress, pairwise
from nltk import stem
from collections import OrderedDict, Counter
import itertools
import numpy as np
import pickle
import string
import re

def parseElement(string_element):
    ''' Parses an element of the form Word_tag1+tag2...|extra_info
        into a (word, [tag1, tag2,...]) tuple. '''
    string_element = string_element.lower()
    # gets rid of extra information elements after the | character
    string_element = re.sub('\|.*', '', string_element)
    word, tags_str = string_element.split('-')
    tags = tags_str.split('+')
    return word, tags

def tagIsValid(tag):
    if tag in constants.TAGET_TAGS:
        return True
    return False

# Words are unique inside this class
class WordTag:
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
        self.loadElementsFromFile(source_filepath)
        self.trans_prob.calculateProbabilities()
        self.init_prob.calculateProbabilities()

    def loadElementsFromFile(self, filepath):
        with open(filepath, 'r', encoding=constants.ENCODING) as f:
            lines = f.readlines()
            num_lines = len(lines)
            for line_num, line in enumerate(lines):
                self.add_line(line)
                percent_done = line_num/(num_lines - 1)
                update_progress(percent_done)

    def addLine(self, line):
        line_tags = []
        initial_tag = None
        for element in line.split(' '):
            word, tags_list = parse_element(element)
            for tag in tags_list:
                if tagIsValid(tag):
                    line_tags.append(tag)
                    initial_tag == tag if initial_tag is None
                    if word in self.word_tag_dict:
                        self.word_tag_dict[word].update(tag)
                    else:
                        self.word_tag_dict[word] = set(tag)
        self.trans_prob.loadTags(line_tags)
        self.init_prob.loadTag(initial_tag)

class Text:
    def __init__(self, filepath):
        self.word_tag_dict = {}
        self.read_file(filepath)
        num_tags = self.get_num_tags()
        transition_probabilities_dict = initialize_trans_prob_dict()
        initial_probabilities_dict = {}
    def readFile(self, filepath):
        with open(filepath, 'r', encoding=constants.ENCODING) as f:
            lines = f.readlines()
            num_lines = len(lines)
            for line_num, line in enumerate(lines):
                self.add_line(line)
                percent_done = line_num/(num_lines - 1)
                update_progress(percent_done)
    # let's change this to get it working...
    def addLine(self, line):
        for element in line.split(' '):
            word, tags_str = element.split('_')
            # gets rid of extra information elements after the | character
            tags_str = re.sub('\|.*', '', tags_str)
            tags = tags_str.split('+')
            if word in self.word_tag_dict:
                self.word_tag_dict[word].update(set(tags))
            else:
                self.word_tag_dict[word] = set(tags)
    def getAllTags(self):
        all_tags = set()
        for word in self.word_tag_dict:
            all_tags.update(self.word_tag_dict[word])
        return all_tags
    def getNum_tags(self):
        return len(self.get_all_tags())
    def updateInitialProbabilities(self, line):
        pass
    def updateTransitionProbabilities(self, line):
        pass
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)
    def getData(self):
        # Word -> (wordarr, tagclass)
        raw_data = [word.get_raw_data() for word in self]
        word_arrays, tag_classes = zip(*raw_data)
        return np.array(word_arrays), np.array(tag_classes)
    def getClasses_frequencies(self):
        tagsets = [word.get_tagset() for word in self]
        c = Counter(tagsets)
        total = sum(c.values())
        freqs = {str(key) : value/total for key, value in c.items()}
        return freqs
    def __getitem__(self, index):
        return self.words_list[index]
    def initializeTrans_prob_dict():
        all_tags = constant.TARGET_TAGS
        transitions = itertools.product(all_tags, all_tags)
        return {trans : 0 for trans in transitions}
    def initializeInit_prob_dict():
        all_tags = constant.TARGET_TAGS
        return {tag : 0 for tag in all_tags}

class Word:
    def __init__(self, word_plus_tags):
        self.word, self.tagset = self.word_tag_separate(word_plus_tags)
        if self.tagset:
            self.word_array = WordArray(self.word)
        else:
            self.word_array = None
    def wordTag_separate(self, word_plus_tags):
        word, *tags_str = word_plus_tags.split('_')
        if tags_str:
            tags = TagSet(tags_str[0])
        else:
            tags = TagSet()
        separated_word = separate_word_from_radical(word)
        return separated_word, tags
    def getRaw_data(self):
        return self.get_array(), self.get_tag_class()
    def getTag_class(self):
        return self.tagset.tag_class
    def getArray(self):
        return self.word_array.array
    def getTagset(self):
        return self.tagset
    def __eq__(self, other):
        if isinstance(other, Word):
            return (self.word == other.word)
        else:
            return (self.word == other)
    def __bool__(self):
        return bool(self.tagset)

class WordArray:
    def __init__(self, separated_word):
        self.letter_dic = self.string_to_dic(separated_word)
        self.array = self.dic_to_array(self.letter_dic)
    def stringTo_dic(self, separated_word):
        SEPARATOR = constants.SEPARATOR
        binary_mask = [0 for each in range(len(separated_word))]
        max_binary = list_to_float([1 for each in binary_mask])
        letter_dic = OrderedDict()
        # ascii_lowercase = 'abcdef...xyz'
        for letter in string.ascii_lowercase + u'àáãéíóôú':
            letter_dic[letter] = list(binary_mask)
        letter_dic[SEPARATOR] = list(binary_mask)
        # populates letter_dic
        for index, letter in enumerate(separated_word):
            if letter in letter_dic:
                letter_dic[letter][index] = 1
        # converts letter_dic to numeric form, also normalizing it
        letter_dic = {letter : list_to_float(vec)/max_binary for letter, vec in letter_dic.items()}
        return letter_dic
    def dicTo_array(self, dic):
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
        self.tag_class = self.get_tag_class()
    def strTo_tags_list(self, tags_string):
        str_list = tags_string.split('+')
        tags_list = []
        for t in str_list:
            new_tag = Tag(t)
            if new_tag:
                tags_list.append(new_tag)
        return tags_list
    def getTags(self):
        return self.tags_list
    def getTag_class(self):
        if self:
            return sum(tag.tag_class for tag in self.tags_list)
        return np.zeros(4)
    def __bool__(self):
        return bool(self.tags_list)
    def __getitem__(self, index):
        if index > len(self.tags_list):
            raise IndexError('{} index too big'.format(index))
        return self.tags_list[index]
    def __eq__(self, other):
        return (set(self.tags_list) == set(other.tags_list))
    def __hash__(self):
        return list_to_float(self.tag_class)
    def __str__(self):
        return ",".join(str(tag) for tag in self)

class Tag:
    def __init__(self, tag_string):
        self.tag = self.tag_strip(tag_string)
        self.tag_class = self.tag_to_class(self.tag)
    def tagStrip(self, tag_string):
        target_tags = constants.TARGET_TAGS
        for tt in target_tags:
            # If the tag is composite, strip it
            # e.g. ADV-KS -> ADV
            if tt == tag_string.split('-')[0]:
                return tt
        return None
    def tagToClass(self, tag):
        tags_classes = constants.TAGS_CLASSES
        if tag in tags_classes:
            return tags_classes[tag]
        return None
    def __bool__(self):
        return bool(self.tag)
    def __eq__(self, other):
        if self.tag == other.tag:
            return True
        return False
    def __hash__(self):
        return hash(self.tag)
    def __str__(self):
        return self.tag
