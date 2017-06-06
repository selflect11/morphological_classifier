from . import constants
import string
from nltk import stem
import numpy as np
from collections import OrderedDict

SEPARATOR = constants.SEPARATOR

def separate_word(word):
    # run RSLP algorithm on word
    word = word.lower()
    stemmer = stem.rslp.RSLPStemmer()
    radical = stemmer.stem(word)
    rest = word.split(radical, 1)[1]
    if not rest:
        print('Could not separate word {}'.format(word))
        return word
    return radical + SEPARATOR + rest

def word_to_array(separated_word):
    binary_mask = [0 for each in range(len(separated_word))]
    max_binary = list_to_float([1 for each in binary_mask])
    letter_dict = OrderedDict()
    # ascii_lowercase = 'abcdef...xyz'
    for letter in string.ascii_lowercase:
        letter_dict[letter] = list(binary_mask)
    letter_dict[SEPARATOR] = list(binary_mask)
    # populates letter_dict
    for index, letter in enumerate(separated_word):
        letter_dict[letter][index] = 1 
    # converts letter_dict to numeric form, also normalizing it
    word_array = np.array([list_to_float(vec)/max_binary for vec in letter_dict.values()], dtype=constants.D_TYPE)
    return word_array

def parse_word(word):
   return word_to_array(separate_word(word))
