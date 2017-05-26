import string
from nltk import stem
import numpy as np

SEPARATOR = '$'

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

def word_to_dict(separated_word):
    binary_mask = [0 for each in range(len(separated_word))]
    max_binary = list_to_float([1 for each in binary_mask])
    word_vec = {letter : list(binary_mask) for letter in string.ascii_lowercase}
    word_vec[SEPARATOR] = list(binary_mask)
    # populates word_vec
    for index, letter in enumerate(separated_word):
        word_vec[letter][index] = 1 
    # converts word_vec to numeric form, also normalizing it
    word_vec = {letter : list_to_float(vec)/max_binary for letter, vec in word_vec.items()}
    return word_vec

def list_to_float(lst):
    # big endian std
    num = 0
    for index, coeff in enumerate(reversed(lst)):
        num += coeff * (2**index)
    return num

def parse_word(word):
    sep_word = separate_word(word)
    letter_dic = word_to_dic(sep_word)
    array = np.array(list(letter_dic.values()), dtype='float64')
    return array
