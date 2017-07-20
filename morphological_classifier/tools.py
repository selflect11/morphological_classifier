from . import constants
from nltk import stem
from itertools import tee
import sys

def list_to_float(lst):
    # big endian std
    num = 0
    for index, coeff in enumerate(lst):
        num += coeff * (2**index)
    return int(num)

def separate_word_from_radical(word_str):
        # RSLP algorithm
        separator = constants.SEPARATOR
        word_str = word_str.lower()
        stemmer = stem.rslp.RSLPStemmer()
        radical = stemmer.stem(word_str)
        rest = get_desinence(word_str, radical)
        if not rest:
            #print('Couldnt separate word {}'.format(word_str))
            return word_str
        return radical + separator + rest

def get_desinence(word, radical):
    if not radical:
        return word
    start, mid, end = word.partition(radical)
    if end:
        return end
    return get_desinence(word, radical[:-1])

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def update_progress(progress):
    barLength = 30 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
