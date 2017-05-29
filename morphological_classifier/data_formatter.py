from . import constants
from . import word_parser
import numpy as np

def tags_to_class(tags_list):
    # receives [tag1, tag3, ...]
    # returns [1, 0, 1, 0, ..., 0]
    tags_classes = constants.TAGS_CLASSES
    data_type = constants.D_TYPE
    num_classes = constants.NUM_CLASSES
    hybrid_class = np.zeros(num_classes, dtype=data_type)
    for tag in tags_list:
        hybrid_class += tags_classes[tag]
    return hybrid_class

def tag_translator(tag):
    target_tags = constants.TARGET_TAGS 
    for tt in target_tags:
        # if the tag is composite, strip it
        # e.g. ADV-KS -> ADV
        if tt == tag.split('-')[0]:
            return tt
    # if not a target tag, throw it out
    return None

def word_tag_separate(word_plus_tag):
    word, tags_str = word_plus_tag.split('_')
    tags = tags_str.split('+')
    tags = map(tag_translator, tags)
    # gets rid of empty tags
    tags = list(filter(lambda tag: tag is not None, tags))
    return word, tags

def text_to_dict(text):
    # returns {word1 : [tag11, tag12,...], word2 : [tag21, tag22, ...] ...}
    words_plus_tags = text.split(' ')
    target_tags = constants.TARGET_TAGS
    word_tags_dict = {word : tags for word, tags in map(word_tag_separate, words_plus_tags)}
    return word_tags_dict

def parse_word_dict(word_dict):
    # {..., wordn : [tagn1, tagn2, ...], ...} -> { ..., vecn : classn, ...}
    parsed_word_dict = {word_parser.parse_word(word) : tags_to_class(tags) for word, tags in word_dict.items()}
    return parsed_word_dict

# maybe not the best idea...
def dict_to_file(src_filepath, target_filepath):
    try:
        with open(src_filepath, 'r') as fr:
            data = fr.read()
        formatted_data = text_to_dict(data)
        with open(target_filepath, 'w') as fw:
            for word, tags in formatted_data.items():
                fw.write(word + ":" + ",".join(tags) + "\n")
    except IOError:
        print('Could not open file')
