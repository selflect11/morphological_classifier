import constants
import numpy as np

def tag_translator(tag):
    target_tags = constants.TARGET_TAGS 
    for tt in target_tags:
        # if the tag is composite, strip it
        # e.g. ADV-KS -> ADV
        if tt == tag.split('-')[0]:
            return tt
    # if not a target tag, throw it out
    return ''

def word_tag_separate(word_plus_tag):
    word, tags_str = word_plus_tag.split('_')
    tags = tags_str.split('+')
    tags = map(tag_translator, tags)
    return word, tags

def format_data(text):
    # return dict
    # {word1 : (tag11, tag12,...), word2 : (tag21, tag22, ...) ...}
    words_plus_tags = text.split(' ')
    target_tags = constants.TARGET_TAGS
    word_tags_dic = {word : tags for word, tags in map(word_tag_separate, words_plus_tags)}
    return word_tags_dic

def convert_data_for_training(src_filepath, target_filepath):
    with open(src_filepath, 'r') as fr:
        data = fr.read()
    formatted_data = format_data(data)
    with open(target_filepath, 'w') as fw:
        for word, tags in formatted_data.items():
            fw.write(word + ":" + ",".join(tags))
