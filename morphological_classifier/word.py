from . import constants
from . import word_parser
from nltk import stem

class Word():
    def __init__(self, word_plus_tags):
        pass
    def word_tag_separate(self, word_plus_tag):
        word, tags_str = word_plus_tags.split('_')
        tags = 
        
    def separate_word(self, word_str):
        # Runs RSLP algorithm
        separator = constants.SEPARATOR
        word_str = word_str.lower()
        stemmer = stem.rslp.RSLPStemmer()
        radical = stemmer.stem(word_str)
        rest = word.split(radical, 1)[1]
        if not rest:
            print('Couldnt separate word {}'.format(word_str))
            return word
        return radical + separator + rest
    def string_to_array(separated_word):
        binary_mask = [0 for each in range(len(separated_word))]
        max_binary = word_parser.list_to_float([1 for each in binary_maks])
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

class TagSet():
    def __init__(self, tags_string):
        self.tag_set = self.str_to_tags(tags_string)
    def tag_separator(self, tags_string):
        tags_list = tags_string.split('+')
        return tags_list
    def str_to_tags(self, tags_list):
        tag_set = []
        for tag in tags_list:
            new_tag = Tag(tag)
            if new_tag:
                tag_set.append(new_tag)
        return tag_set
    def get_hybrid_class(self):
        return sum(tag.tag_class for tag in self.tag_set)

class Tag():
    target_tags = constants.TARGET_TAGS
    def __init__(self, tag_string):
        self.tag = self.tag_strip(tag_string)
        self.tag_class = self.tag_to_class(self.tag)
    def tag_strip(self, tag_string):
        for tt in target_tags:
            # If the tag is composite, strip it
            # e.g. ADV-KS -> ADV
            if tt == tag_string.split('-')[0]:
                return tt
        return None
    def tag_to_class(tag):
        tags_classes = constants.TAGS_CLASSES
        return tags_classes[tag]
    def __bool__(self):
        if self.tag:
            return True
        else:
            return False
