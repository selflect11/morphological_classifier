# -*- coding: iso-8859-1 -*-
# TRAINING: Loads training.txt into a Text() data structure
#           Gets the list of WordArrays and list of Target classes
#           Feeds X,Y to fit
# TESTING:  Tries to predict the test.txt
import numpy as np
import pickle
from . import constants
from .data_structures import Text, Word
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import os, sys

def load_text(data_filepath):
    text_save_filepath = ''
    txt = Text()
    txt.read_file(data_filepath)
    with open(text_save_filepath, 'wb') as f:
        pickle.dump(txt.__dict__, f)
    return txt

class MorphologicalClassifier:
    def __init__(self, Text, Model=None):
        if not Model:
            self.Model = MLPClassifier(solver = 'lbfgs', alpha=1e-5, hidden_layer_sizes=(34,2), random_state=1)
        else:
            self.Model = Model
        self.Text = Text
    def predict(self, word_str):
        new_word = Word(word_str + '_N')
        entry = new_word.get_array()
        predicted_array = self.Model.predict(entry.reshape(1,-1))
        return tags_from_array(predicted_array)
    def plot(self):
        pass
    def get_accuracy(self, n_splits=5):
        X, y = self.Text.get_data()
        scores = cross_val_score(self.Model, X, y, cv=n_splits, scoring='f1_weighted')
        return "Accuracy %0.2f (+/- %0.2f)".format(scores.mean(), scores.std()*2)
    def train_model(self):
        X, y = self.Text.get_data()
        self.Model.fit(X,y)
    def save(self, save_filepath):
        with open(save_filepath, 'wb') as f:
            pickle.dump(self.Model.__dict__, f)
    def load(self, load_filepath):
        with open(load_filepath, 'rb') as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
    def report(self, filepath):
        #with open(filepath, 'w') as f:
            # or something like it...
            # low priority
            #json.dump(self.TrainText.get_classes_frequencies(), f)
        pass

# Helper functions
def tags_from_array(array):
    array = array[0] if len(array)==1 else array
    array = np.array(array)
    tags_dict = constants.TAGS_CLASSES
    tags = constants.TARGET_TAGS
    classes = constants.TARGET_CLASSES
    tag_list = []
    for index, coeff in enumerate(array):
        if np.isclose(tags_dict[tags[index]][index], coeff, atol=0.1):
            tag_list.append(tags[index])
    if not tag_list:
        closest = min(classes, key=lambda v: np.linalg.norm(v - array))
        for tag, arr in tags_dict.items():
            if np.array_equal(arr, closest):
                return tag
    return '_'.join(tag_list)

def setup_paths():
    path = os.path.abspath('../')
    sys.path.append(path)

if __name__ == '__main__':
    setup_paths()
    root = '../data/'
    data_paf = os.path.join(root, 'macmorpho-total.txt')
    txt_paf = os.path.join(root, 'txt_obj.p')
    archs = [(35,1), (35,2), (35,3), (27,1), (27,2), (27,3)]
    txt = Text()
    try:
        txt.load(txt_paf)
    except Exception:
        txt.read_file(data_paf)
        txt.save(txt_paf)
    X, y = txt.get_data()
    for arch in archs:
        clf = MPLClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=arch, random_state=1)
        model = MorphologicalClassifier(clf, txt)
        model.run_model(10)
        model.save(os.path.join(root, 'model_{}.p'.format('_'.join(str(i) for i in arch))))
