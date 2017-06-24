# -*- coding: iso-8859-1 -*-
# TRAINING: Loads training.txt into a Text() data structure
#           Gets the list of WordArrays and list of Target classes
#           Feeds X,Y to fit
# TESTING:  Tries to predict the test.txt
import numpy as np
import pickle
from . import constants
from .data_structures import Text
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import os

def load_text(data_filepath):
    text_save_filepath = ''
    txt = Text()
    txt.read_file(data_filepath)
    with open(text_save_filepath, 'wb') as f:
        pickle.dump(txt.__dict__, f)
    return txt

# for model in models:
    # classifier = MorphologicalClassifier(model)
    # classifier.load_text(paf)
    # classifier.train()
    # classifier.test()

class MorphologicalClassifier:
    def __init__(self, Model, Text):
        self.model = Model
        self.text = Text
    def predict(self, word):
        new_word = Word(word)
        if new_word:
            # gets the array
            entry = new_word.get_array()
            return get_closest_class(self.model.predict(entry))
        return None
    def plot(self):
        pass
    def run_model(self, n_splits=5):
        scores = cross_val_score(self.model, self.data, self.target, cv=n_splits)
        return "Accuracy %0.2f (+/- %0.2f)".format(scores.mean(), scores.std()*2)
    def save(self, save_filepath):
        with open(save_filepath, 'wb') as f:
            pickle.dump(self.model.__dict__, f)
    def report(self, filepath):
        #with open(filepath, 'w') as f:
            # or something like it...
            # low priority
            #json.dump(self.TrainText.get_classes_frequencies(), f)
        pass

# Helper functions
def get_closest_class(array):
    # returns 'ADV', 'V', 'N' or 'ADJ'
    tags = constants.TARGET_CLASSES
    # {tag1 : class1, ...}
    tags_dict = constants.TAGS_CLASSES
    closest = min(tags, key = lambda cls: np.linalg.norm(cls - array))
    for tag, arr in tags_dict.items():
        if np.array_equal(arr, closest):
            return tag

if __name__ == '__main__':
    root = 'C:/Users/Volpi\\Google Drive\\TCC/morphological_classifier/data/'
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
