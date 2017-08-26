import unittest
import tempfile
from os import remove
from os.path import join
from morphological_classifier import classifier, perceptrons


class TestClassifier(unittest.TestCase):
    '''Lets test the loading of weights,
    because as of right now it has weird behaviour'''
    def setUp(self):
        self.pcp = perceptrons.PerceptronTagger()
        self.mc = classifier.MorphologicalClassifier(self.pcp)
        self.root = '/home/volpi/Documents/grive/TCC/morphological_classifier'

    def test_load(self):
        data_train_paf = join(self.root, 'data/macmorpho-mini.txt')
        save_file = tempfile.NamedTemporaryFile(delete=False)
        save_file.close()

        self.mc.train(data_train_paf)
        old_weights = self.mc.tagger.model.weights
        self.mc.save(save_file.name)

        new_mc = classifier.MorphologicalClassifier(self.pcp)
        new_mc.load(save_file.name)
        new_weights = new_mc.tagger.model.weights

        remove(save_file.name)

        self.assertDictEqual(old_weights, new_weights)


if __name__ == '__main__':
    unittest.main()
