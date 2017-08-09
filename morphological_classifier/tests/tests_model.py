# -*- coding: iso-8859-1 -*-
import unittest
from ..model import MorphologicalClassifier
from .. import constants, tools
import tempfile


class TestModel(unittest.TestCase):
    def setUp(self):
        lines = [
            'A_N b_V c_V d_N',
            'A_N b_V c_N d_N',
            'A_N b_V c_N d_N',
            'A_N b_V c_V d_ADV']
        with tempfile.NamedTemporaryFile(mode='w', encoding=constants.ENCODING, delete=False) as temp:
            temp.write('\n'.join(lines))
        self.mc = MorphologicalClassifier(temp.name)
        self.test_phrase = 'A b c d'

    def test_get_most_likely_tags(self):
        self.assertEqual(self.mc.get_most_likely_tags(self.test_phrase),
                        ('N', 'V', 'V', 'N'))

    def test_get_tags_phrase(self):
        self.assertEqual(self.mc.get_tags_phrase(self.test_phrase),
                        [{'N'}, {'V'}, {'V', 'N'}, {'N', 'ADV'}])


if __name__ == '__main__':
    unittest.main()
