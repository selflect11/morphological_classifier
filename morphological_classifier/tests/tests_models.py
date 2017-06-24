# -*- coding: iso-8859-1 -*-
from .. import constants, tools
from ..data_structures import Text, Word, WordArray, TagSet, Tag
from ..models import get_closest_class
import numpy as np
import unittest

class TestTools(unittest.TestCase):
    def test_closest_class(self):
        in_out = [('ADJ', [0.99, 0,0.1,-0.1]),
                ('ADV', [0.03, 0.989, -0.04, 0.0022]),
                ('V', [0.01,-0.0049, 0.875, 0.001]),
                ('N', [0,0,0,1])
                ]
        for answer, arr in in_out:
            arr = np.array(arr)
            tryout = get_closest_class(arr)
            with self.subTest(tryout = tryout):
                self.assertEqual(answer, tryout)
    def test_update_progress(self):
        up = tools.update_progress
        paf = 'C:/Users/Volpi\\Google Drive\\TCC/morphological_classifier/data/macmorpho-mini.txt'
        txt = Text()
        with open(paf, 'r', encoding=constants.ENCODING) as f:
            lines = f.readlines()
            num_lines = len(lines)
        for line_num, line in enumerate(lines):
            percentage = line_num/(num_lines - 1)
            txt.add_line(line)
            up(percentage)
        
if __name__ == '__main__':
    unittest.main()
