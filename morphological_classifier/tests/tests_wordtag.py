# -*- coding: iso-8859-1 -*-
from .. import constants, tools
from ..lexicon import parse_word_tag, WordTags
import unittest
import tempfile
import os
import shutil


class TestWordTag(unittest.TestCase):
    def test_add_line(self):
        line = 'Um_N singelo_ADJ teste_N do_ART+PREP caralho_N ,_PU motherfucker_ADJ|EST'
        fd, temp_path = tempfile.mkstemp()
        wt = WordTags(temp_path)
        wt.add_line(line)
        os.close(fd)
        os.remove(temp_path)

        self.assertEqual(wt['um'], {'N'})
        self.assertEqual(wt['singelo'], {'ADJ'})
        self.assertEqual(wt['teste'], {'N'})
        self.assertEqual(wt['do'], {'ART', 'PREP'})
        self.assertEqual(wt['caralho'], {'N'})
        self.assertEqual(wt['motherfucker'], {'ADJ'})

    def test_load_word_tags_from_file(self):
        pass

    def test_setup(self):
        pass

class TestParseWordTag(unittest.TestCase):
    # Check if this covers it all
    def test_parse_word_tag(self):
        self.assertEqual(parse_word_tag('Word_ADJ|Extra'), ('word', ['ADJ']))
        self.assertEqual(parse_word_tag('Vord_ADJ+V|Extra'), ('vord', ['ADJ', 'V']))


if __name__ == '__main__':
    unittest.main()
