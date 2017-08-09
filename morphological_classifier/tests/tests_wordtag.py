# -*- coding: iso-8859-1 -*-
from .. import constants, tools
from ..lexicon import parse_word_tag, WordTags
import unittest
import tempfile
import os
import shutil


class TestWordTag(unittest.TestCase):
    def setUp(self):
        self.line = 'Um_N singelo_ADJ teste_N do_ART+PREP balacobaco_N ,_PU brother_ADJ-KV'

    def test_add_line(self):
        fd, temp_path = tempfile.mkstemp()
        wt = WordTags(temp_path)
        wt.add_line(self.line)
        os.close(fd)
        os.remove(temp_path)

        self.assertEqual(wt['um'], {'N'})
        self.assertEqual(wt['singelo'], {'ADJ'})
        self.assertEqual(wt['teste'], {'N'})
        self.assertEqual(wt['do'], {'ART', 'PREP'})
        self.assertEqual(wt['balacobaco'], {'N'})
        self.assertEqual(wt['brother'], {'ADJ'})

    def test_load_word_tags_from_file(self):
        with tempfile.NamedTemporaryFile(mode='w', encoding=constants.ENCODING, delete=False) as temp:
            temp.write(self.line)
        wt = WordTags(temp.name)

        self.assertEqual(wt['um'], {'N'})
        self.assertEqual(wt['singelo'], {'ADJ'})
        self.assertEqual(wt['teste'], {'N'})
        self.assertEqual(wt['do'], {'ART', 'PREP'})
        self.assertEqual(wt['balacobaco'], {'N'})
        self.assertEqual(wt['brother'], {'ADJ'})


class TestParseWordTag(unittest.TestCase):
    # Check if this covers it all
    def test_parse_word_tag(self):
        self.assertEqual(parse_word_tag('Word_ADJ-KS'), ('word', ['ADJ']))
        self.assertEqual(parse_word_tag('Vord_ADJ+V-KP'), ('vord', ['ADJ', 'V']))
        self.assertEqual(parse_word_tag('Salto_N'), ('salto', ['N']))
        self.assertEqual(parse_word_tag('sete_ADJ'), ('sete', ['ADJ']))
        self.assertEqual(parse_word_tag('da_ART+PREP'), ('da', ['ART', 'PREP']))


if __name__ == '__main__':
    unittest.main()
