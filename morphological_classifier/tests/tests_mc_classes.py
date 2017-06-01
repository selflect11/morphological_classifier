from .. import data_formatter
from .. import word_parser
from .. import constants
from .. import word
import numpy as np
import unittest
import tempfile
import os

class TestTagSet(unittest.TestCase):
    global TagSet
    TagSet = word.TagSet
    def test_tag_set(self):
        #self.assertCountEqual(TagSet('ADJ+N').tag_set, ['ADJ', 'N'])
        pass
    def test_tag_separator(self):
        self.assertCountEqual(TagSet().tag_separator('ADJ+N'), ['ADJ', 'N'])
        self.assertCountEqual(TagSet().tag_separator('ADJ'), ['ADJ',])
        self.assertCountEqual(TagSet().tag_separator('T+ADJ+N'), ['T', 'ADJ', 'N'])
    def test_str_to_tags(self):
        self.assertCountEqual(TagSet().str_to_tags(['ADJ','ADV']), [word.Tag('ADJ'), word.Tag('ADV')])

class TestTagClass(unittest.TestCase):
    global Tag 
    Tag = word.Tag
    def test_tag_extract(self):
        self.assertEqual(Tag('V').tag, 'V')
        self.assertEqual(Tag('ADV').tag, 'ADV')
        self.assertEqual(Tag('N').tag, 'N')
        self.assertEqual(Tag('ADJ').tag, 'ADJ')
        self.assertEqual(Tag('ADJ-KS').tag, 'ADJ')
        self.assertIsNone(Tag('Test').tag)
    def test_class_extract(self):
        # not working
        pass
    def test_bool_tag(self):
        self.assertTrue(bool(Tag('V')))
        self.assertTrue(bool(Tag('ADJ')))
        self.assertTrue(bool(Tag('ADV')))
        self.assertTrue(bool(Tag('N')))
        self.assertTrue(bool(Tag('N-TST')))
        self.assertFalse(bool(Tag('Test')))
    def test_equality(self):
        self.assertEqual(Tag('V'), Tag('V'))
        self.assertEqual(Tag('ADJ'), Tag('ADJ'))
        self.assertNotEqual(Tag('ADV'), Tag('V'))

if __name__ == '__main__':
    unittest.main()
