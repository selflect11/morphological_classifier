from .. import constants
from .. import word
import numpy as np
import unittest
import tempfile
import os

class TestWord(unittest.TestCase):
    global Word, TagSet
    Word = word.Word
    TagSet = word.TagSet
    def test_get_word(self):
        self.assertEqual(Word('Teste_Testtag').word, 'test$e')
        self.assertEqual(Word('Teste_Testtag+Testtag2').word, 'test$e')
    def test_get_tags(self):
        self.assertEqual(Word('Test_N+V').tags, TagSet('N+V'))

class TestTagSet(unittest.TestCase):
    global TagSet
    TagSet = word.TagSet
    def test_tag_separator(self):
        self.assertEqual(TagSet('ADJ-KS+V+Test'), TagSet('ADJ+V'))
        self.assertCountEqual(TagSet().tag_separator('ADJ+N'), ['ADJ', 'N'])
        self.assertCountEqual(TagSet().tag_separator('ADJ'), ['ADJ',])
        self.assertCountEqual(TagSet().tag_separator('T+ADJ+N'), ['T', 'ADJ', 'N'])
    def test_str_to_tags(self):
        self.assertCountEqual(TagSet().str_to_tags(['ADJ','ADV']), [word.Tag('ADJ'), word.Tag('ADV')])
        self.assertCountEqual(TagSet().str_to_tags(['Test', 'N', 'V-KS']), [word.Tag('V'), word.Tag('N')])
    def test_get_hybrid_class(self):
        verb_array = np.array([0,0,1,0])
        adj_verb_array = np.array([1,0,1,0])
        for index, coeff in enumerate(TagSet('V').tag_class):
            with self.subTest(coeff = coeff):
                self.assertAlmostEqual(verb_array[index], coeff)
        for index, coeff in enumerate(TagSet('V+ADJ-KW').tag_class):
            with self.subTest(coeff = coeff):
                self.assertAlmostEqual(verb_array[index], coeff)
    def test_bool(self):
        self.assertTrue(TagSet('N'))
        self.assertTrue(TagSet('N+V'))
        self.assertTrue(TagSet('N-KS+V'))
        self.assertFalse(TagSet(''))
        self.assertFalse(TagSet('Testtag'))

class TestTag(unittest.TestCase):
    global Tag 
    Tag = word.Tag
    def test_tag_extract(self):
        self.assertEqual(Tag('V').tag, 'V')
        self.assertEqual(Tag('ADV').tag, 'ADV')
        self.assertEqual(Tag('N').tag, 'N')
        self.assertEqual(Tag('ADJ').tag, 'ADJ')
        self.assertEqual(Tag('ADJ-KS').tag, 'ADJ')
        self.assertIsNone(Tag('Test').tag)
    def test_bool(self):
        self.assertTrue(Tag('V'))
        self.assertTrue(Tag('ADJ'))
        self.assertTrue(Tag('ADV'))
        self.assertTrue(Tag('N'))
        self.assertTrue(Tag('N-TST'))
        self.assertFalse(Tag('Test'))
    def test_equality(self):
        self.assertEqual(Tag('V'), Tag('V'))
        self.assertEqual(Tag('ADJ'), Tag('ADJ'))
        self.assertNotEqual(Tag('ADV'), Tag('V'))

if __name__ == '__main__':
    unittest.main()
