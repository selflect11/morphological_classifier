# -*- coding: iso-8859-1 -*-
from .. import constants, tools
from ..data_structures import Text, Word, WordArray, TagSet, Tag
import numpy as np
import unittest
import tempfile
import string
import os

class TestText(unittest.TestCase):
    def test_add_line(self):
        txt = Text()
        txt.add_line('Teste_N terra_N nova_ADJ ._PF')
        answer = [Word('Teste_N'), Word('terra_N'), Word('nova_ADJ')]
        for index, word in enumerate(txt):
            with self.subTest(word = word):
                self.assertEqual(word, answer[index])
    def test_read_file(self):
        # ver questão dos acentos...
        in_str = 'Teste_N maior_ADJ ,_P agora_N é_V pra_PREP valer_V'
        with tempfile.NamedTemporaryFile(mode='w', encoding=constants.ENCODING, delete=False) as f:
            f.write(in_str)
        txt = Text()
        txt.read_file(f.name)
        answer = [Word('teste_N'), Word('maior_ADJ'), Word('agora_N'), Word('é_V'), Word('valer_V')]
        for index, word in enumerate(txt):
            with self.subTest(word = word):
                self.assertEqual(word, answer[index])
    def test_get_data(self):
        txt = Text()
        line = 'Teste_N novo_N'
        txt.add_line(line)
        word_arrays, tag_classes = txt.get_data()
        # setup answer
        word_strs = line.split(' ')
        words = [Word(ws) for ws in word_strs]
        answer_word_arrays = [w.get_array() for w in words]
        answer_tag_classes = [w.get_tag_class() for w in words]
        # tests
        for index_i, entry in enumerate(word_arrays):
            for index_j, coeff in enumerate(entry):
                with self.subTest(coeff = coeff):
                    self.assertAlmostEqual(coeff, answer_word_arrays[index_i][index_j], places=3)
        for index_i, entry in enumerate(tag_classes):
            for index_j, coeff in enumerate(entry):
                with self.subTest(coeff = coeff):
                    self.assertAlmostEqual(coeff, answer_tag_classes[index_i][index_j], places=3)

class TestWord(unittest.TestCase):
    def test_get_tags(self):
        self.assertEqual(Word('Teste_N+V').tag_set, TagSet('N+V'))
        self.assertEqual(Word('Teste_ADJ+N+V').tag_set, TagSet('N+V+ADJ'))
        self.assertEqual(Word('Teste_ADJ+N-KS+V').tag_set, TagSet('N+V+ADJ'))
        self.assertEqual(Word('Teste_ADJ+N-KS+V-DES').tag_set, TagSet('N+V+ADJ'))
        self.assertEqual(Word('Teste_N+V+TestTag').tag_set, TagSet('N+V'))
    def test_bool(self):
        self.assertTrue(Word('Teste_N'))
        self.assertTrue(Word('Teste_N+V'))
        self.assertTrue(Word('Teste_ADJ'))
        self.assertTrue(Word('Teste_ADV'))
        self.assertTrue(Word('Teste_ADJ-APP'))
        self.assertFalse(Word('Teste_T'))
        self.assertFalse(Word('Teste_NUM'))
    def test_separate_word_from_radical(self):
        self.assertEqual(Word('Teste_N'), 'test$e')
        self.assertEqual(Word('Terra_N'), 'terr$a')
        self.assertEqual(Word('Bonita_ADJ'), 'bonit$a')
        self.assertEqual(Word('Furiosamente_ADV'), 'furios$amente')
        self.assertEqual(Word('Amei_V'), 'ame$i')
        self.assertEqual(Word('Amar_V'), 'am$ar')

class TestWordArray(unittest.TestCase):
    def test_string_to_dict(self):
        sep_word = 'terr$a'
        war = WordArray(sep_word)
        # after some lengthy calculations...
        answer = {
            'a' : 1.59e-2,
            'e' : 2.54e-1,
            'r' : 1.9e-1,
            't' : 5.08e-1,
            '$' : 3.17e-2,
        }
        for letter in string.ascii_lowercase:
            with self.subTest(letter = letter):
                if letter in sep_word:
                    # doesn't work for more than 3 decimal places
                    self.assertAlmostEqual(war[letter], answer[letter], places=3)
                else:
                    self.assertAlmostEqual(war[letter], 0)

class TestTagSet(unittest.TestCase):
    def test_str_to_tags(self):
        self.assertCountEqual(TagSet('ADJ+ADV').tags_list, [Tag('ADJ'), Tag('ADV')])
        self.assertCountEqual(TagSet('Test+N+V-KS'), [Tag('V'), Tag('N')])
    def test_get_hybrid_class(self):
        verb_array = constants.TAGS_CLASSES['V']
        adj_verb_array = constants.TAGS_CLASSES['ADJ'] + constants.TAGS_CLASSES['V']
        for index, coeff in enumerate(TagSet('V').tag_class):
            with self.subTest(coeff = coeff):
                self.assertAlmostEqual(verb_array[index], coeff)
        for index, coeff in enumerate(TagSet('V+ADJ-KW').tag_class):
            with self.subTest(coeff = coeff):
                self.assertAlmostEqual(adj_verb_array[index], coeff)
    def test_bool(self):
        self.assertTrue(TagSet('N'))
        self.assertTrue(TagSet('N+V'))
        self.assertTrue(TagSet('N-KS+V'))
        self.assertFalse(TagSet(''))
        self.assertFalse(TagSet('Testtag'))

class TestTag(unittest.TestCase):
    def test_tag_extract(self):
        self.assertEqual(Tag('V').tag, 'V')
        self.assertEqual(Tag('ADV').tag, 'ADV')
        self.assertEqual(Tag('N').tag, 'N')
        self.assertEqual(Tag('ADJ').tag, 'ADJ')
        self.assertEqual(Tag('ADJ-KS').tag, 'ADJ')
        self.assertEqual(Tag('N-WE').tag, 'N')
        self.assertNotEqual(Tag('N').tag, 'ADJ')
        self.assertIsNone(Tag('Test').tag)
        self.assertIsNone(Tag('NUM').tag)
        self.assertIsNone(Tag('ART').tag)
        self.assertIsNone(Tag('NPROP').tag)
    def test_bool(self):
        self.assertTrue(Tag('V'))
        self.assertTrue(Tag('ADJ'))
        self.assertTrue(Tag('ADV'))
        self.assertTrue(Tag('N'))
        self.assertTrue(Tag('N-TST'))
        self.assertFalse(Tag('Test'))
        self.assertFalse(Tag('NUM'))
        self.assertFalse(Tag('ART'))
    def test_equality(self):
        self.assertEqual(Tag('V'), Tag('V'))
        self.assertEqual(Tag('ADJ'), Tag('ADJ'))
        self.assertNotEqual(Tag('ADV'), Tag('V'))

class TestTools(unittest.TestCase):
    def test_list_to_float(self):
        ltf = tools.list_to_float
        self.assertEqual(ltf([1]), 1)
        self.assertEqual(ltf([1,1]), 3)
        self.assertEqual(ltf([1,0,1]), 5)
        self.assertEqual(ltf([1,1,0,1]), 13)
        self.assertEqual(ltf([1,1,0,0,1]), 25)
        self.assertEqual(ltf([1,0,1,1,1,1,0,1,0,1,1,1]), 3031)

if __name__ == '__main__':
    unittest.main()
