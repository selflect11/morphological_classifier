from context import morphological_classifier as mc
import unittest

class TestDataFormatter(unittest.TestCase):
    def test_tag_translator(self):
        f = mc.data_formatter.tag_translator
        self.assertEqual(f('V'), 'V')
        self.assertEqual(f('V-KS'), 'V')
    def test_word_tag_separate(self):
        f = mc.data_formatter.word_tag_separate 
        self.assertEqual(f('Teste_Tag1', ('Teste', ['Tag1'])))
        self.assertEqual(f('Teste_Tag1+Tag2', ('Teste', ['Tag1', 'Tag2'])))
        self.assertEqual(f('Teste_Tag1+Tag2-crap', ('Teste', ['Tag1', 'Tag2'])))
        self.assertEqual(f('Teste_Tag1-crap+Tag2', ('Teste', ['Tag1', 'Tag2'])))
    def test_format_data(self):
        raise NotImplementedError
    def test_convert_data_for_training(self):
        raise NotImplementedError

if __name__ == '__main__':
    unittest.main()
