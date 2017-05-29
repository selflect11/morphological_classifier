from .. import data_formatter
from .. import word_parser
import unittest
import tempfile
import os

# target tags = [ADV, ADJ, V, N]

class TestDataFormatter(unittest.TestCase):
    def test_tag_translator(self):
        tt = data_formatter.tag_translator
        self.assertEqual(tt('ADJ'), 'ADJ')
        self.assertEqual(tt('V'), 'V')
        self.assertEqual(tt('V-KS'), 'V')
        self.assertIsNone(tt('TEST'))
        self.assertIsNone(tt('NUM'))
        self.assertIsNone(tt('ART'))
        self.assertIsNone(tt('NPROP'))
    def test_word_tag_separate(self):
        wts = data_formatter.word_tag_separate 
        self.assertEqual(wts('Teste_ADJ'), ('Teste', ['ADJ']))
        self.assertEqual(wts('Teste_ADJ+N'), ('Teste', ['ADJ', 'N']))
        self.assertEqual(wts('Teste_ADV+V-crap'), ('Teste', ['ADV', 'V']))
        self.assertEqual(wts('Teste_N-crap+ADV'), ('Teste', ['N', 'ADV']))
        self.assertEqual(wts('Teste_FAKETAG1+FAKETAG2'), ('Teste', []))
    def test_text_to_dict(self):
        ttd = data_formatter.text_to_dict
        self.assertEqual(ttd('Pequeno_ADJ teste_N'),
            {'Pequeno' : ['ADJ',], 'teste' : ['N',]})
        self.assertEqual(ttd('Um_N teste_N foi_V bem_ADJ sucedido_V'), 
            {'Um' : ['N',], 'teste' : ['N',], 'foi' : ['V',], 'bem' : ['ADJ',], 'sucedido' : ['V',]})
    def test_tags_to_class(self):
        ttc = data_formatter.tags_to_class
        self.addTypeEqualityFunc(np.array, np.testing.assert_array_almost_equal)
        self.assertEqual(ttc(['ADJ']), [1, 0, 0, 0])
    def test_parse_word_dict(self):
        pwd = data_formatter.parse_word_dict
        # ...
        pass
    def test_dict_to_file(self):
        dict_to_file = data_formatter.dict_to_file
        def string_extract(IN_STR):
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as src_file, tempfile.NamedTemporaryFile(mode='w', delete=False) as target_file: 
                src_file.write(IN_STR)
            # write to file
            dict_to_file(src_file.name, target_file.name)
            # read it's contents
            with open(target_file.name, 'r') as tf:
                written_string = tf.read()
            os.remove(src_file.name)
            os.remove(target_file.name)
            return written_string
        self.assertEqual(string_extract('teste_N'), 'teste:N\n')
        self.assertEqual(string_extract('teste_N+ADJ'), 'teste:N,ADJ\n')
        # big tests
        big_string = 'teste_N+ADJ maior_ADJ'
        string_for_training = string_extract(big_string)
        for each in 'teste:N,ADJ\n', 'maior:ADJ\n':
            with self.subTest(each = each):
                self.assertIn(each, string_for_training)
        # 2
        big_string2 = 'Um_NUM teste_N bem_ADV maior_ADJ'
        string_for_training2 = string_extract(big_string2)
        for each in 'teste:N', 'bem:ADV', 'maior:ADJ':
            with self.subTest(each = each):
                self.assertIn(each, string_for_training2)
        self.assertNotIn("Um:NUM", string_for_training2)

class TestWordParser(unittest.TestCase):
    def test_separate_word(self):
        sw = word_parser.separate_word
        self.assertEqual(sw('terra'), 'terr$a')
        self.assertEqual(sw('brilhante'), 'brilh$ante')
        self.assertEqual(sw('palavra'), 'palavr$a')
        self.assertEqual(sw('malandramente'), 'malandr$amente')
    def test_list_to_float(self):
        ltf = word_parser.list_to_float
        self.assertEqual(ltf([1]), 1)
        self.assertEqual(ltf([1,1]), 3)
        self.assertEqual(ltf([1,0,1]), 5)
        self.assertEqual(ltf([1,1,0,1]), 13)
        self.assertEqual(ltf([1,1,0,0,1]), 25)
        self.assertEqual(ltf([1,0,1,1,1,1,0,1,0,1,1,1]), 3031)
    def test_word_to_array(self):
        wta = word_parser.word_to_array
        separated_word = 'terr$a'
        result_array = wta(separated_word)
        # after lengthy calculations, this is what we got...
        answer_array = np.array([
            1.59E-2,    #a
            0.0,        #b
            0.0,        #c
            0.0,        #d
            2.54E-1,    #e
            0.0,		#f
            0.0,		#g
            0.0,		#h
            0.0,		#i
            0.0,		#j
            0.0,		#k
            0.0,		#l
            0.0,		#m
            0.0,		#n
            0.0,		#o
            0.0,		#p
            0.0,		#q
            1.90E-1,	#r
            0.0,		#s
            5.08E-1,	#t
            0.0,		#u
            0.0,		#v
            0.0,		#w
            0.0,		#x
            0.0,		#y
            0.0,		#z
            3.17E-2,	#$
        ], dtype = constants.D_TYPE)
        for index, item in enumerate(result_array):
            with self.subTest(item = item):
                # doesn't work for more than 3 decimal places
                self.assertAlmostEqual(item, answer_array[index], places=3)
            
if __name__ == '__main__':
    unittest.main()
