# -*- coding: iso-8859-1 -*-
import unittest
import numpy.testing as npt
from ..markov_chain import TransitionProbabilities as TransProb, InitialProbabilities as InitProb
from .. import constants, tools


class TestTransitionProbabilities(unittest.TestCase):
    def setUp(self):
        self.tp = TransProb()
        self.tp.probabilities_dict = {
            (0, 0): 0, (0, 1): 0, (0, 2): 0,
            (1, 0): 0, (1, 1): 0, (1, 2): 0,
            (2, 0): 0, (2, 1): 0, (2, 2): 0,
            }

    def tearDown(self):
        self.tp = TransProb()

    def test_total_num_transitions(self):
        self.tp.probabilities_dict = {
            (0, 0): 1,
            (0, 1): 2,
            (1, 0): 4,
            (1, 1): 8,
            }

        self.assertEqual(self.tp.total_num_transitions(0), 3)
        self.assertEqual(self.tp.total_num_transitions(1), 12)
        self.assertEqual(self.tp.total_num_transitions(2), 0)

    def test_update_count(self):
        self.tp.update_count((0, 1))
        self.tp.update_count((1, 0))
        self.tp.update_count((1, 0))

        self.assertEqual(self.tp[0, 0], 0)
        self.assertEqual(self.tp[0, 1], 1)
        self.assertEqual(self.tp[1, 0], 2)
        self.assertEqual(self.tp[1, 1], 0)
        self.assertEqual(self.tp[2, 0], 0)
        self.assertEqual(self.tp[2, 1], 0)
        self.assertEqual(self.tp[2, 2], 0)
        with self.assertRaises(KeyError):
            self.tp.update_count((3, 0))

    def test_add_transitions(self):
        seq = [1, 0, 1, 1, 0]
        self.tp.add_transitions(seq)

        self.assertEqual(self.tp[1, 0], 2)
        self.assertEqual(self.tp[0, 1], 1)
        self.assertEqual(self.tp[1, 1], 1)

    def test_get_probability(self):
        seq = [0, 0, 1, 0, 1, 1, 0, 1, 0, 1]
        self.tp.add_transitions(seq)
        # does't return a probability right away
        # 'calculateProbabilities()' must be called
        self.assertEqual(self.tp[(0, 0)], 1)
        self.assertEqual(self.tp[(0, 1)], 4)
        self.assertEqual(self.tp[(1, 0)], 3)
        self.assertEqual(self.tp[(1, 1)], 1)

    def test_calculate_probabilities(self):
        seq = [0, 0, 1, 0, 1, 0, 2, 0, 2, 2, 0]
        self.tp.add_transitions(seq)
        self.tp.calculate_probabilities()

        npt.assert_approx_equal(self.tp[(0, 0)], 1/5)
        npt.assert_approx_equal(self.tp[(0, 1)], 2/5)
        npt.assert_approx_equal(self.tp[(0, 2)], 2/5)
        npt.assert_approx_equal(self.tp[(1, 0)], 1)
        npt.assert_approx_equal(self.tp[(1, 1)], 0)
        npt.assert_approx_equal(self.tp[(1, 2)], 0)
        npt.assert_approx_equal(self.tp[(2, 0)], 2/3)
        npt.assert_approx_equal(self.tp[(2, 1)], 0)
        npt.assert_approx_equal(self.tp[(2, 2)], 1/3)


class TestInitialProbabilities(unittest.TestCase):
    def setUp(self):
        self.ip = InitProb()
        self.ip.probabilities_dict = {
            0: 0,
            1: 0,
            2: 0,
            }

    def tearDown(self):
        self.ip = InitProb()

    def test_update_count(self):
        self.ip.update_count(0)
        self.ip.update_count(1)
        self.ip.update_count(1)

        self.assertEqual(self.ip[0], 1)
        self.assertEqual(self.ip[1], 2)
        self.assertEqual(self.ip[2], 0)

    def test_add_tag(self):
        self.ip.add_tag(0)
        self.ip.add_tag(1)
        self.ip.add_tag(1)
        self.ip.add_tag(2)
        self.ip.add_tag(2)
        self.ip.add_tag(2)
        self.ip.add_tag(2)

        self.assertEqual(self.ip[0], 1)
        self.assertEqual(self.ip[1], 2)
        self.assertEqual(self.ip[2], 4)

    def test_get_probability(self):
        self.ip.add_tag(0)
        self.ip.add_tag(1)
        self.ip.add_tag(1)
        self.ip.add_tag(2)
        self.ip.add_tag(2)
        self.ip.add_tag(2)
        self.ip.add_tag(2)

        self.assertEqual(self.ip[0], 1)
        self.assertEqual(self.ip[1], 2)
        self.assertEqual(self.ip[2], 4)

    def test_calculate_probabilities(self):
        tags = [0, 1, 2, 0, 1, 1, 0, 0, 0, 2]
        for tag in tags:
            self.ip.add_tag(tag)
        self.ip.calculate_probabilities()

        npt.assert_approx_equal(self.ip[0], 0.5)
        npt.assert_approx_equal(self.ip[1], 0.3)
        npt.assert_approx_equal(self.ip[2], 0.2)


if __name__ == '__main__':
    unittest.main()
