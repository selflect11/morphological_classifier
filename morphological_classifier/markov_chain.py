# -*- coding: iso-8859-1 -*-
from . import constants
from .tools import pairwise
import itertools
import numpy as np

#   tp = TransitionProbabilities()
#   for each sequence in file:
#       tp.update_probs(sequence)
#   #for state, state+1 in pairwise(sequence):
#   tp.get_probability(transition)
#   >> 0.05
#
#
#
#
#
#

class TransitionProbabilities:
    def __init__(self):
        all_tags = constants.TAGET_TAGS
        self.probabilities_dict = {(i,j) : 0 for i,j in itertools.product(all_tags, all_tags)}
    def calculate_probabilities(self):
        count_dict = self.probabilities_dict
        prob_dict = {}
        for curr_state, next_state in count_dict:
            count = count_dict[curr_state, next_state]
            prob_dict[curr_state, next_state] = count/state_total_count(curr_state)
        self.probabilitites_dict = prob_dict
        return None
    def __getitem__(self, key):
        pass
    @staticmethod
    def state_total_count(state):
        return sum(value for (curr_state, next_state), value in self.probabilities_dict.items() if curr_state == state)

class InitialProbabilities:
    def __init__(self):
        pass
