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

class TransitionProbabilities:
    def __init__(self):
        ''' Reads a series of transitions once
            and then calculates the probabilities
            of each pair of transitions. '''
        all_tags = constants.TAGET_TAGS
        self.probabilities_dict = {(i, j) : 0 for i, j in itertools.product(all_tags, all_tags)}

    # supposed to be run only once
    # maybe I'll add a decorator
    def calculateProbabilities(self):
        count_dict = self.probabilities_dict
        prob_dict = {}
        for curr_state, next_state in count_dict:
            count = count_dict[curr_state, next_state]
            prob_dict[curr_state, next_state] = count/stateTransitionsCount(curr_state)
        self.probabilitites_dict = prob_dict

    def loadTags(self, tags_list):
        for t1, t2 in pairwise(tags_list):
            self.updateCount((t1, t2))

    def updateCount(self, transition_tuple):
        self.probabilities_dict[transition_tuple] += 1

    def getProbability(self, transition_tuple):
        return self.probabilities_dict[transition_tuple]

    @staticmethod
    def stateTransitionsCount(state):
        return sum(value for (curr_state, next_state), value in self.probabilities_dict.items() if curr_state == state)

class InitialProbabilities:
    def __init__(self):
        pass
