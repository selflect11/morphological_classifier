# -*- coding: iso-8859-1 -*-
from . import constants
from .tools import pairwise, safe_division
import itertools


class ProbabilityMatrix:
    def __init__(self, states):
        self.probabilities_dict = {
            state: 0 for state in states
        }

    def update_count(self, entry):
        self.probabilities_dict[entry] += 1

    def items(self):
        return self.probabilities_dict.items()

    def __getitem__(self, entry):
        return self.probabilities_dict[entry]

    def __repr__(self):
        return self.probabilities_dict.__repr__()


class TransitionProbabilities(ProbabilityMatrix):
    def __init__(self):
        ''' Reads a series of transitions once
            and then calculates the probabilities
            of each pair of transitions. '''
        all_tags = constants.TARGET_TAGS
        all_transitions = itertools.product(all_tags, all_tags)
        ProbabilityMatrix.__init__(self, all_transitions)

    def calculate_probabilities(self):
        ''' After the dictionary has been loaded
            with all the counts of each transition,
            computes the probability based on the
            relative frequency of each transition
            given a current state.
            Supposed to be called only once. '''
        count_dict = self.probabilities_dict
        prob_dict = {}
        for curr_state, next_state in count_dict:
            transitions_count = count_dict[curr_state, next_state]
            total_transitions_count = self.total_num_transitions(curr_state)
            prob_dict[curr_state, next_state] = (
                safe_division(transitions_count, total_transitions_count)
                )
        self.probabilities_dict = prob_dict

    def add_transitions(self, tags_list):
        for t1, t2 in pairwise(tags_list):
            self.update_count((t1, t2))

    def total_num_transitions(self, state):
        return sum(value for (curr_state, next_state), value
                    in self.probabilities_dict.items()
                    if curr_state == state)


class InitialProbabilities(ProbabilityMatrix):
    def __init__(self):
        ''' Reads a series of starting tags
            and calculates the probability of
            each tag starting. '''
        all_tags = constants.TARGET_TAGS
        ProbabilityMatrix.__init__(self, all_tags)

    # supposed to be called only once
    def calculate_probabilities(self):
        count_dict = self.probabilities_dict
        total_count = sum(count_dict.values())
        prob_dict = {
            state: safe_division(count, total_count)
            for state, count
            in count_dict.items()
            }
        self.probabilities_dict = prob_dict

    def add_tag(self, tag):
        ''' Wrapper for the update_count() method.
            Needed because of how the other classes
            see this class. '''
        self.update_count(tag)
