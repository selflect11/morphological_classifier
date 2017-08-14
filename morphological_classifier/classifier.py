# -*- coding: utf-8 -*-

class MorphologicalClassifier:
    def __init__(self, tagger):
        self.tagger = tagger

    def classify(self, phrase):
        return tuple(map(self.tagger.tag, phrase.split()))
