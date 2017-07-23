# -*- coding: iso-8859-1 -*-
#   Mines the data for statistics on tags and (word : {tags}) pairs

class WordTagDictionary:
    def __init__(self, filepath):
        pass
    def getAllTags(self):
        pass
    def getTag(self, word):
        pass
    def getTransitionProbability(self, tag_list):
        pass

#   w1 w2 w3 w4
#   WTD.getTag(w1) = [t1]
#   WTD.getTag(w2) = [t21, t22]
#   WTD.getTag(w3) = [t31, t32]
#   WTD.getTag(w4) = [t4]
#   max(t1 t21 t31 t4,
#       t1 t22 t31 t4,
#       t1 t21 t32 t4,
#       t1 t22 t32 t4)
#   transition matrix
#           tag1    tag2    tag3    ...     tagn
#   tag1    p11     p12     p13     ...     p1n
#   tag2    p21     p22     p23     ...     p2n
#   tag3    p31     p32     p33     ...     p3n
#   ...
#   tagn    pn1     pn2     pn3     ...     pnn
#   ==
#   max(p(t1->t21) * p(t21->t31) * p(t31->t4),
#       p(t1->t22) * p(t22->t31) * p(t31->t4),
#       p(t1->t21) * p(t21->t32) * p(t32->t4),
#       p(t1->t22) * p(t22->t32) * p(t32->t4))
