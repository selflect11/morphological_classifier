#   Mines the data for statistics on tags and (word : {tags}) pairs

class WordTagDictionary:
    def __init__(self):
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
#   tag.getProbability()
