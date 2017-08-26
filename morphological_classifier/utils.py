import sys
from collections import defaultdict
from nltk.stem import RSLPStemmer

def get_suffix(word):
    return word[-3:]
#    st = RSLPStemmer()
#    radical = st.stem(word)
#    affix, radical, suffix = str.partition(word, radical)
#    return suffix

#only needed for dumbass serialization
def defaultdict_float():
    return defaultdict(float)

def safe_division(a, b):
    if b == 0:
        return 0
    return a/b

def update_progress(progress):
    # Modify this to change the length of the progress bar
    barLength = 30
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
