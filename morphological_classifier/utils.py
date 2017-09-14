import sys
from collections import defaultdict
from nltk.stem import RSLPStemmer
import re
from statistics import *


def get_scores_from_text(text_line):
    scores_pat = re.compile('\s*(\d\.\d{2})\s*')
    tag_pat = re.compile('^\s*(.+?)\s')
    scores = re.findall(scores_pat, text_line)

    tag_match_obj = re.search(tag_pat, text_line)
    tag_name = ''
    if tag_match_obj and scores:
        return tag_match_obj.group(1), map(float, scores)
    return None, (None, None, None)

def get_total_scores_from_text(text_line):
    total_pat = re.compile(
        'avg / total\s+(\d\.\d{2})\s+(\d\.\d{2})\s+(\d\.\d{2})')
    match_obj = re.search(total_pat, text_line)
    if match_obj:
        return map(float, match_obj.groups())
    return None, None, None

def get_mean_stdev(lst):
    return mean(lst), stdev(lst)

def get_suffix(word):
    return word[-3:]
#    st = RSLPStemmer()
#    radical = st.stem(word)
#    affix, radical, suffix = str.partition(word, radical)
#    if suffix:
#        return suffix
#    else:
#        return word[-3:]

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
