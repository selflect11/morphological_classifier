import numpy as np

D_TYPE = 'float64'
ENCODING = 'utf-8'
SEPARATOR = '$'
TARGET_TAGS = [
    'ADJ',
    'ADV',
    'V',
    'N',
]
NUM_CLASSES = len(TARGET_TAGS)
TARGET_CLASSES = [
    np.array([1,0,0,0], dtype=D_TYPE),
    np.array([0,1,0,0], dtype=D_TYPE),
    np.array([0,0,1,0], dtype=D_TYPE),
    np.array([0,0,0,1], dtype=D_TYPE),
]
TAGS_CLASSES = dict(zip(TARGET_TAGS, TARGET_CLASSES))
