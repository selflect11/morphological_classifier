import numpy as np

DEBUGGING = True
D_TYPE = 'float64'
ENCODING = 'iso-8859-1'
SEPARATOR = '$'
TARGET_TAGS = [
    'ADJ',
    'ADV',
    'V',
    'N',
]
NUM_CLASSES = len(TARGET_TAGS)
# TARGET_CLASSES = [ [1,0,0,0], [0,1,0,0],... ]
TARGET_CLASSES = [
    np.array(
        [1 if i == j else 0 for i in range(NUM_CLASSES)],
        dtype = D_TYPE
    ) for j in range(NUM_CLASSES)
]
TAGS_CLASSES = dict(zip(TARGET_TAGS, TARGET_CLASSES))
