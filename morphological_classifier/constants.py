ENCODING = 'iso-8859-1'
SEPARATOR = '$'
TARGET_TAGS = [
    'ADJ',
    'ADV',
    'V',
    'N',
]
TARGET_CLASSES = [
    np.array([1 if i == j else 0 for i in range(len(TARGET_TAGS))]) for j in range(len(TARGET_TAGS))
]
TAGS_CLASSES = dict(zip(TARGET_TAGS, TARGET_CLASSES))
