def list_to_float(lst):
    # big endian std
    num = 0
    for index, coeff in enumerate(reversed(lst)):
        num += coeff * (2**index)
    return num

def asciify(error):
    return map[error.object[error.start]], error.end
def setup_asciify():
    codecs.register_error('asciify', asciify)
def asciify_word(word):
    setup_asciify()
    word = word.lower()
    return word.encode('ascii', 'asciify')
