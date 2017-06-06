def list_to_float(lst):
    # big endian std
    num = 0
    for index, coeff in enumerate(reversed(lst)):
        num += coeff * (2**index)
    return num

