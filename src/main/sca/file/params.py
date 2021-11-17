TRACE_SIZE = 134016
TEXT_SIZE = 16
KEY_SIZE = 16
BYTE_SIZE = 256

def str_hex_bytes():
    '''
    Returns hex values of a byte.
    '''
    half_val = [str(n) for n in range(10)]
    half_val = half_val + ['a','b','c','d','e','f']
    out = []

    for high in half_val:
        for low in half_val:
            out.append(f'{high}{low}')

    return out