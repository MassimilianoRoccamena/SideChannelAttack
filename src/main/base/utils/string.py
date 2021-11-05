def cat2(s0, s1):
    '''
    Concatenate 2 strings.
    '''
    return f'{s0}{s1}'
    
def cat3(s0, s1, s3):
    '''
    Concatenate 3 strings.
    '''
    return f'{s0}{s1}{s3}'

def upper1(s):
    '''
    Beginning character become uppercase
    '''
    prefix = s[0].upper()
    suffix = s[1:]
    return f'{prefix}{suffix}'

def upper_identifier(s, sep):
    '''
    Separated string become hungarian notation like
    '''
    splitted = s.split(sep)
    output = ''
    for w in splitted:
        output = f'{output}{upper1(w)}'
    return output