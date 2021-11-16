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

def upper_identifier(s, sep='_'):
    '''
    Separated string become hungarian notation like
    '''
    splitted = s.split(sep)
    output = ''
    for w in splitted:
        output = f'{output}{upper1(w)}'
    return output

def lower1(s):
    '''
    Beginning character become lowercase
    '''
    prefix = s[0].lower()
    suffix = s[1:]
    return f'{prefix}{suffix}'

def lower_identifier(s, sep='_'):
    '''
    Hungarian notiation like string become separated
    '''
    output = ''
    first = True
    start, end = 0, 0
    for i in range(len(s)):
        end = i
        if s[i].isupper():
            if start != end:
                suffix = lower1(s[start:end])
                if not first:
                    suffix = f'{sep}{suffix}'
                else:
                    first = False
                output = f'{output}{suffix}'
                start = i
    suffix = lower1(s[start:len(s)])
    if not first:
        suffix = f'{sep}{suffix}'
    else:
        first = False
    output = f'{output}{suffix}'
    return output