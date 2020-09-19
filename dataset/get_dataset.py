from .KETTS_male import KETTS_30m
from .KETTS_female import KETTS_30f
from .KETTS76 import KETTS76
from .KNTTS import KNTTS
from .BC2013 import BC2013
from .LJSpeech11 import LJSpeech11
from .ETRI import ETRI

def get_dataset(dbname, which_set='train'):
    if dbname == 'KETTS76':
        return KETTS76(which_set=which_set)
    if dbname == 'KNTTS':
        return KNTTS(which_set=which_set)
    elif dbname == 'KETTS_male':
        return KETTS_30m(which_set=which_set)
    elif dbname == 'KETTS_female':
        return KETTS_30f(which_set=which_set)
    elif dbname == 'BC2013':
        return BC2013(which_set=which_set)
    elif dbname == 'LJSpeech11':
        return LJSpeech11(which_set=which_set)
    elif dbname == 'ETRI':
        return ETRI(which_set)
    else:
        raise ValueError(f'{dbroot}')
