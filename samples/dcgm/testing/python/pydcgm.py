
#Bring classes into this namespace
from DcgmHandle import *
from DcgmGroup import *
from DcgmStatus import *
from DcgmSystem import *
from DcgmFieldGroup import *

'''
Define a unique exception type we will return so that callers can distinguish our exceptions from python standard ones
'''
class DcgmException(Exception):
    pass