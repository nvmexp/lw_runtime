##
# Python bindings for the internal API of DCGM library (dcgm_fields_internal.h)
##

from ctypes import *
from ctypes.util import find_library
import dcgm_structs

# Provides access to functions
dcgmFP = dcgm_structs._dcgmGetFunctionPointer


