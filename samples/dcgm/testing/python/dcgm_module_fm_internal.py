##
# Python bindings for the internal Fabric Manager APIs
##

from ctypes import *
from ctypes.util import find_library
import dcgm_structs

DCGM_EMBEDDED_HANDLE = c_void_p(0x7fffffff)

class dcgmUuid_t (Structure):
    _fields_ = [
        ('bytes', c_ubyte * 16),
    ]

    def __init__(self, a, b, c, d0, d1, d2, d3, d4, d5, d6, d7):
        self.bytes[0]  = (a)       & 0xff
        self.bytes[1]  = (a >>  8) & 0xff
        self.bytes[2]  = (a >> 16) & 0xff
        self.bytes[3]  = (a >> 24) & 0xff
        self.bytes[4]  = (b)       & 0xff
        self.bytes[5]  = (b >>  8) & 0xff
        self.bytes[6]  = (c)       & 0xff
        self.bytes[7]  = (c >>  8) & 0xff
        self.bytes[8]  = d0        & 0xff
        self.bytes[9]  = d1        & 0xff
        self.bytes[10] = d2        & 0xff
        self.bytes[11] = d3        & 0xff
        self.bytes[12] = d4        & 0xff
        self.bytes[13] = d5        & 0xff
        self.bytes[14] = d6        & 0xff
        self.bytes[15] = d7        & 0xff

    def __hash__(self):
        return reduce(lambda acc, y: acc * 2 + y, self.bytes, 0)

# GUIDS for Fabric Manager internal APIs
ETID_DCGMModuleFMInternal = dcgmUuid_t(0x5bdeafe8, 0xb2f3, 0x4fd5, 0xb3, 0x8c, 0xfe, 0x9b, 0x9d, 0xc3, 0x8f, 0x1e)
g_etblDCGMModuleFMInternal = [
    "dcgmGetSupportedFabricPartitions",  #  0
    "dcgmActivateFabricPartition",       #  1
    "dcgmDeactivateFabricPartition"      #  2
] 

class struct_dcgmExportTable_t(Structure):
    _fields_ = [
        ('struct_size', c_ulong),
        ('fptr', c_void_p * 1000),
    ]

    def get_function_pointer_by_index(self, index):
        if index * sizeof(c_void_p) + sizeof(c_ulong) >= self.struct_size:
            raise IndexError("index %d falls beyond export table which is of size (%d bytes, count %d)" %
                    (index, self.struct_size, (self.struct_size - sizeof(c_ulong)) / sizeof(c_void_p)))
        prot = CFUNCTYPE(c_int)
        result = prot(self.fptr[index])
        if not result:
            raise LookupError("function under index %d was removed (is NULL)" % (index))
        return result

    def get_function_pointer_by_name(self, name):
        # g_etblDCGM needs to be set separately for this function to work properly
        try:
            return self.get_function_pointer_by_index(self.g_etblDCGM.index(name))
        except LookupError, e:
            raise LookupError("Failed to query pointer of function %s: %s" % (name, e.message))

    def __hash__(self):
        return self.struct_size

dcgmExportTable_t = POINTER(struct_dcgmExportTable_t)

# Utils
_dcgmIntCheckReturn = dcgm_structs._dcgmCheckReturn

def dcgmModuleFMInternalGetExportTable(uuid):
    export_table = dcgmExportTable_t()
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmInternalGetExportTable")
    ret = fn(byref(export_table), byref(uuid))
    _dcgmIntCheckReturn(ret)
    return export_table[0]

def etblModuleFMInternal():
    etbl = dcgmModuleFMInternalGetExportTable(ETID_DCGMModuleFMInternal)
    etbl.g_etblDCGM = g_etblDCGMModuleFMInternal
    return etbl

""" 
Corresponding Calls
"""

# This method is used to query all the available fabric partitions in an LWSwitch based system.
def dcgmGetSupportedFabricPartitions(pDcgmHandle, pDcgmFabricPartition):
    fn = etblModuleFMInternal().get_function_pointer_by_name("dcgmGetSupportedFabricPartitions")
    ret = fn(pDcgmHandle, pDcgmFabricPartition)
    _dcgmIntCheckReturn(ret)
    return ret

# This method is used to activate a supported fabric partition in an LWSwitch based system.
def dcgmActivateFabricPartition(pDcgmHandle, partitionId):
    fn = etblModuleFMInternal().get_function_pointer_by_name("dcgmActivateFabricPartition")
    ret = fn(pDcgmHandle, c_uint(partitionId))
    _dcgmIntCheckReturn(ret)
    return ret

# This method is used to deactivate a previously activated fabric partition in an LWSwitch based system.
def dcgmDeactivateFabricPartition(pDcgmHandle, partitionId):
    fn = etblModuleFMInternal().get_function_pointer_by_name("dcgmDeactivateFabricPartition")
    ret = fn(pDcgmHandle, c_uint(partitionId))
    _dcgmIntCheckReturn(ret)
    return ret
