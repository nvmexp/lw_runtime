##
# Python bindings for the internal API of DCGM library (dcgm_client_internal.h)
##

import dcgm_structs
import dcgm_structs_internal
import dcgm_agent_internal
from ctypes import *


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


# GUIDS for internal APIs
ETID_DCGMClientInternal = dcgmUuid_t(0x2c9eabc4, 0x4dc3, 0x2f5d, 0xb7, 0x45, 0xbb, 0x71, 0x9f, 0x26, 0xcf, 0xb5)
g_etblDCGMClientInternal = [
    "dcgmClientSaveCacheManagerStats",          # 1
    "dcgmClientLoadCacheManagerStats",          # 2
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
dcgmDeviceConfig_t  = dcgm_structs.c_dcgmDeviceConfig_v1
dcgmRecvUpdates_t = dcgm_structs._dcgmRecvUpdates_t
dcgmStatsFileType_t = dcgm_structs_internal._dcgmStatsFileType_t
dcgmInjectFieldValue_t = dcgm_structs_internal.c_dcgmInjectFieldValue_v1
dcgmPolicyViolation_t = dcgm_structs.c_dcgmPolicyViolation_v1
dcgmPolicy_t = dcgm_structs.c_dcgmPolicy_v1
dcgmDiagResponse_t = dcgm_structs.c_dcgmDiagResponse_v3


def dcgmInternalGetExportTable(uuid):
    export_table = dcgmExportTable_t()
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmInternalGetExportTable")
    ret = fn(byref(export_table), byref(uuid))
    _dcgmIntCheckReturn(ret)
    return export_table[0]

def etblDcgmClientInternal():
    etbl = dcgmInternalGetExportTable(ETID_DCGMClientInternal)
    etbl.g_etblDCGM = g_etblDCGMClientInternal
    return etbl


""" 
Corresponding Calls
"""

# This method is used to save the DcgmCacheManager's stats to a file local to the host engine
def dcgmClientSaveCacheManagerStats(pDcgmHandle, filename, fileType):
    fn = etblDcgmClientInternal().get_function_pointer_by_name("dcgmClientSaveCacheManagerStats")
    ret = fn(pDcgmHandle, filename, dcgmStatsFileType_t(fileType))
    _dcgmIntCheckReturn(ret)
    return ret

# This method is used to load the DcgmCacheManager's stats from a file local to the host engine
def dcgmClientLoadCacheManagerStats(pDcgmHandle, filename, fileType):
    fn = etblDcgmClientInternal().get_function_pointer_by_name("dcgmClientLoadCacheManagerStats")
    ret = fn(pDcgmHandle, filename, dcgmStatsFileType_t(fileType))
    _dcgmIntCheckReturn(ret)
    return ret
