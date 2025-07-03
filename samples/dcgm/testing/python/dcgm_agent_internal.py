##
# Python bindings for the internal API of DCGM library (dcgm_agent_internal.h)
##

from ctypes import *
from ctypes.util import find_library
import dcgm_structs
import dcgm_structs_internal


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

# GUIDS for internal APIs
ETID_DCGMEngineInternal = dcgmUuid_t(0x7c3efec4, 0x9fc9, 0x5e6c, 0xb3, 0x37, 0xfe, 0x79, 0x7e, 0x22, 0xe7, 0xd4)
g_etblDCGMEngineInternal = [
    "dcgmServerRun",                      #  0
    "dcgmGetLatestValuesForFields",       #  1
    "dcgmGetMultipleValuesForField",      #  2
    "dcgmGetFieldValuesSince",            #  3
    "dcgmWatchFieldValue",                #  4
    "dcgmUnwatchFieldValue",              #  5
    "",                                   #  6
    "dcgmIntrospectGetFieldMemoryUsage",  #  7
    "dcgmMetadataStateSetRunInterval",    #  8
    "dcgmIntrospectGetFieldExecTime",     #  9
    "",                                   #  10
    "dcgmVgpuConfigSet",                  #  11
    "dcgmVgpuConfigGet",                  #  12
    "dcgmVgpuConfigEnforce",              #  13
    "dcgmGetVgpuDeviceAttributes",        #  14
    "dcgmGetVgpuInstanceAttributes",      #  15
    "dcgmStopDiagnostic",                 #  16
] 

# GUIDs for Internal testing support table
ETID_DCGMEngineTestInternal = dcgmUuid_t(0x8c4eabc6, 0x2ea8, 0x4e7d, 0xa3, 0x58, 0xef, 0x81, 0x4d, 0x21, 0xc3, 0xa5)
g_etblDCGMEngineTestInternal = [
    "dcgmInjectFieldValue",               #  0
    "dcgmIsPolicyManagerRunning",         #  1
    "dcgmGetCacheManagerFieldInfo",       #  2
    "dcgmCreateFakeEntities",             #  3
    "dcgmInjectEntityFieldValue",         #  4
    "dcgmSetEntityLwLinkLinkState"        #  5
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

# TODO create class to check errors when there is a list of DCGM Internal Errors
# class DCGMIntError(dcgm_structs.DCGM_ST_INIT_ERROR):
  # pass

def dcgmInternalGetExportTable(uuid):
    export_table = dcgmExportTable_t()
    fn = dcgm_structs._dcgmGetFunctionPointer("dcgmInternalGetExportTable")
    ret = fn(byref(export_table), byref(uuid))
    _dcgmIntCheckReturn(ret)
    return export_table[0]

def etblDcgmEngineInternal():
    etbl = dcgmInternalGetExportTable(ETID_DCGMEngineInternal)
    etbl.g_etblDCGM = g_etblDCGMEngineInternal
    return etbl

def etblDcgmEngineTestInternal():
    etbl = dcgmInternalGetExportTable(ETID_DCGMEngineTestInternal)
    etbl.g_etblDCGM = g_etblDCGMEngineTestInternal
    return etbl

""" 
Corresponding Calls
"""

def dcgmServerRun(portNumber, socketPath, isConnectionTcp):
    fn = etblDcgmEngineInternal().get_function_pointer_by_name("dcgmServerRun")
    ret = fn(portNumber, socketPath, isConnectionTcp)
    _dcgmIntCheckReturn(ret)
    return ret

def dcgmGetLatestValuesForFields(dcgmHandle, gpuId, fieldIds):
    fn = etblDcgmEngineInternal().get_function_pointer_by_name("dcgmGetLatestValuesForFields")
    field_values = (dcgm_structs.c_dcgmFieldValue_v1 * len(fieldIds))()
    id_values = (c_uint * len(fieldIds))(*fieldIds)
    ret = fn(dcgmHandle, c_int(gpuId), id_values, c_uint(len(fieldIds)), field_values)
    _dcgmIntCheckReturn(ret)
    return field_values

def dcgmGetMultipleValuesForField(dcgmHandle, gpuId, fieldId, maxCount, startTs, endTs, order):
    fn = etblDcgmEngineInternal().get_function_pointer_by_name("dcgmGetMultipleValuesForField")
    localMaxCount = c_int(maxCount) #Going to pass by ref
    #Make space to return up to maxCount records
    max_field_values = (dcgm_structs.c_dcgmFieldValue_v1 * maxCount)()
    ret = fn(dcgmHandle, c_int(gpuId), c_uint(fieldId), byref(localMaxCount), c_int64(startTs), c_int64(endTs), c_uint(order), max_field_values)
    _dcgmIntCheckReturn(ret)
    localMaxCount = localMaxCount.value #Colwert to int
    #We may have gotten less records back than we requested. If so, truncate our array
    return max_field_values[:localMaxCount]

# This method is used to tell the cache manager to watch a field value
def dcgmWatchFieldValue(dcgmHandle, gpuId, fieldId, updateFreq, maxKeepAge, maxKeepEntries):
    fn = etblDcgmEngineInternal().get_function_pointer_by_name("dcgmWatchFieldValue")
    ret = fn(dcgmHandle, c_int(gpuId), c_uint(fieldId), c_longlong(updateFreq), c_double(maxKeepAge), c_int(maxKeepEntries))
    _dcgmIntCheckReturn(ret)
    return ret

# This method is used to tell the cache manager to unwatch a field value
def dcgmUnwatchFieldValue(dcgmHandle, gpuId, fieldId, clearCache):
    fn = etblDcgmEngineInternal().get_function_pointer_by_name("dcgmUnwatchFieldValue")
    ret = fn(dcgmHandle, c_int(gpuId), c_uint(fieldId), c_int(clearCache))
    _dcgmIntCheckReturn(ret)
    return ret

def dcgmInjectFieldValue(dcgmHandle, gpuId, value):
    fn = etblDcgmEngineTestInternal().get_function_pointer_by_name("dcgmInjectFieldValue")
    ret = fn(dcgmHandle, c_uint(gpuId), byref(value))
    _dcgmIntCheckReturn(ret)
    return ret

def dcgmInjectEntityFieldValue(dcgmHandle, entityGroupId, entityId, value):
    fn = etblDcgmEngineTestInternal().get_function_pointer_by_name("dcgmInjectEntityFieldValue")
    ret = fn(dcgmHandle, c_uint(entityGroupId), c_uint(entityId), byref(value))
    _dcgmIntCheckReturn(ret)
    return ret

def dcgmSetEntityLwLinkLinkState(dcgmHandle, entityGroupId, entityId, linkId, linkState):
    linkStateStruct = dcgm_structs_internal.c_dcgmSetLwLinkLinkState_v1()
    linkStateStruct.version = dcgm_structs_internal.dcgmSetLwLinkLinkState_version1
    linkStateStruct.entityGroupId = entityGroupId
    linkStateStruct.entityId = entityId
    linkStateStruct.linkId = linkId
    linkStateStruct.linkState = linkState
    fn = etblDcgmEngineTestInternal().get_function_pointer_by_name("dcgmSetEntityLwLinkLinkState")
    ret = fn(dcgmHandle, byref(linkStateStruct))
    _dcgmIntCheckReturn(ret)
    return ret

def dcgmGetCacheManagerFieldInfo(dcgmHandle, gpuId, fieldId):
    fn = etblDcgmEngineTestInternal().get_function_pointer_by_name("dcgmGetCacheManagerFieldInfo")
    cmfi = dcgm_structs_internal.dcgmCacheManagerFieldInfo_v3()

    cmfi.gpuId = gpuId
    cmfi.fieldId = fieldId

    ret = fn(dcgmHandle, byref(cmfi))
    _dcgmIntCheckReturn(ret)
    return cmfi


def dcgmCreateFakeEntities(dcgmHandle, entityGroupId, numToCreate):
    fn = etblDcgmEngineTestInternal().get_function_pointer_by_name("dcgmCreateFakeEntities")
    
    cfe = dcgm_structs_internal.c_dcgmCreateFakeEntities_v1()

    cfe.version = dcgm_structs_internal.dcgmCreateFakeEntities_version1
    cfe.entityGroupId = entityGroupId
    cfe.numToCreate = numToCreate

    ret = fn(dcgmHandle, byref(cfe))
    _dcgmIntCheckReturn(ret)

    #Return the entiy IDs
    return cfe.entityId[0:numToCreate]


#First parameter below is the return type
dcgmFieldValueEnumeration_f = CFUNCTYPE(c_int32, c_uint32, POINTER(dcgm_structs.c_dcgmFieldValue_v1), c_int32, c_void_p)

def dcgmGetFieldValuesSince(dcgmHandle, groupId, sinceTimestamp, fieldIds, enumCB, userData):
    fn = etblDcgmEngineInternal().get_function_pointer_by_name("dcgmGetFieldValuesSince")
    c_fieldIds = (c_uint32 * len(fieldIds))(*fieldIds)
    c_nextSinceTimestamp = c_int64()
    ret = fn(dcgmHandle, groupId, c_int64(sinceTimestamp), c_fieldIds, c_int32(len(fieldIds)), byref(c_nextSinceTimestamp), enumCB, py_object(userData))
    dcgm_structs._dcgmCheckReturn(ret)
    return c_nextSinceTimestamp.value

def dcgmMetadataStateSetRunInterval(dcgmHandle, runIntervalMs):
    fn = etblDcgmEngineInternal().get_function_pointer_by_name("dcgmMetadataStateSetRunInterval")
    ret = fn(dcgmHandle, c_uint(runIntervalMs))
    _dcgmIntCheckReturn(ret)

def dcgmIntrospectGetFieldExecTime(dcgm_handle, fieldId, waitIfNoData=True):
    fn = etblDcgmEngineInternal().get_function_pointer_by_name("dcgmIntrospectGetFieldExecTime")
    
    execTime = dcgm_structs.c_dcgmIntrospectFullFieldsExecTime_v1()
    execTime.version = dcgm_structs.dcgmIntrospectFullFieldsExecTime_version1
    
    ret = fn(dcgm_handle, fieldId, byref(execTime), waitIfNoData)
    dcgm_structs._dcgmCheckReturn(ret)
    return execTime

def dcgmIntrospectGetFieldMemoryUsage(dcgm_handle, fieldId, waitIfNoData=True):
    fn = etblDcgmEngineInternal().get_function_pointer_by_name("dcgmIntrospectGetFieldMemoryUsage")
    
    memInfo = dcgm_structs.c_dcgmIntrospectFullMemory_v1()
    memInfo.version = dcgm_structs.dcgmIntrospectFullMemory_version1
    
    ret = fn(dcgm_handle, fieldId, byref(memInfo), waitIfNoData)
    dcgm_structs._dcgmCheckReturn(ret)
    return memInfo

def dcgmVgpuConfigSet(dcgm_handle, group_id, configToSet, status_handle):
    fn = etblDcgmEngineInternal().get_function_pointer_by_name("dcgmVgpuConfigSet")
    configToSet.version = dcgm_structs.dcgmDeviceVgpuConfig_version1
    ret = fn(dcgm_handle, group_id, byref(configToSet), status_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret

def dcgmVgpuConfigGet(dcgm_handle, group_id, reqCfgType, count, status_handle):
    fn = etblDcgmEngineInternal().get_function_pointer_by_name("dcgmVgpuConfigGet")

    vgpu_config_values_array = count * dcgm_structs.c_dcgmDeviceVgpuConfig_v1
    c_config_values = vgpu_config_values_array()

    for index in range(0, count):
        c_config_values[index].version = dcgm_structs.dcgmDeviceVgpuConfig_version1

    ret = fn(dcgm_handle, group_id, reqCfgType, count, c_config_values, status_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return map(None, c_config_values[0:count])

def dcgmVgpuConfigEnforce(dcgm_handle, group_id, status_handle):
    fn = etblDcgmEngineInternal().get_function_pointer_by_name("dcgmVgpuConfigEnforce")
    ret = fn(dcgm_handle, group_id, status_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret

def dcgmGetVgpuDeviceAttributes(dcgm_handle, gpuId):
    fn = etblDcgmEngineInternal().get_function_pointer_by_name("dcgmGetVgpuDeviceAttributes")
    device_values = dcgm_structs.c_dcgmVgpuDeviceAttributes_v6()
    device_values.version = dcgm_structs.dcgmVgpuDeviceAttributes_version6
    ret = fn(dcgm_handle, c_int(gpuId), byref(device_values))
    dcgm_structs._dcgmCheckReturn(ret)
    return device_values

def dcgmGetVgpuInstanceAttributes(dcgm_handle, vgpuId):
    fn = etblDcgmEngineInternal().get_function_pointer_by_name("dcgmGetVgpuInstanceAttributes")
    device_values = dcgm_structs.c_dcgmVgpuInstanceAttributes_v1()
    device_values.version = dcgm_structs.dcgmVgpuInstanceAttributes_version1
    ret = fn(dcgm_handle, c_int(vgpuId), byref(device_values))
    dcgm_structs._dcgmCheckReturn(ret)
    return device_values

def dcgmStopDiagnostic(dcgm_handle):
    fn = etblDcgmEngineInternal().get_function_pointer_by_name("dcgmStopDiagnostic")
    ret = fn(dcgm_handle)
    dcgm_structs._dcgmCheckReturn(ret)
    return ret
