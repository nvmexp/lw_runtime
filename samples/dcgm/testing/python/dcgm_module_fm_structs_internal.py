##
# Python bindings for "dcgm_module_fm_structs_internal.h"
##

from ctypes import *
import dcgm_structs
from dcgm_fields import _PrintableStructure

DCGM_MAX_FABRIC_PARTITIONS = 31

class c_dcgmFabricPartitionGpuInfo_t(_PrintableStructure):
    _fields_ = [
        ('physicalId', c_uint),
        ('uuid', c_char * dcgm_structs.DCGM_DEVICE_UUID_BUFFER_SIZE),
        ('pciBusId', c_char * dcgm_structs.DCGM_MAX_STR_LENGTH)
    ]

class c_dcgmFabricPartitionInfo_t(_PrintableStructure):
    _fields_ = [
        ('partitionId', c_uint),
        ('isActive', c_uint),
        ('numGpus', c_uint),
        ('gpuInfo', c_dcgmFabricPartitionGpuInfo_t * dcgm_structs.DCGM_MAX_NUM_DEVICES)
    ]

class c_dcgmFabricPartitionList_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('numPartitions', c_uint),
        ('partitionInfo', c_dcgmFabricPartitionInfo_t * DCGM_MAX_FABRIC_PARTITIONS)
    ]

dcgmFabricPartitionList_version1 = dcgm_structs.make_dcgm_version(c_dcgmFabricPartitionList_v1, 1)

class c_dcgmActivatedFabricPartitionList_v1(_PrintableStructure):
    _fields_ = [
        ('version', c_uint),
        ('numPartitions', c_uint),
        ('partitionIds', c_uint * DCGM_MAX_FABRIC_PARTITIONS)
    ]

dcgmActivatedFabricPartitionList_version1 = dcgm_structs.make_dcgm_version(c_dcgmActivatedFabricPartitionList_v1, 1)
