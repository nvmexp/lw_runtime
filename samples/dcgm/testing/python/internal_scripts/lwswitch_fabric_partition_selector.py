
try:
    import pydcgm
except ImportError:
    print("Unable to find pydcgm. You need to add the location of "
          "pydcgm.py to your environment as PYTHONPATH=$PYTHONPATH:[path-to-pydcgm.py]")

import sys
import os
import time
import dcgm_structs
import dcgm_module_fm_structs_internal
import dcgm_module_fm_internal
from string import Template
import argparse
from ctypes import *
from string import Template

# initialize the argument parser
argParser = argparse.ArgumentParser(description='HGX-2 Fabric Partition Selector for multitenancy')

# list all the supported arguments
# --help or -h is by default present in argparse library
argParser.add_argument('--hostname',
                       type=str,
                       default='localhost',
                       help='hostname or IP address of lw-hostengine to connect. Defaults to localhost')

# mutually exclusive options, ie one option at a time
argGroup = argParser.add_mutually_exclusive_group(required=True)
argGroup.add_argument('--list',
                      action='store_true',
                      help='query all the available fabric partitions')

argGroup.add_argument('--activate',
                      type=int,
                      choices=range(0, 1000),
                      help='activate a supported fabric partition')

argGroup.add_argument('--deactivate',
                      type=int,
                      help='deactivate a previously activated fabric partition')

dcgmHandle = 0

def dumpReturnStatus(optionCtx, retStatus):
    statusInfoTemplate = Template("""<return: $optionCtx completed with status: $retStatus </return>""")
    statsStr = statusInfoTemplate.substitute(optionCtx=optionCtx, retStatus=retStatus)
    print(statsStr)

def dumpFabricPartitions(partitionList):
    gpuInfoTemplate = Template(
"""        <gpu> <physicalId> $physicalId </physicalId>  <uuid> $uuid </uuid>  <pciBusId> $pciBusId </pciBusId>  </gpu>""")

    partitionInfoTemplate = Template( 
"""  <partition> \n    <partitionId> $partitionId </partitionId> <isActive> $isActive </isActive> <numGpus> $numGpus </numGpus> \n      <gpus> $gpus \n      </gpus> \n  </partition>""")

    partitionListTemplate = Template( """<partitions>  <version> $version </version>  <numPartitions> $numPartitions </numPartitions> $partitions \n</partitions>""")

    partitionInfoStr = ""
    for partIdx in range(0, partitionList.numPartitions):
        partitionInfo = partitionList.partitionInfo[partIdx]
        gpuInfoStr = ""        
        for gpuIdx in range(0, partitionInfo.numGpus):
            gpuInfo = partitionInfo.gpuInfo[gpuIdx]
            tempGpuInfoStr = gpuInfoTemplate.substitute(physicalId=gpuInfo.physicalId, uuid=gpuInfo.uuid, pciBusId=gpuInfo.pciBusId)
            gpuInfoStr = gpuInfoStr + "\n" + tempGpuInfoStr
        tempPartitionInfoStr = partitionInfoTemplate.substitute(partitionId=partitionInfo.partitionId, isActive=partitionInfo.isActive, gpus=gpuInfoStr, numGpus=partitionInfo.numGpus)
        partitionInfoStr = partitionInfoStr + "\n" + tempPartitionInfoStr

    partitionListStr = partitionListTemplate.substitute(version=partitionList.version, numPartitions=partitionList.numPartitions, partitions=partitionInfoStr)
    print(partitionListStr)

def initializeHandles(hostname):
    global dcgmHandle
    dcgmHandle = pydcgm.DcgmHandle(ipAddress=hostname)

def listFabricPartitions():
    partitionList = dcgm_module_fm_structs_internal.c_dcgmFabricPartitionList_v1()
    partitionList.version = dcgm_module_fm_structs_internal.dcgmFabricPartitionList_version1
    ret = dcgm_module_fm_internal.dcgmGetSupportedFabricPartitions(dcgmHandle.handle, byref(partitionList))
    dumpReturnStatus("List all supported fabric partitions", ret)
    # dump the output in XML/JSON format
    if ret != dcgm_structs.DCGM_ST_OK or partitionList.numPartitions == 0:
        return

    dumpFabricPartitions(partitionList)

def activateFabricPartition(partitionId):
    ret = dcgm_module_fm_internal.dcgmActivateFabricPartition(dcgmHandle.handle, partitionId)
    dumpReturnStatus("Activate fabric partition", ret)

def deactivateFabricPartition(partitionId):
    ret = dcgm_module_fm_internal.dcgmDeactivateFabricPartition(dcgmHandle.handle, partitionId)
    dumpReturnStatus("Deactivate fabric partition", ret)

def main():
    cmdOptions = argParser.parse_args()

    # connec to dcgm instance
    initializeHandles(cmdOptions.hostname)

    # dispatch based on the options provided
    if cmdOptions.list:
        listFabricPartitions()

    if cmdOptions.activate is not None:
        activateFabricPartition(cmdOptions.activate)

    if cmdOptions.deactivate is not None:
        deactivateFabricPartition(cmdOptions.deactivate)

if __name__ == "__main__":
    main()
