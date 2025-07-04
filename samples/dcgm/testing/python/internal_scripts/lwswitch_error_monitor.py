

try:
    import pydcgm
except ImportError:
    print("Unable to find pydcgm. You need to add the location of "
          "pydcgm.py to your environment as PYTHONPATH=$PYTHONPATH:[path-to-pydcgm.py]")

import sys
import os
import time
import dcgm_field_helpers
import dcgm_fields
import dcgm_structs


class LwSwitchErrorMonitor:
    def __init__(self, hostname):
        self._pidPostfix = "_" + str(os.getpid()) #Add this to any names so we can run multiple instances
        self._updateIntervalSecs = 5.0 #How often to print out new rows
        self._hostname = hostname
        self._InitFieldLists()
        self._InitHandles()
        
    def _InitFieldLists(self):
        #LWSwitch error field Ids
        self._lwSwitchErrorFieldIds = []
        self._lwSwitchErrorFieldIds.append(dcgm_fields.DCGM_FI_DEV_LWSWITCH_FATAL_ERRORS)
        self._lwSwitchErrorFieldIds.append(dcgm_fields.DCGM_FI_DEV_LWSWITCH_NON_FATAL_ERRORS)
        
        #GPU error field Ids
        self._gpuErrorFieldIds = []
        self._gpuErrorFieldIds.append(dcgm_fields.DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_TOTAL)
        self._gpuErrorFieldIds.append(dcgm_fields.DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_TOTAL)
        self._gpuErrorFieldIds.append(dcgm_fields.DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_TOTAL)
        self._gpuErrorFieldIds.append(dcgm_fields.DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_TOTAL)
        self._gpuErrorFieldIds.append(dcgm_fields.DCGM_FI_DEV_XID_ERRORS)
        self._gpuErrorFieldIds.append(dcgm_fields.DCGM_FI_DEV_GPU_LWLINK_ERRORS)
        #self._gpuErrorFieldIds.append(dcgm_fields.DCGM_FI_DEV_GPU_TEMP) #Will always generate output

    def _InitHandles(self):
        self._dcgmHandle = pydcgm.DcgmHandle(ipAddress=self._hostname)
        
        groupName = "error_mon_gpus" + self._pidPostfix
        self._allGpusGroup = pydcgm.DcgmGroup(self._dcgmHandle, groupName=groupName, groupType=dcgm_structs.DCGM_GROUP_DEFAULT)
        print("Found %d GPUs" % (len(self._allGpusGroup.GetEntities())))

        groupName = "error_mon_lwswitches" + self._pidPostfix
        self._allLwSwitchesGroup = pydcgm.DcgmGroup(self._dcgmHandle, groupName=groupName, groupType=dcgm_structs.DCGM_GROUP_DEFAULT_LWSWITCHES)
        print("Found %d LwSwitches" % len(self._allLwSwitchesGroup.GetEntities()))

        fgName = "error_mon_lwswitches" + self._pidPostfix
        self._lwSwitchErrorFieldGroup = pydcgm.DcgmFieldGroup(self._dcgmHandle, name=fgName, fieldIds=self._lwSwitchErrorFieldIds)
        
        fgName = "error_mon_gpus" + self._pidPostfix
        self._gpuErrorFieldGroup = pydcgm.DcgmFieldGroup(self._dcgmHandle, name=fgName, fieldIds=self._gpuErrorFieldIds)

        updateFreq = int(self._updateIntervalSecs / 2.0) * 1000000
        maxKeepAge = 3600.0 #1 hour
        maxKeepSamples = 0 #Rely on maxKeepAge

        self._lwSwitchWatcher = dcgm_field_helpers.DcgmFieldGroupEntityWatcher(
            self._dcgmHandle.handle, self._allLwSwitchesGroup.GetId(), 
            self._lwSwitchErrorFieldGroup, dcgm_structs.DCGM_OPERATION_MODE_AUTO,
            updateFreq, maxKeepAge, maxKeepSamples, 0)
        self._gpuWatcher = dcgm_field_helpers.DcgmFieldGroupEntityWatcher(
            self._dcgmHandle.handle, self._allGpusGroup.GetId(), 
            self._gpuErrorFieldGroup, dcgm_structs.DCGM_OPERATION_MODE_AUTO,
            updateFreq, maxKeepAge, maxKeepSamples, 0)

    def _GetLatestGpuErrorSamples(self):
        numErrors = 0
        nowStr = time.strftime("%m/%d/%Y %H:%M:%S") 
        
        self._gpuWatcher.GetMore()
        for entityGroupId in self._gpuWatcher.values.keys():
            for entityId in self._gpuWatcher.values[entityGroupId]:
                for fieldId in self._gpuWatcher.values[entityGroupId][entityId]:
                    for value in self._gpuWatcher.values[entityGroupId][entityId][fieldId].values:
                        if not value.isBlank and value.value > 0:
                            fieldMeta = dcgm_fields.DcgmFieldGetById(fieldId)
                            print "%s: Got error for GPU %d, field Id %s, value %d" % (nowStr, entityId, fieldMeta.tag, int(value.value))
                            numErrors += 1
        
        self._gpuWatcher.EmptyValues()
        if numErrors == 0:
            print "%s: No GPU errors." % nowStr

    def _GetLatestSwitchErrorSamples(self):
        numErrors = 0
        nowStr = time.strftime("%m/%d/%Y %H:%M:%S") 

        self._lwSwitchWatcher.GetMore()
        for entityGroupId in self._lwSwitchWatcher.values.keys():
            for entityId in self._lwSwitchWatcher.values[entityGroupId]:
                for fieldId in self._lwSwitchWatcher.values[entityGroupId][entityId]:
                    for value in self._lwSwitchWatcher.values[entityGroupId][entityId][fieldId].values:
                        if not value.isBlank and value.value > 0:
                            fieldMeta = dcgm_fields.DcgmFieldGetById(fieldId)
                            print "%s: Got error for LwSwitch %d, field Id %s, value %d" % (nowStr, entityId, fieldMeta.tag, int(value.value))
                            numErrors += 1
        
        self._lwSwitchWatcher.EmptyValues()
        if numErrors == 0:
            print "%s: No Switch errors." % nowStr

    def _MonitorOneCycle(self):
        self._GetLatestGpuErrorSamples()
        self._GetLatestSwitchErrorSamples()

    def Monitor(self):
        self._gpuWatcher.EmptyValues()
        self._lwSwitchWatcher.EmptyValues()

        try:
            while True:
                self._MonitorOneCycle()
                time.sleep(self._updateIntervalSecs)
        except KeyboardInterrupt:
            print "Got CTRL-C. Exiting"
            return



def main():
    hostname = "localhost"

    if len(sys.argv) > 1:
        hostname = sys.argv[1]

    print ("Using hostname " + hostname)

    errorMonitor = LwSwitchErrorMonitor(hostname)
    errorMonitor.Monitor()

if __name__ == "__main__":
    main()
