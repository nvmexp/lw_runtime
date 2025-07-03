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

class LwSwitchCounterMonitor:
    def __init__(self, hostname):
        self._pidPostfix = "_" + str(os.getpid()) #Add this to any names so we can run multiple instances
        self._updateIntervalSecs = 30.0 #How often to print out new rows
        self._hostname = hostname
        self.LWSWITCH_NUM_LINKS = 18
        self._InitFieldLists()
        self._InitHandles()

    def _InitFieldLists(self):
        self._lwSwitchLatencyFieldIds = []
        #get the low/medium/high/max latency bucket field ids, each switch port has 4 values.
        #the field ids are contiguous, where first 4 ids are for port0, next 4 for port1 and so on.
        for i in range(dcgm_fields.DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P00, dcgm_fields.DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P17+1, 1):
            self._lwSwitchLatencyFieldIds.append(i)

        #need two lists because there is gap between bandwidth0 and bandwidth1 field Ids.
        #each counter has two values, TX_0 and RX_0. 
        #the field ids are contiguous, where first 2 ids are for port0, next 2 for port1 and so on.
        self._lwSwitchBandwidth0FieldIds = []
        for i in range(dcgm_fields.DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P00, dcgm_fields.DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P17+1, 1):
            self._lwSwitchBandwidth0FieldIds.append(i)

        #get bandwidth counter1 field ids, ie TX_1, RX_1
        self._lwSwitchBandwidth1FieldIds = []
        for i in range(dcgm_fields.DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P00, dcgm_fields.DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P17+1, 1):
            self._lwSwitchBandwidth1FieldIds.append(i)

    def _InitHandles(self):
        self._dcgmHandle = pydcgm.DcgmHandle(ipAddress=self._hostname)

        groupName = "bandwidth_mon_lwswitches" + self._pidPostfix
        self._allLwSwitchesGroup = pydcgm.DcgmGroup(self._dcgmHandle, groupName=groupName, groupType=dcgm_structs.DCGM_GROUP_DEFAULT_LWSWITCHES)
        print("Found %d LWSwitches" % len(self._allLwSwitchesGroup.GetEntities()))

        fgName = "latency_mon_lwswitches" + self._pidPostfix
        self._lwSwitchLatencyFieldGroup = pydcgm.DcgmFieldGroup(self._dcgmHandle, name=fgName, fieldIds=self._lwSwitchLatencyFieldIds)

        fgName = "bandwidth0_mon_lwswitches" + self._pidPostfix
        self._lwSwitchBandwidth0FieldGroup = pydcgm.DcgmFieldGroup(self._dcgmHandle, name=fgName, fieldIds=self._lwSwitchBandwidth0FieldIds)

        fgName = "bandwidth1_mon_lwswitches" + self._pidPostfix
        self._lwSwitchBandwidth1FieldGroup = pydcgm.DcgmFieldGroup(self._dcgmHandle, name=fgName, fieldIds=self._lwSwitchBandwidth1FieldIds)

        updateFreq = int(self._updateIntervalSecs / 2.0) * 1000000
        maxKeepAge = 3600.0 #1 hour
        maxKeepSamples = 0 #Rely on maxKeepAge

        self._lwSwitchLatencyWatcher = dcgm_field_helpers.DcgmFieldGroupEntityWatcher(
            self._dcgmHandle.handle, self._allLwSwitchesGroup.GetId(), 
            self._lwSwitchLatencyFieldGroup, dcgm_structs.DCGM_OPERATION_MODE_AUTO,
            updateFreq, maxKeepAge, maxKeepSamples, 0)
        self._lwSwitchBandwidth0Watcher = dcgm_field_helpers.DcgmFieldGroupEntityWatcher(
            self._dcgmHandle.handle, self._allLwSwitchesGroup.GetId(), 
            self._lwSwitchBandwidth0FieldGroup, dcgm_structs.DCGM_OPERATION_MODE_AUTO,
            updateFreq, maxKeepAge, maxKeepSamples, 0)
        self._lwSwitchBandwidth1Watcher = dcgm_field_helpers.DcgmFieldGroupEntityWatcher(
            self._dcgmHandle.handle, self._allLwSwitchesGroup.GetId(), 
            self._lwSwitchBandwidth1FieldGroup, dcgm_structs.DCGM_OPERATION_MODE_AUTO,
            updateFreq, maxKeepAge, maxKeepSamples, 0)

    def _MonitorOneCycle(self):
        numErrors = 0
        nowStr = time.strftime("%m/%d/%Y %H:%M:%S") 
        self._lwSwitchLatencyWatcher.GetMore()
        self._lwSwitchBandwidth0Watcher.GetMore()
        self._lwSwitchBandwidth1Watcher.GetMore()
        #3D dictionary of [entityGroupId][entityId][fieldId](DcgmFieldValueTimeSeries)
        # where entityId = SwitchID
        for entityGroupId in self._lwSwitchLatencyWatcher.values.keys():
            for entityId in self._lwSwitchLatencyWatcher.values[entityGroupId]:
                latencyFieldId = dcgm_fields.DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P00
                for linkIdx in range(0, self.LWSWITCH_NUM_LINKS):
                    # if the link is not enabled, then the corresponding latencyFieldId key value will be
                    # empty, so skip those links.
                    if self._lwSwitchLatencyWatcher.values[entityGroupId][entityId].has_key(latencyFieldId):
                        latencyLow = self._lwSwitchLatencyWatcher.values[entityGroupId][entityId][latencyFieldId].values[-1].value
                        latencyFieldId += 1
                        latencyMed = self._lwSwitchLatencyWatcher.values[entityGroupId][entityId][latencyFieldId].values[-1].value
                        latencyFieldId += 1
                        latencyHigh = self._lwSwitchLatencyWatcher.values[entityGroupId][entityId][latencyFieldId].values[-1].value
                        latencyFieldId += 1
                        latencyMax = self._lwSwitchLatencyWatcher.values[entityGroupId][entityId][latencyFieldId].values[-1].value
                        latencyFieldId += 1
                        print ("SwitchID %d LinkIdx %d Latency Low %d Medium %d High %d Max %d"
                                % (entityId, linkIdx, latencyLow, latencyMed, latencyHigh, latencyMax))
                    else:
                        latencyFieldId += 4;
        for entityGroupId in self._lwSwitchBandwidth0Watcher.values.keys():
            for entityId in self._lwSwitchBandwidth0Watcher.values[entityGroupId]:
                bandwidth0FieldId = dcgm_fields.DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P00
                bandwidth1FieldId = dcgm_fields.DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P00
                for linkIdx in range(0, self.LWSWITCH_NUM_LINKS):
                    # if the link is not enabled, then the corresponding bandwidth0FieldId and
                    # bandwidth1FieldId key values will be empty, so skip those links.
                    if self._lwSwitchBandwidth0Watcher.values[entityGroupId][entityId].has_key(bandwidth0FieldId):
                        counter0Tx = self._lwSwitchBandwidth0Watcher.values[entityGroupId][entityId][bandwidth0FieldId].values[-1].value
                        counter1Tx = self._lwSwitchBandwidth1Watcher.values[entityGroupId][entityId][bandwidth1FieldId].values[-1].value
                        bandwidth0FieldId += 1
                        bandwidth1FieldId += 1
                        counter0Rx = self._lwSwitchBandwidth0Watcher.values[entityGroupId][entityId][bandwidth0FieldId].values[-1].value
                        counter1Rx = self._lwSwitchBandwidth1Watcher.values[entityGroupId][entityId][bandwidth1FieldId].values[-1].value
                        bandwidth0FieldId += 1
                        bandwidth1FieldId += 1
                        print ("SwitchID %d LinkIdx %d counter0Tx %d counter0Rx %d counter1Tx %d counter1Rx %d"
                                % (entityId, linkIdx, counter0Tx, counter0Rx, counter1Tx, counter1Rx))
                    else:
                        bandwidth0FieldId += 2
                        bandwidth1FieldId += 2

        self._lwSwitchLatencyWatcher.EmptyValues()
        self._lwSwitchBandwidth0Watcher.EmptyValues()
        self._lwSwitchBandwidth1Watcher.EmptyValues()

    def Monitor(self):
        self._lwSwitchLatencyWatcher.EmptyValues()
        self._lwSwitchBandwidth0Watcher.EmptyValues()
        self._lwSwitchBandwidth1Watcher.EmptyValues()

        try:
            while True:
                self._MonitorOneCycle()
                time.sleep(self._updateIntervalSecs)
        except KeyboardInterrupt:
            print ("Got CTRL-C. Exiting")
            return



def main():
    if len(sys.argv) > 1:
        hostname = sys.argv[1]
    else:
        hostname = "localhost"

    counterMonitor = LwSwitchCounterMonitor(hostname)
    print ("Using hostname %s and update interval as %d secs " % (hostname, counterMonitor._updateIntervalSecs))
    counterMonitor.Monitor()

if __name__ == "__main__":
    main()
