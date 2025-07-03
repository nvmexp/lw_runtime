import subprocess
import signal, os
import pydcgm
import dcgm_fields
import dcgm_structs
import threading
from DcgmReader import DcgmReader

if 'DCGM_TESTING_FRAMEWORK' in os.elwiron:
    try:
        import collectd_tester_api_cosmos as collectd
    except:
        import collectd
else:
    import collectd

# Set default values for the hostname and the library path
g_dcgmLibPath = '/usr/lib'
g_dcgmHostName = 'localhost'

# Override settings through the environment since this is imported into collectd and not exelwted
if 'DCGM_HOSTNAME' in os.elwiron:
    g_dcgmHostName = os.elwiron['DCGM_HOSTNAME']

if 'DCGMLIBPATH' in os.elwiron:
    g_dcgmLibPath = os.elwiron['DCGMLIBPATH']

g_dcgmIgnoreFields = [dcgm_fields.DCGM_FI_DEV_UUID, dcgm_fields.DCGM_FI_DEV_PCI_BUSID] #Fields not to publish

g_publishFieldIds = [
    dcgm_fields.DCGM_FI_DEV_PCI_BUSID, #Do we want this published or not?
    dcgm_fields.DCGM_FI_DEV_UUID, #Needed for plugin instance
    dcgm_fields.DCGM_FI_DEV_POWER_USAGE,
    dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
    dcgm_fields.DCGM_FI_DEV_SM_CLOCK,
    dcgm_fields.DCGM_FI_DEV_GPU_UTIL,
    dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING,
    dcgm_fields.DCGM_FI_DEV_RETIRED_SBE,
    dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
    dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_TOTAL,
    dcgm_fields.DCGM_FI_DEV_ECC_DBE_AGG_TOTAL,
    dcgm_fields.DCGM_FI_DEV_FB_TOTAL,
    dcgm_fields.DCGM_FI_DEV_FB_FREE,
    dcgm_fields.DCGM_FI_DEV_FB_USED,
    dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
    dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_TOTAL,
    dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
    dcgm_fields.DCGM_FI_DEV_XID_ERRORS,
    dcgm_fields.DCGM_FI_DEV_MEM_CLOCK,
    dcgm_fields.DCGM_FI_DEV_MEMORY_TEMP,
    dcgm_fields.DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION,
    dcgm_fields.DCGM_FI_DEV_MEM_COPY_UTIL,
    dcgm_fields.DCGM_FI_DEV_POWER_VIOLATION,
    dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION,
    dcgm_fields.DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_TOTAL,
    dcgm_fields.DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_TOTAL,
    dcgm_fields.DCGM_FI_DEV_LWLINK_BANDWIDTH_TOTAL,
    dcgm_fields.DCGM_FI_DEV_PCIE_TX_THROUGHPUT,
    dcgm_fields.DCGM_FI_DEV_PCIE_RX_THROUGHPUT,
    dcgm_fields.DCGM_FI_DEV_SERIAL, #Needed for plugin instance
    dcgm_fields.DCGM_FI_DEV_MINOR_NUMBER #Needed for plugin instance

    ]

class DcgmCollectd(DcgmReader):
    ###########################################################################
    def __init__(self):
        DcgmReader.__init__(self, fieldIds=g_publishFieldIds, ignoreList=g_dcgmIgnoreFields, fieldGroupName='dcgm_collectd')
        self.m_gpuIdToUUId = {} # FieldId => dcgm_fields.dcgm_field_meta_t

    ###########################################################################
    def SetupGpuIdUUIdMappings(self):
        '''
        Populate the m_gpuIdToUUId map
        '''
        gpuIds = self.m_dcgmGroup.GetGpuIds()
        for gpuId in gpuIds:
            gpuInfo = self.m_dcgmSystem.discovery.GetGpuAttributes(gpuId)
            self.m_gpuIdToUUId[gpuId] = gpuInfo.identifiers.uuid

    ###########################################################################
    def LwstomDataHandler(self, fvs):
        # Fill the gpuId to UUId dictionary
        if not self.m_gpuIdToUUId:
            self.SetupGpuIdUUIdMappings()

        value = collectd.Values(type='gauge')
        value.plugin = 'dcgm_collectd'

        for gpuId in fvs.keys():
            gpuFv = fvs[gpuId]

            uuid = self.m_gpuIdToUUId[gpuId]
            value.plugin_instance = '%s' % (uuid)

            typeInstance = str(gpuId)

            for fieldId in gpuFv.keys():
                # Skip ignore list
                if fieldId in self.m_dcgmIgnoreFields:
                    continue

                fieldTag = self.m_fieldIdToInfo[fieldId].tag
                val = gpuFv[fieldId][-1]

                #Skip blank values. Otherwise, we'd have to insert a placeholder blank value based on the fieldId
                if val.isBlank:
                    continue

                valTimeSec1970 = (val.ts / 1000000) #Round down to 1-second for now
                valueArray = [val.value, ]

                # Updating write() to dispatch() as per API documentation (https://collectd.org/documentation/manpages/collectd-python.5.shtml#values)
                value.dispatch(type=fieldTag, type_instance=typeInstance, time=valTimeSec1970, values=valueArray, plugin=value.plugin)

                #collectd.info("gpuId %d, tag %s, value %s" % (gpuId, fieldTag, str(val.value)))

    ###########################################################################
    def LogInfo(self, msg):
        collectd.info(msg)

    ###########################################################################
    def LogError(self, msg):
        collectd.info(msg)

###############################################################################
##### Wrapper the Class methods for collectd callbacks
###############################################################################
def init_dcgm():
    g_dcgmCollectd.Init(libpath=g_dcgmLibPath)

###############################################################################
def shutdown_dcgm():
    g_dcgmCollectd.Shutdown()

###############################################################################
def read_dcgm(data=None):
    g_dcgmCollectd.Process()


###############################################################################
##### Main
###############################################################################
g_dcgmCollectd = DcgmCollectd()

# Register the callback functions with collectd
collectd.register_init(init_dcgm)
collectd.register_read(read_dcgm)
collectd.register_shutdown(shutdown_dcgm)
