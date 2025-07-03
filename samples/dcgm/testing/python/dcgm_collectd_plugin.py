import subprocess
import signal, os
import pydcgm
import dcgm_fields
import dcgm_structs
import threading
from DcgmReader import DcgmReader

if 'DCGM_TESTING_FRAMEWORK' in os.elwiron:
    try:
        import collectd_tester_api as collectd
    except:
        import collectd
else:
    import collectd

# Set default values for the hostname and the library path
g_dcgmLibPath = '/usr/lib'
g_dcgmHostName = 'localhost'

# Add overriding through the environment instead of through the
if 'DCGM_HOSTNAME' in os.elwiron:
    g_dcgmHostName = os.elwiron['DCGM_HOSTNAME']

if 'DCGMLIBPATH' in os.elwiron:
    g_dcgmLibPath = os.elwiron['DCGMLIBPATH']

g_dcgmIgnoreFields = [dcgm_fields.DCGM_FI_DEV_UUID] #Fields not to publish

g_publishFieldIds = [
    dcgm_fields.DCGM_FI_DEV_UUID, #Needed for plugin instance
    dcgm_fields.DCGM_FI_DEV_POWER_USAGE,
    dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
    dcgm_fields.DCGM_FI_DEV_SM_CLOCK,
    dcgm_fields.DCGM_FI_DEV_GPU_UTIL,
    dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING,
    dcgm_fields.DCGM_FI_DEV_RETIRED_SBE,
    dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
    dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_TOTAL,
    dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
    dcgm_fields.DCGM_FI_DEV_ECC_SBE_AGG_TOTAL,
    dcgm_fields.DCGM_FI_DEV_ECC_DBE_AGG_TOTAL,
    dcgm_fields.DCGM_FI_DEV_FB_TOTAL,
    dcgm_fields.DCGM_FI_DEV_FB_FREE,
    dcgm_fields.DCGM_FI_DEV_FB_USED,
    dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
    dcgm_fields.DCGM_FI_DEV_POWER_VIOLATION,
    dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION,
    dcgm_fields.DCGM_FI_DEV_XID_ERRORS,
    dcgm_fields.DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
    dcgm_fields.DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_TOTAL,
    dcgm_fields.DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_TOTAL,
    dcgm_fields.DCGM_FI_DEV_MEM_CLOCK,
    dcgm_fields.DCGM_FI_DEV_MEMORY_TEMP,
    dcgm_fields.DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION,
    dcgm_fields.DCGM_FI_DEV_MEM_COPY_UTIL,
    dcgm_fields.DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_TOTAL,
    dcgm_fields.DCGM_FI_DEV_LWLINK_BANDWIDTH_TOTAL,
    dcgm_fields.DCGM_FI_DEV_PCIE_TX_THROUGHPUT,
    dcgm_fields.DCGM_FI_DEV_PCIE_RX_THROUGHPUT
    ]

class DcgmCollectdPlugin(DcgmReader):
    ###########################################################################
    def __init__(self):
        DcgmReader.__init__(self, fieldIds=g_publishFieldIds, ignoreList=g_dcgmIgnoreFields, fieldGroupName='collectd_plugin')

    ###########################################################################
    def LwstomDataHandler(self, fvs):
        value = collectd.Values(type='gauge')  # pylint: disable=no-member
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

                value.dispatch(type=fieldTag, type_instance=typeInstance, time=valTimeSec1970, values=valueArray, plugin=value.plugin)

                collectd.debug("gpuId %d, tag %s, value %s" % (gpuId, fieldTag, str(val.value)))  # pylint: disable=no-member

    ###########################################################################
    def LogInfo(self, msg):
        collectd.info(msg)  # pylint: disable=no-member

    ###########################################################################
    def LogError(self, msg):
        collectd.error(msg)  # pylint: disable=no-member

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

def register_collectd_callbacks():
    collectd.register_init(init_dcgm)  # pylint: disable=no-member
    collectd.register_read(read_dcgm)  # pylint: disable=no-member
    collectd.register_shutdown(shutdown_dcgm)  # pylint: disable=no-member

###############################################################################
##### Main
###############################################################################
g_dcgmCollectd = DcgmCollectdPlugin()
register_collectd_callbacks()
