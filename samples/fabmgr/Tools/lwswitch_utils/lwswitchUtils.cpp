#include <stdio.h>
#include <stdlib.h>
#include <commandline.h>
#include "lwswitchUtilsParser.h"
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>
#include "errno.h"

#include "lwos.h"
#include "logging.h"

#include "lwtypes.h"
#include "lwlink.h"
#include "ctrl/ctrl2080/ctrl2080lwlink.h"
#include "ctrl_dev_lwswitch.h"
#include "ioctl_lwswitch.h"
#include "ioctl_dev_lwswitch.h"

#include "ioctl_dev_internal_lwswitch.h"
#include "ioctl_lwswitch.h"
#include "ioctl_dev_lwswitch.h"
#include "ioctl_common_lwswitch.h"

#include "lr10/dev_route_ip.h"
#include "lr10/dev_ingress_ip.h"
#include "lr10/dev_egress_ip.h"
#include "lr10/dev_tstate_ip.h"
#include "lr10/dev_sourcetrack_ip.h"
#include "lr10/dev_nxbar_tc_global_ip.h"
#include "lr10/dev_nxbar_tile_ip.h"

extern "C"
{
#include "lwswitch_user_api.h"
}

// Globals
#define UUID_BUFFER_SIZE 80
lwswitch_device *gSwitchDevicePtr[LWSWITCH_MAX_DEVICES];
LWSWITCH_GET_DEVICES_V2_PARAMS gDeviceInfo;

struct all_args lwswitchUtilsArgs[] = {

        {
                LWSWITCH_UTILS_CMD_HELP,
                "-h",
                "--help",
                "\t\tDisplays help information",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                LWSWITCH_UTILS_CMD_LIST_LWSWITCH,
                "",
                "--list-all-lwswitches",
                "\t\tList all LWSwitches in the system",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                LWSWITCH_UTILS_CMD_GET_FATAL_ERROR_SCOPE,
                "",
                "--get_error_scope",
                "\t\tGet fatal error scope from all LWSwitches in the system",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                LWSWITCH_UTILS_CMD_INJECT_NON_FATAL_ERROR,
                "",
                "--inject-non-fatal-error",
                "\t\tInject non fatal error on a LWSwitch port",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                LWSWITCH_UTILS_CMD_INJECT_FATAL_DEVICE_ERROR,
                "",
                "--inject-fatal-device-error",
                "\t\tInject fatal error on a LWSwitch port, which requires device reset to recover",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                LWSWITCH_UTILS_CMD_INJECT_FATAL_PORT_ERROR,
                "",
                "--inject-fatal-port-error",
                "\t\tInject fatal error on a LWSwitch port, which requires port reset to recover",
                "\n\t",
                CMDLINE_OPTION_NO_VALUE_ALLOWED
        },
        {
                LWSWITCH_UTILS_CMD_SWITCH_INSTANCE,
                "",
                "--switch-instance",
                "\t\tLWSwitch instance",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
        {
                LWSWITCH_UTILS_CMD_SWITCH_INSTANCE,
                "",
                "--port-number",
                "\t\tLWSwitch port number",
                "\n\t",
                CMDLINE_OPTION_VALUE_REQUIRED
        },
};

/*****************************************************************************
 Method to Display Usage Info
 *****************************************************************************/
static void lwswitchUtilsUsage(void * pCmdLine) {
    printf("\n\n");
    printf("    Usage: lwswitchUtils [options]\n");
    printf("\n");
    printf("    Options include:\n");
    cmdline_printOptionsSummary(pCmdLine, 0);
    printf("\n");
    printf("    Please email lwdatools@lwpu.com with any questions,\n"
            "    bug reports, etc.");
    printf("\n\n");
    exit(0);
}

/*****************************************************************************
 Method to Display Help Message
 *****************************************************************************/
static void lwswitchUtilsDisplayHelpMessage(void * pCmdLine) {
    printf("\n");
    printf("    LWSwitch error generation utility. It only works with debug driver build.\n");

    lwswitchUtilsUsage(pCmdLine);
}

int cleanup(LwswitchUtilsCmdParser_t *pCmdParser, int status)
{
    (void)lwswitchUtilsCmdParserDestroy(pCmdParser);
    return status;
}

int checkDriverVersion(void)
{
    LWSWITCH_GET_DEVICES_V2_PARAMS *deviceInfo = &gDeviceInfo;

    LW_STATUS status = lwswitch_api_get_devices(deviceInfo);
    if (status != LW_OK) {
        if (status == LW_ERR_LIB_RM_VERSION_MISMATCH) {
            fprintf(stderr, "lwswitch_UTILS version is incompatible with LWSwitch driver. "
                    "Please update with matching LWPU driver package");
            exit(-1);
        }
        // all other errors, log the error code and bail out
        fprintf(stderr, "failed to query device information from LWSwitch driver, return status: %d\n",
                status);
        exit(-1);
    }

    if (deviceInfo->deviceCount <= 0) {
        fprintf(stderr, "no LWSwitches found\n");
        exit(-1);
    }

    return 0;
}

void listAllSwitches(void)
{
    char uuidBuf[UUID_BUFFER_SIZE];

    for (uint32_t i = 0; i < gDeviceInfo.deviceCount; i++) {
        LWSWITCH_DEVICE_INSTANCE_INFO_V2 &switchInfo = gDeviceInfo.info[i];
        memset(uuidBuf, 0, UUID_BUFFER_SIZE);
        lwswitch_uuid_to_string(&gDeviceInfo.info[i].uuid, uuidBuf, UUID_BUFFER_SIZE);

        printf("  Index: %d", switchInfo.deviceInstance);
        printf("  Instance: %d", switchInfo.deviceInstance);
        printf("  UUID: %s\n", uuidBuf);
        printf("  Physical Id:       %d\n", switchInfo.physId);
        printf("  PCI Bus ID:        %x:%x:%x:%x\n", switchInfo.pciDomain,  switchInfo.pciBus, switchInfo.pciDevice, switchInfo.pciFunction);
        printf("  Driver State:      %d\n", switchInfo.driverState);
        printf("  Device State:      %d\n", switchInfo.deviceState);
        printf("  Driver Reason:     %d\n\n", switchInfo.deviceReason);
    }
}

void openAllSwitches(void)
{
    for (uint32_t i = 0; i < gDeviceInfo.deviceCount; i++) {
        gSwitchDevicePtr[i] = NULL;

        if (gDeviceInfo.info[i].deviceReason != LWSWITCH_DEVICE_BLACKLIST_REASON_NONE) {
            // the lwswitch is degraded or excluded
            // open and ioctls would fail on degraded or excluded lwswitch
            continue;
        }

        char uuidBuf[UUID_BUFFER_SIZE];
        memset(uuidBuf, 0, UUID_BUFFER_SIZE);
        lwswitch_uuid_to_string(&gDeviceInfo.info[i].uuid, uuidBuf, UUID_BUFFER_SIZE);

        LW_STATUS retVal = lwswitch_api_create_device(&gDeviceInfo.info[i].uuid, &gSwitchDevicePtr[i]);
        if ( retVal != LW_OK ) {
            // the tool opens lwswitch at the beginning to get platform arch
            // this could fail because the switch is excluded or degraded
            fprintf(stderr, "cannot open handle to LWSwitch index: %d pci bus id: %d\n",
                    gDeviceInfo.info[i].deviceInstance, gDeviceInfo.info[i].pciBus);
            gSwitchDevicePtr[i] = NULL;
        }
    }
}

void closeAllSwitches(void)
{
    for (uint32_t i = 0; i < LWSWITCH_MAX_DEVICES; i++) {
        if (gSwitchDevicePtr[i] != NULL) {
            lwswitch_api_free_device(&gSwitchDevicePtr[i]);
            gSwitchDevicePtr[i] = NULL;
        }
    }
}

void reg_wr(lwswitch_device *switch_dev, LwU32 engine, LwU32 instance, LwU32 offset, LwU32 val)
{
    LWSWITCH_REGISTER_WRITE wr;
    LW_STATUS status;

    wr.engine = engine;
    wr.instance = instance;
    wr.offset = offset;
    wr.bcast = 0;
    wr.val = val;

    status = lwswitch_api_control(switch_dev, IOCTL_LWSWITCH_REGISTER_WRITE, &wr, sizeof(LWSWITCH_REGISTER_WRITE));
}

void reg_rd(lwswitch_device *switch_dev, LwU32 engine, LwU32 instance, LwU32 offset, LwU32 *val)
{
    LWSWITCH_REGISTER_READ rd;
    LW_STATUS status;

    rd.engine = engine;
    rd.instance = instance;
    rd.offset = offset;
    rd.val = 0;

    status = lwswitch_api_control(switch_dev, IOCTL_LWSWITCH_REGISTER_READ, &rd, sizeof(LWSWITCH_REGISTER_READ));
    *val = rd.val;
}

void getSwitchFatalErrorScope(uint32_t switchInstance)
{
    if (switchInstance >= gDeviceInfo.deviceCount) {
        fprintf(stderr, "Invalid switch instance %d.\n", switchInstance);
        return;
    }

    lwswitch_device *pSwitchDev = gSwitchDevicePtr[switchInstance];
    if (!pSwitchDev) {
        fprintf(stderr, "Null switch instance %d.\n", switchInstance);
        return;
    }

    // get device and port reset status

    LWSWITCH_GET_FATAL_ERROR_SCOPE_PARAMS getFatalErrorScopeParams;
    memset( &getFatalErrorScopeParams, 0, sizeof(LWSWITCH_GET_FATAL_ERROR_SCOPE_PARAMS) );

    LW_STATUS status =  lwswitch_api_control(pSwitchDev,
                                             IOCTL_LWSWITCH_GET_FATAL_ERROR_SCOPE,
                                             &getFatalErrorScopeParams,
                                             sizeof(LWSWITCH_GET_FATAL_ERROR_SCOPE_PARAMS));
    if (status == LW_OK) {
        printf("switch instance %d: device %d port reset: ", switchInstance, getFatalErrorScopeParams.device);
        for (uint32_t port = 0; port < LWSWITCH_MAX_PORTS; port++) {
            if (getFatalErrorScopeParams.port[port]) {
                printf(" %d ", port);
            }
        }
        printf("\n");
    } else {
        fprintf(stderr, "failed to get LWSwitch fatal error scope with error %d\n", status);
    }
}

void getAllSwitchFatalErrorScope(void)
{
    for (uint32_t i = 0; i < LWSWITCH_MAX_DEVICES; i++) {
        if (gSwitchDevicePtr[i] != NULL) {
            getSwitchFatalErrorScope(i);
        }
    }
}

void injectSwitchError(uint32_t switchInstance, uint32_t portNum,
                       LWSwitch_err_block_t errBlock, bool fatalError)
{
    if (switchInstance >= gDeviceInfo.deviceCount) {
        fprintf(stderr, "Invalid switch instance %d.\n", switchInstance);
        return;
    }

    lwswitch_device *pSwitchDev = gSwitchDevicePtr[switchInstance];
    if (!pSwitchDev) {
        fprintf(stderr, "Null switch instance %d.\n", switchInstance);
        return;
    }

    switch (errBlock) {
    case NPORT_ROUTE:
        if (fatalError) {
            reg_wr(pSwitchDev, REGISTER_RW_ENGINE_NPORT, portNum,
                   LW_ROUTE_ERR_REPORT_INJECT_0,
                   DRF_NUM(_ROUTE, _ERR_REPORT_INJECT_0, _LWS_ECC_DBE_ERR, 1));
        } else {
            reg_wr(pSwitchDev, REGISTER_RW_ENGINE_NPORT, portNum,
                   LW_ROUTE_ERR_REPORT_INJECT_0,
                   DRF_NUM(_ROUTE, _ERR_REPORT_INJECT_0, _LWS_ECC_LIMIT_ERR, 1));
        }
        break;

    case NPORT_INGRESS:
        if (fatalError) {
            reg_wr(pSwitchDev, REGISTER_RW_ENGINE_NPORT, portNum,
                   LW_INGRESS_ERR_REPORT_INJECT_0,
                   DRF_NUM(_INGRESS, _ERR_REPORT_INJECT_0, _CMDDECODEERR, 1));
        } else {
            reg_wr(pSwitchDev, REGISTER_RW_ENGINE_NPORT, portNum,
                   LW_INGRESS_ERR_REPORT_INJECT_0,
                   DRF_NUM(_INGRESS, _ERR_REPORT_INJECT_0, _ACLFAIL, 1));
        }
        break;

    case NPORT_EGRESS:
        if (fatalError) {
            reg_wr(pSwitchDev, REGISTER_RW_ENGINE_NPORT, portNum,
                   LW_EGRESS_ERR_REPORT_INJECT_0,
                   DRF_NUM(_EGRESS, _ERR_REPORT_INJECT_0, _EGRESSBUFERR, 1));
        } else {
            reg_wr(pSwitchDev, REGISTER_RW_ENGINE_NPORT, portNum,
                   LW_EGRESS_ERR_REPORT_INJECT_0,
                   DRF_NUM(_EGRESS, _ERR_REPORT_INJECT_0, _PRIVRSPERR, 1));
        }
        break;

    case NPORT_TSTATE:
        if (fatalError) {
            reg_wr(pSwitchDev, REGISTER_RW_ENGINE_NPORT, portNum,
                   LW_TSTATE_ERR_REPORT_INJECT_0,
                   DRF_NUM(_TSTATE, _ERR_REPORT_INJECT_0, _TAGPOOL_ECC_DBE_ERR, 1));
        } else {
            reg_wr(pSwitchDev, REGISTER_RW_ENGINE_NPORT, portNum,
                   LW_TSTATE_ERR_REPORT_INJECT_0,
                   DRF_NUM(_TSTATE, _ERR_REPORT_INJECT_0, _TAGPOOL_ECC_LIMIT_ERR, 1));
        }
        break;

    case NPORT_SOURCETRACK:
        if (fatalError) {
            reg_wr(pSwitchDev, REGISTER_RW_ENGINE_NPORT, portNum,
                   LW_SOURCETRACK_ERR_REPORT_INJECT_0,
                   DRF_NUM(_SOURCETRACK, _ERR_REPORT_INJECT_0, _SOURCETRACK_TIME_OUT_ERR, 1));
        } else {
            reg_wr(pSwitchDev, REGISTER_RW_ENGINE_NPORT, portNum,
                   LW_SOURCETRACK_ERR_REPORT_INJECT_0,
                   DRF_NUM(_SOURCETRACK, _ERR_REPORT_INJECT_0, _CREQ_TCEN0_CRUMBSTORE_ECC_LIMIT_ERR, 1));
        }
        break;

    default:
        fprintf(stderr, "Invalid error block %d.\n", switchInstance);
        break;
    }
}

void injectDeviceResetScopeFatalError(uint32_t switchInstance, uint32_t portNum)
{
    injectSwitchError(switchInstance, portNum, NPORT_EGRESS, true);
}

void injectPortResetScopeFatalError(uint32_t switchInstance, uint32_t portNum)
{
    injectSwitchError(switchInstance, portNum, NPORT_INGRESS, true);
}

void injectNonFatalError(uint32_t switchInstance, uint32_t portNum)
{
    injectSwitchError(switchInstance, portNum, NPORT_ROUTE, false);
}

int main(int argc, char **argv)
{
    FMIntReturn_t ret = FM_INT_ST_OK;
    LwswitchUtilsCmdParser_t *pCmdParser;

    pCmdParser = lwswitchUtilsCmdParserInit(argc, argv, lwswitchUtilsArgs,
                                             LWSWITCH_UTILS_CMD_COUNT, lwswitchUtilsUsage,
                                             lwswitchUtilsDisplayHelpMessage);
    if (NULL == pCmdParser) {
        return FM_INT_ST_BADPARAM;
    }

    ret = lwswitchUtilsCmdProcessing(pCmdParser);
    if (FM_INT_ST_OK != ret) {
        if (ret == FM_INT_ST_BADPARAM){
            fprintf(stderr, "Unable to process command: bad command line parameter. \n");
        } else {
            fprintf(stderr, "Unable to process command: generic error. \n");
        }
        lwswitchUtilsCmdParserDestroy(pCmdParser);
        return ret;
    }

    checkDriverVersion();

    if (pCmdParser->mListSwitches) {
        listAllSwitches();
        return ret;
    }

    openAllSwitches();

    if (pCmdParser->mGetSwitchErrorScope) {
        getAllSwitchFatalErrorScope();
    }

    if (pCmdParser->mInjectSwitchError) {

        if (pCmdParser->mFatalScope == NON_FATAL) {
            injectNonFatalError(pCmdParser->mSwitchDevInstance, pCmdParser->mSwitchPortNum);
        } else if (pCmdParser->mFatalScope == FATAL_DEVICE) {
            injectDeviceResetScopeFatalError(pCmdParser->mSwitchDevInstance, pCmdParser->mSwitchPortNum);
        } else if (pCmdParser->mFatalScope == FATAL_PORT) {
            injectPortResetScopeFatalError(pCmdParser->mSwitchDevInstance, pCmdParser->mSwitchPortNum);
        }
    }

    closeAllSwitches();
    cleanup(pCmdParser, 0);
    return ret;
}

