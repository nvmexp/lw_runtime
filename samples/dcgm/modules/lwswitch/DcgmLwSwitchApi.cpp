

extern "C" {
#include <stdio.h>
#include <stdint.h>
#include "dcgm_lwswitch_internal.h"
#include "lwcm_util.h"
#include "lwcmvalue.h"
#include "logging.h"
}

// Wrap each dcgmFunction with apiEnter and apiExit
#define DCGM_ENTRY_POINT(dcgmFuncname, tsapiFuncname, argtypes, fmt, ...)             \
        static dcgmReturn_t tsapiFuncname argtypes ;                                          \
        dcgmReturn_t DECLDIR dcgmFuncname argtypes                                            \
        {                                                                                     \
            dcgmReturn_t result;                                                              \
            result = apiEnter();                                                              \
            if (result != DCGM_ST_OK)                                                         \
            {                                                                                 \
                return result;                                                                \
            }                                                                                 \
            result = tsapiFuncname(__VA_ARGS__);                                              \
            apiExit();                                                                        \
            return result;                                                                    \
        }

#define DCGM_INT_ENTRY_POINT(dcgmFuncname, tsapiFuncname, argtypes, fmt, ...)         \
        static dcgmReturn_t tsapiFuncname argtypes ;                                  \
        static dcgmReturn_t dcgmFuncname argtypes                                     \
        {                                                                             \
            dcgmReturn_t result;                                                      \
            result = apiEnter();                                                      \
            if (result != DCGM_ST_OK)                                                 \
            {                                                                         \
                return result;                                                        \
            }                                                                         \
            result = tsapiFuncname(__VA_ARGS__);                                      \
            apiExit();                                                                \
            return result;                                                            \
        }

extern "C" {
#include "dcgm_lwswitch_entry_point.h"
}

#include "DcgmModuleApi.h"
#include "dcgm_lwswitch_structs.h"

// Instructions:
//
// - Try to make Export Tables backward binary compatible
// - Number all internal functions. Otherwise it's hard to make integrations properly
// - Don't remove rows. Deprecate old functions by putting NULL instead
// - When you do integrations make sure to pad missing functions with NULLs
// - Never renumber functions when integrating. Numbers of functions should always match the
//   module_* numbering
DCGM_INIT_EXTERN_CONST etblDCGMLwSwitchInternal g_etblDCGMLwSwitchInternal =
{
        sizeof (g_etblDCGMLwSwitchInternal),
        dcgmLwSwitchStart,                // 1
        dcgmLwSwitchShutdown,             // 2
};


/*****************************************************************************/
dcgmReturn_t tsapiLwSwitchStart(dcgmHandle_t pDcgmHandle, dcgm_lwswitch_msg_start_t *startMsg)
{
    dcgmReturn_t dcgmReturn;

    if(startMsg->header.version != dcgm_lwswitch_msg_start_version)
    {
        PRINT_ERROR("", "dcgm_lwswitch_msg_start_version version mismatch");
        return DCGM_ST_VER_MISMATCH;
    }

    startMsg->header.length = sizeof(*startMsg);
    startMsg->header.moduleId = DcgmModuleIdLwSwitch;
    startMsg->header.subCommand = DCGM_LWSWITCH_SR_START;

    dcgmReturn = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &startMsg->header);

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t tsapiLwSwitchShutdown(dcgmHandle_t pDcgmHandle, dcgm_lwswitch_msg_shutdown_t *shutdownMsg)
{
    dcgmReturn_t dcgmReturn;

    if(shutdownMsg->header.version != dcgm_lwswitch_msg_shutdown_version)
    {
        PRINT_ERROR("", "dcgm_lwswitch_shutdown_version version mismatch");
        return DCGM_ST_VER_MISMATCH;
    }

    shutdownMsg->header.length = sizeof(*shutdownMsg);
    shutdownMsg->header.moduleId = DcgmModuleIdLwSwitch;
    shutdownMsg->header.subCommand = DCGM_LWSWITCH_SR_SHUTDOWN;

    dcgmReturn = dcgmModuleSendBlockingFixedRequest(pDcgmHandle, &shutdownMsg->header);

    return dcgmReturn;
}

/*****************************************************************************/
