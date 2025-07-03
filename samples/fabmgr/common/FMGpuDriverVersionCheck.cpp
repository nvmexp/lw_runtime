
#include "lwRmApi.h"
#include "lwos.h"
#include "ctrl0000.h"
#include "ctrl0000gpu.h"

#include "fm_log.h"
#include "FMGpuDriverVersionCheck.h"

//
//
// Since Fabric Manager issue direct IOCTLs to the Drivers (RM, LWSwitch Driver, LWLinkCoreLib),
// it must maintain the application binary interface (ABI) compatibility with the Driver. Also
// FM and Driver is built from same change list (CL) and they go in pairs. This means, typically
// when updating the TRD Driver, Fabric Manager needs to be updated to the matching package
// as well and vice versa.
//
// However, certain customer wants to update only the Fabric Manager in order to get a very
// targeted FM only fix and don't update their Driver (due to longer driver qualification time).
// Due to the above Fabric Manager and Driver dependency, this Fabric Manager only update
// request will be evaluated on a case by case and validated against the Driver version
// requested to white list/support.
//
//
// Fill the Driver version string to be whitelisted in the below array.
//
// This is common code as this information will be used by
//  1) Fabric Manager
//  2) LWSwitch Audit Tool
//

// this value can be modified if required when the whitelisted of versions are added
// not defining a certain size for an array will result in compilation issues
static const int NUM_WHITE_LISTED_VERSIONS = 1;

FMGpuDriverVersionCheck::RMLibVersionInfo_t FMGpuDriverVersionCheck::mGpuDrvWhitelistedVersions[NUM_WHITE_LISTED_VERSIONS] =
{
    // add version string like { "445.45" },  { "450.50" }, { "450.44.01" }
};

#define UINT2PTR(v)((void*)(uintptr_t)(v))

FMGpuDriverVersionCheck::FMGpuDriverVersionCheck()
{
    mRmClientHandle = 0;
}

FMGpuDriverVersionCheck::~FMGpuDriverVersionCheck()
{
    // do nothing
}

/******************************************************************************************
 Allocate an RM handle and validate the version. If the version doesn’t match or not
 whitelisted, then this function will throw an exception. The RM handle allocated
 will be closed after that.
******************************************************************************************/
void
FMGpuDriverVersionCheck::checkGpuDriverVersionCompatibility(std::string errorCtx)
{
    // first allocate RM device handle
    openGpuDriverHandle();

    // this can throw exception, catch and close our handle.
    try {
        checkRMLibVersionCompatibility(errorCtx);
    } catch (const std::runtime_error &e) {
        // close the handle
        closeGpuDriverHandle();
        // pass the exception to called
        throw std::runtime_error(e.what());
    }

    // normal case, close our handle once the version is validated.
    closeGpuDriverHandle();
}

void
FMGpuDriverVersionCheck::checkRMLibVersionCompatibility(std::string errorCtx)
{
    // MODS GDM build does not have access to driver
#if defined(LW_MODS_GDM_BUILD)
    return;
#else
    LwU32 rmResult;
    LW0000_CTRL_SYSTEM_GET_BUILD_VERSION_PARAMS rmVersionParams = { 0 };
    char pDriverVersionBuffer[RM_VERSION_STRING_SIZE] = {0};
    char pVersionBuffer[RM_VERSION_STRING_SIZE] = {0};
    char pTitleBuffer[RM_VERSION_STRING_SIZE] = {0};

    rmVersionParams.sizeOfStrings = RM_VERSION_STRING_SIZE;
    rmVersionParams.pVersionBuffer = LW_PTR_TO_LwP64(pVersionBuffer);
    rmVersionParams.pTitleBuffer = LW_PTR_TO_LwP64(pTitleBuffer);
    rmVersionParams.pDriverVersionBuffer = LW_PTR_TO_LwP64(pDriverVersionBuffer);

    // first get the lwrrently loaded RM driver version
    rmResult = LwRmControl(mRmClientHandle, mRmClientHandle, LW0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION,
                           &rmVersionParams, sizeof(rmVersionParams));
    if (LWOS_STATUS_SUCCESS != rmResult) {
        // treat this as not supported version
        std::ostringstream ss;
        ss << "getting LWPU GPU driver version information failed with error " << lwstatusToString(rmResult);
        FM_LOG_ERROR("%s", ss.str().c_str());
        FM_SYSLOG_ERR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    // parse major/minor of RM/RMLib version
    const char* rmDriverVersionString = (const char*)(UINT2PTR(rmVersionParams.pDriverVersionBuffer));

    // get the RMLib version with which this FM instance is compiled/linked
    RMLibVersionInfo_t fmRMLibVersionInfo = {{0}};
    strncpy(fmRMLibVersionInfo.version, (const char*)LW_VERSION_STRING, RM_VERSION_STRING_SIZE);

    // compare the RM version with our compiled/linked RMLib version
    if (strncmp(rmDriverVersionString, fmRMLibVersionInfo.version, RM_VERSION_STRING_SIZE) == 0) {
        // version matched exactly, so we can continue
        return;
    }

    // no exact version match, check our whitelist for compatible driver version
    for (int idx=0; idx<NUM_WHITE_LISTED_VERSIONS; idx++) {
        RMLibVersionInfo_t tempVersion = mGpuDrvWhitelistedVersions[idx];
        if (strncmp(rmDriverVersionString, tempVersion.version, RM_VERSION_STRING_SIZE) == 0) {
            // whitelisted version match.
            std::ostringstream ss;
            ss << errorCtx << " LWPU GPU driver interface version is " << fmRMLibVersionInfo.version
               << " and driver version is " << rmDriverVersionString
               << ". Continuing as the driver version is whitelisted for compatibility";
            FM_LOG_INFO("%s", ss.str().c_str());
            return;
        }
    }

    // no version match, we can't continue
    std::ostringstream ss;
    ss << errorCtx << " LWPU GPU driver interface version " << fmRMLibVersionInfo.version
       << " don't match with driver version " << rmDriverVersionString
       <<". Please update with matching LWPU driver package.";

    FM_LOG_ERROR("%s", ss.str().c_str());
    FM_SYSLOG_ERR("%s", ss.str().c_str());
    throw std::runtime_error(ss.str());
#endif // LW_MODS_GDM_BUILD
}

void
FMGpuDriverVersionCheck::openGpuDriverHandle(void)
{
    // MODS GDM build does not have access to driver
#if defined(LW_MODS_GDM_BUILD)
    return;
#else
    LwU32 rmResult;

    // allocate an rmclient object
    rmResult = LwRmAllocRoot(&mRmClientHandle);
    if (LWOS_STATUS_SUCCESS != rmResult) {
        FM_LOG_CRITICAL("failed to allocate handle (client) to LWPU GPU driver with error:%s",
                        lwstatusToString(rmResult));
        // throw error based on common return values
        switch (rmResult) {
            case LW_ERR_OPERATING_SYSTEM: {
                throw std::runtime_error("failed to allocate handle (client) to LWPU GPU driver. "
                                         "Make sure that the LWPU driver is installed and running");
                break;
            }
            case LW_ERR_INSUFFICIENT_PERMISSIONS: {
                throw std::runtime_error("failed to allocate handle (client) to LWPU GPU driver. "
                                         "Make sure that the current user has permission to access device files");
                break;
            }
            default: {
                throw std::runtime_error("failed to allocate handle (client) to LWPU GPU driver");
                break;
            }
        } // end of switch
    }
#endif // LW_MODS_GDM_BUILD
}

void
FMGpuDriverVersionCheck::closeGpuDriverHandle(void)
{
    // MODS GDM build does not have access to driver
#if defined(LW_MODS_GDM_BUILD)
    return;
#else
    LwRmFree(mRmClientHandle, LW01_NULL_OBJECT, mRmClientHandle);
    mRmClientHandle = 0;
#endif // LW_MODS_GDM_BUILD
}

