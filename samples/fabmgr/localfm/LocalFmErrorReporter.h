/*
 *  Copyright 2018-2022 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#pragma once
 
#include "fabricmanager.pb.h"
#include "FMErrorCodesInternal.h"
#include "FMCommonTypes.h"
#include "LocalFMGpuMgr.h"
 
extern "C"
{
    #include "lwswitch_user_api.h"
}
 
/*****************************************************************************/
/*  Local Fabric Manager errors                                              */
/*  Act as an interface to read error report from LWSwitch driver, RM  and   */
/*  report the same to Global FM.                                            */
/*  Global FM can on demand request error from Local FM and the              */
/*  corresponding GPB message handler uses this class's methods to retrieve  */
/*  the current information.                                           */
/*****************************************************************************/
 
class LocalFMErrorReporter
{
    friend class LocalFMGpuMgr;

public:
    LocalFMErrorReporter(LocalFabricManagerControl *pLfm,
                         LocalFMLWLinkDevRepo *linkDevRepo);

    ~LocalFMErrorReporter();

    void processLWSwitchFatalErrorEvent(FMUuid_t &switchUuid);
    void processLWSwitchNonFatalErrorEvent(FMUuid_t &switchUuid);
    void reportSwitchFatalErrorScope(FMUuid_t &switchUuid);

private:
    
    void checkForGpuLWLinkError(void);
    static void gpuErrorCallbackWrapper(void *cbArgs);
    void reportGpuLWLinkRecoveryError(FMUuid_t gpuUuid);
    void reportGpuLWLinkFatalError(FMUuid_t gpuUuid, uint32_t errorLinkIndex);
    void reportSwitchLWLinkRecoveryError(uint32_t physicalId,
                                         LWSWITCH_ERROR &switchError);
    void reportSwitchErrors(FMUuid_t &switchUuid, LWSWITCH_ERROR_SEVERITY_TYPE errorType);

    void getSwitchErrors(FMUuid_t &switchUuid,
                         LWSWITCH_ERROR_SEVERITY_TYPE errorType,
                         std::queue < SwitchError_struct * > *errQ);

    lwswitch::fmMessage* buildSwitchErrorMsg(LWSWITCH_ERROR_SEVERITY_TYPE errorType,
                                             std::queue < SwitchError_struct * > *errQ );

    void reportGpuErrorInfo(FMUuid_t gpuUuid, uint32_t errorLinkIndex, int errorCode);

    LocalFabricManagerControl *mpLfm;
    LocalFMLWLinkDevRepo *mLWLinkDevRepo;
};

