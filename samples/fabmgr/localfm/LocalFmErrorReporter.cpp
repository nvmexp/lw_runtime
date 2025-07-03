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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/poll.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <map>
#include <functional>
 
#include "FMErrorCodesInternal.h"
#include "LocalFmErrorReporter.h"
#include "FmServerConnection.h"
#include "fabricmanager.pb.h"
#include "FMErrorCodesInternal.h"
#include "fm_log.h"
#include "LocalFMGpuMgr.h"
 
#include "ctrl0000gpu.h"
#include "ctrl2080lwlink.h"
#include "class/cl2080.h"


/*****************************************************************************/

LocalFMErrorReporter::LocalFMErrorReporter(LocalFabricManagerControl *pLfm,
                                           LocalFMLWLinkDevRepo *linkDevRepo)
{
    FMIntReturn_t retVal;
 
    mLWLinkDevRepo = linkDevRepo;
    mpLfm = pLfm;

    if ((mpLfm->getFabricMode() != FM_MODE_SHARED_LWSWITCH))
    {
        retVal = mpLfm->mFMGpuMgr->subscribeForGpuEvents(LW2080_NOTIFIERS_LWLINK_ERROR_FATAL,
                                                         &gpuErrorCallbackWrapper, (void*)this);
        if ( retVal != FM_INT_ST_OK ) 
        {
            std::ostringstream ss;
            ss << "failed to register watch requests for GPU non-fatal LWLink errors";
            FM_LOG_ERROR("%s", ss.str().c_str());
            throw std::runtime_error(ss.str());
        } 
 
        retVal = mpLfm->mFMGpuMgr->subscribeForGpuEvents(LW2080_NOTIFIERS_LWLINK_ERROR_RECOVERY_REQUIRED,
                                                         &gpuErrorCallbackWrapper, (void*)this);
        if ( retVal != FM_INT_ST_OK )
        {
            std::ostringstream ss;
            ss << "failed to register watch requests for GPU fatal LWLink errors";
            FM_LOG_ERROR("%s", ss.str().c_str());
            throw std::runtime_error(ss.str());
        }
    }
};
 
LocalFMErrorReporter::~LocalFMErrorReporter()
{
    // unregister GPU error polling requests 
    if (mpLfm->getFabricMode() != FM_MODE_SHARED_LWSWITCH)
    {
        mpLfm->mFMGpuMgr->unsubscribeGpuEvents(LW2080_NOTIFIERS_LWLINK_ERROR_FATAL,
                                               &gpuErrorCallbackWrapper, (void*)this);
        mpLfm->mFMGpuMgr->unsubscribeGpuEvents(LW2080_NOTIFIERS_LWLINK_ERROR_RECOVERY_REQUIRED,
                                               &gpuErrorCallbackWrapper, (void*)this);
    }
}

void
LocalFMErrorReporter::processLWSwitchFatalErrorEvent(FMUuid_t &switchUuid)
{
    // report fatal errors to GFM
    reportSwitchErrors(switchUuid, LWSWITCH_ERROR_SEVERITY_FATAL);
    
    // report fatal error scope to GFM
    reportSwitchFatalErrorScope(switchUuid);
}

void
LocalFMErrorReporter::processLWSwitchNonFatalErrorEvent(FMUuid_t &switchUuid)
{
    // report non-fatal errors to GFM
    reportSwitchErrors(switchUuid, LWSWITCH_ERROR_SEVERITY_NONFATAL);
}

/*
 * Send switch error asynchronously to GFM
 */
void
LocalFMErrorReporter::reportSwitchErrors(FMUuid_t &switchUuid, LWSWITCH_ERROR_SEVERITY_TYPE errorType)
{
    std::queue < SwitchError_struct * > errQ;
    lwswitch::fmMessage   *pFmMessage;
    FMIntReturn_t          ret;
 
    getSwitchErrors( switchUuid, errorType, &errQ );
    if ( errQ.size() == 0 ) return;
 
    pFmMessage = buildSwitchErrorMsg( errorType, &errQ );
    if ( !pFmMessage ) return;
 
    // send non fatal errors to global fabric manager
    ret = mpLfm->SendMessageToGfm( pFmMessage, true );
    if ( ret != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("request to send LWSwitch error message to fabric manager failed with error %d", ret);
    }
 
    if ( pFmMessage )  delete pFmMessage;
}

/*
 * Get switch errors from the switch interface
 *
 * Allocate SwitchError_structs on the errQ
 * The caller of the function should free the memory
 */
void
LocalFMErrorReporter::getSwitchErrors(FMUuid_t &switchUuid,
                                      LWSWITCH_ERROR_SEVERITY_TYPE errorType,
                                      std::queue < SwitchError_struct * > *errQ)
{
    LocalFMSwitchInterface *pInterface = mpLfm->switchInterfaceAt( switchUuid );
 
    if ( !pInterface || !errQ )
    {
        FM_LOG_ERROR("error reporting: failed to get LWSwitch driver interface object for uuid %s",
                      switchUuid.bytes); 
        return;
    }

    if ( mpLfm->isSwitchDegraded(pInterface->getSwitchPhysicalId()) )
    {
        FM_LOG_INFO("not reading errors for LWSwitch physical id %d as it is degraded.",
                    pInterface->getSwitchPhysicalId());
        return;
    }

    uint32_t physicalId = pInterface->getSwitchPhysicalId();
    LWSWITCH_GET_ERRORS_PARAMS ioctlParams;
    uint32_t i;

    do {
        if ( pInterface->getSwitchErrors(errorType, ioctlParams) != FM_INT_ST_OK )
        {
            // error is already logged
            return;
        }

        for ( i = 0; i < ioctlParams.errorCount; i++)
        {
            // LWLink Recovery errors are reported as non-fatal and requires
            // some treatment at GlobalFM. So report the additional data as
            // seperate recovery message. But continue reporting the switch error
            // to GlobalFM, so that the same will be published into cache manager
            if (ioctlParams.error[i].error_value == LWSWITCH_ERR_HW_DLPL_TX_RECOVERY_LONG)
            {
                reportSwitchLWLinkRecoveryError( physicalId, ioctlParams.error[i] );
            }
            SwitchError_struct *switchErr = new SwitchError_struct;
            switchErr->switchPhysicalId = physicalId;
            memcpy( &(switchErr->switchError), &(ioctlParams.error[i]), sizeof(LWSWITCH_ERROR) );
            errQ->push( switchErr );
        }
    } while (ioctlParams.errorCount != 0);
}

/*
 * build switch error message to global fabric manager
 * Caller need to free the FM message
 */
lwswitch::fmMessage*
LocalFMErrorReporter::buildSwitchErrorMsg(LWSWITCH_ERROR_SEVERITY_TYPE errorType,
                                          std::queue < SwitchError_struct * > *errQ)
{
    if ( !errQ )
        return NULL;
 
    SwitchError_struct          *swErr;
    lwswitch::switchError       *lwSwithErr = NULL;
    lwswitch::switchErrorInfo   *info = NULL;
 
    lwswitch::switchErrorReport *report = new lwswitch::switchErrorReport();
 
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
 
    if ( errorType == LWSWITCH_ERROR_SEVERITY_FATAL ) {
        pFmMessage->set_type( lwswitch::FM_LWSWITCH_ERROR_FATAL );
    } else {
        pFmMessage->set_type( lwswitch::FM_LWSWITCH_ERROR_NON_FATAL );
    }
 
    pFmMessage->set_allocated_errorreport( report );
 
    while ( !errQ->empty() ) {
 
        swErr = errQ->front();
 
        lwSwithErr = report->add_switcherror();
        lwSwithErr->set_switchphysicalid( swErr->switchPhysicalId );
 
        info = lwSwithErr->add_errorinfo();
        info->set_errorvalue( swErr->switchError.error_value );
        if ( errorType == LWSWITCH_ERROR_SEVERITY_FATAL ){
            info->set_severity(lwswitch::LWSWITCH_ERROR_SEVERITY_FATAL);
        } else {
            info->set_severity(lwswitch::LWSWITCH_ERROR_SEVERITY_NONFATAL);
        }
        info->set_errorsrc( (enum lwswitch::switchErrorSrc) swErr->switchError.error_src );
        info->set_instance( swErr->switchError.instance );
        info->set_subinstance( swErr->switchError.subinstance );
        info->set_time( swErr->switchError.time );
        info->set_resolved( (bool)swErr->switchError.error_resolved );
 
        errQ->pop();
        delete swErr;
    }
 
    return pFmMessage;
}

void
LocalFMErrorReporter::gpuErrorCallbackWrapper(void *args) 
{
    fmSubscriberCbArguments_t *errorArgs = (fmSubscriberCbArguments_t*) args;
    LocalFMErrorReporter *fmsr = (LocalFMErrorReporter*) errorArgs->subscriberCtx;
    fmGpuErrorInfoCtx_t *cbFmGpuErrorArgs = (fmGpuErrorInfoCtx_t*) errorArgs->args;
    FMUuid_t gpuUuid = cbFmGpuErrorArgs->gpuUuid;
    int gpuError = cbFmGpuErrorArgs->gpuError;
    uint64_t errorLinkIndex = cbFmGpuErrorArgs->errorLinkIndex;
    fmsr->reportGpuErrorInfo(gpuUuid, errorLinkIndex, gpuError);
}

void 
LocalFMErrorReporter::reportGpuErrorInfo(FMUuid_t gpuUuid, uint32_t errorLinkIndex, int errorCode)
{
    switch(errorCode) {
        case LW2080_NOTIFIERS_LWLINK_ERROR_FATAL:
            reportGpuLWLinkFatalError(gpuUuid, errorLinkIndex);
            break;
        case LW2080_NOTIFIERS_LWLINK_ERROR_RECOVERY_REQUIRED:
            reportGpuLWLinkRecoveryError(gpuUuid);
            break;
        default:
            FM_LOG_ERROR("error reporting: received unknown GPU LWLink error information from GPU Driver");
            break;
    }
}

void
LocalFMErrorReporter::reportGpuLWLinkRecoveryError(FMUuid_t gpuUuid)
{
    uint64 deviceID;
    FMIntReturn_t fmResult;
    lwswitch::lwlinkErrorRecoveryMsg *recoveryMsg;
    unsigned int activeMask = 0;
    unsigned int statusMask = 0;
    FMPciInfo_t fmGpuPciInfo;
    fmResult = mpLfm->mFMGpuMgr->getAllLwLinkStatus(gpuUuid, statusMask, activeMask);
 
    if (fmResult != FM_INT_ST_OK) {
        FM_LOG_WARNING("error reporting: failed to get GPU LWLink status information from GPU Driver");
        return;
    }

    recoveryMsg = new lwswitch::lwlinkErrorRecoveryMsg();
    int rmlink = 0;
    for (int linkidx = 0; linkidx < MAX_LWLINKS_PER_GPU; linkidx++) {
        if (!(statusMask & (1U << linkidx))) {
            continue;
        }
 
        if (!(activeMask & (1U) << linkidx)) {
            recoveryMsg->add_linkindex(rmlink);
        }
        rmlink++;
    }
    
    mpLfm->getGpuPciInfo(gpuUuid, fmGpuPciInfo);
 
    //get Device ID
    lwlink_pci_dev_info pci_info;
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::lwlinkErrorMsg *errorMsg = new lwswitch::lwlinkErrorMsg();
 
    pci_info.domain = fmGpuPciInfo.domain;
    pci_info.bus = fmGpuPciInfo.bus;
    pci_info.device = fmGpuPciInfo.device;
    pci_info.function = fmGpuPciInfo.function;    
 
    deviceID = mLWLinkDevRepo->getDeviceId( pci_info );
 
    recoveryMsg->set_gpuorswitchid(deviceID);
 
    errorMsg->set_allocated_recoverymsg( recoveryMsg );
 
    pFmMessage->set_type( lwswitch::FM_LWLINK_ERROR_GPU_RECOVERY );
    pFmMessage->set_allocated_lwlinkerrormsg( errorMsg );
 
    fmResult = mpLfm->SendMessageToGfm( pFmMessage, true );
 
    if ( fmResult != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("request to send GPU LWLink recovery error message to fabric manager failed with error %d", 
                     fmResult);
    }
 
    delete pFmMessage;
}

void
LocalFMErrorReporter::reportGpuLWLinkFatalError(FMUuid_t gpuUuid, uint32_t errorLinkIndex)
{
    uint64 deviceID;
    FMIntReturn_t fmResult = FM_INT_ST_OK;
    lwlink_pci_dev_info pci_info;
    FMPciInfo_t fmGpuPciInfo;
 
    mpLfm->getGpuPciInfo(gpuUuid, fmGpuPciInfo);
 
    pci_info.domain = fmGpuPciInfo.domain;
    pci_info.bus = fmGpuPciInfo.bus;
    pci_info.device = fmGpuPciInfo.device;
    pci_info.function = fmGpuPciInfo.function;    
 
    deviceID = mLWLinkDevRepo->getDeviceId( pci_info );
    lwswitch::lwlinkErrorGpuFatalMsg *fatalMsg = new lwswitch::lwlinkErrorGpuFatalMsg();
    fatalMsg->set_gpuorswitchid( deviceID );

    // TODO fill in the link that are in error when it is available from the RM event notification
    // fatalMsg->set_linkindex(errorLinkIndex);

    // fill the lwlink error message    
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::lwlinkErrorMsg *errorMsg = new lwswitch::lwlinkErrorMsg();
    errorMsg->set_allocated_gpufatalmsg( fatalMsg );
 
    // fill the outer FM Message
    pFmMessage->set_type( lwswitch::FM_LWLINK_ERROR_GPU_FATAL );
    pFmMessage->set_allocated_lwlinkerrormsg( errorMsg );
 
    fmResult = mpLfm->SendMessageToGfm( pFmMessage, true );
 
    if ( fmResult != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("request to send GPU LWLink fatal error message to fabric manager failed with error %d", 
                     fmResult);
    }
 
    delete pFmMessage;
}
 
void
LocalFMErrorReporter::reportSwitchLWLinkRecoveryError(uint32_t physicalId,
                                                      LWSWITCH_ERROR &switchError)
{
    int ret;
 
    // report the device and link id information to GFM. Based on the
    // that, GFM will retrain the specific connection
 
    // get the device id information using PCI information
    LocalFMSwitchInterface *pInterface = mpLfm->switchInterfaceAt( physicalId );
    if ( !pInterface )
    {
        FM_LOG_ERROR("error reporting: failed to get LWSwitch driver interface object for physical Id  %d",
                      physicalId); 
        return;
    }
 
    FMPciInfo_t switchPciInfo = pInterface->getSwtichPciInfo();
    lwlink_pci_dev_info pciInfo;
    pciInfo.domain = switchPciInfo.domain;
    pciInfo.bus = switchPciInfo.bus;
    pciInfo.device = switchPciInfo.device;
    pciInfo.function = switchPciInfo.function;
    uint64 deviceId = mLWLinkDevRepo->getDeviceId( pciInfo );
 
    // fill the recovery error message details
    lwswitch::lwlinkErrorRecoveryMsg *recoveryMsg = new lwswitch::lwlinkErrorRecoveryMsg();
    recoveryMsg->set_gpuorswitchid( deviceId );
    recoveryMsg->add_linkindex ( switchError.instance );
 
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::lwlinkErrorMsg *errorMsg = new lwswitch::lwlinkErrorMsg();
    errorMsg->set_allocated_recoverymsg( recoveryMsg );
 
    // fill the outer FM Message
    pFmMessage->set_type( lwswitch::FM_LWLINK_ERROR_LWSWITCH_RECOVERY );
    pFmMessage->set_allocated_lwlinkerrormsg( errorMsg );
 
    ret = mpLfm->SendMessageToGfm( pFmMessage, true );
    if ( ret != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("request to send LWSwitch LWLink recovery error message to fabric manager failed with error %d", ret);
    }
 
    delete pFmMessage;
}

// get the switch error scope from the driver and report to GFM
void
LocalFMErrorReporter::reportSwitchFatalErrorScope(FMUuid_t &switchUuid)
{
    LocalFMSwitchInterface *pSwitchInterface = mpLfm->switchInterfaceAt( switchUuid );

    if ( !pSwitchInterface )
    {
        FM_LOG_ERROR("error reporting: failed to get LWSwitch driver interface object for uuid %s",
                      switchUuid.bytes);
        return;
    }

    int portIndex;
    bool deviceScope = false;
    bool resetNeeded = false;
    bool portsNeedReset[LWSWITCH_MAX_PORTS];
    memset(portsNeedReset, 0, (sizeof(bool)*LWSWITCH_MAX_PORTS));

    // get device and port reset status
    switchIoctl_t ioctlStruct;
    LWSWITCH_GET_FATAL_ERROR_SCOPE_PARAMS getFatalErrorScopeParams;

    ioctlStruct.type        = IOCTL_LWSWITCH_GET_FATAL_ERROR_SCOPE;
    ioctlStruct.ioctlParams = &getFatalErrorScopeParams;
    ioctlStruct.paramSize   = sizeof(LWSWITCH_GET_FATAL_ERROR_SCOPE_PARAMS);

    memset( &getFatalErrorScopeParams, 0, sizeof(LWSWITCH_GET_FATAL_ERROR_SCOPE_PARAMS) );
    if ( pSwitchInterface->doIoctl( &ioctlStruct ) != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("failed to get fatal error scope from lwswitch %s.", pSwitchInterface->getUuid().bytes);
        return;
    }

    deviceScope = getFatalErrorScopeParams.device;
    if (deviceScope) {
        resetNeeded = true;
    }

    for (portIndex = 0; portIndex < LWSWITCH_MAX_PORTS; portIndex++)
    {
        portsNeedReset[portIndex] = getFatalErrorScopeParams.port[portIndex];
        if (portsNeedReset[portIndex])
        {
            resetNeeded = true;
        }
    }

    if (resetNeeded == false)
    {
        // nothing needs to reset, no need to send message to GFM
        return;
    }

    // device or ports need to reset, send the reset scope message to GFM
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::switchFatalErrorScope *scopeMsg = new lwswitch::switchFatalErrorScope();

    pFmMessage->set_allocated_fatalerrorscope( scopeMsg );
    pFmMessage->set_type( lwswitch::FM_LWSWITCH_FATAL_ERROR_SCOPE );

    // fill the switch fatal error scope
    scopeMsg->set_switchphysicalid( pSwitchInterface->getSwitchPhysicalId() );
    scopeMsg->set_devicescope( deviceScope );

    for (portIndex = 0; portIndex < LWSWITCH_MAX_PORTS; portIndex++ )
    {
        if ( portsNeedReset[portIndex] )
        {
            scopeMsg->add_switchports(portIndex);
        }
    }

    int ret = mpLfm->SendMessageToGfm( pFmMessage, true );
    if ( ret != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("request to send LWSwitch fatal error scope message to fabric manager failed with error %d", ret);
    }

    delete pFmMessage;
}

