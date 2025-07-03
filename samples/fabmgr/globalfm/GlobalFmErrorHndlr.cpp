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
#include <iostream>
#include <sstream>

#include <g_lwconfig.h>
#include "fm_log.h"
#include "GlobalFabricManager.h"
#include "GFMHelper.h"
#include "GlobalFmErrorHndlr.h"
#include "FMAutoLock.h"

GlobalFMErrorHndlr::GlobalFMErrorHndlr(GlobalFabricManager *pGfm,
                                       uint32_t nodeId,
                                       uint32_t partitionId,
                                       GlobalFMErrorSource errSource,
                                       GlobalFMErrorTypes  errType,
                                       lwswitch::fmMessage &errMsg)
{
    mGfm = pGfm;
    mErrorInfo.nodeId = nodeId;
    mErrorInfo.partitionId = partitionId;
    mErrorInfo.errSource = errSource;
    mErrorInfo.errType = errType;
    mErrorInfo.errMsg = errMsg;
}

GlobalFMErrorHndlr::~GlobalFMErrorHndlr()
{
    // nothing as of now
}

void
GlobalFMErrorHndlr::processErrorMsg(void)
{
    switch(mErrorInfo.errType) {
        case ERROR_TYPE_LWLINK_RECOVERY: {
            handleErrorLWLinkRecovery();
            break;
        }
        case ERROR_TYPE_LWLINK_FATAL: {
            handleErrorLWLinkFatal();
            break;
        }
        case ERROR_TYPE_LWSWITCH_FATAL_SCOPE: {
            handleErrorSwitchFatalScope();
            break;
        }
        case ERROR_TYPE_CONFIG_NODE_FAILED:
        case ERROR_TYPE_CONFIG_SWITCH_FAILED:
        case ERROR_TYPE_CONFIG_SWITCH_PORT_FAILED:
        case ERROR_TYPE_CONFIG_GPU_FAILED:
        case ERROR_TYPE_CONFIG_TIMEOUT: {
            handleErrorConfigFailed();
            break;
        }
        // TODO This code doesn't lwrrently get called. 
        case ERROR_TYPE_SOCKET_DISCONNECTED: {
            handleErrorSocketDisconnected();
            break;
        }
        // TODO This code doesn't lwrrently get called. 
        case ERROR_TYPE_HEARTBEAT_FAILED: {
            handleErrorHeartbeatFailed();            
            break;
        }
        default: {
            FM_LOG_ERROR( "unknown error type information received in fabric manager error handler" );
            break;
        }
    }
}

/*************************************************************************************
 * Handle an LWLink Long Recovery error message. When H/W trigger this error, 
 * the associated LWLink connection will transition to SAFE mode. As part of the
 * error handling, GFM will identify the associated connection and initiate a
 * re-train request to restore the connection to Active (HS) mode.
 *
 **************************************************************************************/
void
GlobalFMErrorHndlr::handleErrorLWLinkRecovery(void)
{
    int ret;
    uint32 nodeId = mErrorInfo.errMsg.nodeid();
    const lwswitch::lwlinkErrorMsg &errorMsg = mErrorInfo.errMsg.lwlinkerrormsg();
    const lwswitch::lwlinkErrorRecoveryMsg &recoveryMsg = errorMsg.recoverymsg();
    // first get the LWLink device which generated this recovery error
    FMLWLinkDevInfo lwLinkDevInfo;
    ret = mGfm->mLWLinkDevRepo.getDeviceInfo( nodeId, recoveryMsg.gpuorswitchid(), lwLinkDevInfo );
    if ( !ret ) {
        FM_LOG_WARNING( "failed to find LWLink device which generated LWLink recovery error on " NODE_ID_LOG_STR " %d", nodeId);
        return;
    }

    // log the recovery error information depending on which side saw the error
    std::ostringstream errStr;
    if (lwLinkDevInfo.getDeviceType () == lwlink_device_type_lwswitch) {
        // on LWSwitch side
        errStr << "found LWLink recovery error on LWSwitch " << NODE_ID_LOG_STR << " " << nodeId << " pci bus id: " << lwLinkDevInfo.getDevicePciBusId();
        
    } else {
        // on GPU side
        errStr << "found LWLink recovery error on GPU " << NODE_ID_LOG_STR << " " << nodeId << " pci bus id: " << lwLinkDevInfo.getDevicePciBusId();
    }
    FM_LOG_ERROR("%s", errStr.str().c_str());
    FM_SYSLOG_ERR("%s", errStr.str().c_str());

    // find the reported connections and retrain.
    for ( int i=0; i< recoveryMsg.linkindex_size(); i++ ) {
        FMLWLinkEndPointInfo endPoint;
        endPoint.nodeId = nodeId;
        endPoint.gpuOrSwitchId = recoveryMsg.gpuorswitchid();
        endPoint.linkIndex = recoveryMsg.linkindex( i );
        FMLWLinkDetailedConnInfo *connInfo = mGfm->mLWLinkConnRepo.getConnectionInfo( endPoint );
        if ( connInfo != NULL ) {
            // found the connection. re-train the connection
            std::stringstream outStr;
            connInfo->dumpConnInfo( &outStr, mGfm, mGfm->mLWLinkDevRepo);
            FMLWLinkEndPointInfo masterEndpoint = connInfo->getMasterEndPointInfo();
            FMLWLinkEndPointInfo slaveEndpoint = connInfo->getSlaveEndPointInfo();
            FMLWLinkDevInfo masterDevInfo;
            FMLWLinkDevInfo slaveDevInfo;
            mGfm->mLWLinkDevRepo.getDeviceInfo( masterEndpoint.nodeId, masterEndpoint.gpuOrSwitchId, masterDevInfo );
            mGfm->mLWLinkDevRepo.getDeviceInfo( slaveEndpoint.nodeId, slaveEndpoint.gpuOrSwitchId, slaveDevInfo );

            if (((mGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) || (mGfm->getFabricMode() == FM_MODE_VGPU)) &&
                (masterDevInfo.getDeviceType() != slaveDevInfo.getDeviceType()))
            {
                // Do not re-train access link in shared fabric mode
                // as GPUs are not with the service VM for retraining
                FM_LOG_INFO("skipping re-training LWLink connection due to shared LWSwitch multitenancy mode %s\n",
                           outStr.str().c_str());
                FM_SYSLOG_NOTICE("skipping re-training LWLink connection due to shared LWSwitch multitenancy mode %s\n",
                           outStr.str().c_str());
                
                outStr.str(""); // clear previous string
                continue;
            }
            FM_LOG_INFO("re-training following LWLink connection due to LWLink recovery error %s\n",
                       outStr.str().c_str());
            FM_SYSLOG_NOTICE("re-training following LWLink connection due to LWLink recovery error %s\n",
                             outStr.str().c_str());

            GFMHelper::trainLWLinkConnection(mGfm, mGfm->mLinkTrainIntf, mGfm->mLWLinkConnRepo, mGfm->mLWLinkDevRepo,
                                             connInfo, LWLINK_TRAIN_SAFE_TO_HIGH);
            outStr.str(""); // clear previous string
            connInfo->dumpConnAndStateInfo( &outStr, mGfm, mGfm->mLWLinkDevRepo);
            FM_LOG_INFO("re-training completed for LWLink connection %s\n", outStr.str().c_str());
            FM_SYSLOG_NOTICE("re-training completed for LWLink connection %s\n", outStr.str().c_str());            
        }

    }
}

/*************************************************************************************
 * Handle an LWLink Fatal error message. 
 *  To recover from an LWLink Fatal error, we need a GPU/LWSwtich Link level reset.
 *  The current version will just clear the FM session state so that no further LWCA
 *  clients are allowed by RM. Later the administrator will either reboot the machine
 *  or reset the devices out-of-band and restart FabricManager.
 *
 **************************************************************************************/
void
GlobalFMErrorHndlr::handleErrorLWLinkFatal(void)
{
    FMPciInfo_t pciInfo = {0};
    uint32_t nodeId = mErrorInfo.errMsg.nodeid();

    if (mErrorInfo.errSource == ERROR_SOURCE_GPU) {
        // first get the GPU which is reporting this error
        int ret;

        const lwswitch::lwlinkErrorMsg &errorMsg = mErrorInfo.errMsg.lwlinkerrormsg();
        const lwswitch::lwlinkErrorGpuFatalMsg &gpuFatalMsg = errorMsg.gpufatalmsg();
        FMLWLinkDevInfo lwLinkDevInfo;
        ret = mGfm->mLWLinkDevRepo.getDeviceInfo( nodeId, gpuFatalMsg.gpuorswitchid(), lwLinkDevInfo );
        if ( !ret ) {
            FM_LOG_WARNING( "failed to find GPU LWLink device which generated LWLink fatal error on " NODE_ID_LOG_STR " %d", nodeId);
            return;
        }
        
        std::ostringstream ss;
        pciInfo = lwLinkDevInfo.getPciInfo();
        ss << "found LWLink fatal error on GPU with " << NODE_ID_LOG_STR << " " << nodeId << " pci bus id: " << pciInfo.busId;
        FM_LOG_ERROR("%s", ss.str().c_str());
        FM_SYSLOG_ERR("%s", ss.str().c_str());

        bool errorOnInactiveLWLinks = true;;
        if (!gpuFatalMsg.has_linkindex()) {
            // the error message does not carry the GPU LWLink index
            // cannot check if the error link are inactive or not
            errorOnInactiveLWLinks = false;
        } else {
            if (lwLinkDevInfo.isLinkActive(gpuFatalMsg.linkindex())) {
                // the fatal LWLink error is active
                errorOnInactiveLWLinks = false;
            }
        }

        // All fatal errors are on inactive LWLinks
        if (errorOnInactiveLWLinks) {
            // This error could be due to link training failures at init time
            // in bare metal or full passthrough mode,
            // degraded mode manager would degrade the device accordingly.
            // the error is already logged at parseLWSwitchErrorReportMsg()
            FM_LOG_INFO("ignoring GPU LWLink fatal error on " NODE_ID_LOG_STR " %d as the detected error is on a Non Active LWLink port.", nodeId);
            return;
        }

    } else {
        // LWSwitch LWLink fatal errors are not reported specifically. It will be 
        // reported as normal LWSwitch fatal errors (ERROR_TYPE_FATAL)
        FM_LOG_ERROR( "LWSwitch LWLink fatal error is reported to wrong handler ");
        return;
    }

    if ((mGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) || (mGfm->getFabricMode() == FM_MODE_VGPU))
    {
        FM_LOG_INFO("the ports in error will get reset as part of corresponding partition reset.");

        // Do not clear FM session on the service VM in shared fabric mode
        // Fatal error in one partition should not affect activation of other partitions
        // FM session needs to be set, as GPU health check depends on it.
        return;
    }

    std::ostringstream ss2;
    if (LWSWITCH_ARCH_TYPE_SV10 == mGfm->getSwitchArchType()) {
        // pre GA100, GPU reset is not supported
        // send clear FM session
        ss2 << "Aborting all the LWCA jobs and leaving the system uninitialized due to GPU LWLink fatal error. " << endl
            << "New LWCA job launch will fail on this system. Please refer to your system user guide for recovery procedure.";
        // send clear FM session to the specified node (for single node) or to all nodes (for multi-node)
        mGfm->sendDeInitToAllFabricNodes();
    } else {
        // For GA100 and later, GPU reset is supported
        // Do not clear FM session
        // Log and instruct the user to reset the GPU
        FMGpuInfo_t gpuInfo = {0};
        mGfm->getGpuInfo(nodeId, pciInfo, gpuInfo);

        ss2 << "GPU " << NODE_ID_LOG_STR << " " << nodeId << " index " << gpuInfo.gpuIndex << " pci bus id " <<  pciInfo.busId << " experienced an LWLink fatal error "
            << "and running LWCA jobs will be affected. " << endl
            << "Resetting the specified GPU may clear the issue. Please refer to your system user guide for GPU reset instructions.";
    }

    FM_LOG_ERROR("%s", ss2.str().c_str());
    FM_SYSLOG_ERR("%s", ss2.str().c_str());
}

/*************************************************************************************
 * Handle a Fatal error scope reported by LWSwitch
 *  To recover from a Fatal error, we need a GPU/LWSwtich Link level reset.
 *
 *  When fatal error scope requires an access port reset:
 *  Do not clear FM session state, log and instruct the user to reset the affected GPU
 *  All access ports that are connected to this GPU will be reset in the process.
 *
 *  When fatal error scope requires a device reset or trunk port reset:
 *  Clear the FM session state so that no further LWCA
 *  clients are allowed by RM. Later the administrator will either reboot the machine
 *  or reset the devices out-of-band and restart FabricManager.
 *
 **************************************************************************************/
void
GlobalFMErrorHndlr::handleErrorSwitchFatalScope(void)
{
    // Log fatal error scope
    lwswitch::switchFatalErrorScope fatalErrorScopeMsg = mErrorInfo.errMsg.fatalerrorscope();

    uint32_t nodeId = mErrorInfo.errMsg.nodeid();
    uint32_t physicalId = fatalErrorScopeMsg.switchphysicalid();
    bool deviceScope = fatalErrorScopeMsg.has_devicescope() ? fatalErrorScopeMsg.devicescope() : false;

    std::list<uint32_t> portsNeedReset;
    portsNeedReset.clear();
    for (int i = 0; i < fatalErrorScopeMsg.switchports_size(); i++) {
        portsNeedReset.push_back(fatalErrorScopeMsg.switchports(i));
    }

    FMPciInfo_t pciInfo = {0};
    mGfm->getLWSwitchPciBdf(nodeId, physicalId, pciInfo);

    if (deviceScope) {
        std::ostringstream ss;
        ss  << "an LWSwitch fatal error oclwrred on " << NODE_ID_LOG_STR << " " << nodeId << " physical id: " << physicalId
            << " pci bus id: " << pciInfo.busId << " and requires full LWSwitch reset to recover.";
        FM_LOG_ERROR("%s", ss.str().c_str());
        FM_SYSLOG_ERR("%s", ss.str().c_str());
    }

    //
    // check if all fatal errors are on inactive LWLinks. do not check in shared lwswitch
    // mode, as link status in shared switch mode are not up to date
    //
    if (!deviceScope && (mGfm->getFabricMode() != FM_MODE_SHARED_LWSWITCH) &&
        isLWSwitchFatalErrorOnInactiveLWLinks(nodeId, physicalId, portsNeedReset)) {
        // The error could be due to link training failures at init time
        // in bare metal or full passthrough mode,
        // degraded mode manager would degrade the device accordingly.
        FM_LOG_INFO("ignoring LWSwitch LWLink fatal error on " NODE_ID_LOG_STR " %d as the detected error is on a Non Active LWLink port.", nodeId);
        return;
    }

    // only log port reset scope error, when the device does not need to be reset.
    if (!deviceScope && portsNeedReset.size() > 0) {
        std::ostringstream ss;
        ss  << "a fatal error oclwrred on LWSwitch port(s) ";

        std::list<uint32_t>::iterator it;
        int portCount = portsNeedReset.size();
        for ( it = portsNeedReset.begin(); it != portsNeedReset.end(); it++, portCount-- ) {
            uint32_t portIndex = *it;

            ss << portIndex;
            if (portCount > 1) {
                ss << ",";
            } else {
                // last port
                ss << " " ;
            }
        }

        ss  << "on LWSwitch " << NODE_ID_LOG_STR << " " << nodeId << " physical id: " << physicalId << " pci bus id: "
            << pciInfo.busId << " and requires corresponding ports reset to recover.";

        FM_LOG_ERROR("%s", ss.str().c_str());
        FM_SYSLOG_ERR("%s", ss.str().c_str());
    }

    if ((mGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) || (mGfm->getFabricMode() == FM_MODE_VGPU))
    {
        //
        // these port errors are expected in Shared LWSwitch VM shutdown (partition deactivation) case,
        // as one side of the LWLink gets reset (like Hypervisor SBRing GPU as part of VM shutdown).
        //
        if (mGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) {
            FM_LOG_INFO("if these port errors are reported as part of partition tear down, then it is expected "
                        "and corresponding LWSwitch ports will get reset as part of partition deactivation");
        }

        // Do not clear FM session on the service VM in shared fabric mode
        // Fatal error in one partition should not affect activation of other partitions
        // FM session needs to be set, as GPU health check depends on it.
        return;
    }

    std::set<uint32_t> gpusToReset;
    gpusToReset.clear();
    bool allErrorsOnAccessPort = isLWSwitchFatalErrorOnAccessLWLinks(nodeId, physicalId, portsNeedReset, gpusToReset);

    //
    // clear FM Session for following cases
    // 1. If the reported error scope requires a switch reset
    // 2. If the reported error scope requires a port reset and that port is a Trunk port
    // 3. For older platform Willow, as there is no per port reset/GPU reset supported.
    //
    if ( deviceScope ||
         ((portsNeedReset.size() > 0) && !allErrorsOnAccessPort) ||
         ((portsNeedReset.size() > 0) && (LWSWITCH_ARCH_TYPE_SV10 == mGfm->getSwitchArchType()))) {
        // send clear FM session
        std::ostringstream ss;
        ss  << "Aborting all the LWCA jobs and leaving the system uninitialized as reported fatal error requires resetting LWSwitch "
            << "with " << NODE_ID_LOG_STR << " " << nodeId <<  " physical id: " << physicalId << " pci bus id: " << pciInfo.busId << endl
            << "New LWCA job launch will fail on this system. Please refer to your system user guide for recovery procedure.";

        FM_LOG_ERROR("%s", ss.str().c_str());
        FM_SYSLOG_ERR("%s", ss.str().c_str());
        // send clear FM session to all nodes
        mGfm->sendDeInitToAllFabricNodes();
        return;
    }

    // The fatal errors are all on access ports, no need to clear FM session
    // log and instruct the user to reset the affected GPUs.
    for (std::set<uint32_t>::iterator it = gpusToReset.begin(); it != gpusToReset.end(); it++) {
        uint32_t enumIndex = *it;
        FMPciInfo_t pciInfo = {0};
        std::ostringstream ss;

        mGfm->getGpuPciBdf(nodeId, enumIndex, pciInfo);

        ss  << "LWSwitch port connected to GPU " << NODE_ID_LOG_STR << " " << nodeId << " index " << enumIndex << " pci bus id " <<  pciInfo.busId
            << " experienced an LWLink fatal error and requires port reset to recover."
            << " All the running LWCA jobs on this GPU will be affected." << endl
            << "Resetting the specified GPU may clear the issue. Please refer to your system user guide for GPU reset instructions.";

        FM_LOG_ERROR("%s", ss.str().c_str());
        FM_SYSLOG_ERR("%s", ss.str().c_str());
    }
}

void
GlobalFMErrorHndlr::handleErrorConfigFailed(void)
{
    switch(mErrorInfo.errType) {
        case ERROR_TYPE_CONFIG_NODE_FAILED: {
            // a node global configuration failed
            FM_LOG_ERROR("failed to configure LWSwitch/GPU global settings");
            FM_SYSLOG_ERR("failed to configure LWSwitch/GPU global settings");
            break;
        }
        case ERROR_TYPE_CONFIG_SWITCH_FAILED: {
            FM_LOG_ERROR("failed to configure LWSwitch settings ");            
            FM_SYSLOG_ERR("failed to configure LWSwitch settings ");
            break;
        }
        case ERROR_TYPE_CONFIG_SWITCH_PORT_FAILED: {
            FM_LOG_ERROR("failed to configure LWSwitch port");
            FM_SYSLOG_ERR("failed to configure LWSwitch port");
            break;
        }
        case ERROR_TYPE_CONFIG_GPU_FAILED: {
            FM_LOG_ERROR("failed to configure GPU settings");            
            FM_SYSLOG_ERR("failed to configure GPU settings");
            break;
        }
        default: {
            // shouldn't land here
            FM_LOG_ERROR("fabric manager error handler is called for unhandled config error type %d", mErrorInfo.errType);
            break;
       }
    }

    // set error state
    mGfm->setNodeConfigError(mErrorInfo.nodeId);

    // send clear FM session to all nodes
    mGfm->sendDeInitToAllFabricNodes();
}

// TODO This code doesn't lwrrently get called. 
void
GlobalFMErrorHndlr::handleErrorSocketDisconnected(void)
{
    FM_LOG_ERROR("lost socket connection between global and local fabric manager instance");
    FM_SYSLOG_ERR("lost socket connection between global and local fabric manager instance");
}

// TODO This code doesn't lwrrently get called. 
void
GlobalFMErrorHndlr::handleErrorHeartbeatFailed(void)
{
    FM_SYSLOG_ERR("failed heartbeat between global and local fabric manager instance");
    FM_LOG_ERROR("failed heartbeat between global and local fabric manager instance");
}

bool
GlobalFMErrorHndlr::isLWSwitchFatalErrorOnInactiveLWLinks(uint32_t nodeId,
                                                          uint32_t physicalId,
                                                          std::list<uint32_t> &portList)
{
    bool errorOnInactiveLWLinks = true;
    std::list<uint32_t>::iterator it;

    for ( it = portList.begin(); it != portList.end(); it++ ) {

        uint32_t portIndex = *it;

        uint64 lwLinkSwitchId;
        if (mGfm->getLWSwitchLWLinkDriverId(nodeId, physicalId, lwLinkSwitchId) == false) {
            errorOnInactiveLWLinks = false;
            break;
        }

        FMLWLinkDevInfo lwLinkDevInfo;
        if (mGfm->mLWLinkDevRepo.getDeviceInfo(nodeId, lwLinkSwitchId, lwLinkDevInfo) == false) {
            errorOnInactiveLWLinks = false;
            break;
        }


        if (lwLinkDevInfo.isLinkActive(portIndex)) {
            // the link is in active state
            errorOnInactiveLWLinks = false;
            break;
        }
    }

    // all link fatal errors are in non activate state
    return errorOnInactiveLWLinks;
}

bool
GlobalFMErrorHndlr::isLWSwitchFatalErrorOnAccessLWLinks(uint32_t nodeId,
                                                        uint32_t physicalId,
                                                        std::list<uint32_t> &portList,
                                                        std::set<uint32_t> &gpusToReset)
{
    bool errorOnAcessLWLinks = true;
    gpusToReset.clear();

    std::list<uint32_t>::iterator it;
    for ( it = portList.begin(); it != portList.end(); it++ ) {

        uint32_t portIndex = *it;

        accessPort *portInfo = mGfm->mpParser->getAccessPortInfo(nodeId, physicalId, portIndex);
        if (!portInfo || !portInfo->has_farpeerid()) {
            // the port in error is not a access port
            errorOnAcessLWLinks = false;
            break;
        } else {
            uint32_t enumIndex;
            if (!mGfm->getGpuEnumIndex(nodeId, portInfo->farpeerid(), enumIndex)) {
                // cannot find the GPU PCI info
                errorOnAcessLWLinks = false;
                break;
            }

            // add the GPU enumIndex to the set
            gpusToReset.insert(enumIndex);
        }

    }

    // all link fatal errors are on access ports
    return errorOnAcessLWLinks;
}

