/*
 *  Copyright 2021-2022 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#include <poll.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#include <map>
#include <list>

#include <g_lwconfig.h>

#include "fm_log.h"
#include "lwtypes.h"
#include "FMCommonTypes.h"
#include "LocalFabricManager.h"
#include "FMErrorCodesInternal.h"

#include "ctrl/ctrl0000/ctrl0000gpu.h"

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
#include "LocalFmMulticastHndlr.h"

//
//This class used RmClientHandle and FM session from LocalFMGpuMgr and assumes these will not be ilwalidated 
//during the lifetiime of this class
//
LocalFmMulticastHndlr::LocalFmMulticastHndlr(LocalFabricManagerControl *pLocalFmControl)
     :mpLfm(pLocalFmControl),
      mHandleFmClient(mpLfm->mFMGpuMgr->getRmClientHandle()),
      mHandleFmSession(mpLfm->mFMGpuMgr->getFmSessionHandle())
{
    lwosInitializeCriticalSection( &mLock );
    FM_LOG_DEBUG( "LocalFmMulticastHndlr constructor called");
    //
    // TODO: RM Multicast messages yet to be defined
    //
    // The RM request and response messages will be going through lwlink inband interface
    // Should reqister for driver notification when a message from RM is received from
    // the lwlink inband interface.
    //
}

LocalFmMulticastHndlr::~LocalFmMulticastHndlr()
{
    // TODO: RM Multicast events yet to be defined
    // mLocalFabricManagerControl->mFMGpuMgr->unsubscribeGpuEvents(LW000F_NOTIFIERS_FABRIC_EVENT,
    //                                                             &processEvents, (void *)this);
    lwosDeleteCriticalSection( &mLock );
    FM_LOG_DEBUG( "LocalFmMulticastHndlr destructor called");
}

void
LocalFmMulticastHndlr::processEvents(void *args)
{
    // TODO: RM Multicast events yet to be defined
    return ;
}

FMIntReturn_t
LocalFmMulticastHndlr::sendGroupCreateReqMsg(uint32_t numOfGpus, uint32_t memSize)
{
    FMIntReturn_t rc;
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::multicastGroupCreateRequest *pMsg = new lwswitch::multicastGroupCreateRequest();

    pFmMessage->set_type( lwswitch::FM_MULTICAST_GROUP_CREATE_REQ );
    pFmMessage->set_allocated_mcgroupcreatereq( pMsg );

    pMsg->set_numofgpus(numOfGpus);
    pMsg->set_memsize(memSize);

    rc =  mpLfm->SendMessageToGfm(pFmMessage, false);
    if (rc != FM_INT_ST_OK) {
        FM_LOG_ERROR("failed to send group create request to global fabric manager.");
    }

    delete( pFmMessage );
    return rc;
}

FMIntReturn_t
LocalFmMulticastHndlr::sendGroupBindReqMsg(uint64_t mcHandle, std::list<char*> gpuUuidList)
{
    FMIntReturn_t rc;
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::multicastGroupBindRequest *pMsg = new lwswitch::multicastGroupBindRequest();

    pFmMessage->set_type( lwswitch::FM_MULTICAST_GROUP_BIND_REQ );
    pFmMessage->set_allocated_mcgroupbindreq( pMsg );

    pMsg->set_mchandle(mcHandle);

    std::list<char*>::iterator it;
    for ( it = gpuUuidList.begin(); it != gpuUuidList.end(); it++) {
        char* gpuUUid = *it;
        if (gpuUUid) {
            pMsg->add_uuid(gpuUUid);
        }
    }

    rc =  mpLfm->SendMessageToGfm(pFmMessage, false);
    if (rc != FM_INT_ST_OK) {
        FM_LOG_ERROR("failed to send group bind request for handle 0x%lu to global fabric manager",
                     mcHandle);
    }

    delete( pFmMessage );
    return rc;
}

FMIntReturn_t
LocalFmMulticastHndlr::sendGroupSetupCompleteAckMsg(uint64_t mcHandle,
                                                    lwswitch::configStatus rspCode)
{
    FMIntReturn_t rc;
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::multicastGroupSetupCompleteAck *pMsg = new lwswitch::multicastGroupSetupCompleteAck();

    pFmMessage->set_type( lwswitch::FM_MULTICAST_GROUP_SETUP_COMPLETE_ACK );
    pFmMessage->set_allocated_mcgroupsetupcompleteack( pMsg );

    pMsg->set_mchandle(mcHandle);
    pMsg->set_rspcode(rspCode);

    rc =  mpLfm->SendMessageToGfm(pFmMessage, false);
    if (rc != FM_INT_ST_OK) {
        FM_LOG_ERROR("failed to send group set up complete ack for handle 0x%lu to global fabric manager.",
                     mcHandle);
    }

    delete( pFmMessage );
    return rc;
}

FMIntReturn_t
LocalFmMulticastHndlr::sendGroupReleaseReqMsg(uint64_t mcHandle, std::list<char*> gpuUuidList)
{
    FMIntReturn_t rc;
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::multicastGroupReleaseRequest *pMsg = new lwswitch::multicastGroupReleaseRequest();

    pFmMessage->set_type( lwswitch::FM_MULTICAST_GROUP_RELEASE_REQ );
    pFmMessage->set_allocated_mcgroupreleasereq( pMsg );

    pMsg->set_mchandle(mcHandle);

    std::list<char*>::iterator it;
    for ( it = gpuUuidList.begin(); it != gpuUuidList.end(); it++) {
        char* gpuUUid = *it;
        if (gpuUUid) {
            pMsg->add_uuid(gpuUUid);
        }
    }

    rc =  mpLfm->SendMessageToGfm(pFmMessage, false);
    if (rc != FM_INT_ST_OK) {
        FM_LOG_ERROR("failed to send group release request for handle 0x%lu to global fabric manager",
                     mcHandle);
    }

    delete( pFmMessage );
    return rc;
}

FMIntReturn_t
LocalFmMulticastHndlr::sendGroupReleaseCompleteAckMsg(uint64_t mcHandle, lwswitch::configStatus rspCode)
{
    FMIntReturn_t rc;
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::multicastGroupReleaseCompleteAck *pMsg = new lwswitch::multicastGroupReleaseCompleteAck();

    pFmMessage->set_type( lwswitch:: FM_MULTICAST_GROUP_RELEASE_COMPLETE_ACK );
    pFmMessage->set_allocated_mcgroupreleasecompleteack( pMsg );

    pMsg->set_mchandle(mcHandle);
    pMsg->set_rspcode(rspCode);

    rc =  mpLfm->SendMessageToGfm(pFmMessage, false);
    if (rc != FM_INT_ST_OK) {
        FM_LOG_ERROR("failed to send group release complete ack for hanlde 0x%lu to global fabric manager.",
                     mcHandle);
    }

    delete( pFmMessage );
    return rc;
}

void
LocalFmMulticastHndlr::handleGroupCreateRspMsg(lwswitch::fmMessage *pFmMessage)
{
    // TODO: notify RM when RM control call is ready
    return ;
}

void
LocalFmMulticastHndlr::handleGroupBindRspMsg(lwswitch::fmMessage *pFmMessage)
{
    // TODO: notify RM when RM control call is ready
    return ;
}

void
LocalFmMulticastHndlr::handleGroupSetupCompleteReqMsg(lwswitch::fmMessage *pFmMessage)
{
    const lwswitch::multicastGroupSetupCompleteRequest &request = pFmMessage->mcgroupsetupcompletereq();

    // TODO: notify RM when RM control call is ready

    if (mpLfm->isSimMode()) {
        // RM multicast is not ready for emulation, use the unicast API to set FLA
        for (int i = 0; i < request.uuid().size(); i++) {
            FMUuid_t fmUuid;
            strncpy(fmUuid.bytes, request.uuid(i).c_str(), (FM_UUID_BUFFER_SIZE-1));

            if (request.has_parentmcflaaddrvalid() && request.parentmcflaaddrvalid()) {
                mpLfm->mFMGpuMgr->setGpuFabricFLA(fmUuid,
                                                  request.parentmcflaaddrbase(),
                                                  request.parentmcflaaddrrange());

            } else if (request.has_mcflaaddrvalid() && request.mcflaaddrvalid()) {
                mpLfm->mFMGpuMgr->setGpuFabricFLA(fmUuid,
                                                  request.mcflaaddrbase(),
                                                  request.mcflaaddrrange());
            }
        }
    }
    return ;
}

void
LocalFmMulticastHndlr::handleGroupReleaseRspMsg(lwswitch::fmMessage *pFmMessage)
{
    // TODO: notify RM when RM control call is ready
    return ;
}

void
LocalFmMulticastHndlr::handleGroupReleaseCompleteReqMsg(lwswitch::fmMessage *pFmMessage)
{
    const lwswitch::multicastGroupReleaseCompleteRequest &request = pFmMessage->mcgroupreleasecompletereq();

    // TODO: notify RM when RM control call is ready

    if (mpLfm->isSimMode()) {
        // RM multicast is not ready for emulation, use the unicast API to clear FLA

        if (mpLfm->isSimMode()) {
            // RM multicast is not ready for emulation, use the unicast API to set FLA
            for (int i = 0; i < request.uuid().size(); i++) {
                FMUuid_t fmUuid;
                strncpy(fmUuid.bytes, request.uuid(i).c_str(), (FM_UUID_BUFFER_SIZE-1));

                if (request.has_parentmcflaaddrvalid() && request.parentmcflaaddrvalid()) {
                    mpLfm->mFMGpuMgr->clearGpuFabricFLA(fmUuid,
                                                        request.parentmcflaaddrbase(),
                                                        request.parentmcflaaddrrange());

                } else if (request.has_mcflaaddrvalid() && request.mcflaaddrvalid()) {
                    mpLfm->mFMGpuMgr->clearGpuFabricFLA(fmUuid,
                                                        request.mcflaaddrbase(),
                                                        request.mcflaaddrrange());
                }
            }
        }
    }

    return ;
}

void
LocalFmMulticastHndlr::handleMessage(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "message type %d", pFmMessage->type());
    lwosInitializeCriticalSection(&mLock);

    switch ( pFmMessage->type() )
    {
        case lwswitch::FM_MULTICAST_GROUP_CREATE_RSP:
        {
            handleGroupCreateRspMsg(pFmMessage);
            break;
        }
        case lwswitch::FM_MULTICAST_GROUP_BIND_RSP:
        {
            handleGroupBindRspMsg(pFmMessage);
            break;
        }
        case lwswitch::FM_MULTICAST_GROUP_RELEASE_RSP:
        {
            handleGroupReleaseRspMsg(pFmMessage);
            break;
        }
        case lwswitch::FM_MULTICAST_GROUP_SETUP_COMPLETE_REQ:
        {
            handleGroupSetupCompleteReqMsg(pFmMessage);
            break;
        }
        case lwswitch::FM_MULTICAST_GROUP_RELEASE_COMPLETE_REQ:
        {
            handleGroupReleaseCompleteReqMsg(pFmMessage);
            break;
        }
        default:
            FM_LOG_ERROR( "unknown multicast message type %d", pFmMessage->type());
            break;
        }
}

#endif
