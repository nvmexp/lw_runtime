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
#include <g_lwconfig.h>
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)

#include <string.h>
#include <map>
#include <sstream>
#include "fm_log.h"
#include "lwtypes.h"
#include "lwRmApi.h"
#include "lwos.h"
#include "lwos.h"
#include "FMCommonTypes.h"
#include "LocalFMMemMgrExporter.h"
#include "LocalFMMemMgrImporter.h"
#include "memmgr.pb.h"
#include "LocalFabricManager.h"
#include "FMHandleGenerator.h"
#include "FMUtils.h"

#include "ctrl/ctrl00fa.h"

/*
 * Locks are taken at the entry point of exporter or importer this includes following functions just before the
 * first piece of code that accesses IMEX data structures:
 * Exporter:
 *  - handleUnimportRequest()
 *  - handleImportRequest()
 * Importer:
 *  - sendImportRequest()
 *  - sendUnimportRequest()
 *  - handleImportResponse()
 *  - handleUnimportResponse()
 *  - processRequestTimeout()
 *
 * Locks are released whenever exiting exporter or importer. This includes
 * - return from entry point functions listed above (Note: Autolock can only handle this case)
 * - before calls into sendFatalErrorToAllNodes(), SendMessageToPeerLFM()
 * - before importer calls exporter function or vice versa
 *
 * We have three threads of exelwtion:
 * - message processing
 * - GPU events
 * - timeouts
 *
 * In addition it is possible that simultaneously one application is doing a single node import+export and
 * another application is doing multi-node import+export.
 *
 * Also Note:
 * - message processing thread takes the libEvent lock before taking an IMEX lock
 * - GPU event thread takes an IMEX lock before taking and libEvent lock(in SendMessageToPeerLFM)
 *
 * To prevent deadlocks IMEX locks are released before sending/receiving any messages from the same context
 *
 * Future plan is to restructure the code such that locks are taken and released within the same function. This
 * can be accomplished by restructing the code and writing set/get functiions for each data structure.
 */


// This class uses RmClientHandle and FM session from LocalFMGpuMgr and assumes 
// these will not be ilwalidated during the lifetiime of this class
LocalFMMemMgrExporter::LocalFMMemMgrExporter( LocalFMCoOpMgr *pLocalCoopMgr,
                                              LocalFabricManagerControl *pLocalFmControl)
    :mFMLocalCoOpMgr( pLocalCoopMgr ),
     mLocalFmControl( pLocalFmControl ), 
     mHandleFmClient( mLocalFmControl->mFMGpuMgr->getRmClientHandle() ), 
     mHandleFmSession( mLocalFmControl->mFMGpuMgr->getFmSessionHandle() )
{
    FM_LOG_DEBUG( "LocalFMMemMgrExporter constructor called" );
    lwosInitializeCriticalSection( &mLock );
    mImporter = nullptr;
    mEnableMessageProcessing = true;
}

LocalFMMemMgrExporter::~LocalFMMemMgrExporter()
{
    FM_LOG_DEBUG( "LocalFMMemMgrExporter destructor called" );
    for( auto it = mExportObjectMap.begin(); it != mExportObjectMap.end(); it++ )
    {
        delete( it->second );
    }
    lwosDeleteCriticalSection( &mLock );
}

bool
LocalFMMemMgrExporter::readPageTableEntries( exportObjectData *data )
{    
    LwU32 retVal;
    LW00FA_CTRL_DESCRIBE_PARAMS  describeParams = { 0 };
    describeParams.offset = 0;

    // Each describe call returns upto LW00FA_CTRL_DESCRIBE_PFN_ARRAY_SIZE(lwrrently 512) PFNs.
    // We need to keep calling LW00FA_CTRL_CMD_DESCRIBE until all PFNs have been read.
    // When the last set of  PFNs are read the last LW00FA_CTRL_CMD_DESCRIBE call will give the remaining PFNs and
    // retVal will be LW_OK and we'll break out of the loop.
    // If LW00FA_CTRL_CMD_DESCRIBE return something other than LW_OK it is an error.
    retVal = LwRmControl( mHandleFmClient, data->exportObjectRefHandle, LW00FA_CTRL_CMD_DESCRIBE,
                          &describeParams, sizeof( describeParams ) ); 

    while( retVal == LW_OK )
    {
        FM_LOG_DEBUG( "params.offset=%llu params.numPfns=%u params.totalPfns=%llu",describeParams.offset,
                      describeParams.numPfns, describeParams.totalPfns );

        // store all PFNs read in rsp msg
        for( unsigned int i = 0; i < describeParams.numPfns; i++ ) 
        {
            data->ffn.push_back( describeParams.pfnArray[i] );
            FM_LOG_DEBUG( "PFN%llu=%d",i  + describeParams.offset, describeParams.pfnArray[i] );
        }

        if( ( describeParams.offset + describeParams.numPfns ) >= describeParams.totalPfns )
        {
            // All PFNs have been read successfully
	       break;
        }
        describeParams.offset += describeParams.numPfns;

        retVal = LwRmControl( mHandleFmClient, data->exportObjectRefHandle, LW00FA_CTRL_CMD_DESCRIBE,
                              &describeParams, sizeof( describeParams ) ); 
    }

    if( retVal != LW_OK )
    {
        FM_LOG_ERROR( "memory export describe failed error %s", lwstatusToString( retVal ) );
        FM_SYSLOG_ERR( "memory export describe failed error %s", lwstatusToString( retVal ) );
        return false;
    }

    // PFNs successfully read at this point. set flags and pageSize
    data->memFlags = describeParams.memFlags;
    data->kind = describeParams.attrs.kind;
    data->pageSize = describeParams.attrs.pageSize;
    data->size = describeParams.attrs.size;
    return true;
}

bool
LocalFMMemMgrExporter::sendImportResponse( const lwswitch::memoryFlaImportReq &reqMsg, 
                                           exportObjectData *data,
                                           uint32 peerNodeId, lwswitch::memoryReqErrors errCode )
{
    FMIntReturn_t retVal;
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::memoryFlaImportRsp *rspMsg = new lwswitch::memoryFlaImportRsp();

    rspMsg->set_errcode( errCode );
    rspMsg->set_importeventid( reqMsg.importeventid() ); 

    // exportObjectData parameter will be NULL if the import return code (errCode) is not success. 
    // access data only if errCode is set to MEMORY_REQ_SUCCESS
    if( errCode == lwswitch::MEMORY_REQ_SUCCESS )
    {
        rspMsg->set_pagesize( data->pageSize );
        rspMsg->set_memflags( data->memFlags );
        rspMsg->set_size( data->size );
        rspMsg->set_kind( data->kind );
        for( auto it = data->ffn.begin(); it != data->ffn.end(); it++ )
        {
            rspMsg->add_pageframenumbers( *it );
        }
    }

    pFmMessage->set_allocated_memoryflaimportrsp( rspMsg );
    pFmMessage->set_type( lwswitch::FM_MEMORY_FLA_IMPORT_RSP );
    pFmMessage->set_version( FABRIC_MANAGER_VERSION );
    pFmMessage->set_nodeid( mLocalFmControl->getLocalNodeId() );

    if( pFmMessage->nodeid() == peerNodeId )
    {
        lwosLeaveCriticalSection(&mLock);
        bool rv = mImporter->handleImportResponse( pFmMessage );
        delete( pFmMessage );
        return rv;
    }

    lwosLeaveCriticalSection(&mLock);
    retVal = mFMLocalCoOpMgr->SendMessageToPeerLFM( peerNodeId, pFmMessage, false );
    delete( pFmMessage );

    std::string uuidStr = FMUtils::colwertUuidToHexStr( reqMsg.exportuuid().data(), LW_FABRIC_UUID_LEN );
    if( retVal != FM_INT_ST_OK )
    {
        std::ostringstream ss;
        ss << "send import response message to peer Memory Manager on node " << peerNodeId << " failed.";
        ss << " UUID FAB-" << uuidStr.c_str() << " import event id " << reqMsg.importeventid(); 
        ss << " GPU id " << reqMsg.exportgpuid() << " index " << reqMsg.index(); 
        ss << " error " << retVal;
        
        FM_LOG_ERROR( "fatal error, %s", ss.str().c_str() );
        FM_SYSLOG_ERR( "fatal error, %s", ss.str().c_str() );
        sendFatalErrorToAllNodes( lwswitch::MEMORY_FLA_GENERIC_ERROR, ss.str().c_str() );
        return false;
    }
    FM_LOG_DEBUG( "Sent Import response to node %d UUID FAB-%s importEventId=%lu", 
                  peerNodeId, uuidStr.c_str(), reqMsg.importeventid() );
    return true;
}

bool
LocalFMMemMgrExporter::sendUnimportResponse( const lwswitch::memoryFlaUnimportReq &reqMsg,
                                             uint32 peerNodeId, 
                                             lwswitch::memoryReqErrors errCode )
{

    FMIntReturn_t retVal;
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::memoryFlaUnimportRsp *rspMsg = new lwswitch::memoryFlaUnimportRsp();

    rspMsg->set_unimporteventid( reqMsg.unimporteventid() );
    rspMsg->set_errcode( errCode );

    pFmMessage->set_allocated_memoryflaunimportrsp( rspMsg );
    pFmMessage->set_type( lwswitch::FM_MEMORY_FLA_UNIMPORT_RSP );
    pFmMessage->set_version( FABRIC_MANAGER_VERSION );
    pFmMessage->set_nodeid( mLocalFmControl->getLocalNodeId() );

    if( pFmMessage->nodeid() == peerNodeId )
    {
        bool returlwal = mImporter->handleUnimportResponse( pFmMessage );
        delete( pFmMessage );
        return returlwal;   
    }

    retVal =  mFMLocalCoOpMgr->SendMessageToPeerLFM( peerNodeId, pFmMessage, false );
    delete( pFmMessage );

    if( retVal != FM_INT_ST_OK )
    {
        std::ostringstream ss;
        ss << "send unimport response message to peer Memory Manager on node " << peerNodeId << " failed.";
        ss << " import event id " << reqMsg.importeventid() << " unimport event id " << reqMsg.unimporteventid();
        ss << " error " << retVal;

        FM_LOG_ERROR( "fatal error, %s", ss.str().c_str() );
        FM_SYSLOG_ERR( "fatal error, %s", ss.str().c_str() );
        sendFatalErrorToAllNodes( lwswitch::MEMORY_FLA_GENERIC_ERROR, ss.str().c_str() );
        return false;
    }
    FM_LOG_DEBUG( "Sent unimport response to node %d import event id %lu unimport event id %lu", 
                   peerNodeId, reqMsg.importeventid(), reqMsg.unimporteventid() );
    return true;
}

bool
LocalFMMemMgrExporter::handleImportRequest( lwswitch::fmMessage *pFmMessage )
{
    FM_LOG_DEBUG( "handleImportRequest called" );
    if( !pFmMessage->has_memoryflaimportreq() )
    {

        std::ostringstream ss;
        ss << "import request failed: memory import request not found in message from node id " << pFmMessage->nodeid();
        // This is a code sanity check and is highly unlikely to happen
        FM_LOG_ERROR( "fatal error, %s", ss.str().c_str() );
        FM_SYSLOG_ERR( "fatal error, %s", ss.str().c_str() );
        sendFatalErrorToAllNodes( lwswitch::MEMORY_FLA_GENERIC_ERROR, ss.str().c_str() );
        return false;
    }

    const lwswitch::memoryFlaImportReq &impReq = pFmMessage->memoryflaimportreq();

    if( !impReq.has_exportuuid() )
    {
        std::ostringstream ss;
        ss << "import request failed: export object UUID not found in message from node id " << pFmMessage->nodeid();
        // This is a code sanity check and is highly unlikely to happen
        FM_LOG_ERROR( "fatal error, %s", ss.str().c_str() );
        FM_SYSLOG_ERR( "fatal error, %s", ss.str().c_str() );
        sendFatalErrorToAllNodes( lwswitch::MEMORY_FLA_GENERIC_ERROR, ss.str().c_str() );
        return false;
    }

    std::string uuidStr = FMUtils::colwertUuidToHexStr( impReq.exportuuid().data(), LW_FABRIC_UUID_LEN );
    if( impReq.has_exportgpuid() == false ) 
    {
        std::ostringstream ss;
        ss << "import request from node id " << pFmMessage->nodeid() << " for UUID FAB-" << uuidStr.c_str();
        ss << " received with missing GPU id.";
        // This is a code sanity check and is highly unlikely to happen
        FM_LOG_ERROR( "fatal error, %s", ss.str().c_str() );
        FM_SYSLOG_ERR( "fatal error, %s", ss.str().c_str() );
        sendFatalErrorToAllNodes( lwswitch::MEMORY_FLA_GENERIC_ERROR, ss.str().c_str() );
        return false;
    }

    importEventIdKey importKey = { impReq.importeventid(), ( uint16 )pFmMessage->nodeid() };
    // If importEventIdKey already exists, it means we got an import request for the same event id more than once 
    lwosEnterCriticalSection(&mLock);
    if( mImportEventIdMap.find( importKey ) != mImportEventIdMap.end() )
    {
        std::ostringstream ss;
        ss << "import request for UUID FAB-" << uuidStr.c_str() << " from node id " << pFmMessage->nodeid();
        ss << "for duplicate import event id " << impReq.importeventid();
        // This is a code sanity check and is highly unlikely to happen
        FM_LOG_ERROR( "fatal error, %s", ss.str().c_str() );
        FM_SYSLOG_ERR( "fatal error, %s", ss.str().c_str() );
        lwosLeaveCriticalSection(&mLock);
        sendFatalErrorToAllNodes( lwswitch::MEMORY_FLA_GENERIC_ERROR, ss.str().c_str() );
        return false;
    }
    FM_LOG_DEBUG( "memory import request received from node id %d with GPU id %d UUID FAB-%s Import Event id %lu", 
                  pFmMessage->nodeid(), impReq.exportgpuid(), uuidStr.c_str(), impReq.importeventid() );

    exportObjectKey key;
    key.index = impReq.index();
    memcpy( &key.exportUuid, impReq.exportuuid().data(), LW_FABRIC_UUID_LEN );

    std::map< exportObjectKey, exportObjectData* >::iterator it;
    it = mExportObjectMap.find( key );
    exportObjectData *data = nullptr;
    if( it != mExportObjectMap.end() )
    {
        // found a cached entry for this exported object
        data = it->second;
        if( data->exportGpuId != impReq.exportgpuid()) 
        {
            std::ostringstream ss;
            ss << "memory import request received from node id " << pFmMessage->nodeid();
            ss << " for UUID FAB-" << uuidStr.c_str() << " import event id " << impReq.importeventid();
            ss << " with wrong GPU id " << impReq.exportgpuid() << " expected GPU id " << data->exportGpuId;
            FM_LOG_ERROR( "%s", ss.str().c_str() );
            FM_SYSLOG_ERR( "%s", ss.str().c_str() );
            return sendImportResponse( impReq, data, pFmMessage->nodeid(), lwswitch::GPUD_ID_MISMATCH );
        }
    }
    else
    {
        data = new exportObjectData ;

        // dup object
        LW00FA_ALLOCATION_PARAMETERS exportedRefAllocParams = { 0 };
        memcpy( &exportedRefAllocParams.exportUuid , &key.exportUuid, LW_FABRIC_UUID_LEN );

        exportedRefAllocParams.index = key.index;
        exportedRefAllocParams.flags = 0;

        if( FMHandleGenerator::allocHandle( data->exportObjectRefHandle ) == false )
        {
            FM_LOG_ERROR( "unable to allocate handle to dup exported object for UUID FAB-%s for request from node id %d", 
                          uuidStr.c_str(), pFmMessage->nodeid() );
            FM_SYSLOG_ERR( "unable to allocate handle to dup exported object for UUID FAB-%s for request from node id %d", 
                           uuidStr.c_str(), pFmMessage->nodeid() );
            // TODO we are out of 2^32 handles, no more memory import possible
            delete( data );
            return sendImportResponse( impReq, nullptr, pFmMessage->nodeid(), lwswitch::HANDLE_ALLOC_FAIL );
        }

        LwU32 retVal;
        retVal = LwRmAlloc( mHandleFmClient, 
                            mLocalFmControl->mFMGpuMgr->getMemSubDeviceHandleForGpuId( impReq.exportgpuid() ),
                            data->exportObjectRefHandle, LW_MEMORY_FABRIC_EXPORTED_REF, &exportedRefAllocParams );
        if( retVal != LW_OK )
        {
            FM_LOG_ERROR( "export object dup for UUID FAB-%s for request from node id %d failed with error %s",
                          uuidStr.c_str(), pFmMessage->nodeid(), lwstatusToString( retVal ) );
            FM_SYSLOG_ERR( "export object dup for UUID FAB-%s for request from node id %d failed with error %s",
                           uuidStr.c_str(), pFmMessage->nodeid(), lwstatusToString( retVal ) );
            FM_LOG_DEBUG( " mHandleFmClient=%d getMemSubDeviceHandleForGpuId=%d, exportObjectRefHandle=%d",
                          mHandleFmClient,
                          mLocalFmControl->mFMGpuMgr->getMemSubDeviceHandleForGpuId( impReq.exportgpuid()),
                          data->exportObjectRefHandle );
            FM_LOG_DEBUG( " exportedRefAllocParams.index = %d exportedRefAllocParams.flags = %d, UUID=%s",
                          exportedRefAllocParams.index, exportedRefAllocParams.flags,
                          FMUtils::colwertUuidToHexStr( &exportedRefAllocParams.exportUuid, 
                                                     LW_FABRIC_UUID_LEN ).c_str());
            delete( data );
            return sendImportResponse( impReq, nullptr, pFmMessage->nodeid(), lwswitch::EXPORT_OBJECT_DUP_FAIL );
        }
        
        // read PFNs
        if( readPageTableEntries( data ) == false )
        {
            std::stringstream ss;
            ss << "unable to read page table entries for UUID FAB-" << uuidStr.c_str();
            ss << " for import request from node id " << pFmMessage->nodeid();
            FM_LOG_ERROR( "fatal error, %s", ss.str().c_str() );
            FM_SYSLOG_ERR( "fatal error, %s", ss.str().c_str() );
            delete( data );
            // fatal error, after dup success reading page table entries should not fail
            lwosLeaveCriticalSection(&mLock);
            sendFatalErrorToAllNodes( lwswitch::MEMORY_FLA_GENERIC_ERROR, ss.str().c_str() );
            return false;
        }
        // save export object UUID and index
        data->key = key;

        // save GPU id of exporting GPU
        data->exportGpuId = impReq.exportgpuid();

        // add entry to mExportObjectMap
        mExportObjectMap[ key ] = data;
    }

    // When node id already exists, it means we got an import request for the same object 
    // from the same node more than once.
    if( data->nodeIds.find( pFmMessage->nodeid() ) != data->nodeIds.end() )
    {
        // send fatal error to all nodes as this is an unrecoverable code bug??
        std::stringstream ss;
        ss << "received duplicate import request for UUID FAB-" << uuidStr.c_str() ;
        ss << " from node id " << pFmMessage->nodeid();
        FM_LOG_ERROR( "fatal error, %s", ss.str().c_str() );
        FM_SYSLOG_ERR( "fatal error, %s", ss.str().c_str() );
        lwosLeaveCriticalSection(&mLock);
        sendFatalErrorToAllNodes( lwswitch::MEMORY_FLA_GENERIC_ERROR, ss.str().c_str() );
        return false;
    }

    // add node to list of importing nodes
    data->nodeIds.insert( pFmMessage->nodeid() );

    // increment import count
    data->importCount++;

    // create entry in mImportEventIdMap
    mImportEventIdMap[ importKey ] = data;

    return sendImportResponse( impReq, data, pFmMessage->nodeid(), lwswitch::MEMORY_REQ_SUCCESS );
}

bool
LocalFMMemMgrExporter::handleUnimportRequest( lwswitch::fmMessage *pFmMessage )
{
    FM_LOG_DEBUG( "handleUnimportRequest called" );
    if( !pFmMessage->has_memoryflaunimportreq() )
    {

        std::ostringstream ss;
        ss << "unimport request failed: memory unimport request not found in message from node id ";
        ss << pFmMessage->nodeid();
        // This is a code sanity check and is highly unlikely to happen
        FM_LOG_ERROR( "fatal error, %s", ss.str().c_str() );
        FM_SYSLOG_ERR( "fatal error, %s", ss.str().c_str() );
        sendFatalErrorToAllNodes( lwswitch::MEMORY_FLA_GENERIC_ERROR, ss.str().c_str() );
        return false;
    }

    const lwswitch::memoryFlaUnimportReq &unimpReq = pFmMessage->memoryflaunimportreq();
    FM_LOG_DEBUG( "memory unimport request received from node id %d Import Event id %lu Unimport Event id %lu",
                  pFmMessage->nodeid(), unimpReq.importeventid(), unimpReq.unimporteventid() );
    // create entry in mImportEventIdMap
    importEventIdKey importKey = { unimpReq.importeventid(), ( uint16 )pFmMessage->nodeid() };

    lwosEnterCriticalSection(&mLock);
    auto it = mImportEventIdMap.find( importKey );
    if ( it == mImportEventIdMap.end() )
    {
        // unimport key not found
        FM_LOG_DEBUG( "unimport key for import event id %lu node id %d not found",
                      unimpReq.importeventid(), pFmMessage->nodeid());
        lwosLeaveCriticalSection(&mLock);
        sendUnimportResponse( unimpReq, pFmMessage->nodeid(), lwswitch::UNIMPORT_OBJECT_NOT_FOUND );
        return false;
    }

    exportObjectData *data = it->second;
    // The pointer to export object data has been copied. Now the importKey can be removed from mImportEventIdMap
    mImportEventIdMap.erase( importKey );
    exportObjectKey exportEntryKey = data->key;

    data->importCount--;
    data->nodeIds.erase( pFmMessage->nodeid() );

    if( data->importCount == 0)
    {
        LwRmFree( mHandleFmClient, mHandleFmClient, data->exportObjectRefHandle );
        FMHandleGenerator::freeHandle( data->exportObjectRefHandle );
        delete( data );
        mExportObjectMap.erase( exportEntryKey );
    }
    lwosLeaveCriticalSection(&mLock);
    sendUnimportResponse( unimpReq, pFmMessage->nodeid(), lwswitch::MEMORY_REQ_SUCCESS );

    return true;
}

void
LocalFMMemMgrExporter::handleMessage( lwswitch::fmMessage *pFmMessage )
{
    if( mEnableMessageProcessing == false )
    {
        FM_LOG_DEBUG( "message processing disabled" );
        return;
    }

    // Exporter needs to call importer when the the node exporting memory is the same as the node importing memory
    if( mImporter == nullptr )
    {
        FM_LOG_ERROR( "importer not set yet, ignoring import request %d", pFmMessage->type() ); 
        FM_SYSLOG_ERR( "importer not set yet, ignoring import request %d", pFmMessage->type() ); 
        return;
    }
    
    // All messages are processed serially
    FM_LOG_DEBUG( "message type %d", pFmMessage->type());
    switch( pFmMessage->type() )
    {
    case lwswitch::FM_MEMORY_FLA_IMPORT_REQ:
        FM_LOG_DEBUG( "message type FM_MEMORY_FLA_IMPORT_REQ reveived" );
        if( handleImportRequest( pFmMessage ) )
        {
            FM_LOG_DEBUG( "handleImportRequest success" );
        }
        else
        {
            FM_LOG_DEBUG( "handleImportRequest failed" );
        }
        break;
        
    case lwswitch::FM_MEMORY_FLA_UNIMPORT_REQ:
        FM_LOG_DEBUG( "message type FM_MEMORY_FLA_UNIMPORT_REQ received" );
        if( handleUnimportRequest( pFmMessage ) )
        {
            FM_LOG_DEBUG( "handleUnimportRequest success" );
        }
        else
        {
            FM_LOG_DEBUG( "handleUnimportRequest failed" );
        }
        break;

    case lwswitch::FM_MEMORY_FLA_FATAL_ERROR_MSG:
        FM_LOG_DEBUG( "message type FM_MEMORY_FLA_FATAL_ERROR_MSG received" );
        handleFatalErrorMsg( pFmMessage );
        break;

    default:
        FM_LOG_ERROR( "unknown message type %d received from node id %d by Memory Manager", 
                      pFmMessage->type(), pFmMessage->nodeid() );
        FM_SYSLOG_ERR( "unknown message type %d received from node id %d by Memory Manager", 
                       pFmMessage->type(), pFmMessage->nodeid() );
        break;
    }
}

void
LocalFMMemMgrExporter::sendFatalErrorToOneNode( uint16 peerNodeId,
                                                lwswitch::memoryFlaFatalErrors errCode,
                                                const char *errString )
{
    FMIntReturn_t retVal;
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::memoryFlaFatalErrorMsg *errMsg = new lwswitch::memoryFlaFatalErrorMsg();

    errMsg->set_errcode( errCode );
    errMsg->set_errmessage( errString ); 

    pFmMessage->set_allocated_memoryflafatalerrormsg( errMsg );
    pFmMessage->set_type( lwswitch::FM_MEMORY_FLA_FATAL_ERROR_MSG );
    pFmMessage->set_version( FABRIC_MANAGER_VERSION );
    pFmMessage->set_nodeid( mLocalFmControl->getLocalNodeId() );

    if( pFmMessage->nodeid() == peerNodeId )
    {
        handleFatalErrorMsg( pFmMessage );
        delete( pFmMessage );
        return;
    }

    retVal = mFMLocalCoOpMgr->SendMessageToPeerLFM( peerNodeId, pFmMessage, false );
    delete( pFmMessage );

    if( retVal != FM_INT_ST_OK )
    {
        FM_LOG_ERROR( "send fatal error message to peer Memory Manager failed, peer node id %d error %d message %s", 
                      peerNodeId, retVal, errString );
        FM_SYSLOG_ERR( "send fatal error message to peer Memory Manager failed, peer node id %d error %d message %s", 
                       peerNodeId, retVal, errString );
        return;
    }
    FM_LOG_DEBUG( "Sent fatal error message to node %d message=%s", peerNodeId, errString );
}

void
LocalFMMemMgrExporter::sendFatalErrorToAllNodes( lwswitch::memoryFlaFatalErrors errCode, const char *errString )
{
    // TODO send message to GFM as well
    // loop on all LFM nodes
    std::set< uint32 > peerNodeIds;
    mFMLocalCoOpMgr->getPeerNodeIds( peerNodeIds );
    for( auto it = peerNodeIds.begin(); it != peerNodeIds.end(); it++ )
    {
        sendFatalErrorToOneNode( *it, errCode, errString );
    }
}

void
LocalFMMemMgrExporter::handleFatalErrorMsg( lwswitch::fmMessage *pFmMessage )
{
    const lwswitch::memoryFlaFatalErrorMsg &fatalErrMsg = pFmMessage->memoryflafatalerrormsg();

    // log fatal error message
    FM_LOG_ERROR( "fatal error seen on node %d error = %d error message: %s",
                  pFmMessage->nodeid(), fatalErrMsg.errcode(), fatalErrMsg.errmessage().c_str() );
    FM_SYSLOG_ERR( "fatal error seen on node %d error = %d error message: %s",
                   pFmMessage->nodeid(), fatalErrMsg.errcode(), fatalErrMsg.errmessage().c_str() );
    mLocalFmControl->onConfigDeInitReqRcvd();
}

void
LocalFMMemMgrExporter::disableMessageProcessing()
{
    if( mEnableMessageProcessing == true )
    {
        FM_LOG_ERROR( "memory exporter disabling processing for memory import/unimport messages" );
        FM_SYSLOG_ERR( "memory exporter disabling processing for memory import/unimport messages" );
        mEnableMessageProcessing = false;
    }
}
#endif
