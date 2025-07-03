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
#pragma once

#include <g_lwconfig.h>
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)

#include "fabricmanager.pb.h"
#include "memmgr.pb.h"
#include "class/cl00fa.h"
#include "lwos.h"
#include <map>
#include <vector>
#include <array>
#include <set>
#include "FMHandleGenerator.h"


class LocalFabricManagerControl;
class LocalFMCoOpMgr;
class LocalFMMemMgrImporter;

class LocalFMMemMgrExporter
{
    struct exportObjectKey {
        std::array<uint8,LW_FABRIC_UUID_LEN>  exportUuid;
        uint16 index;

        inline bool operator < ( const exportObjectKey& l ) const {
            if( l.index == index )
                return l.exportUuid < exportUuid;
            else
                return l.index < index;
        }
    };

    struct exportObjectData {
        uint32                  exportGpuId;
        std::vector< uint32 >   ffn;
        uint32                  pageSize;
        uint32                  memFlags;
        uint64                  size;
        uint32                  kind;
        uint16                  importCount;
        std::set< uint16 >      nodeIds;
        uint32                  exportObjectRefHandle;
        // This key is required at unimport time in case importCount drops to 0 and we need to delete this entry.
        // So we need a way at unimport time to go from importEventIdKey to exportObjectData to exportObjectKey
        exportObjectKey         key;

        exportObjectData() {
            importCount = 0;
        }
    };


    struct importEventIdKey {
        uint64  importEventId;
        uint16  importNodeId;

        inline bool operator < ( const importEventIdKey& l ) const {
            if( l.importNodeId == importNodeId )
                return l.importEventId < importEventId;
            else
                return l.importNodeId < importNodeId;
        }

    };

    // Both mExportObjectMap & mImportEventIdMap have a pointer to exportObjectData
    // This is because during import request processing we only get exportObjectKey where as during unimport
    // we only get importEventIdKey. However, exportObjectData needs to be modified at both import and unimport time.
    std::map< exportObjectKey, exportObjectData* > mExportObjectMap;
    std::map< importEventIdKey, exportObjectData* > mImportEventIdMap;

    LocalFMCoOpMgr *mFMLocalCoOpMgr;
    LocalFabricManagerControl *mLocalFmControl;
    LocalFMMemMgrImporter *mImporter;

    LwHandle mHandleFmClient;
    LwHandle mHandleFmSession;
    LWOSCriticalSection mLock;
    bool mEnableMessageProcessing;

public:
    LocalFMMemMgrExporter( LocalFMCoOpMgr *pLocalCoopMgr, LocalFabricManagerControl *pLocalFmControl);
    ~LocalFMMemMgrExporter();
    void setImporter(LocalFMMemMgrImporter *importer) {
        mImporter = importer;
    };

    void handleMessage( lwswitch::fmMessage *pFmMessage );
    bool handleUnimportRequest( lwswitch::fmMessage *pFmMessage );
    bool handleImportRequest( lwswitch::fmMessage *pFmMessage );
    void disableMessageProcessing();
    void sendFatalErrorToAllNodes( lwswitch::memoryFlaFatalErrors errCode, const char *errString );
private:
    bool sendImportResponse( const lwswitch::memoryFlaImportReq &reqMsg, exportObjectData *data, uint32 peerNodeId, 
                             lwswitch::memoryReqErrors errCode );
    bool sendUnimportResponse( const lwswitch::memoryFlaUnimportReq &reqMsg, uint32 peerNodeId,
                               lwswitch::memoryReqErrors errCode );

    bool readPageTableEntries( exportObjectData *data );
    
    void sendFatalErrorToOneNode( uint16 peerNodeId, lwswitch::memoryFlaFatalErrors errCode, const char *errString );

    void handleFatalErrorMsg( lwswitch::fmMessage *pFmMessage );
};
#endif
