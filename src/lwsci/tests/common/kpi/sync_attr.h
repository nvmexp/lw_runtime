//! \file
//! \brief LwSciSync attr.
//!
//! \copyright
//! Copyright (c) 2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef SYNC_ATTR_H
#define SYNC_ATTR_H

#include <vector>

#include "lwscisync.h"
#include "lwscisync_internal.h"

class SyncAttr {
public:
    std::vector<LwSciSyncAttrKeyValuePair> signalerKeyVal;
    std::vector<LwSciSyncAttrKeyValuePair> waiterKeyVal;
    std::vector<LwSciSyncInternalAttrKeyValuePair> signalerIntKeyVal;
    std::vector<LwSciSyncInternalAttrKeyValuePair> waiterIntKeyVal;
};

class VolvoSyncAttr: public SyncAttr {
public:
    VolvoSyncAttr() {
        signalerKeyVal.push_back({
            LwSciSyncAttrKey_RequiredPerm,
            (void*)&signalPerm,
            sizeof(signalPerm)
        });

        waiterKeyVal.push_back({
            LwSciSyncAttrKey_RequiredPerm,
            (void*)&waiterPerm,
            sizeof(waiterPerm) });

        signalerKeyVal.push_back({
            LwSciSyncAttrKey_NeedCpuAccess,
            (void*)&cpuFlag,
            sizeof(cpuFlag)
        });

        waiterKeyVal.push_back({
            LwSciSyncAttrKey_NeedCpuAccess,
            (void*)&cpuFlag,
            sizeof(cpuFlag) });

        signalerIntKeyVal.push_back({
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
            (void*)&primitiveType,
            sizeof(primitiveType)
        });
        signalerIntKeyVal.push_back({
            LwSciSyncInternalAttrKey_SignalerPrimitiveCount,
            (void*)&primitiveCount,
            sizeof(primitiveCount)
        });

        waiterIntKeyVal.push_back({
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
            (void*)&primitiveType,
            sizeof(primitiveType)
        });
    };
private:
    LwSciSyncAccessPerm signalPerm = LwSciSyncAccessPerm_SignalOnly;
    LwSciSyncAccessPerm waiterPerm = LwSciSyncAccessPerm_WaitOnly;
    LwSciSyncInternalAttrValPrimitiveType primitiveType =
        LwSciSyncInternalAttrValPrimitiveType_Syncpoint;
    uint32_t primitiveCount = 1U;
    bool cpuFlag = true;
};

#endif // SYNC_ATTR_H
