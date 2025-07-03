//! \file
//! \brief LwSciSync kpi perf test.
//!
//! \copyright
//! Copyright (c) 2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef TEST_H
#define TEST_H

#include "lwscisync.h"
#include "lwscisync_internal.h"

#include "sync_attr.h"
#include "kpitimer.h"
#include "util.h"

class SyncTest
{
public:
    virtual void run(void) = 0;

    virtual ~SyncTest() {
        if (syncModule != nullptr) {
            LwSciSyncModuleClose(syncModule);
        }
        if (signalerList != nullptr) {
            LwSciSyncAttrListFree(signalerList);
        }
        if (waiterList != nullptr) {
            LwSciSyncAttrListFree(waiterList);
        }
    };

protected:
    SyncTest() = default;

    KPItimer timer;
    LwSciSyncModule syncModule{ nullptr };
    LwSciSyncAttrList signalerList{ nullptr };
    LwSciSyncAttrList waiterList{ nullptr };
};

class ModuleOpen : public SyncTest
{
public:
    ModuleOpen() = default;
    virtual ~ModuleOpen() = default;

    virtual void run(void)
    {
        KPIStart(&timer);
        LwSciError err = LwSciSyncModuleOpen(&syncModule);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    };
};

class AttrListCreate : public SyncTest
{
public:
    AttrListCreate() = default;
    virtual ~AttrListCreate() = default;

    virtual void run(void)
    {
        CHECK_LWSCIERR(LwSciSyncModuleOpen(&syncModule));

        KPIStart(&timer);
        LwSciError err = LwSciSyncAttrListCreate(syncModule, &signalerList);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    };
};

class AttrListSetAttrs_Signaler : public SyncTest
{
public:
    AttrListSetAttrs_Signaler() = default;
    virtual ~AttrListSetAttrs_Signaler() = default;

    virtual void run(void)
    {
        CHECK_LWSCIERR(LwSciSyncModuleOpen(&syncModule));
        CHECK_LWSCIERR(LwSciSyncAttrListCreate(syncModule, &signalerList));

        VolvoSyncAttr attr;

        KPIStart(&timer);
        LwSciError err = LwSciSyncAttrListSetAttrs(
                            signalerList,
                            attr.signalerKeyVal.data(),
                            attr.signalerKeyVal.size());
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    };
};

class AttrListSetAttrs_Waiter : public SyncTest
{
public:
    AttrListSetAttrs_Waiter() = default;
    virtual ~AttrListSetAttrs_Waiter() = default;

    virtual void run(void)
    {
        CHECK_LWSCIERR(LwSciSyncModuleOpen(&syncModule));
        CHECK_LWSCIERR(LwSciSyncAttrListCreate(syncModule, &waiterList));

        VolvoSyncAttr attr;

        KPIStart(&timer);
        LwSciError err = LwSciSyncAttrListSetAttrs(
                            waiterList,
                            attr.waiterKeyVal.data(),
                            attr.waiterKeyVal.size());
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    }
};

class AttrListSetInternalAttrs_Signaler : public SyncTest
{
public:
    AttrListSetInternalAttrs_Signaler() = default;
    virtual ~AttrListSetInternalAttrs_Signaler() = default;

    virtual void run(void)
    {
        CHECK_LWSCIERR(LwSciSyncModuleOpen(&syncModule));
        CHECK_LWSCIERR(LwSciSyncAttrListCreate(syncModule, &signalerList));

        VolvoSyncAttr attr;
        CHECK_LWSCIERR(LwSciSyncAttrListSetAttrs(
                        signalerList,
                        attr.signalerKeyVal.data(),
                        attr.signalerKeyVal.size()));

        KPIStart(&timer);
        LwSciError err = LwSciSyncAttrListSetInternalAttrs(
                            signalerList,
                            attr.signalerIntKeyVal.data(),
                            attr.signalerIntKeyVal.size());
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    }
};

class AttrListSetInternalAttrs_Waiter : public SyncTest
{
public:
    AttrListSetInternalAttrs_Waiter() = default;
    virtual ~AttrListSetInternalAttrs_Waiter() = default;

    virtual void run(void)
    {
        CHECK_LWSCIERR(LwSciSyncModuleOpen(&syncModule));

        VolvoSyncAttr attr;

        CHECK_LWSCIERR(LwSciSyncAttrListCreate(syncModule, &waiterList));
        CHECK_LWSCIERR(LwSciSyncAttrListSetAttrs(
                        waiterList,
                        attr.waiterKeyVal.data(),
                        attr.waiterKeyVal.size()));

        KPIStart(&timer);
        LwSciError err = LwSciSyncAttrListSetInternalAttrs(
                            waiterList,
                            attr.waiterIntKeyVal.data(),
                            attr.waiterIntKeyVal.size());
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    }
};

class AttrListReconcile : public SyncTest
{
public:
    AttrListReconcile() = default;

    virtual ~AttrListReconcile()
    {
        if (newConflictList != nullptr) {
            LwSciSyncAttrListFree(newConflictList);
        }
        if (reconciledList != nullptr) {
            LwSciSyncAttrListFree(reconciledList);
        }
    }

    virtual void run(void)
    {
        setupAttrLists();

        LwSciSyncAttrList unreconciledList[2] =
            { signalerList,  waiterList };

        KPIStart(&timer);
        LwSciError err = LwSciSyncAttrListReconcile(
                            unreconciledList,
                            2U,
                            &reconciledList,
                            &newConflictList);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    }

protected:
    void setupAttrLists()
    {
        CHECK_LWSCIERR(LwSciSyncModuleOpen(&syncModule));

        VolvoSyncAttr attr;

        CHECK_LWSCIERR(LwSciSyncAttrListCreate(syncModule, &signalerList));
        CHECK_LWSCIERR(LwSciSyncAttrListSetAttrs(
                        signalerList,
                        attr.signalerKeyVal.data(),
                        attr.signalerKeyVal.size()));
        CHECK_LWSCIERR(LwSciSyncAttrListSetInternalAttrs(
                        signalerList,
                        attr.signalerIntKeyVal.data(),
                        attr.signalerIntKeyVal.size()));

        CHECK_LWSCIERR(LwSciSyncAttrListCreate(syncModule, &waiterList));
        CHECK_LWSCIERR(LwSciSyncAttrListSetAttrs(
                        waiterList,
                        attr.waiterKeyVal.data(),
                        attr.waiterKeyVal.size()));
        CHECK_LWSCIERR(LwSciSyncAttrListSetInternalAttrs(
                        waiterList,
                        attr.waiterIntKeyVal.data(),
                        attr.waiterIntKeyVal.size()));
    };

    LwSciSyncAttrList newConflictList{ nullptr };
    LwSciSyncAttrList reconciledList{ nullptr };
};

class ObjAlloc : public AttrListReconcile
{
public:
    ObjAlloc() = default;

    virtual ~ObjAlloc()
    {
        if (syncObj != nullptr) {
            LwSciSyncObjFree(syncObj);
        }
    };

    virtual void run(void)
    {
        reconcileAttrList();

        KPIStart(&timer);
        LwSciError err =
            LwSciSyncObjAlloc(reconciledList, &syncObj);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    };

protected:
    void reconcileAttrList()
    {
        setupAttrLists();
        LwSciSyncAttrList unreconciledList[2] =
            { signalerList,  waiterList };
        CHECK_LWSCIERR(LwSciSyncAttrListReconcile(
            unreconciledList,
            2U,
            &reconciledList,
            &newConflictList));
    };

    LwSciSyncObj syncObj{ nullptr };
};

class ObjGetPrimitiveType : public ObjAlloc
{
public:
    ObjGetPrimitiveType() = default;
    virtual ~ObjGetPrimitiveType() = default;

    virtual void run(void)
    {
        reconcileAttrList();
        CHECK_LWSCIERR(LwSciSyncObjAlloc(reconciledList, &syncObj));

        LwSciSyncInternalAttrValPrimitiveType type;

        KPIStart(&timer);
        LwSciError err = LwSciSyncObjGetPrimitiveType(syncObj, &type);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    };
};

class ObjGetNumPrimitives : public ObjAlloc
{
public:
    ObjGetNumPrimitives() = default;
    virtual ~ObjGetNumPrimitives() = default;

    virtual void run(void)
    {
        reconcileAttrList();
        CHECK_LWSCIERR(LwSciSyncObjAlloc(reconciledList, &syncObj));

        uint32_t numPrimitives = 0U;

        KPIStart(&timer);
        LwSciError err = LwSciSyncObjGetNumPrimitives(syncObj, &numPrimitives);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    };
};

class ObjDup : public ObjAlloc
{
public:
    ObjDup() = default;

    virtual ~ObjDup()
    {
        if (syncObjDup != nullptr) {
            LwSciSyncObjFree(syncObjDup);
        }
    };

    virtual void run(void)
    {
        reconcileAttrList();
        CHECK_LWSCIERR(LwSciSyncObjAlloc(reconciledList, &syncObj));

        KPIStart(&timer);
        LwSciError err = LwSciSyncObjDup(syncObj, &syncObjDup);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
    };

protected:
    LwSciSyncObj syncObjDup{ nullptr };
};

class ObjRef : public ObjAlloc
{
public:
    ObjRef() = default;
    virtual ~ObjRef() = default;

    virtual void run(void)
    {
        reconcileAttrList();
        CHECK_LWSCIERR(LwSciSyncObjAlloc(reconciledList, &syncObj));

        KPIStart(&timer);
        LwSciError err = LwSciSyncObjRef(syncObj);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
        LwSciSyncObjFree(syncObj);
    };
};

class FenceExtract : public ObjAlloc
{
public:
    FenceExtract() = default;
    virtual ~FenceExtract() = default;

    virtual void run(void)
    {
        LwSciSyncFence syncFence = LwSciSyncFenceInitializer;
        uint64_t fenceId = 0;
        uint64_t fenceValue = 0;

        reconcileAttrList();
        CHECK_LWSCIERR(LwSciSyncObjAlloc(reconciledList, &syncObj));
        CHECK_LWSCIERR(LwSciSyncObjGenerateFence(syncObj, &syncFence));

        KPIStart(&timer);
        LwSciError err = LwSciSyncFenceExtractFence(&syncFence, &fenceId, &fenceValue);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
        LwSciSyncFenceClear(&syncFence);
    };
};

class FenceUpdate : public ObjAlloc
{
public:
    FenceUpdate() = default;
    virtual ~FenceUpdate() = default;

    virtual void run(void)
    {
        LwSciSyncFence syncFence = LwSciSyncFenceInitializer;
        LwSciSyncFence newFence = LwSciSyncFenceInitializer;
        uint64_t fenceId = 0;
        uint64_t fenceValue = 0;

        reconcileAttrList();
        CHECK_LWSCIERR(LwSciSyncObjAlloc(reconciledList, &syncObj));
        CHECK_LWSCIERR(LwSciSyncObjGenerateFence(syncObj, &syncFence));
        CHECK_LWSCIERR(LwSciSyncFenceExtractFence(&syncFence, &fenceId, &fenceValue));

        KPIStart(&timer);
        LwSciError err = LwSciSyncFenceUpdateFence(syncObj, fenceId, fenceValue, &newFence);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
        LwSciSyncFenceClear(&syncFence);
        LwSciSyncFenceClear(&newFence);
    };
};

class FenceDup : public ObjAlloc
{
public:
    FenceDup() = default;
    virtual ~FenceDup() = default;

    virtual void run(void)
    {
        LwSciSyncFence syncFence = LwSciSyncFenceInitializer;
        LwSciSyncFence fenceDup = LwSciSyncFenceInitializer;
        uint64_t fenceId = 0;
        uint64_t fenceValue = 0;

        reconcileAttrList();
        CHECK_LWSCIERR(LwSciSyncObjAlloc(reconciledList, &syncObj));
        CHECK_LWSCIERR(LwSciSyncObjGenerateFence(syncObj, &syncFence));

        KPIStart(&timer);
        LwSciError err = LwSciSyncFenceDup(&syncFence, &fenceDup);
        KPIEnd(&timer);
        CHECK_LWSCIERR(err);
        LwSciSyncFenceClear(&syncFence);
        LwSciSyncFenceClear(&fenceDup);
    };
};
#endif // TEST_H
