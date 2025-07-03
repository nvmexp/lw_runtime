/*
 * Copyright (c) 2022 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include "lwscibuf_c2c_internal.h"
#include "lwscisync_test_attribute_list.h"
#include "lwscisync_interprocess_test.h"
#include "lwscisync_c2cExpDesc.h"

class LwSciSyncC2C : public LwSciSyncInterProcessTest
{
};

static LwSciError fillCpuSignalerAttr(
    LwSciSyncAttrList signalerAttrList)
{
    LwSciError err = LwSciError_Success;

    bool cpuSignaler = true;
    LwSciSyncAccessPerm signalerAccessPerm = LwSciSyncAccessPerm_SignalOnly;
    LwSciSyncAttrKeyValuePair signalerKeyValue[] = {
        {    .attrKey = LwSciSyncAttrKey_NeedCpuAccess,
             .value = (void*) &cpuSignaler,
             .len = sizeof(cpuSignaler),
        },
        {    .attrKey = LwSciSyncAttrKey_RequiredPerm,
             .value = (void*) &signalerAccessPerm,
             .len = sizeof(signalerAccessPerm),
        },
    };

    err =  LwSciSyncAttrListSetAttrs(signalerAttrList, signalerKeyValue,
        sizeof(signalerKeyValue)/sizeof(LwSciSyncAttrKeyValuePair));
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}

static LwSciError fillWaiter_copyDoneConsPublic(
    LwSciSyncAttrList waiterAttrList)
{
    LwSciError err = LwSciError_Success;

    bool cpuWaiter = true;
    LwSciSyncAccessPerm waiterAccessPerm = LwSciSyncAccessPerm_WaitOnly;

    LwSciSyncAttrKeyValuePair waiterKeyValue[] = {
        {    .attrKey = LwSciSyncAttrKey_NeedCpuAccess,
             .value = (void*) &cpuWaiter,
             .len = sizeof(cpuWaiter),
        },
        {    .attrKey = LwSciSyncAttrKey_RequiredPerm,
             .value = (void*) &waiterAccessPerm,
             .len = sizeof(waiterAccessPerm),
        },
    };

    err =  LwSciSyncAttrListSetAttrs(waiterAttrList, waiterKeyValue,
        sizeof(waiterKeyValue)/sizeof(LwSciSyncAttrKeyValuePair));
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}

static LwSciError fillWaiter_copyDoneConsInternal(
    LwSciSyncAttrList waiterAttrList)
{
    LwSciError err = LwSciError_Success;

#if (defined(__x86_64__))
    LwSciSyncInternalAttrValPrimitiveType primitiveInfo[] =
        { LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore };
#else
    LwSciSyncInternalAttrValPrimitiveType primitiveInfo[] =
        { LwSciSyncInternalAttrValPrimitiveType_Syncpoint,
          LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore};
#endif
    LwSciSyncInternalAttrKeyValuePair waiterInternalKeyValue[] = {
        {    .attrKey = LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
             .value = (void*) primitiveInfo,
             .len = sizeof(primitiveInfo),
        },
    };

    err = LwSciSyncAttrListSetInternalAttrs(waiterAttrList, waiterInternalKeyValue,
        sizeof(waiterInternalKeyValue)/sizeof(LwSciSyncInternalAttrKeyValuePair));
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}

static LwSciError fillWaiter_copyDoneCons(
    LwSciSyncAttrList waiterAttrList)
{
    LwSciError err = LwSciError_Success;

    err = fillWaiter_copyDoneConsPublic(waiterAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }

    err = fillWaiter_copyDoneConsInternal(waiterAttrList);
    if (err != LwSciError_Success) {
        goto fail;
    }

fail:
    return err;
}

/** @jama{TBD} C2C reconciliation test
 * In this test, the waiter reconciles and verifies primitive info
 * of the signaler.
 */
TEST_F(LwSciSyncC2C, DISABLED_C2CReconcilex86Orin)
{
    LwSciError error = LwSciError_Success;
#if (defined(__x86_64__))
    LwSciSyncInternalAttrValPrimitiveType recPrimitiveType =
        LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore;
#else
    LwSciSyncInternalAttrValPrimitiveType recPrimitiveType =
        LwSciSyncInternalAttrValPrimitiveType_Syncpoint;
#endif

    auto peer = std::make_shared<LwSciSyncIpcPeer>();
    peers.push_back(peer);
    /* TODO: Replace this with INTER_CHIP ipcEndpoint */
    peer->SetUp("lwscisync_a_0");

    // consReadsDoneProdObj - Producer fills in C2C-copy waiter attributes
    auto consReadsDoneProdObjWaitAttrList = peer->createAttrList(&error);
    ASSERT_EQ(error, LwSciError_Success);
    error = LwSciSyncFillAttrsIndirectChannelC2c(
        peer->ipcEndpoint->getEndpoint(),
        consReadsDoneProdObjWaitAttrList.get(),
        LwSciSyncAccessPerm_WaitOnly);
    ASSERT_EQ(LwSciError_Success, error);

    auto consReadsDoneProdObjSigAttrList =
        peer->importUnreconciledList(consReadsDoneProdSigAttrListDesc, &error);
    ASSERT_EQ(error, LwSciError_Success);

    auto consReadsDoneProdObjRecList = LwSciSyncPeer::reconcileLists(
        {consReadsDoneProdObjSigAttrList.get(),
         consReadsDoneProdObjWaitAttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(consReadsDoneProdObjRecList.get(), nullptr);

    peer->verifyInternalAttr(consReadsDoneProdObjRecList.get(),
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
            recPrimitiveType);
    peer->verifyInternalAttr(consReadsDoneProdObjRecList.get(),
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
            recPrimitiveType);

    // copyDoneConsObj - Consumer fills in engine waiter attributes
    auto copyDoneConsObjWaitAttrList = peer->createAttrList(&error);
    ASSERT_EQ(error, LwSciError_Success);

    error = fillWaiter_copyDoneCons(copyDoneConsObjWaitAttrList.get());
    ASSERT_EQ(LwSciError_Success, error);

    auto copyDoneConsObjSigAttrList =
        peer->importUnreconciledList(copyDoneConsObjSigAttrListDesc, &error);
    ASSERT_EQ(error, LwSciError_Success);

    auto copyDoneConsObjRecList = LwSciSyncPeer::reconcileLists(
        {copyDoneConsObjSigAttrList.get(),
         copyDoneConsObjWaitAttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    ASSERT_NE(copyDoneConsObjRecList.get(), nullptr);

    peer->verifyInternalAttr(copyDoneConsObjRecList.get(),
            LwSciSyncInternalAttrKey_SignalerPrimitiveInfo,
            recPrimitiveType);
    peer->verifyInternalAttr(copyDoneConsObjRecList.get(),
            LwSciSyncInternalAttrKey_WaiterPrimitiveInfo,
            recPrimitiveType);
}

/** @jama{TBD} In this test, export descriptors are generated for attrList
 *  to be transfered to remote end.
 */
TEST_F(LwSciSyncC2C, DISABLED_GenerateC2CExpDesc)
{
    LwSciError error = LwSciError_Success;
    FILE *ptr_fp = NULL;
    uint8_t *desc = NULL;

    auto peer = std::make_shared<LwSciSyncIpcPeer>();
    peers.push_back(peer);
    /* TODO: Replace this with INTER_CHIP ipcEndpoint */
    peer->SetUp("lwscisync_a_1");

    // consReadsDoneProdObj - Consumer fills in CPU signaler attributes
    auto consReadsDoneProdObjSigAttrList = peer->createAttrList(&error);
    ASSERT_EQ(error, LwSciError_Success);
    error = fillCpuSignalerAttr(
        consReadsDoneProdObjSigAttrList.get());
    ASSERT_EQ(LwSciError_Success, error);

    // Export unreconciled signaler list to the waiter
    auto consReadsDoneProdSigAttrListDesc = peer->exportUnreconciledList(
            {consReadsDoneProdObjSigAttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);

    // Copy the export desc to file
    if ((ptr_fp = fopen("c2cExpDesc.h", "wb")) == NULL)
    {
        printf("Unable to open file!\n");
        exit(1);
    } else {
        printf("Opened file successfully for writing.\n");
    }

    desc = (uint8_t *)consReadsDoneProdSigAttrListDesc.first.get();

    fprintf(ptr_fp, "%s", "#include <cstdint>\n");
    fprintf(ptr_fp, "%s", "const std::vector<unsigned char> consReadsDoneProdSigAttrListDesc = {");
    for (int i = 0; i < consReadsDoneProdSigAttrListDesc.second; i++) {
        fprintf(ptr_fp, "%d", *desc);
        if (i != (consReadsDoneProdSigAttrListDesc.second-1))
            fprintf(ptr_fp, "%s", ", ");
        desc++;
    }
    fprintf(ptr_fp, "%s", "};\n\n");

    // copyDoneConsObj - Producer fills in C2C-copy signaler attributes
    auto copyDoneConsObjSigAttrList = peer->createAttrList(&error);
    ASSERT_EQ(error, LwSciError_Success);
    error = LwSciSyncFillAttrsIndirectChannelC2c(
        peer->ipcEndpoint->getEndpoint(),
        copyDoneConsObjSigAttrList.get(),
        LwSciSyncAccessPerm_SignalOnly);
    ASSERT_EQ(LwSciError_Success, error);

    // Export unreconciled signaler list to the waiter
    auto copyDoneConsObjSigAttrListDesc = peer->exportUnreconciledList(
            {copyDoneConsObjSigAttrList.get()}, &error);
    ASSERT_EQ(error, LwSciError_Success);
    desc = (uint8_t *)copyDoneConsObjSigAttrListDesc.first.get();

    fprintf(ptr_fp, "%s", "const std::vector<unsigned char> copyDoneConsObjSigAttrListDesc = {");
    for (int i = 0; i < copyDoneConsObjSigAttrListDesc.second; i++) {
        fprintf(ptr_fp, "%d", *desc);
        if (i != (copyDoneConsObjSigAttrListDesc.second-1))
            fprintf(ptr_fp, "%s", ", ");
        desc++;
    }
    fprintf(ptr_fp, "%s", "};\n\n");
    fclose(ptr_fp);
}
