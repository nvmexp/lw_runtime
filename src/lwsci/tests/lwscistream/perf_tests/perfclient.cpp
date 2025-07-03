//! \file
//! \brief LwSciStream test client declaration.
//!
//! \copyright
//! Copyright (c) 2019-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include "perfclient.h"

extern TestArg testArg;

PerfClient::PerfClient(LwSciBufModule buf,
                       LwSciSyncModule sync):
    bufModule(buf),
    syncModule(sync)
{
    syncs.resize(NUM_ELEMENTS, nullptr);

    for (uint32_t i{ 0U }; i < NUM_PACKETS; i++) {
        packets[i].buffers.fill(nullptr);
    }
}

PerfClient::~PerfClient(void)
{
    if (waitContext != nullptr) {
        LwSciSyncCpuWaitContextFree(waitContext);
        waitContext = nullptr;
    }

    for (uint32_t i{ 0U }; i < NUM_PACKETS; i++) {
        for (uint32_t j{ 0U }; j < NUM_ELEMENTS; j++) {
            if (packets[i].buffers[j] != nullptr) {
                LwSciBufObjFree(packets[i].buffers[j]);
                packets[i].buffers[j] = nullptr;
            }
        }
    }

    for (uint32_t i{ 0U }; i < testArg.numSyncs; i++) {
        if (syncs[i] != nullptr) {
            LwSciSyncObjFree(syncs[i]);
            syncs[i] = nullptr;
        }
    }

    if (endpointHandle != 0U) {
        LwSciStreamBlockDelete(endpointHandle);
        endpointHandle = 0U;
    }
}

void PerfClient::setCpuSyncAttrList(LwSciSyncAccessPerm cpuPerm,
                                    LwSciSyncAttrList attrList)
{
    LwSciSyncAttrKeyValuePair keyValue[2];
    bool cpuSync{ true };
    keyValue[0].attrKey = LwSciSyncAttrKey_NeedCpuAccess;
    keyValue[0].value = (void*)&cpuSync;
    keyValue[0].len = sizeof(cpuSync);
    keyValue[1].attrKey = LwSciSyncAttrKey_RequiredPerm;
    keyValue[1].value = (void*)&cpuPerm;
    keyValue[1].len = sizeof(cpuPerm);
    CHECK_LWSCIERR(LwSciSyncAttrListSetAttrs(attrList, keyValue, 2));
}

void PerfClient::setupSync(void)
{
    // Query max number of sync objects supported by LwSciStream.
    int32_t maxNumSync{ 0 };
    CHECK_LWSCIERR(LwSciStreamAttributeQuery(
                    LwSciStreamQueryableAttrib_MaxSyncObj,
                    &maxNumSync));

    LwSciStreamEventType event;

    // Send sync requirement
    LwSciSyncAttrList waiterAttrList{ nullptr };

    if (testArg.numSyncs > 0U) {
        CHECK_LWSCIERR(LwSciSyncCpuWaitContextAlloc(syncModule, &waitContext));

        CHECK_LWSCIERR(LwSciSyncAttrListCreate(syncModule, &waiterAttrList));
        setCpuSyncAttrList(LwSciSyncAccessPerm_WaitOnly, waiterAttrList);
    }

    assert(testArg.numSyncs <= NUM_ELEMENTS);
    for (uint32_t i{ 0U }; i < testArg.numSyncs; i++) {
        CHECK_LWSCIERR(
            LwSciStreamBlockElementWaiterAttrSet(
                endpointHandle, i, waiterAttrList));
    }
    CHECK_LWSCIERR(
        LwSciStreamBlockSetupStatusSet(
            endpointHandle, LwSciStreamSetup_WaiterAttrExport, true));

    LwSciSyncAttrListFree(waiterAttrList);
}

void PerfClient::recvWaiterAttr(void)
{
    LwSciSyncAttrList signalerAttrList{ nullptr };
    uint32_t usedNumSyncs{ 0U };

    if (testArg.numSyncs > 0U) {
        CHECK_LWSCIERR(LwSciSyncAttrListCreate(syncModule, &signalerAttrList));
        setCpuSyncAttrList(LwSciSyncAccessPerm_SignalOnly, signalerAttrList);
        for (uint32_t i{ 0U }; i < NUM_ELEMENTS; i++) {
            if (i < testArg.numSyncs) {
                LwSciSyncAttrList eventSyncAttrList;
                CHECK_LWSCIERR(
                    LwSciStreamBlockElementWaiterAttrGet(
                        endpointHandle, i, &eventSyncAttrList));
                if (eventSyncAttrList != nullptr) {
                    // Reconcile sync attr with sync requirement from the other end
                    LwSciSyncAttrList unreconciledList[2]{ signalerAttrList,
                                                           eventSyncAttrList };
                    LwSciSyncAttrList reconciledList{ nullptr };
                    LwSciSyncAttrList newConflictList{ nullptr };
                    CHECK_LWSCIERR(LwSciSyncAttrListReconcile(unreconciledList,
                                                              2U,
                                                              &reconciledList,
                                                              &newConflictList));
                    LwSciSyncObjAlloc(reconciledList, &syncs[i]);
                    LwSciSyncAttrListFree(reconciledList);
                    LwSciSyncAttrListFree(newConflictList);
                    LwSciSyncAttrListFree(eventSyncAttrList);
                    usedNumSyncs++;
                }
            }
        }

        LwSciSyncAttrListFree(signalerAttrList);
    }
    CHECK_LWSCIERR(
        LwSciStreamBlockSetupStatusSet(
            endpointHandle, LwSciStreamSetup_WaiterAttrImport, true));

    // Send sync objects
    for (uint32_t i{ 0U }; i < usedNumSyncs; i++) {
        CHECK_LWSCIERR(LwSciStreamBlockElementSignalObjSet(
            endpointHandle, i, syncs[i]));
    }
    CHECK_LWSCIERR(
        LwSciStreamBlockSetupStatusSet(
            endpointHandle, LwSciStreamSetup_SignalObjExport, true));
}

void PerfClient::recvSignalObj(void)
{
    CHECK_LWSCIERR(
        LwSciStreamBlockSetupStatusSet(
            endpointHandle, LwSciStreamSetup_SignalObjImport, true));
}

void PerfClient::sendEndpointElements(void)
{
    // Query max number of elements per packet supported by LwSciStream.
    int32_t maxNumElement{ 0 };
    CHECK_LWSCIERR(LwSciStreamAttributeQuery(
                    LwSciStreamQueryableAttrib_MaxElements,
                    &maxNumElement));
    assert(NUM_ELEMENTS < maxNumElement);

    // Send buffer attr
    for (uint32_t i{ 0U }; i < NUM_ELEMENTS; i++) {
        LwSciBufAttrList bufAttrLists{ nullptr };
        CHECK_LWSCIERR(LwSciBufAttrListCreate(bufModule, &bufAttrLists));

        setEndpointBufAttr(bufAttrLists);

        CHECK_LWSCIERR(LwSciStreamBlockElementAttrSet(endpointHandle, i,
                                                      bufAttrLists));

        LwSciBufAttrListFree(bufAttrLists);
    }

    // Indicate element specification is done
    CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
        endpointHandle, LwSciStreamSetup_ElementExport, true));
}

void PerfClient::recvAllocatedElements(void)
{
    LwSciStreamEventType event;

    // Indicate element import is done
    CHECK_LWSCIERR(LwSciStreamBlockSetupStatusSet(
        endpointHandle, LwSciStreamSetup_ElementImport, true));
}

void PerfClient::recvPacket(void)
{
    // Receive a new packet
    assert(numRecvPackets < NUM_PACKETS);
    Packet *packet = &packets[numRecvPackets];

    // Retrieve packet handle
    CHECK_LWSCIERR(
        LwSciStreamBlockPacketNewHandleGet(endpointHandle,
                                            &packet->handle));
    packet->cookie = static_cast<LwSciStreamCookie>(++numRecvPackets);
    packet->buffers.fill(nullptr);

    // Retrieve all packet buffers
    for (uint32_t i {0U}; NUM_ELEMENTS > i; ++i) {
        CHECK_LWSCIERR(
            LwSciStreamBlockPacketBufferGet(endpointHandle,
                                            packet->handle, i,
                                            &packet->buffers[i]));
    }

    // Send the packet status to the pool.
    CHECK_LWSCIERR(
        LwSciStreamBlockPacketStatusSet(endpointHandle,
                                        packet->handle,
                                        packet->cookie,
                                        LwSciError_Success));
}

void PerfClient::recvPacketComplete(void)
{
    // Receive packet completion event
    LwSciStreamEventType event;
    CHECK_LWSCIERR(LwSciStreamBlockEventQuery(endpointHandle,
                                              QUERY_TIMEOUT,
                                              &event));
    assert(LwSciStreamEventType_PacketsComplete == event);

    // Indicate packet import complete
    CHECK_LWSCIERR(
        LwSciStreamBlockSetupStatusSet(endpointHandle,
                                       LwSciStreamSetup_PacketImport,
                                       true));
}

void PerfClient::recvSetupComplete(void)
{
    LwSciStreamEventType event;
    CHECK_LWSCIERR(LwSciStreamBlockEventQuery(endpointHandle,
                                              QUERY_TIMEOUT,
                                              &event));
    assert(event == LwSciStreamEventType_SetupComplete);
}
