//! \file
//! \brief LwSciStream APIs unit testing - Allocate Packet.
//!
//! \copyright
//! Copyright (c) 2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#include "lwscistreamtest.h"

constexpr uint32_t COOKIE_VALUE = 8U;

//==============================================
// Define AllocatePacket test suite
//==============================================

class AllocatePacket :
    public LwSciStreamTest
{
protected:
    virtual void SetUp()
    {
        ASSERT_EQ(LwSciError_Success, LwSciBufModuleOpen(&bufModule));
        makeRawBufferAttrList(bufModule, rawBufAttrList);

        ASSERT_EQ(LwSciError_Success, LwSciSyncModuleOpen(&syncModule));

        prodSynchronousOnly = false;
        cpuWaiterAttrList(syncModule, prodSyncAttrList);

        consSynchronousOnly = false;
        cpuWaiterAttrList(syncModule, consSyncAttrList);
    }
};

//==============================================================================
// Test Case 1 : Create two packets for a mailbox stream with one consumer, when
// tried to create third packet should result in failure.
//==============================================================================

TEST_F(AllocatePacket, CreatePacket)
{
    // Create a mailbox stream with one consumer and a pool having capacity of two packets
    createBlocks(QueueType::Mailbox, 1, 2);
    connectStream();

    // Setup packet attributes
    packetAttrSetup();

    // Create two packets
    createPacket();
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    // Choose a new pool cookie for new packet
    LwSciStreamPacket packetHandle;
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_VALUE);
    ASSERT_EQ(LwSciError_Overflow,
                LwSciStreamPoolPacketCreate(pool, poolCookie, &packetHandle));
}

//==============================================================================
// Test Case 2 : Create two packets for a mailbox stream with two consumers,
// when tried to create third packet should result in failure.
//==============================================================================

TEST_F(AllocatePacket, CreatePacketMulticast)
{
    // Create a mailbox stream with two consumer and a pool having capacity of two packets
    createBlocks(QueueType::Mailbox, 2, 2);
    connectStream();

    // Setup packet attributes
    packetAttrSetup();

    // Create two packets
    createPacket();
    checkPacketStatus();

    // Create and exchange sync objects
    createSync();

    // Choose a new pool cookie for new packet
    LwSciStreamPacket packetHandle;
    LwSciStreamCookie poolCookie
                = static_cast<LwSciStreamCookie>(COOKIE_VALUE);
    ASSERT_EQ(LwSciError_Overflow,
                LwSciStreamPoolPacketCreate(pool, poolCookie, &packetHandle));
}
