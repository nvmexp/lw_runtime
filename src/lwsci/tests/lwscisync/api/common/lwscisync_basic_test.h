/*
 * Copyright (c) 2021-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#ifndef INCLUDED_LWSCISYNC_BASIC_TEST_H
#define INCLUDED_LWSCISYNC_BASIC_TEST_H

#include "lwscisync_peer.h"
#include <memory>
#include <vector>

class LwSciSyncBasicTest : public ::testing::Test
{
public:
    void SetUp() override
    {
        peer.SetUp();
    }

    void TearDown() override
    {
        peer.TearDown();
    }

    LwSciSyncPeer peer;
};

#endif // INCLUDED_LWSCISYNC_BASIC_TEST_H
