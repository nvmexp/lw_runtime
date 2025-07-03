/*
 * lwscibuf_basic_test.h
 *
 * Copyright (c) 2020-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#ifndef INCLUDED_LWSCIBUF_BASIC_TEST_H
#define INCLUDED_LWSCIBUF_BASIC_TEST_H

#include "lwscibuf_test_integration.h"
#include "lwscibuf_peer.h"
#include <memory>
#include <vector>

class LwSciBufBasicTest : public ::testing::Test
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

    LwSciBufPeer peer;
};

#endif // INCLUDED_LWSCIBUF_BASIC_TEST_H
