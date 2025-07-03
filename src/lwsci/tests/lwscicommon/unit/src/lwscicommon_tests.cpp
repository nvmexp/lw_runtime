/*
 * Copyright (c) 2019, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#define GTEST_HAS_RTTI 0
#include "gtest/gtest.h"

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    //TODO: use "threadsafe" setting. "fast" setting spews warning
    // when running death tests.
    // (Lwrrently, with threadsafe setting, on safety QNX gtest fails
    //  to intercept the abort() call, as a result the process will abort
    //  in death tests).
    ::testing::FLAGS_gtest_death_test_style="fast";

    return RUN_ALL_TESTS();
}