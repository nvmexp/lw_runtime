/*
 * Copyright (c) 2019-2020 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */
#ifndef INCLUDED__TEST_INFO_H
#define INCLUDED__TEST_INFO_H

#include <stdint.h>

class TestInfo
{

private:
    TestInfo() : submitSize(1024), reportProgress(false), verbose(false)
    {
    }

public:
    TestInfo(const TestInfo&) = delete;
    TestInfo(TestInfo&&) = delete;
    TestInfo& operator=(const TestInfo&) = delete;
    TestInfo& operator=(TestInfo&&) = delete;

    uint32_t submitSize;
    bool reportProgress;
    bool showDescription;
    bool verbose;
    uint32_t gpuIndex[2];

public:
    static TestInfo* get()
    {
        static TestInfo instance;
        return &instance;
    }
};

#endif // INCLUDED__TEST_INFO_H
