/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __testloop_h
#define __testloop_h

/*
Definitions:

NotSupported    -- Test is not supported by the device it's being run on.
Not Requested   -- Test is either explicitly not requested by the user or part of "debug only" tests which must be explicitly requested.
*/
typedef enum TestSupportEnum
{
    NotSupported = 0,
    Supported,
    NotRequested,
} TestSupport;

TestSupport GetTestSupport(OGTEST *lwrTest);
void InitNonGraphics(void);
void MainLoop(void);

extern OGTEST **lwog_FullTestList;
#define GET_TEST(index)         (lwog_FullTestList[index])

#endif // __testloop_h
