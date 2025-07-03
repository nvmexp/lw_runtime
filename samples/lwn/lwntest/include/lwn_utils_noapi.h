/*
 * Copyright (c) 2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// lwntest util code that is not lwn API or c/c++ specific.

#ifndef __lwn_utils_noapi_h__
#define __lwn_utils_noapi_h__

// Color enums for LWNTestClear()
enum LWNTestResultColor {
    LWNTEST_COLOR_PASS = 0,        // Green   0.0, 1.0, 0.0
    LWNTEST_COLOR_FAIL = 1,        // Red     1.0, 0.0, 0.0
    // WNF = Will-not-fix
    LWNTEST_COLOR_WNF = 2,         // Yellow  1.0, 1.0, 0.0
    LWNTEST_COLOR_UNSUPPORTED = 3, // Blue    0.0, 0.0, 1.0
};

// Utility code to display a test result by clearing the screen.
extern void LWNTestClear(LWNTestResultColor color);

// Calls lwnQueueFinish(g_lwnQueue)
extern void LWNTestFinish();

inline void LWNTestClearAndFinish(LWNTestResultColor color)
{
    LWNTestClear(color);
    LWNTestFinish();
}

// Shortlwt for failing with a clear to red (existed prior to the above methods)
inline void LWNFailTest()
{
    LWNTestClear(LWNTEST_COLOR_FAIL);
}

//////////////////////////////////////////////////////////////////////////
//
//                  LWN LWOGTEST TEST CLASSES
//
// We provide a simplified version of the lwogtest C++ test class methods for
// LWN.  We provide standard functions for the initGraphics() and
// exitGraphics() methods, where initGraphics() sets up render target
// attachments and the scissor/viewport state to the window framebuffer and
// exitGraphics() presents the window framebuffer and cleans up all of the
// tracked API object allocations and other shader resources.  We have
// separate versions for regular tests (CppMethods) and tests that should be
// run with object tracking disabled (CppMethods_NoTracking).  Tests using
// "NoTracking" are responsible for explicitly freeing all LWN objects they
// allocate.
//
extern void lwnDefaultInitGraphics(void);
extern void lwnDefaultExitGraphics(void);

#define LWNTEST_CppMethods()                                        \
    lwString getDescription() const;                                \
    int isSupported(void) const;                                    \
    void doGraphics(void) const;                                    \
    void initGraphics(void) const   { lwnDefaultInitGraphics(); }   \
    void exitGraphics(void) const   { lwnDefaultExitGraphics(); }

#define LWNTEST_CppMethods_NoTracking()                                 \
    lwString getDescription() const;                                    \
    int isSupported(void) const;                                        \
    void doGraphics(void) const;                                        \
    void initGraphics(void) const   { lwnDefaultInitGraphics();         \
                                      DisableLWNObjectTracking(); }     \
    void exitGraphics(void) const   { lwnDefaultExitGraphics(); }

#endif // #ifndef __lwn_utils_noapi_h__
