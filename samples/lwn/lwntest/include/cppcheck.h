/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __CPPCHECK_H__
#define __CPPCHECK_H__

//
// cppcheck.h
//
// A class to check conditions in tests.  This functions much like assert()
// but rather than exit the program when a check fails the image is cleared
// to red and control move on to next test.
//


#ifndef __cplusplus
#error "cppcheck.h can only be used in C++ code"
#endif

class lwogtestCheckMgr {
public:
    lwogtestCheckMgr() : failed(false) { initGraphics(); }
    void initGraphics();
    void exitGraphics();
    void failCheck(const char *condition, const char *filename, int linenum);
    bool hasFailed() { return failed; }
protected:
    bool failed;
};

extern lwogtestCheckMgr lwogtestCheck;

#define LWOGTEST_CHECK(condition) \
    if ((condition) == false) \
        lwogtestCheck.failCheck(#condition, __FILE__, __LINE__);

#endif  // #ifndef __CPPCHECK_H__
