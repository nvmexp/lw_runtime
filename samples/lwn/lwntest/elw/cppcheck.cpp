/*
 * Copyright (c) 2014 - 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "ogtest.h"
#include "cppcheck.h"

lwogtestCheckMgr lwogtestCheck;

void lwogtestCheckMgr::initGraphics() {
    failed = false;
}

void lwogtestCheckMgr::exitGraphics() {
    if (failed) {
#if 0
        glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
        glClear( GL_COLOR_BUFFER_BIT );
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
#endif
    }
}

void lwogtestCheckMgr::failCheck(const char *condition, const char *filename, int linenum)
{
    failed = true;

    fprintf(stdout, "Check failed at %s, line %d: (%s) -- ", filename, linenum, condition);
}

