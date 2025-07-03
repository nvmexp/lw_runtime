/*
 * Copyright (c) 2016-2017, LWPU CORPORATION. All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef _EGLSTREAM_H_
#define _EGLSTREAM_H_

#include "egltest.h"
#include "lwglsi.h"

#if defined(EXTENSION_LIST)
void PrintEGLStreamState(EGLint streamState);
bool  EGLStreamInitSingleProcess(void);
bool  EGLStreamInitCrossP(void);
int  EGLStreamQuery(EGLDisplay display,
                    EGLStreamKHR eglStream,
                    EGLenum attribute,
                    void *value,
                    bool printState=false);
void EGLStreamFini(void);
#endif

#endif // _EGLSTREAM_H_
