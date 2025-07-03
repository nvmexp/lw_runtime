/*
 * Copyright (c) 2016 - 2017 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef _DEFS_H
#define _DEFS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>
#include <sys/syscall.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>

#define LOG_MSG(format, ...) do { \
    printf("[Msg] "); \
    printf(format, ##__VA_ARGS__); \
} while(0)

#define LOG_ERR(format, ...) do { \
    printf("[Error] %s: ", __func__); \
    printf(format, ##__VA_ARGS__); \
} while(0)

#ifndef EGLTEST_DEBUG
#define EGLTEST_DEBUG 0

#if EGLTEST_DEBUG
#define LOG_FUNC(format, ...) do { \
    printf("[TestFunc] %s: ", __func__); \
    printf(format, ##__VA_ARGS__); \
} while(0)

#define LOG_INFO(format, ...) do { \
    printf("pid:%d tid:%lu [Info] %s: ", \
           getpid(), syscall(SYS_gettid), __func__); \
    printf(format, ##__VA_ARGS__); \
} while(0)
#else
#define LOG_FUNC(format, ...)
#define LOG_INFO(format, ...)
#endif // #if EGLTEST_DEBUG
#endif // #ifndef EGLTEST_DEBUG

#endif //_DEFS_H
