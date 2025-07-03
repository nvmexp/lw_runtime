/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include <stdMessages.h>

#ifndef MESSAGE_DEFS_IMPL

#ifdef __cplusplus
extern "C" {
#endif
#define IMPORT_MSG(a) extern const msgMessage a;
#define MSG(a,b,c) IMPORT_MSG(a)

#else

#ifdef __cplusplus
#define IMPORT_MSG(a) extern "C" const msgMessage a;
#else
#define IMPORT_MSG(a) extern const msgMessage a;
#endif

#define MSG(a,b,c) \
    IMPORT_MSG(a) \
    static msgMessageRec  __##a##__= {msg##b,False,False,c}; \
    const msgMessage a= (const msgMessage)&__##a##__;

#endif

