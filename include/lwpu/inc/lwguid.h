/*
 * SPDX-FileCopyrightText: Copyright (c) 2012,2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "lwtypes.h"

/*
 * lwguid.h
 */
#ifndef __LWGUID_H
#define __LWGUID_H

#ifdef __cplusplus
extern "C" {
#endif //__cplusplus

#if defined(LW_WINDOWS) && !defined(LW_MODS)

/*!
 * If the file defines LWSETGUID, this header will define the GUID as a
 * constant. If the file does not define LWSETGUID, this header will define
 * the GUID as an external reference.  This should prevent multiple copies
 * of the same GUID from being defined inside one binary.  Only one c/cpp file
 * should ever use the LWSETGUID for a given name inside LW_GUID_DEF(name,...).
 */
#ifdef LWSETGUID
#define INITGUID
#endif // LWSETGUID

#include <guiddef.h>

// Translate our macro into the windows macro.
#define LW_GUID_DEF(name, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8) \
        DEFINE_GUID(name, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8)

#else // !defined(LW_WINDOWS) && !defined(LW_MODS)

// Also defined in lwcd.h
#ifndef GUID_DEFINED
#define GUID_DEFINED
typedef struct _GUID {
    LwU32   Data1;
    LwU16   Data2;
    LwU16   Data3;
    LwU8    Data4[8];
} GUID, *LPGUID;
#endif

/*!
 * If the file defines LWSETGUID, this header will define the GUID as a
 * constant. If the file does not define LWSETGUID, this header will define
 * the GUID as an external reference.  This should prevent multiple copies
 * of the same GUID from being defined inside one binary.  Only one c/cpp file
 * should ever use the LWSETGUID for a given name inside LW_GUID_DEF(name,...).
 */
#ifdef LWSETGUID

// Define the constant based on the name given
#define LW_GUID_DEF(name, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8)     \
        const GUID name =                                                \
                        { l, w1, w2,                                     \
                                   { b1, b2, b3, b4, b5, b6, b7, b8 } }
#else // !LWSETGUID

// Define an external reference constant
#define LW_GUID_DEF(name, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8)     \
        extern const GUID name

#endif // LWSETGUID

#endif // !defined(LW_WINDOWS) && !defined(LW_MODS)


#ifdef __cplusplus
}
#endif //__cplusplus

#endif // __LWGUID_H

