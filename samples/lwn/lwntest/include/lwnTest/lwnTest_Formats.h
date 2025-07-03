/*
* Copyright (c) 2016, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/
#ifndef __lwnTest_Formats_h__
#define __lwnTest_Formats_h__

#include "lwnUtil/lwnUtil_Interface.h"

#include "lwn/lwn.h"
#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
#include "lwn/lwn_Cpp.h"
#endif

namespace lwnTest {

enum SamplerComponentType {
    COMP_TYPE_FLOAT=0,
    COMP_TYPE_INT=1,
    COMP_TYPE_UNSIGNED=2,
    COMP_TYPE_COUNT=3
};

static const int FLAG_DEPTH = 0x1;
static const int FLAG_STENCIL = 0x2;
static const int FLAG_COMPRESSED = 0x4;
static const int FLAG_TEXTURE = 0x8;
static const int FLAG_VERTEX = 0x10;
static const int FLAG_ASTC = 0x20;
static const int FLAG_RENDER = 0x40;
static const int FLAG_COPYIMAGE = 0x80;
static const int FLAG_PRIVATE = 0x100;

// Information about formats that can be easily captured in a flat struct
struct FormatDesc {
#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_C
    LWNformat format;
#else
    lwn::Format format;
#endif
    const char* formatName;
    int stride;
    SamplerComponentType samplerComponentType;
    uint8_t numBitsPerComponent[4];
    int flags;

    static int numFormats();
    static const FormatDesc* findByFormat(LWNformat fmt);
#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
    static const FormatDesc* findByFormat(lwn::Format fmt);
#endif
};

} // namespace lwnTest

#endif // #ifndef __lwnTest_Formats_h__
