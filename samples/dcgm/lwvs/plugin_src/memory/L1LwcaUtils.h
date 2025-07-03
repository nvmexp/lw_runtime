/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2009,2012,2015,2017 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef __LWDA_TOOLS_H_INCLUDED
#define __LWDA_TOOLS_H_INCLUDED

#include "memory_plugin.h"

// TODO switch all kernels over to standard fixed-size types
// e.g. uint8_t, int64_t
typedef unsigned char      unsigned08;
typedef signed char        signed08;
typedef unsigned short     unsigned16;
typedef signed short       signed16;
typedef unsigned           unsigned32;
typedef signed int         signed32;
typedef unsigned long long unsigned64;
typedef long long          signed64;
typedef unsigned64         device_ptr;

// Ensure that types are the correct size
template< int size > struct EnsureUint;
template<> struct EnsureUint<4>  { };
template<> struct EnsureUint<8>  { };
template<> struct EnsureUint<16> { };
struct EnsureU32   { EnsureUint<sizeof(unsigned32)> ensureU32; };
struct EnsureU64   { EnsureUint<sizeof(unsigned64)> ensureU64; };

template<typename T>
__device__ T GetPtr(device_ptr ptr)
{
    return reinterpret_cast<T>(static_cast<size_t>(ptr));
}

#endif // __LWDA_TOOLS_H_INCLUDED
