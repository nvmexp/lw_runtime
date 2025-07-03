/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#pragma once

#define OPTIX_DEFINE_ABI_VERSION_ONLY
#include <optix_function_table.h>
#undef OPTIX_DEFINE_ABI_VERSION_ONLY

namespace optix_exp {

enum class OptixABI
{
    ABI_18 = 18,
    ABI_19 = 19,
    ABI_20 = 20,
    ABI_21 = 21,
    ABI_22 = 22,
    ABI_23 = 23,
    ABI_24 = 24,
    ABI_25 = 25,
    ABI_26 = 26,
    ABI_27 = 27,
    ABI_28 = 28,
    ABI_29 = 29,
    ABI_30 = 30,
    ABI_31 = 31,
    ABI_32 = 32,
    ABI_33 = 33,
    ABI_34 = 34,
    ABI_35 = 35,
    ABI_36 = 36,
    ABI_37 = 37,
    ABI_38 = 38,
    ABI_39 = 39,
    ABI_40 = 40,
    ABI_41 = 41,
    ABI_42 = 42,
    ABI_43 = 43,
    ABI_44 = 44,
    ABI_45 = 45,
    ABI_46 = 46,
    ABI_47 = 47,
    ABI_48 = 48,
    ABI_49 = 49,
    ABI_50 = 50,
    ABI_51 = 51,
    ABI_52 = 52,
    ABI_53 = 53,
    ABI_54 = 54,
    ABI_55 = 55,
    ABI_56 = 56,
    ABI_57 = 57,
    ABI_58 = 58,
    ABI_59 = 59,
    ABI_60 = 60,
    ABI_MIN = ABI_18,
    ABI_LWRRENT = OPTIX_ABI_VERSION
};

inline bool operator<( OptixABI lhs, OptixABI rhs )
{
    return (int)lhs < (int)rhs;
}

inline bool operator<=( OptixABI lhs, OptixABI rhs )
{
    return (int)lhs <= (int)rhs;
}

inline bool operator>( OptixABI lhs, OptixABI rhs )
{
    return (int)lhs > (int)rhs;
}

inline bool operator>=( OptixABI lhs, OptixABI rhs )
{
    return (int)lhs >= (int)rhs;
}

}  // end namespace optix_exp
