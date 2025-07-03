
//
//  Copyright (c) 2020 LWPU Corporation.  All rights reserved.
//
//  LWPU Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from LWPU Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#pragma once

#include <optix_types.h>

#define SBT_INDEX_DIRECT_CALLABLE 1
#define SBT_INDEX_CONTINUATION_CALLABLE 2

struct Params
{
    unsigned int*          payloads;
    OptixTraversableHandle handle;
    // for testing OPTIX_EXCEPTION_CODE_CALLABLE_PARAMETER_MISMATCH, OPTIX_EXCEPTION_CODE_UNSUPPORTED_PRIMITIVE_TYPE, OPTIX_EXCEPTION_CODE_ILWALID_RAY
    int*                   callSiteLine;
    char*                  lineInfo;
    unsigned int           lineInfoLength;
    // for testing OPTIX_EXCEPTION_CODE_TRAVERSAL_ILWALID_HIT_SBT, OPTIX_EXCEPTION_CODE_TRAVERSAL_ILWALID_MISS_SBT
    unsigned int           sbtOffset;
    unsigned int           missSBTIndex;
    // for testing OPTIX_EXCEPTION_CODE_ILWALID_RAY
    float*                 rayData;
    // for testing OPTIX_EXCEPTION_CODE_CALLABLE_PARAMETER_MISMATCH
    int*                   paramMismatchIntData;
    unsigned int           paramMismatchCallableNameLength;
    char*                  paramMismatchCallableName;
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    char* covered;
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
};

enum PROGRAM_TYPE_PAYLOADS
{
    PROGRAM_TYPE_PAYLOAD_RG = 1 << 0,
    PROGRAM_TYPE_PAYLOAD_IS = 1 << 1,
    PROGRAM_TYPE_PAYLOAD_CH = 1 << 2,
    PROGRAM_TYPE_PAYLOAD_AH = 1 << 3,
    PROGRAM_TYPE_PAYLOAD_MS = 1 << 4,
    PROGRAM_TYPE_PAYLOAD_EX = 1 << 5,
    PROGRAM_TYPE_PAYLOAD_DC = 1 << 6,
    PROGRAM_TYPE_PAYLOAD_CC = 1 << 7
};

enum PAYLOAD_VALUES
{
    PAYLOAD_VAL_0 = 0x0F000000,
    PAYLOAD_VAL_1 = 0x1F000000,
    PAYLOAD_VAL_2 = 0x2F000000,
    PAYLOAD_VAL_3 = 0x3F000000,
    PAYLOAD_VAL_4 = 0x4F000000,
    PAYLOAD_VAL_5 = 0x5F000000,
    PAYLOAD_VAL_6 = 0x6F000000,
    PAYLOAD_VAL_7 = 0x7F000000,
    PAYLOAD_VALUES_UNDEF
};
