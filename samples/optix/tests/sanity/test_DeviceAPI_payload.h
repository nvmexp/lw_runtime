
//
//  Copyright (c) 2019 LWPU Corporation.  All rights reserved.
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

#include <optix_types.h>

struct Params
{
    unsigned int*          payloads;
    OptixTraversableHandle handle;
    float3                 origin;
    float3                 direction;
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    char* covered;
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
};

enum PROGRAM_TYPE_PAYLOADS
{
    PROGRAM_TYPE_PAYLOAD_RG = 1,
    PROGRAM_TYPE_PAYLOAD_IS = 2,
    PROGRAM_TYPE_PAYLOAD_CH = 4,
    PROGRAM_TYPE_PAYLOAD_AH = 8,
    PROGRAM_TYPE_PAYLOAD_MS = 16,
    PROGRAM_TYPE_PAYLOAD_EX = 32,
    PROGRAM_TYPE_PAYLOAD_DC = 64,
    PROGRAM_TYPE_PAYLOAD_CC = 128
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
