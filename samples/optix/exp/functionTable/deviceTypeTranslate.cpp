/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
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

#include <exp/functionTable/deviceTypeTranslate.h>

namespace ABI_v20 {

typedef struct OptixInstance
{
    float        transform[12];      /* affine world-to-object transformation as 3x4 matrix in row-major layout */
    unsigned int instanceId : 24;    /* Application supplied ID */
    unsigned int visibilityMask : 8; /* Visibility mask. If rayMask & instanceMask == 0 the instance is lwlled */
    unsigned int sbtOffset : 24;     /* SBT record offset.  Should be set to 0 for
                                      * instances of Instance Acceleration Structure (IAS)
                                      * objects */
    unsigned int flags : 8;          /* combinations of OptixInstanceFlags */

    size_t traversableHandle; /* Set with a LWdeviceptr to a bottom level acceleration or
                                 a OptixTraversableHandle. */
} OptixInstance;

}

namespace optix_exp
{

size_t translateABI_getOptixInstanceTypeSize( OptixABI inAbi )
{
    if( inAbi <= OptixABI::ABI_20 )
    {
        return sizeof(ABI_v20::OptixInstance);
    }

    return sizeof(OptixInstance);
}

size_t translateABI_getOptixInstanceTraversableHandleFieldOffet( OptixABI inAbi )
{
    if( inAbi <= OptixABI::ABI_20 )
    {
        return offsetof( ABI_v20::OptixInstance, traversableHandle );
    }

    return offsetof( OptixInstance, traversableHandle );
}

}  // end namespace optix_exp
