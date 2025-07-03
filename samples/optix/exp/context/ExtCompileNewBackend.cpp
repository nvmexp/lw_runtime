/*
 * Copyright (c) 2021 LWPU CORPORATION.  All rights reserved.
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

#include <optix_ext_compile_new_backend.h>

#include <exp/context/DeviceContext.h>

extern "C" OptixResult optixExtCompileNewBackendSetEnabled( OptixDeviceContext contextAPI, int enabled )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    try
    {
        optix_exp::ErrorDetails errDetails;
        OptixResult             result = context->setD2IREnabled( enabled, errDetails );
        if( result != OPTIX_SUCCESS )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixExtCompileNewBackendIsEnabled( OptixDeviceContext contextAPI, int* outIsEnabled )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( outIsEnabled );

    try
    {
        *outIsEnabled = context->isD2IREnabled();
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixExtCompileOldBackendFallbackSetEnabled( OptixDeviceContext contextAPI, int enabled )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    try
    {
        optix_exp::ErrorDetails errDetails;
        OptixResult             result = context->setLWPTXFallbackEnabled( enabled, errDetails );
        if( result != OPTIX_SUCCESS )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixExtCompileOldBackendFallbackIsEnabled( OptixDeviceContext contextAPI, int* outIsEnabled )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( outIsEnabled );
    try
    {
        *outIsEnabled = context->isLWPTXFallbackEnabled();
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}
