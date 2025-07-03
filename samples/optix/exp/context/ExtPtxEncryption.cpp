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

#include <exp/context/DeviceContext.h>
#include <exp/context/EncryptionManager.h>
#include <exp/context/ErrorHandling.h>

#include <prodlib/misc/LWTXProfiler.h>

extern "C" OptixResult optixExtPtxEncryptionGetOptixSalt( OptixDeviceContext contextAPI, void* optixSalt, size_t optixSaltSizeInBytes )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::PTX_ENCRYPTION_GET_OPTIX_SALT );
    optix_exp::DeviceContextLogger& clog = context->getLogger();
    OPTIX_CHECK_NULL_ARGUMENT( optixSalt );

    try
    {
        optix_exp::ErrorDetails errDetails;
        if( OptixResult result = context->getEncryptionManager().getOptixSalt( optixSalt, optixSaltSizeInBytes, errDetails ) )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixExtPtxEncryptionSetOptixSalt( OptixDeviceContext contextAPI, const void* optixSalt, size_t optixSaltSizeInBytes )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::PTX_ENCRYPTION_SET_OPTIX_SALT );
    optix_exp::DeviceContextLogger& clog = context->getLogger();
    OPTIX_CHECK_NULL_ARGUMENT( optixSalt );

    try
    {
        optix_exp::ErrorDetails errDetails;
        if( OptixResult result = context->getEncryptionManager().setOptixSalt( optixSalt, optixSaltSizeInBytes, errDetails ) )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixExtPtxEncryptionSetVendorSalt( OptixDeviceContext contextAPI, const void* vendorSalt, size_t vendorSaltSizeInBytes )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::PTX_ENCRYPTION_SET_VENDOR_SALT );
    optix_exp::DeviceContextLogger& clog = context->getLogger();
    OPTIX_CHECK_NULL_ARGUMENT( vendorSalt );

    try
    {
        optix_exp::ErrorDetails errDetails;
        if( OptixResult result = context->getEncryptionManager().setVendorSalt( vendorSalt, vendorSaltSizeInBytes, errDetails ) )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixExtPtxEncryptionSetPublicVendorKey( OptixDeviceContext contextAPI, const void* publicVendorKey, size_t publicVendorKeySizeInBytes )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::PTX_ENCRYPTION_SET_PUBLIC_VENDOR_KEY );
    optix_exp::DeviceContextLogger& clog = context->getLogger();
    OPTIX_CHECK_NULL_ARGUMENT( publicVendorKey );

    try
    {
        optix_exp::ErrorDetails errDetails;
        if( OptixResult result =
                context->getEncryptionManager().setPublicVendorKey( publicVendorKey, publicVendorKeySizeInBytes, errDetails ) )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}
