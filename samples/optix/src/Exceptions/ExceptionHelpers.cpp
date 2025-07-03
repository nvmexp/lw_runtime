// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <Exceptions/ExceptionHelpers.h>

#include <Exceptions/AlreadyMapped.h>
#include <Exceptions/DatabaseFilePermissions.h>
#include <Exceptions/FileNotFound.h>
#include <Exceptions/IllegalSymbol.h>
#include <Exceptions/LaunchFailed.h>
#include <Exceptions/NoDevice.h>
#include <Exceptions/ResourceAlreadyRegistered.h>
#include <Exceptions/ResourceNotRegistered.h>
#include <Exceptions/TimeoutException.h>
#include <Exceptions/TypeMismatch.h>
#include <Exceptions/VariableNotFound.h>
#include <Exceptions/VariableRedeclared.h>
#include <Exceptions/VersionMismatch.h>
#include <prodlib/exceptions/AssertionFailure.h>
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/IlwalidContext.h>
#include <prodlib/exceptions/IlwalidDevice.h>
#include <prodlib/exceptions/IlwalidSource.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/MemoryAllocationFailed.h>
#include <prodlib/exceptions/UnknownError.h>
#include <prodlib/exceptions/ValidationError.h>

using namespace prodlib;

namespace {

RTresult getRTresultFromLwdaError( lwdaError lwdaErrorCode )
{
    switch( lwdaErrorCode )
    {

        case lwdaSuccess:
            return RT_SUCCESS;

        case lwdaErrorIlwalidValue:
            return RT_ERROR_ILWALID_VALUE;

        case lwdaErrorMemoryAllocation:
            return RT_ERROR_MEMORY_ALLOCATION_FAILED;

        default:
            return RT_ERROR_UNKNOWN;
    }
}

} // namespace

#define EXCEPTION_2_RTRESULT( type, code )                                                                             \
    if( dynamic_cast<const type*>( e ) )                                                                               \
        return code;

namespace optix {

RTresult getRTresultFromLWresult( LWresult lwdaErrorCode )
{
    switch( lwdaErrorCode )
    {

        case LWDA_SUCCESS:
            return RT_SUCCESS;

        case LWDA_ERROR_ILWALID_VALUE:
            return RT_ERROR_ILWALID_VALUE;

        case LWDA_ERROR_OUT_OF_MEMORY:
            return RT_ERROR_MEMORY_ALLOCATION_FAILED;

        case LWDA_ERROR_NOT_INITIALIZED:
        case LWDA_ERROR_DEINITIALIZED:
        case LWDA_ERROR_NO_DEVICE:
        case LWDA_ERROR_ILWALID_DEVICE:
            return RT_ERROR_ILWALID_CONTEXT;

        case LWDA_ERROR_ILWALID_IMAGE:
        case LWDA_ERROR_ILWALID_CONTEXT:
        case LWDA_ERROR_CONTEXT_ALREADY_LWRRENT:
        case LWDA_ERROR_MAP_FAILED:
        case LWDA_ERROR_UNMAP_FAILED:
        case LWDA_ERROR_ARRAY_IS_MAPPED:
        case LWDA_ERROR_ALREADY_MAPPED:
        case LWDA_ERROR_NO_BINARY_FOR_GPU:
        case LWDA_ERROR_ALREADY_ACQUIRED:
        case LWDA_ERROR_NOT_MAPPED:
        case LWDA_ERROR_ILWALID_SOURCE:
        case LWDA_ERROR_FILE_NOT_FOUND:
        case LWDA_ERROR_ILWALID_HANDLE:
        case LWDA_ERROR_NOT_FOUND:
        case LWDA_ERROR_NOT_READY:

        case LWDA_ERROR_LAUNCH_FAILED:
        case LWDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
        case LWDA_ERROR_LAUNCH_TIMEOUT:
        case LWDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:

        case LWDA_ERROR_UNKNOWN:

        default:
            return RT_ERROR_UNKNOWN;
    }
}

RTresult getRTresultFromException( const Exception* e )
{
    /*
  RT_TIMEOUT_CALLBACK                  = 0x100,

  RT_ERROR_ILWALID_CONTEXT             = 0x500,
  RT_ERROR_ILWALID_VALUE               = 0x501,
  RT_ERROR_MEMORY_ALLOCATION_FAILED    = 0x502,
  RT_ERROR_TYPE_MISMATCH               = 0x503,
  RT_ERROR_VARIABLE_NOT_FOUND          = 0x504,
  RT_ERROR_VARIABLE_REDECLARED         = 0x505,
  RT_ERROR_ILLEGAL_SYMBOL              = 0x506,
  RT_ERROR_ILWALID_SOURCE              = 0x507,
  RT_ERROR_VERSION_MISMATCH            = 0x508,

  RT_ERROR_OBJECT_CREATION_FAILED      = 0x600,
  RT_ERROR_NO_DEVICE                   = 0x601,
  RT_ERROR_ILWALID_DEVICE              = 0x602,
  RT_ERROR_ILWALID_IMAGE               = 0x603,
  RT_ERROR_FILE_NOT_FOUND              = 0x604,
  RT_ERROR_ALREADY_MAPPED              = 0x605,
  RT_ERROR_ILWALID_DRIVER_VERSION      = 0x606,
  RT_ERROR_CONTEXT_CREATION_FAILED     = 0x607,

  RT_ERROR_RESOURCE_NOT_REGISTERED     = 0x608,
  RT_ERROR_RESOURCE_ALREADY_REGISTERED = 0x609,

  RT_ERROR_LAUNCH_FAILED               = 0x900,
  */
    // Most common errors first
    EXCEPTION_2_RTRESULT( UnknownError, RT_ERROR_UNKNOWN );
    EXCEPTION_2_RTRESULT( LaunchFailed, RT_ERROR_LAUNCH_FAILED );
    {
        const LwdaError* de = dynamic_cast<const LwdaError*>( e );
        if( de )
            return getRTresultFromLWresult( de->getErrorCode() );
    }
    {
        const LwdaRuntimeError* de = dynamic_cast<const LwdaRuntimeError*>( e );
        if( de )
            return getRTresultFromLwdaError( de->getErrorCode() );
    }

    EXCEPTION_2_RTRESULT( TimeoutException, RT_TIMEOUT_CALLBACK );

    EXCEPTION_2_RTRESULT( IlwalidContext, RT_ERROR_ILWALID_CONTEXT );
    EXCEPTION_2_RTRESULT( IlwalidValue, RT_ERROR_ILWALID_VALUE );
    EXCEPTION_2_RTRESULT( MemoryAllocationFailed, RT_ERROR_MEMORY_ALLOCATION_FAILED );
    EXCEPTION_2_RTRESULT( TypeMismatch, RT_ERROR_TYPE_MISMATCH );
    EXCEPTION_2_RTRESULT( VariableNotFound, RT_ERROR_VARIABLE_NOT_FOUND );
    EXCEPTION_2_RTRESULT( VariableRedeclared, RT_ERROR_VARIABLE_REDECLARED );
    EXCEPTION_2_RTRESULT( IllegalSymbol, RT_ERROR_ILLEGAL_SYMBOL );
    EXCEPTION_2_RTRESULT( IlwalidSource, RT_ERROR_ILWALID_SOURCE );
    EXCEPTION_2_RTRESULT( VersionMismatch, RT_ERROR_VERSION_MISMATCH );

    // RT_ERROR_OBJECT_CREATION_FAILED never used *ever*, not even in the docs.
    EXCEPTION_2_RTRESULT( NoDevice, RT_ERROR_NO_DEVICE );
    EXCEPTION_2_RTRESULT( IlwalidDevice, RT_ERROR_ILWALID_DEVICE );
    // RT_ERROR_ILWALID_IMAGE never used *ever*, not even in the docs.
    EXCEPTION_2_RTRESULT( FileNotFound, RT_ERROR_FILE_NOT_FOUND );
    EXCEPTION_2_RTRESULT( AlreadyMapped, RT_ERROR_ALREADY_MAPPED );
    // RT_ERROR_CONTEXT_CREATION_FAILED never throw, but used in rtapi.cpp

    EXCEPTION_2_RTRESULT( ResourceAlreadyRegistered, RT_ERROR_RESOURCE_ALREADY_REGISTERED );
    EXCEPTION_2_RTRESULT( ResourceNotRegistered, RT_ERROR_RESOURCE_NOT_REGISTERED );

    EXCEPTION_2_RTRESULT( DatabaseFilePermissions, RT_ERROR_DATABASE_FILE_PERMISSIONS );

    return RT_ERROR_UNKNOWN;
}

#define THROW_CHILD( type )                                                                                            \
    {                                                                                                                  \
        const type* de = dynamic_cast<const type*>( e );                                                               \
        if( de )                                                                                                       \
            throw * de;                                                                                                \
    }

void rethrowException( const Exception* e )
{
    // Most common first
    THROW_CHILD( UnknownError );

    THROW_CHILD( AlreadyMapped );
    THROW_CHILD( AssertionFailure );
    THROW_CHILD( LwdaError );
    THROW_CHILD( LwdaRuntimeError );
    THROW_CHILD( FileNotFound );
    THROW_CHILD( IllegalSymbol );
    THROW_CHILD( IlwalidContext );
    THROW_CHILD( IlwalidDevice );
    THROW_CHILD( IlwalidSource );
    THROW_CHILD( IlwalidValue );
    THROW_CHILD( LaunchFailed );
    THROW_CHILD( MemoryAllocationFailed );
    THROW_CHILD( NoDevice );
    THROW_CHILD( ResourceAlreadyRegistered );
    THROW_CHILD( ResourceNotRegistered );
    THROW_CHILD( TimeoutException );
    THROW_CHILD( TypeMismatch );
    THROW_CHILD( ValidationError );
    //THROW_CHILD(UnknownError);
    THROW_CHILD( VariableNotFound );
    THROW_CHILD( VariableRedeclared );
    THROW_CHILD( VersionMismatch );
    THROW_CHILD( DatabaseFilePermissions );

    if( !e )
        throw UnknownError( RT_EXCEPTION_INFO, "rethrow: exception is NULL" );

    throw UnknownError( e->getExceptionInfo(), e->getDescription() );
}

} // namespace optix
