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

#include <optix_types.h>

#include <rtcore/interface/types.h>

#include <prodlib/exceptions/Exception.h>

#include <lwca.h>

#include <exception>
#include <iosfwd>
#include <new>
#include <sstream>
#include <string>

namespace optix_exp {

OptixResult colwertError( RtcResult result );

class APIError : public prodlib::Exception
{
  public:
    APIError( const prodlib::ExceptionInfo& exceptionInfo, OptixResult code );
    APIError( const prodlib::ExceptionInfo& exceptionInfo, OptixResult code, const std::string& description );

    static const char* getErrorName( OptixResult code );
    static const char* getErrorString( OptixResult code );

    std::string getDescription() const override;
    Exception*  doClone() const override { return new APIError( *this ); }

  protected:
    OptixResult m_code;
    std::string m_description;
};

class ErrorDetails
{
  public:
    std::string        m_description;
    std::ostringstream m_compilerFeedback;

    // Use corelib::stringf for formatting with arguments
    OptixResult logDetails( OptixResult code, const std::string& description );
    OptixResult logDetails( LWresult code, const std::string& description );
    OptixResult logDetails( RtcResult code, const std::string& description );

  private:
    // Maybe.  Want to set a single break point, but could be by name and get any.
    void logDetailsMessage( const std::string& description );
};

void copyCompileDetails( std::ostringstream& compileDetails, char* logString, size_t* logStringSize );
void copyCompileDetails( const std::string& compileDetails, char* logString, size_t* logStringSize );

#define OPTIX_API_EXCEPTION_CHECK                                                                                      \
    catch( const std::bad_alloc& )                                                                                     \
    {                                                                                                                  \
        clog.sendError( "Unknown std::bad_alloc error" );                                                              \
        return OPTIX_ERROR_HOST_OUT_OF_MEMORY;                                                                         \
    }                                                                                                                  \
    catch( const prodlib::Exception& e )                                                                               \
    {                                                                                                                  \
        clog.sendError( std::string( "Unknown OptiX internal exception: " ) + e.what() );                              \
        return OPTIX_ERROR_UNKNOWN;                                                                                    \
    }                                                                                                                  \
    catch( const std::exception& e )                                                                                   \
    {                                                                                                                  \
        clog.sendError( std::string( "Unknown std::exception: " ) + e.what() );                                        \
        return OPTIX_ERROR_UNKNOWN;                                                                                    \
    }                                                                                                                  \
    catch( ... )                                                                                                       \
    {                                                                                                                  \
        clog.sendError( "Unknown exception" );                                                                         \
        return OPTIX_ERROR_UNKNOWN;                                                                                    \
    }

#define OPTIX_API_EXCEPTION_CHECK_W_LOG_STRING                                                                         \
    catch( const std::bad_alloc& )                                                                                     \
    {                                                                                                                  \
        std::string errMsg = "Unknown std::bad_alloc error";                                                           \
        clog.sendError( errMsg );                                                                                      \
        optix_exp::copyCompileDetails( errMsg, logString, logStringSize );                                             \
        return OPTIX_ERROR_HOST_OUT_OF_MEMORY;                                                                         \
    }                                                                                                                  \
    catch( const prodlib::Exception& e )                                                                               \
    {                                                                                                                  \
        std::string errMsg = std::string( "Unknown OptiX internal exception: " ) + e.what();                           \
        clog.sendError( errMsg );                                                                                      \
        optix_exp::copyCompileDetails( errMsg, logString, logStringSize );                                             \
        return OPTIX_ERROR_UNKNOWN;                                                                                    \
    }                                                                                                                  \
    catch( const std::exception& e )                                                                                   \
    {                                                                                                                  \
        std::string errMsg = std::string( "Unknown std::exception: " ) + e.what();                                     \
        clog.sendError( errMsg );                                                                                      \
        optix_exp::copyCompileDetails( errMsg, logString, logStringSize );                                             \
        return OPTIX_ERROR_UNKNOWN;                                                                                    \
    }                                                                                                                  \
    catch( ... )                                                                                                       \
    {                                                                                                                  \
        std::string errMsg = "Unknown exception";                                                                      \
        clog.sendError( errMsg );                                                                                      \
        optix_exp::copyCompileDetails( errMsg, logString, logStringSize );                                             \
        return OPTIX_ERROR_UNKNOWN;                                                                                    \
    }

#define OPTIX_CHECK_NULL_ARGUMENT( arg )                                                                               \
    if( ( arg ) == nullptr )                                                                                           \
    {                                                                                                                  \
        clog.sendError( #arg " is null" );                                                                             \
        return OPTIX_ERROR_ILWALID_VALUE;                                                                              \
    }

#define OPTIX_CHECK_NULL_ARGUMENT_W_LOG_STRING( arg )                                                                  \
    if( ( arg ) == nullptr )                                                                                           \
    {                                                                                                                  \
        std::string errMsg = #arg " is null";                                                                          \
        clog.sendError( errMsg );                                                                                      \
        optix_exp::copyCompileDetails( errMsg, logString, logStringSize );                                             \
        return OPTIX_ERROR_ILWALID_VALUE;                                                                              \
    }

#define OPTIX_CHECK_ZERO_ARGUMENT( arg )                                                                               \
    if( ( arg ) == 0 )                                                                                                 \
    {                                                                                                                  \
        clog.sendError( #arg " is 0" );                                                                                \
        return OPTIX_ERROR_ILWALID_VALUE;                                                                              \
    }

#define OPTIX_CHECK_ZERO_ARGUMENT_W_LOG_STRING( arg )                                                                  \
    if( ( arg ) == 0 )                                                                                                 \
    {                                                                                                                  \
        std::string errMsg = #arg " is 0";                                                                             \
        clog.sendError( errMsg );                                                                                      \
        optix_exp::copyCompileDetails( errMsg, logString, logStringSize );                                             \
        return OPTIX_ERROR_ILWALID_VALUE;                                                                              \
    }

// Ilwokes implCast on apiArg and attempts to cast it to implArg (of type implType*).
//
// In case of failure, logs a suitable error message to an ErrorDetails object and returns with the error code from implCast.
// Assumes that an instance of ErrorDetails is available as errDetails.
#define OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT2_NULL_OK( implType, implArg, apiArg, apiArgName, apiTypeName )         \
    optix_exp::implType* implArg;                                                                                      \
    if( OptixResult implCastResult = implCast( apiArg, implArg ) )                                                     \
    {                                                                                                                  \
        return errDetails.logDetails( implCastResult, apiArgName " argument is not an " apiTypeName );                 \
    }

// Ilwokes implCast on apiArg and attempts to cast it to implArg (of type implType*).
//
// In case of failure, logs a suitable error message to an ErrorDetails object and returns with the error code from implCast.
// Assumes that an instance of ErrorDetails is available as errDetails.
//
// Similar to OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT2_NULL_OK, but fails for null args.
#define OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT2( implType, implArg, apiArg, apiArgName, apiTypeName )                 \
    if( ( apiArg ) == nullptr )                                                                                        \
    {                                                                                                                  \
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, apiArgName " argument is null" );                     \
    }                                                                                                                  \
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT2_NULL_OK( implType, implArg, apiArg, apiArgName, apiTypeName )

// Ilwokes implCast on arg and attempts to cast it to implType*.
//
// In case of failure, logs a suitable error message to an ErrorDetails object and returns with the error code from implCast.
// Assumes that an instance of ErrorDetails is available as errDetails.
//
// Similar to OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT2, but makes the following assumptions on argument
// names: If the public API name of argument is "foo", then the impementation uses "fooAPI" as name for
// the API type and "foo" as name for the implementation type.
#define OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT( implType, arg, apiTypeName )                                          \
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT2( implType, arg, arg##API, #arg, apiTypeName )

// Ilwokes implCast on context.
//
// In case of failure, logs a suitable error message and returns with the error code from implCast.
// Uses a default constructed intstance of DeviceContextLogger instead of clog.
//
// Similar to OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT, but specialized for contexts since there is not
// yet a usable logger instance when colwerting a context.

class DeviceContext;
OptixResult validateContextAPI( OptixDeviceContext contextAPI, DeviceContext*& context );
#define OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT()                                                                       \
    optix_exp::DeviceContext* context;                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        if( const OptixResult validateResult = validateContextAPI( contextAPI, context ) )                             \
        {                                                                                                              \
            return validateResult;                                                                                     \
        }                                                                                                              \
    } while( false )

OptixResult validateContextAPI( OptixDeviceContext contextAPI, DeviceContext*& context, char* logString, size_t* logStringSize );
#define OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT_W_LOG_STRING()                                                          \
    optix_exp::DeviceContext* context;                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        if( const OptixResult validateResult = validateContextAPI( contextAPI, context, logString, logStringSize ) )   \
        {                                                                                                              \
            return validateResult;                                                                                     \
        }                                                                                                              \
    } while( false )

// Ilwokes implCast on arg and attempts to cast it to implType*.
//
// In case of failure, logs a suitable error message and returns with the error code from implCast.
// Uses a default constructed intstance of DeviceContextLogger instead of clog.
//
// Similar to OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT2, but makes the following assumptions on argument
// names: If the public API name of argument is "foo", then the impementation uses "fooAPI" as name for
// the API type and "foo" as name for the implementation type.
//
// Similar to OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT, but for arbitrary types.
#define OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( implType, arg, apiTypeName )                                  \
    optix_exp::implType* arg;                                                                                          \
    {                                                                                                                  \
        optix_exp::DeviceContextLogger defaultLogger;                                                                  \
        if( arg##API == nullptr )                                                                                      \
        {                                                                                                              \
            defaultLogger.sendError( #arg " argument is null" );                                                       \
            return OPTIX_ERROR_ILWALID_VALUE;                                                                          \
        }                                                                                                              \
        if( OptixResult implCastResult = implCast( arg##API, arg ) )                                                   \
        {                                                                                                              \
            defaultLogger.sendError( #arg " argument is not an " #apiTypeName );                                       \
            return implCastResult;                                                                                     \
        }                                                                                                              \
    }

#define OPTIX_CHECK_SAME_CONTEXT( context, objectContext, apiName )                                                    \
    if( ( context ) != ( objectContext ) )                                                                             \
    {                                                                                                                  \
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, apiName " belongs to different context" );            \
    }

#define OPTIX_FUNCTION_NOT_YET_IMPLEMENTED( funcName )                                                                 \
    clog.sendError( funcName " not yet implemented" );                                                                 \
    return OPTIX_ERROR_NOT_SUPPORTED;

#define OPTIX_CHECK_VALIDATION_MODE_STREAM_STATE( deviceContextPtr, stream )                                           \
    if( deviceContextPtr->hasValidationModeCheckStreamState() )                                                        \
    {                                                                                                                  \
        if( LWresult lwdaErr = corelib::lwdaDriver().LwStreamSynchronize( stream ) )                                   \
        {                                                                                                              \
            errDetails.logDetails( lwdaErr, "Failed to synchronize with given stream" );                               \
            return errDetails.logDetails( OPTIX_ERROR_VALIDATION_FAILURE,                                              \
                                          "Validation mode found given stream in erroneous state" );                   \
        }                                                                                                              \
    }

#define OPTIX_CHECK_VALIDATION_MODE_LWRRENT_LWDA_CONTEXT( deviceContextPtr )                                           \
    if( deviceContextPtr->hasValidationModeCheckLwrrentLwdaContext() )                                                 \
    {                                                                                                                  \
        LWcontext lwrrentContext = NULL;                                                                               \
        if( LWresult lwdaErr = corelib::lwdaDriver().LwCtxGetLwrrent( &lwrrentContext ) )                              \
            return errDetails.logDetails( lwdaErr, "Failed to get current LWCA context" );                             \
        if( deviceContextPtr->getLwdaContext() != lwrrentContext )                                                     \
            return errDetails.logDetails( OPTIX_ERROR_VALIDATION_FAILURE,                                              \
                "Validation mode found current LWCA context does not match the LWCA context"                           \
                " associated with the supplied OptixDeviceContext" );                                                  \
    }

}  // end namespace optix_exp
