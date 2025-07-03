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

#include <exp/context/ErrorHandling.h>

#include <corelib/system/LwdaDriver.h>

#include <cstring>
#include <sstream>

namespace optix_exp {

OptixResult colwertError( RtcResult result )
{
    switch( result )
    {
        case RTC_SUCCESS:
            return OPTIX_SUCCESS;
        case RTC_ERROR_ILWALID_VALUE:
            return OPTIX_ERROR_ILWALID_VALUE;
        case RTC_ERROR_ILWALID_DEVICE_CONTEXT:
            return OPTIX_ERROR_ILWALID_DEVICE_CONTEXT;
        case RTC_ERROR_ILWALID_VERSION: /* generic */
            return OPTIX_ERROR_INTERNAL_ERROR;
        case RTC_ERROR_OUT_OF_MEMORY: /* generic */
            return OPTIX_ERROR_INTERNAL_ERROR;
        case RTC_ERROR_OUT_OF_CONSTANT_SPACE:
            return OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY;
        case RTC_ERROR_ILWALID_STACK_SIZE:
            return OPTIX_ERROR_ILWALID_VALUE;
        case RTC_ERROR_COMPILE_ERROR:
            return OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
        case RTC_ERROR_LINK_ERROR:
            return OPTIX_ERROR_PIPELINE_LINK_ERROR;
        case RTC_ERROR_LAUNCH_FAILURE:
            return OPTIX_ERROR_LAUNCH_FAILURE;
        case RTC_ERROR_NOT_SUPPORTED:
            return OPTIX_ERROR_NOT_SUPPORTED;
        case RTC_ERROR_ALREADY_INITIALIZED: /* generic */
            return OPTIX_ERROR_INTERNAL_ERROR;
        case RTC_ERROR_UNKNOWN:
            return OPTIX_ERROR_UNKNOWN;
    }
    return OPTIX_ERROR_UNKNOWN;
}

OptixResult colwertError( LWresult result )
{
    switch( result )
    {
        case LWDA_SUCCESS:
            return OPTIX_SUCCESS;
        case LWDA_ERROR_DEINITIALIZED:
        case LWDA_ERROR_NOT_INITIALIZED:
            return OPTIX_ERROR_LWDA_NOT_INITIALIZED;
        case LWDA_ERROR_OUT_OF_MEMORY:
            return OPTIX_ERROR_DEVICE_OUT_OF_MEMORY;
        default:
            return OPTIX_ERROR_LWDA_ERROR;
    }
}

APIError::APIError( const prodlib::ExceptionInfo& exceptionInfo, OptixResult code )
    : APIError( exceptionInfo, code, "" )
{
}

APIError::APIError( const prodlib::ExceptionInfo& exceptionInfo, OptixResult code, const std::string& description )
    : prodlib::Exception( exceptionInfo )
    , m_code( code )
    , m_description( description )
{
}

const char* APIError::getErrorName( OptixResult code )
{
    switch( code )
    {
        case OPTIX_SUCCESS:
            return "OPTIX_SUCCESS";
        case OPTIX_ERROR_ILWALID_VALUE:
            return "OPTIX_ERROR_ILWALID_VALUE";
        case OPTIX_ERROR_HOST_OUT_OF_MEMORY:
            return "OPTIX_ERROR_HOST_OUT_OF_MEMORY";
        case OPTIX_ERROR_ILWALID_OPERATION:
            return "OPTIX_ERROR_ILWALID_OPERATION";
        case OPTIX_ERROR_FILE_IO_ERROR:
            return "OPTIX_ERROR_FILE_IO_ERROR";
        case OPTIX_ERROR_ILWALID_FILE_FORMAT:
            return "OPTIX_ERROR_ILWALID_FILE_FORMAT";
        case OPTIX_ERROR_DISK_CACHE_ILWALID_PATH:
            return "OPTIX_ERROR_DISK_CACHE_ILWALID_PATH";
        case OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR:
            return "OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR";
        case OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR:
            return "OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR";
        case OPTIX_ERROR_DISK_CACHE_ILWALID_DATA:
            return "OPTIX_ERROR_DISK_CACHE_ILWALID_DATA";
        case OPTIX_ERROR_LAUNCH_FAILURE:
            return "OPTIX_ERROR_LAUNCH_FAILURE";
        case OPTIX_ERROR_ILWALID_DEVICE_CONTEXT:
            return "OPTIX_ERROR_ILWALID_DEVICE_CONTEXT";
        case OPTIX_ERROR_LWDA_NOT_INITIALIZED:
            return "OPTIX_ERROR_LWDA_NOT_INITIALIZED";
        case OPTIX_ERROR_VALIDATION_FAILURE:
            return "OPTIX_ERROR_VALIDATION_FAILURE";
        case OPTIX_ERROR_ILWALID_PTX:
            return "OPTIX_ERROR_ILWALID_PTX";
        case OPTIX_ERROR_ILWALID_LAUNCH_PARAMETER:
            return "OPTIX_ERROR_ILWALID_LAUNCH_PARAMETER";
        case OPTIX_ERROR_ILWALID_PAYLOAD_ACCESS:
            return "OPTIX_ERROR_ILWALID_PAYLOAD_ACCESS";
        case OPTIX_ERROR_ILWALID_ATTRIBUTE_ACCESS:
            return "OPTIX_ERROR_ILWALID_ATTRIBUTE_ACCESS";
        case OPTIX_ERROR_ILWALID_FUNCTION_USE:
            return "OPTIX_ERROR_ILWALID_FUNCTION_USE";
        case OPTIX_ERROR_ILWALID_FUNCTION_ARGUMENTS:
            return "OPTIX_ERROR_ILWALID_FUNCTION_ARGUMENTS";
        case OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY:
            return "OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY";
        case OPTIX_ERROR_PIPELINE_LINK_ERROR:
            return "OPTIX_ERROR_PIPELINE_LINK_ERROR";
        case OPTIX_ERROR_ILLEGAL_DURING_TASK_EXELWTE:
            return "OPTIX_ERROR_ILLEGAL_DURING_TASK_EXELWTE";
        case OPTIX_ERROR_INTERNAL_COMPILER_ERROR:
            return "OPTIX_ERROR_INTERNAL_COMPILER_ERROR";
        case OPTIX_ERROR_DENOISER_MODEL_NOT_SET:
            return "OPTIX_ERROR_DENOISER_MODEL_NOT_SET";
        case OPTIX_ERROR_DENOISER_NOT_INITIALIZED:
            return "OPTIX_ERROR_DENOISER_NOT_INITIALIZED";
        case OPTIX_ERROR_ACCEL_NOT_COMPATIBLE:
            return "OPTIX_ERROR_ACCEL_NOT_COMPATIBLE";
        case OPTIX_ERROR_PAYLOAD_TYPE_MISMATCH:
            return "OPTIX_ERROR_PAYLOAD_TYPE_MISMATCH";
        case OPTIX_ERROR_PAYLOAD_TYPE_RESOLUTION_FAILED:
            return "OPTIX_ERROR_PAYLOAD_TYPE_RESOLUTION_FAILED";
        case OPTIX_ERROR_PAYLOAD_TYPE_ID_ILWALID:
            return "OPTIX_ERROR_PAYLOAD_TYPE_ID_ILWALID";
        case OPTIX_ERROR_NOT_SUPPORTED:
            return "OPTIX_ERROR_NOT_SUPPORTED";
        case OPTIX_ERROR_UNSUPPORTED_ABI_VERSION:
            return "OPTIX_ERROR_UNSUPPORTED_ABI_VERSION";
        case OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH:
            return "OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH";
        case OPTIX_ERROR_ILWALID_ENTRY_FUNCTION_OPTIONS:
            return "OPTIX_ERROR_ILWALID_ENTRY_FUNCTION_OPTIONS";
        case OPTIX_ERROR_LIBRARY_NOT_FOUND:
            return "OPTIX_ERROR_LIBRARY_NOT_FOUND";
        case OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND:
            return "OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND";
        case OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE:
            return "OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE";
        case OPTIX_ERROR_DEVICE_OUT_OF_MEMORY:
            return "OPTIX_ERROR_DEVICE_OUT_OF_MEMORY";
        case OPTIX_ERROR_LWDA_ERROR:
            return "OPTIX_ERROR_LWDA_ERROR";
        case OPTIX_ERROR_INTERNAL_ERROR:
            return "OPTIX_ERROR_INTERNAL_ERROR";
        case OPTIX_ERROR_UNKNOWN:
            return "OPTIX_ERROR_UNKNOWN";
    }
    return "Unknown OptixResult code";
}

const char* APIError::getErrorString( OptixResult code )
{
    switch( code )
    {
        case OPTIX_SUCCESS:
            return "Success";
        case OPTIX_ERROR_ILWALID_VALUE:
            return "Invalid value";
        case OPTIX_ERROR_HOST_OUT_OF_MEMORY:
            return "Host is out of memory";
        case OPTIX_ERROR_ILWALID_OPERATION:
            return "Invalid operation";
        case OPTIX_ERROR_FILE_IO_ERROR:
            return "File I/O error";
        case OPTIX_ERROR_ILWALID_FILE_FORMAT:
            return "Invalid file format";
        case OPTIX_ERROR_DISK_CACHE_ILWALID_PATH:
            return "Invalid path to disk cache file";
        case OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR:
            return "Disk cache file is not writable";
        case OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR:
            return "Disk cache database error";
        case OPTIX_ERROR_DISK_CACHE_ILWALID_DATA:
            return "Invalid data in disk cache";
        case OPTIX_ERROR_LAUNCH_FAILURE:
            return "Launch failure";
        case OPTIX_ERROR_ILWALID_DEVICE_CONTEXT:
            return "Invalid device context";
        case OPTIX_ERROR_LWDA_NOT_INITIALIZED:
            return "LWCA is not initialized";
        case OPTIX_ERROR_VALIDATION_FAILURE:
            return "Error during validation mode run";
        case OPTIX_ERROR_ILWALID_PTX:
            return "Invalid PTX input";
        case OPTIX_ERROR_ILWALID_LAUNCH_PARAMETER:
            return "Invalid launch parameter";
        case OPTIX_ERROR_ILWALID_PAYLOAD_ACCESS:
            return "Invalid payload access";
        case OPTIX_ERROR_ILWALID_ATTRIBUTE_ACCESS:
            return "Invalid attribute access";
        case OPTIX_ERROR_ILWALID_FUNCTION_USE:
            return "Invalid use of optix device function";
        case OPTIX_ERROR_ILWALID_FUNCTION_ARGUMENTS:
            return "Invalid arguments for optix device function";
        case OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY:
            return "Pipeline is out of constant memory";
        case OPTIX_ERROR_PIPELINE_LINK_ERROR:
            return "Pipeline link error";
        case OPTIX_ERROR_ILLEGAL_DURING_TASK_EXELWTE:
            return "Operation illegal during task exelwtion";
        case OPTIX_ERROR_INTERNAL_COMPILER_ERROR:
            return "Internal compiler error";
        case OPTIX_ERROR_DENOISER_MODEL_NOT_SET:
            return "Denoiser model (weights) not set";
        case OPTIX_ERROR_DENOISER_NOT_INITIALIZED:
            return "Denoiser not initialized";
        case OPTIX_ERROR_ACCEL_NOT_COMPATIBLE:
            return "Acceleration structure not compatible";
        case OPTIX_ERROR_PAYLOAD_TYPE_MISMATCH:
            return "Payload types do not match";
        case OPTIX_ERROR_PAYLOAD_TYPE_RESOLUTION_FAILED:
            return "Payload type cannot be uniquely resolved";
        case OPTIX_ERROR_PAYLOAD_TYPE_ID_ILWALID:
            return "Payload type ID is not valid";
        case OPTIX_ERROR_NOT_SUPPORTED:
            return "Feature not supported";
        case OPTIX_ERROR_UNSUPPORTED_ABI_VERSION:
            return "Unsupported ABI version";
        case OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH:
            return "Function table size mismatch";
        case OPTIX_ERROR_ILWALID_ENTRY_FUNCTION_OPTIONS:
            return "Invalid options to entry function";
        case OPTIX_ERROR_LIBRARY_NOT_FOUND:
            return "Library not found";
        case OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND:
            return "Entry symbol not found";
        case OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE:
            return "Library could not be unloaded";
        case OPTIX_ERROR_DEVICE_OUT_OF_MEMORY:
            return "Device out of memory";
        case OPTIX_ERROR_LWDA_ERROR:
            return "LWCA error";
        case OPTIX_ERROR_INTERNAL_ERROR:
            return "Internal error";
        case OPTIX_ERROR_UNKNOWN:
            return "Unknown error";
    }
    return "Unknown OptixResult code";
}

std::string APIError::getDescription() const
{
    std::ostringstream out;
    out << getErrorName( m_code );
    if( m_description.empty() )
        out << ": " << getInfoDescription( false );
    else
        out << ": \"" << m_description << "\"" << getInfoDescription();
    return out.str();
}

OptixResult ErrorDetails::logDetails( OptixResult code, const std::string& description )
{
    logDetailsMessage( description );
    return code;
}

OptixResult ErrorDetails::logDetails( LWresult code, const std::string& description )
{
    const char* lwdaErrorString;
    corelib::lwdaDriver().LwGetErrorString( code, &lwdaErrorString );
    std::string str = description;
    if( lwdaErrorString )
        str += std::string( " (LWCA error string: " ) + lwdaErrorString + ", LWCA error code: " + std::to_string( code )
               + ")";
    else
        str += std::string( " (LWCA error code: " ) + std::to_string( code ) + ")";
    logDetailsMessage( str );
    return colwertError( code );
}

OptixResult ErrorDetails::logDetails( RtcResult code, const std::string& description )
{
    logDetailsMessage( description );
    return colwertError( code );
}

void ErrorDetails::logDetailsMessage( const std::string& description )
{
    if( !m_description.empty() )
        m_description += "\n";
    m_description += description;
}

void copyCompileDetails( std::ostringstream& compileDetails, char* logString, size_t* logStringSize )
{
    copyCompileDetails( compileDetails.str(), logString, logStringSize );
}

void copyCompileDetails( const std::string& compileDetails, char* logString, size_t* logStringSize )
{
    if( !logString )
        return;
    if( logStringSize == nullptr )
        return;
    if( *logStringSize == 0 )
        return;
    std::strncpy( logString, compileDetails.c_str(), *logStringSize );
    logString[*logStringSize - 1] = 0;
    *logStringSize                = compileDetails.size() + 1;
}
}  // end namespace optix_exp

extern "C" const char* optixGetErrorName( OptixResult result )
{
    return optix_exp::APIError::getErrorName( result );
}

extern "C" const char* optixGetErrorString( OptixResult result )
{
    return optix_exp::APIError::getErrorString( result );
}
