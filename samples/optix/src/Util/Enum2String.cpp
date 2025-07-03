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

#include <Util/Enum2String.h>

#include <Context/Context.h>
#include <prodlib/exceptions/Assert.h>

std::string optix::format2string( RTformat val )
{
    switch( val )
    {
        // clang-format off
    case RT_FORMAT_UNKNOWN: return "RT_FORMAT_UNKNOWN";
    case RT_FORMAT_HALF: return "RT_FORMAT_HALF";
    case RT_FORMAT_HALF2: return "RT_FORMAT_HALF2";
    case RT_FORMAT_HALF3: return "RT_FORMAT_HALF3";
    case RT_FORMAT_HALF4: return "RT_FORMAT_HALF4";
    case RT_FORMAT_FLOAT: return "RT_FORMAT_FLOAT";
    case RT_FORMAT_FLOAT2: return "RT_FORMAT_FLOAT2";
    case RT_FORMAT_FLOAT3: return "RT_FORMAT_FLOAT3";
    case RT_FORMAT_FLOAT4: return "RT_FORMAT_FLOAT4";
    case RT_FORMAT_BYTE: return "RT_FORMAT_BYTE";
    case RT_FORMAT_BYTE2: return "RT_FORMAT_BYTE2";
    case RT_FORMAT_BYTE3: return "RT_FORMAT_BYTE3";
    case RT_FORMAT_BYTE4: return "RT_FORMAT_BYTE4";
    case RT_FORMAT_UNSIGNED_BYTE: return "RT_FORMAT_UNSIGNED_BYTE";
    case RT_FORMAT_UNSIGNED_BYTE2: return "RT_FORMAT_UNSIGNED_BYTE2";
    case RT_FORMAT_UNSIGNED_BYTE3: return "RT_FORMAT_UNSIGNED_BYTE3";
    case RT_FORMAT_UNSIGNED_BYTE4: return "RT_FORMAT_UNSIGNED_BYTE4";
    case RT_FORMAT_SHORT: return "RT_FORMAT_SHORT";
    case RT_FORMAT_SHORT2: return "RT_FORMAT_SHORT2";
    case RT_FORMAT_SHORT3: return "RT_FORMAT_SHORT3";
    case RT_FORMAT_SHORT4: return "RT_FORMAT_SHORT4";
    case RT_FORMAT_UNSIGNED_SHORT: return "RT_FORMAT_UNSIGNED_SHORT";
    case RT_FORMAT_UNSIGNED_SHORT2: return "RT_FORMAT_UNSIGNED_SHORT2";
    case RT_FORMAT_UNSIGNED_SHORT3: return "RT_FORMAT_UNSIGNED_SHORT3";
    case RT_FORMAT_UNSIGNED_SHORT4: return "RT_FORMAT_UNSIGNED_SHORT4";
    case RT_FORMAT_INT: return "RT_FORMAT_INT";
    case RT_FORMAT_INT2: return "RT_FORMAT_INT2";
    case RT_FORMAT_INT3: return "RT_FORMAT_INT3";
    case RT_FORMAT_INT4: return "RT_FORMAT_INT4";
    case RT_FORMAT_UNSIGNED_INT: return "RT_FORMAT_UNSIGNED_INT";
    case RT_FORMAT_UNSIGNED_INT2: return "RT_FORMAT_UNSIGNED_INT2";
    case RT_FORMAT_UNSIGNED_INT3: return "RT_FORMAT_UNSIGNED_INT3";
    case RT_FORMAT_UNSIGNED_INT4: return "RT_FORMAT_UNSIGNED_INT4";
    case RT_FORMAT_LONG_LONG: return "RT_FORMAT_LONG_LONG";
    case RT_FORMAT_LONG_LONG2: return "RT_FORMAT_LONG_LONG2";
    case RT_FORMAT_LONG_LONG3: return "RT_FORMAT_LONG_LONG3";
    case RT_FORMAT_LONG_LONG4: return "RT_FORMAT_LONG_LONG4";
    case RT_FORMAT_UNSIGNED_LONG_LONG: return "RT_FORMAT_UNSIGNED_LONG_LONG";
    case RT_FORMAT_UNSIGNED_LONG_LONG2: return "RT_FORMAT_UNSIGNED_LONG_LONG2";
    case RT_FORMAT_UNSIGNED_LONG_LONG3: return "RT_FORMAT_UNSIGNED_LONG_LONG3";
    case RT_FORMAT_UNSIGNED_LONG_LONG4: return "RT_FORMAT_UNSIGNED_LONG_LONG4";
    case RT_FORMAT_USER: return "RT_FORMAT_USER";
    case RT_FORMAT_BUFFER_ID: return "RT_FORMAT_BUFFER_ID";
    case RT_FORMAT_PROGRAM_ID: return "RT_FORMAT_PROGRAM_ID";
    case RT_FORMAT_UNSIGNED_BC1: return "RT_FORMAT_UNSIGNED_BC1";
    case RT_FORMAT_UNSIGNED_BC2: return "RT_FORMAT_UNSIGNED_BC2";
    case RT_FORMAT_UNSIGNED_BC3: return "RT_FORMAT_UNSIGNED_BC3";
    case RT_FORMAT_UNSIGNED_BC4: return "RT_FORMAT_UNSIGNED_BC4";
    case RT_FORMAT_BC4: return "RT_FORMAT_BC4";
    case RT_FORMAT_UNSIGNED_BC5: return "RT_FORMAT_UNSIGNED_BC5";
    case RT_FORMAT_BC5: return "RT_FORMAT_BC5";
    case RT_FORMAT_UNSIGNED_BC6H: return "RT_FORMAT_UNSIGNED_BC6H";
    case RT_FORMAT_BC6H: return "RT_FORMAT_BC6H";
    case RT_FORMAT_UNSIGNED_BC7: return "RT_FORMAT_UNSIGNED_BC7";
    default:
      RT_ASSERT( !!!"Unknown enumerant specified to format2string" );
            // clang-format on
    }
    return "";
}

std::string optix::gltarget2string( RTgltarget val )
{
    switch( val )
    {
        // clang-format off
    case RT_TARGET_GL_TEXTURE_2D: return "RT_TARGET_GL_TEXTURE_2D";
    case RT_TARGET_GL_TEXTURE_RECTANGLE: return "RT_TARGET_GL_TEXTURE_RECTANGLE";
    case RT_TARGET_GL_TEXTURE_3D: return "RT_TARGET_GL_TEXTURE_3D";
    case RT_TARGET_GL_RENDER_BUFFER: return "RT_TARGET_GL_RENDER_BUFFER";
    default:
      RT_ASSERT( !!!"Unknown enumerant specified to gltarget2string" );
            // clang-format on
    }
    return "";
}

std::string optix::bufferdesc2string( unsigned int val )
{
    // need to handle buffer types and flags
    std::string ret;
    bool        got = false;
    if( val & RT_BUFFER_INPUT )
        ret += "BUFFER_INPUT", got = true;
    if( val & RT_BUFFER_OUTPUT )
        ret += ( got ? std::string( " | " ) : std::string() ) + "BUFFER_OUTPUT";
    if( val & RT_BUFFER_GPU_LOCAL )
        ret += " | BUFFER_GPU_LOCAL";
    if( val & RT_BUFFER_PROGRESSIVE_STREAM )
        ret += "BUFFER_PROGRESSIVE_STREAM";
    return ret;
}

std::string optix::exception2string( RTexception val )
{
    switch( val )
    {
        // clang-format off
        case RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS:      return "RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS";
        case RT_EXCEPTION_USER_EXCEPTION_CODE_OUT_OF_BOUNDS: return "RT_EXCEPTION_USER_EXCEPTION_CODE_OUT_OF_BOUNDS";
        case RT_EXCEPTION_TRACE_DEPTH_EXCEEDED:              return "RT_EXCEPTION_TRACE_DEPTH_EXCEEDED";
        case RT_EXCEPTION_PROGRAM_ID_ILWALID:                return "RT_EXCEPTION_PROGRAM_ID_ILWALID";
        case RT_EXCEPTION_TEXTURE_ID_ILWALID:                return "RT_EXCEPTION_TEXTURE_ID_ILWALID";
        case RT_EXCEPTION_BUFFER_ID_ILWALID:                 return "RT_EXCEPTION_BUFFER_ID_ILWALID";
        case RT_EXCEPTION_INDEX_OUT_OF_BOUNDS:               return "RT_EXCEPTION_INDEX_OUT_OF_BOUNDS";
        case RT_EXCEPTION_STACK_OVERFLOW:                    return "RT_EXCEPTION_STACK_OVERFLOW";
        case RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS:        return "RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS";
        case RT_EXCEPTION_ILWALID_RAY:                       return "RT_EXCEPTION_ILWALID_RAY";
        case RT_EXCEPTION_INTERNAL_ERROR:                    return "RT_EXCEPTION_INTERNAL_ERROR";
        case RT_EXCEPTION_USER:                              return "RT_EXCEPTION_USER";
        case RT_EXCEPTION_ALL:                               return "RT_EXCEPTION_ALL";
        default:
              RT_ASSERT( !!!"Unknown enumerant specified to exception2string" );
            // clang-format on
    }
    return "";
}

std::string optix::exceptionFlags2string( unsigned int val )
{
    if( Context::getExceptionEnabled( val, RT_EXCEPTION_ALL ) )
        return "RT_EXCEPTION_ALL";

    std::string result;

    if( Context::getExceptionEnabled( val, RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS ) )
        result += "RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS ";
    if( Context::getExceptionEnabled( val, RT_EXCEPTION_USER_EXCEPTION_CODE_OUT_OF_BOUNDS ) )
        result += "RT_EXCEPTION_USER_EXCEPTION_CODE_OUT_OF_BOUNDS ";
    if( Context::getExceptionEnabled( val, RT_EXCEPTION_TRACE_DEPTH_EXCEEDED ) )
        result += "RT_EXCEPTION_TRACE_DEPTH_EXCEEDED ";
    if( Context::getExceptionEnabled( val, RT_EXCEPTION_PROGRAM_ID_ILWALID ) )
        result += "RT_EXCEPTION_PROGRAM_ID_ILWALID ";
    if( Context::getExceptionEnabled( val, RT_EXCEPTION_TEXTURE_ID_ILWALID ) )
        result += "RT_EXCEPTION_TEXTURE_ID_ILWALID ";
    if( Context::getExceptionEnabled( val, RT_EXCEPTION_BUFFER_ID_ILWALID ) )
        result += "RT_EXCEPTION_BUFFER_ID_ILWALID ";
    if( Context::getExceptionEnabled( val, RT_EXCEPTION_INDEX_OUT_OF_BOUNDS ) )
        result += "RT_EXCEPTION_INDEX_OUT_OF_BOUNDS ";
    if( Context::getExceptionEnabled( val, RT_EXCEPTION_STACK_OVERFLOW ) )
        result += "RT_EXCEPTION_STACK_OVERFLOW ";
    if( Context::getExceptionEnabled( val, RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS ) )
        result += "RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS ";
    if( Context::getExceptionEnabled( val, RT_EXCEPTION_ILWALID_RAY ) )
        result += "RT_EXCEPTION_ILWALID_RAY ";
    if( Context::getExceptionEnabled( val, RT_EXCEPTION_INTERNAL_ERROR ) )
        result += "RT_EXCEPTION_INTERNAL_ERROR ";
    if( Context::getExceptionEnabled( val, RT_EXCEPTION_USER ) )
        result += "RT_EXCEPTION_USER ";

    return result.empty() ? result : result.substr( 0, result.size() - 1 );
}
