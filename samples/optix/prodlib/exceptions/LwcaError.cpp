// Copyright LWPU Corporation 2008
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <lwda_runtime_api.h>

#include <prodlib/exceptions/LwdaError.h>

#include <sstream>

using namespace prodlib;


namespace {
int getLwrrentLwdaDevice()
{
    int device;
    if( lwdaGetDevice( &device ) == lwdaSuccess )
        return device;
    return -1;
}
}

LwdaError::LwdaError( const ExceptionInfo& info, const std::string& function_name, LWresult lwdaErrorCode, const std::string& additionalText )
    : Exception( info )
    , m_functionName( function_name )
    , m_lwdaErrorCode( lwdaErrorCode )
    , m_additionalText( additionalText )
{
    m_lwdaDeviceNumber = getLwrrentLwdaDevice();
}

std::string LwdaError::getDescription() const
{
    std::ostringstream out;
    out << "Encountered a LWCA error: " << m_functionName << " returned (" << m_lwdaErrorCode << "): ";

    switch( m_lwdaErrorCode )
    {
        case LWDA_SUCCESS:  // 0
            out << "Success";
            break;

        case LWDA_ERROR_ILWALID_VALUE:  // 1
            out << "Invalid value";
            break;

        case LWDA_ERROR_OUT_OF_MEMORY:  // 2
            out << "Out of memory";
            break;

        case LWDA_ERROR_NOT_INITIALIZED:  // 3
            out << "Not initialized";
            break;

        case LWDA_ERROR_DEINITIALIZED:  // 4
            out << "Deinitialized";
            break;

        case LWDA_ERROR_PROFILER_DISABLED:  // 5
            out << "Profiler disabled";
            break;

        // Deprecated as of LWCA 5.0
        // case LWDA_ERROR_PROFILER_NOT_INITIALIZED: // 6
        // case LWDA_ERROR_PROFILER_ALREADY_STARTED: // 7
        // case LWDA_ERROR_PROFILER_ALREADY_STOPPED: // 8

        case LWDA_ERROR_NO_DEVICE:  // 100
            out << "No device";
            break;

        case LWDA_ERROR_ILWALID_DEVICE:  // 101
            out << "Invalid device";
            break;

        case LWDA_ERROR_ILWALID_IMAGE:  // 200
            out << "Invalid image";
            break;

        case LWDA_ERROR_ILWALID_CONTEXT:  // 201
            out << "Invalid context";
            break;

        case LWDA_ERROR_CONTEXT_ALREADY_LWRRENT:  // 202
            out << "Context already current";
            break;

        case LWDA_ERROR_MAP_FAILED:  // 205
            out << "Map failed";
            break;

        case LWDA_ERROR_UNMAP_FAILED:  // 206
            out << "Unmap failed";
            break;

        case LWDA_ERROR_ARRAY_IS_MAPPED:  // 207
            out << "Array is mapped";
            break;

        case LWDA_ERROR_ALREADY_MAPPED:  // 208
            out << "Already mapped";
            break;

        case LWDA_ERROR_NO_BINARY_FOR_GPU:  // 209
            out << "No binary for GPU";
            break;

        case LWDA_ERROR_ALREADY_ACQUIRED:  // 210
            out << "Already acquired";
            break;

        case LWDA_ERROR_NOT_MAPPED:  // 211
            out << "Not mapped";
            break;

        case LWDA_ERROR_NOT_MAPPED_AS_ARRAY:  // 212
            out << "Not mapped as array";
            break;

        case LWDA_ERROR_NOT_MAPPED_AS_POINTER:  // 213
            out << "Not mapped as pointer";
            break;

        case LWDA_ERROR_ECC_UNCORRECTABLE:  // 214
            out << "ECC uncorrectable";
            break;

        case LWDA_ERROR_UNSUPPORTED_LIMIT:  // 215
            out << "Unsupported limit";
            break;

        case LWDA_ERROR_CONTEXT_ALREADY_IN_USE:  // 216
            out << "Context already in use";
            break;

        case LWDA_ERROR_PEER_ACCESS_UNSUPPORTED:  // 217
            out << "Peer acces unsupported";
            break;

#if( LWDA_VERSION >= 6000 )
        case LWDA_ERROR_ILWALID_PTX:  // 218
            out << "Invalid ptx";
            break;
#endif

#if( LWDA_VERSION >= 6050 )
        case LWDA_ERROR_ILWALID_GRAPHICS_CONTEXT:  // 219
            out << "Invalid graphics context";
            break;
#endif

        case LWDA_ERROR_ILWALID_SOURCE:  // 300
            out << "Invalid source";
            break;

        case LWDA_ERROR_FILE_NOT_FOUND:  // 301
            out << "File not found";
            break;

        case LWDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:  // 302
            out << "Shared object symbol not found";
            break;

        case LWDA_ERROR_SHARED_OBJECT_INIT_FAILED:  // 303
            out << "Shared object init failed";
            break;

        case LWDA_ERROR_OPERATING_SYSTEM:  // 304
            out << "Operating system call failed";
            break;

        case LWDA_ERROR_ILWALID_HANDLE:  // 400
            out << "Invalid handle";
            break;

        case LWDA_ERROR_NOT_FOUND:  // 500
            out << "Not found";
            break;

        case LWDA_ERROR_NOT_READY:  // 600
            out << "Not ready";
            break;

        case LWDA_ERROR_ILLEGAL_ADDRESS:  // 700
            out << "Illegal address";
            break;

        case LWDA_ERROR_LAUNCH_OUT_OF_RESOURCES:  // 701
            out << "Launch out of resources";
            break;

        case LWDA_ERROR_LAUNCH_TIMEOUT:  // 702
            out << "Launch timeout";
            break;

        case LWDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:  // 703
            out << "Launch incompatible texturing";
            break;

        case LWDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:  // 704
            out << "Peer access already enabled";
            break;

        case LWDA_ERROR_PEER_ACCESS_NOT_ENABLED:  // 705
            out << "Peer access not enabled";
            break;

        case LWDA_ERROR_PRIMARY_CONTEXT_ACTIVE:  // 708
            out << "Primary device for context already initialized";
            break;

        case LWDA_ERROR_CONTEXT_IS_DESTROYED:  // 709
            out << "Context has been destroyed";
            break;

        case LWDA_ERROR_ASSERT:  // 710
            out << "Device side assert triggered";
            break;

        case LWDA_ERROR_TOO_MANY_PEERS:  // 711
            out << "Too many peers";
            break;

        case LWDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:  // 712
            out << "Host memory already registered";
            break;

        case LWDA_ERROR_HOST_MEMORY_NOT_REGISTERED:  // 713
            out << "Host memory not registered";
            break;

#if( LWDA_VERSION >= 6000 )
        case LWDA_ERROR_HARDWARE_STACK_ERROR:  // 714
            out << "Device stack error (stack error or exceeded stack size limit)";
            break;

        case LWDA_ERROR_ILLEGAL_INSTRUCTION:  // 715
            out << "Illegal instruction";
            break;

        case LWDA_ERROR_MISALIGNED_ADDRESS:  // 716
            out << "Misaligned address";
            break;

        case LWDA_ERROR_ILWALID_ADDRESS_SPACE:  // 717
            out << "Invalid address space";
            break;

        case LWDA_ERROR_ILWALID_PC:  // 718
            out << "Invalid program counter";
            break;
#endif

        case LWDA_ERROR_LAUNCH_FAILED:  // 719 (700 for pre LWCA 6)
            out << "Launch failed";
            break;

        case LWDA_ERROR_NOT_PERMITTED:  // 800
            out << "Operation not permitted";
            break;

        case LWDA_ERROR_NOT_SUPPORTED:  // 801
            out << "Operation not supported on current system or device";
            break;

        case LWDA_ERROR_UNKNOWN:
        default:
            out << "Unknown";
            break;
    }

    return out.str() + getInfoDescription() + m_additionalText;
}

LWresult LwdaError::getErrorCode() const
{
    return m_lwdaErrorCode;
}

int LwdaError::getLwdaDeviceNumber() const
{
    return m_lwdaDeviceNumber;
}

LwdaRuntimeError::LwdaRuntimeError( const ExceptionInfo& info, const std::string& function_name, lwdaError lwdaErrorCode )
    : Exception( info )
    , m_functionName( function_name )
    , m_lwdaErrorCode( lwdaErrorCode )
{
    m_lwdaDeviceNumber = getLwrrentLwdaDevice();
}

std::string LwdaRuntimeError::getDescription() const
{
    std::ostringstream out;
    out << "Encountered a LWCA error: " << m_functionName << " returned (" << m_lwdaErrorCode << "): ";

    out << lwdaGetErrorString( m_lwdaErrorCode );

    return out.str() + getInfoDescription();
}

lwdaError LwdaRuntimeError::getErrorCode() const
{
    return m_lwdaErrorCode;
}

int LwdaRuntimeError::getLwdaDeviceNumber() const
{
    return m_lwdaDeviceNumber;
}
