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

#include <prodlib/exceptions/RTCoreError.h>

#include <sstream>

using namespace prodlib;

RTCoreError::RTCoreError( const prodlib::ExceptionInfo& info,
                          const std::string&            functionName,
                          RtcResult                     rtcoreErrorCode,
                          const std::string&            additionalText )
    : Exception( info )
    , m_functionName( functionName )
    , m_rtcoreErrorCode( rtcoreErrorCode )
    , m_additionalText( additionalText )
{
}

std::string RTCoreError::getDescription() const
{
    std::ostringstream out;
    out << "Encountered a rtcore error: " << m_functionName << " returned (" << m_rtcoreErrorCode << "): ";

    switch( m_rtcoreErrorCode )
    {
        case RTC_SUCCESS:
            out << "Success";
            break;
        case RTC_ERROR_ILWALID_VALUE:
            out << "Invalid value";
            break;
        case RTC_ERROR_ILWALID_DEVICE_CONTEXT:
            out << "Invalid device context";
            break;
        case RTC_ERROR_ILWALID_VERSION:
            out << "Invalid version";
            break;
        case RTC_ERROR_OUT_OF_MEMORY:
            out << "Out of memory";
            break;
        case RTC_ERROR_OUT_OF_CONSTANT_SPACE:
            out << "Out of constant space";
            break;
        case RTC_ERROR_ILWALID_STACK_SIZE:
            out << "Invalid stack size";
            break;
        case RTC_ERROR_COMPILE_ERROR:
            out << "Compile error";
            break;
        case RTC_ERROR_LINK_ERROR:
            out << "Link error";
            break;
        case RTC_ERROR_LAUNCH_FAILURE:
            out << "Launch failure";
            break;
        case RTC_ERROR_NOT_SUPPORTED:
            out << "Not supported";
            break;
        case RTC_ERROR_ALREADY_INITIALIZED:
            out << "Library was already initialized with different parameters";
            break;
        case RTC_ERROR_UNKNOWN:
            out << "Unknown";
            break;
    }

    return out.str() + getInfoDescription() + m_additionalText;
}
