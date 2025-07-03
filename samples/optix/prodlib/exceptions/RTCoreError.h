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

#pragma once

#include <prodlib/exceptions/Exception.h>
#include <rtcore/interface/types.h>

namespace prodlib {
class RTCoreError : public prodlib::Exception
{
  public:
    RTCoreError( const prodlib::ExceptionInfo& info,
                 const std::string&            functionName,
                 RtcResult                     rtcoreErrorCode,
                 const std::string&            additionalText = "" );
    virtual ~RTCoreError() throw() {}
    virtual std::string getDescription() const;

    RtcResult getRtcoreErrorCode() const { return m_rtcoreErrorCode; }

  private:
    virtual Exception* doClone() const { return new RTCoreError( *this ); }

    std::string m_functionName;
    RtcResult   m_rtcoreErrorCode;
    std::string m_additionalText;  // Additional information about the error
};

}  // end namespace prodlib
