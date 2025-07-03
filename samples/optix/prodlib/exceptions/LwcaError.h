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

#include <lwca.h>
#include <driver_types.h>

#include <string>

namespace prodlib {

class LwdaError : public Exception
{
  public:
    LwdaError( const ExceptionInfo& info,
               const std::string&   functionName,
               LWresult             lwdaErrorCode,
               const std::string&   additionalText = "" );
    ~LwdaError() noexcept override                 = default;
    std::string      getDescription() const override;
    virtual LWresult getErrorCode() const;
    int              getLwdaDeviceNumber() const;

  private:
    Exception* doClone() const override { return new LwdaError( *this ); }

    std::string m_functionName;
    LWresult    m_lwdaErrorCode;
    std::string m_additionalText;    // Additional information about the error
    int         m_lwdaDeviceNumber;  // Device ordinal from the current device when the exception
                                     // is thrown. -1 if LWCA can't tell us.
};

class LwdaRuntimeError : public Exception
{
  public:
    LwdaRuntimeError( const ExceptionInfo& info, const std::string& functionName, lwdaError lwdaErrorCode );
    ~LwdaRuntimeError() noexcept override = default;
    std::string       getDescription() const override;
    virtual lwdaError getErrorCode() const;
    int               getLwdaDeviceNumber() const;

  private:
    Exception* doClone() const override { return new LwdaRuntimeError( *this ); }

    std::string m_functionName;
    lwdaError   m_lwdaErrorCode;
    int         m_lwdaDeviceNumber;  // Device ordinal from the current device when the exception
                                     // is thrown. -1 if LWCA can't tell us.
};

inline void lwdaDriverThrow( LWresult err, const char* expr, const char* file, unsigned int line )
{
    if( err != LWDA_SUCCESS )
        throw LwdaError( ExceptionInfo( file, line, true ), expr, err );
}


inline void lwdaRuntimeThrow( lwdaError err, const char* expr, const char* file, unsigned int line )
{
    if( err != lwdaSuccess )
        throw LwdaRuntimeError( ExceptionInfo( file, line, true ), expr, err );
}

inline void checkLwdaError( LWresult result, const char* expr, const char* file, unsigned int line, LWresult* returnResult )
{
    if( returnResult )
    {
        *returnResult = result;
    }
    else if( result != LWDA_SUCCESS )
    {
        throw LwdaError( ExceptionInfo( file, line, true ), expr, result );
    }
}

}  // end namespace prodlib

#define CALL_LWDA_DRIVER_THROW( call ) prodlib::lwdaDriverThrow( call, #call, RT_FILE_NAME, RT_LINE )
#define CALL_LWDA_RUNTIME_THROW( call ) prodlib::lwdaRuntimeThrow( call, #call, RT_FILE_NAME, RT_LINE )
