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

#pragma once

#include <internal/optix_declarations.h>

#include <string>


namespace prodlib {
class Exception;
}

namespace optix {

class Context;

class ErrorManager
{
  public:
    ErrorManager();
    ~ErrorManager();

    void setErrorString( const std::string& funcname, const prodlib::Exception& e );
    void setErrorString( const std::string& funcname, const std::exception& e );
    void setErrorString( const std::string& funcname, const std::string& error, RTresult errorCode );

    std::string        getErrorString( RTresult ) const;
    static const char* getErrorString_static( RTresult );

  private:
    std::string m_lastErrorString;
    RTresult    m_lastErrorCode;
};
}  // end of optix namespace
