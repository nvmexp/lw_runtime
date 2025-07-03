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

#include <prodlib/exceptions/BasicException.h>
#include <prodlib/system/Knobs.h>  // for RT_DSTRING

#include <sstream>


namespace optix {
class TypeMismatch : public prodlib::BasicException
{
  public:
    TypeMismatch( const prodlib::ExceptionInfo& info, const std::string& description )
        : BasicException( info, description )
    {
    }

    template <class Value>
    TypeMismatch( const prodlib::ExceptionInfo& info, const std::string& description, Value value );

    ~TypeMismatch() throw() override {}

  private:
    prodlib::Exception* doClone() const override { return new TypeMismatch( *this ); }
};

template <class Value>
TypeMismatch::TypeMismatch( const prodlib::ExceptionInfo& info, const std::string& description, Value value )
    : BasicException( info, std::string() )
{
    std::ostringstream out;
    out << description << value;
    m_description = RT_DSTRING( out.str() );
}

}  // end namespace optix
