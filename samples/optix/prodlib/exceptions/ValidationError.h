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

#include <prodlib/exceptions/IlwalidContext.h>

#include <sstream>


namespace prodlib {
class ValidationError : public IlwalidContext
{
  public:
    ValidationError( const prodlib::ExceptionInfo& info, const std::string& errorString );
    ~ValidationError() throw() override {}

    template <class Value>
    ValidationError( const prodlib::ExceptionInfo& info, const std::string& description, Value value );

    template <class Value>
    ValidationError( const prodlib::ExceptionInfo& info, const std::string& description, Value value0, Value value1 );

    std::string getDescription() const override;

  private:
    Exception* doClone() const override { return new ValidationError( *this ); }
};

template <class Value>
ValidationError::ValidationError( const prodlib::ExceptionInfo& info, const std::string& description, Value value )
    : IlwalidContext( info, std::string() )
{
    std::ostringstream out;
    out << description << value;
    m_description = out.str();
}

template <class Value>
ValidationError::ValidationError( const prodlib::ExceptionInfo& info, const std::string& description, Value value0, Value value1 )
    : IlwalidContext( info, std::string() )
{
    std::ostringstream out;
    out << description << value0 << " " << value1;
    m_description = out.str();
}

}  // end namespace prodlib
