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

#include <string>


namespace prodlib {

#if defined( DEBUG ) || defined( DEVELOP )
#if !defined( RT_FILE_NAME )
#define RT_FILE_NAME __FILE__
#endif
#define RT_LINE __LINE__
#define RT_EXCEPTION_INFO prodlib::ExceptionInfo( RT_FILE_NAME, RT_LINE, true )
#else
#if !defined( RT_FILE_NAME )
#define RT_FILE_NAME "<internal>"
#endif
#define RT_LINE 0
#define RT_EXCEPTION_INFO prodlib::ExceptionInfo( RT_FILE_NAME, RT_LINE, false )
#endif


class ExceptionInfo
{
  public:
    ExceptionInfo( const char* filename, unsigned int lineno, bool showInfo );
    std::string getDescription() const;

  private:
    const char* const m_filename;
    unsigned int      m_lineno   = 0;
    bool              m_showInfo = false;
};


// Any descendant of Exception should have:
// 1. virtual ~Class() throw();
// 2. virtual Exception* doClone() const { return new Class(*this); }
class Exception : public std::exception
{
  public:
    Exception( const ExceptionInfo& exceptionInfo );
    ~Exception() throw() override {}  // std::exception has the throw() decorator on it, so we must too.

    Exception*          clone() const;
    virtual std::string getDescription() const = 0;
    virtual std::string getInfoDescription( bool prependSeparator = true ) const;
    ExceptionInfo getExceptionInfo() const;

    const std::string& getStacktrace() const;

    /// From std::exception
    const char* what() const throw() override
    {
        m_what = getDescription();
        return m_what.c_str();
    }

  private:
    ExceptionInfo      m_exceptionInfo;
    virtual Exception* doClone() const = 0;

  protected:
    std::string walkStackTrace() const;

    std::string         m_stackTrace;
    mutable std::string m_what;
};

}  // end namespace prodlib
