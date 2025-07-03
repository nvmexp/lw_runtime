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

#include <prodlib/exceptions/Backtrace.h>
#include <prodlib/exceptions/Exception.h>

#include <sstream>

using namespace prodlib;


ExceptionInfo::ExceptionInfo( const char* const filename, unsigned int lineno, bool showInfo )
    : m_filename( filename )
    , m_lineno( lineno )
    , m_showInfo( showInfo )
{
}

std::string ExceptionInfo::getDescription() const
{
    std::stringstream ss;

    if( m_showInfo )
    {
        ss << "file: " << m_filename << ", line: " << m_lineno;
    }

    return ss.str();
}

Exception::Exception( const ExceptionInfo& exceptionInfo )
    : m_exceptionInfo( exceptionInfo )
{
    m_stackTrace = walkStackTrace();
}

Exception* Exception::clone() const
{
    Exception* c = doClone();
    // Make sure we get the stack from the original Exception object in case it accidentilly
    // was created during the clone process.
    c->m_stackTrace = m_stackTrace;
    return c;
}

std::string Exception::getInfoDescription( bool prependSeparator ) const
{
    // If requested, prepend a separator iff the exception info is available.
    if( prependSeparator )
    {
        const std::string info = m_exceptionInfo.getDescription();
        if( !info.empty() )
            return std::string( ", " ) + info;
        else
            return std::string();
    }
    else
    {
        return m_exceptionInfo.getDescription();
    }
}

ExceptionInfo Exception::getExceptionInfo() const
{
    return m_exceptionInfo;
}

const std::string& Exception::getStacktrace() const
{
    return m_stackTrace;
}

std::string Exception::walkStackTrace() const
{
    // Skip first 3 to avoid putting the backtrace calling code
    // in the trace
    std::vector<std::string> trace = prodlib::backtrace( 3, 128 );

    std::ostringstream strm;

    if( trace.size() > 4 )
    {
        // Omit the bottom four frames, since they are just
        // crt entry points
        // TODO: On windows, we might want to exclude the top 4 entries,
        // because they are CRT startup funcitons that execute before main
        for( std::size_t i = 0; i < trace.size(); ++i )
        {
            strm << "\t(" << i << ") " << trace[i] << '\n';
        }
    }

    return strm.str();
}
