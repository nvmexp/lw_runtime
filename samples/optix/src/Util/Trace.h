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

#include <corelib/system/Timer.h>
#include <prodlib/system/Logger.h>

#include <iosfwd>
#include <list>
#include <sstream>
#include <string>

#define TRMARK( x )                                                                                                    \
    if( trace.is_valid() )                                                                                             \
    trace->marker( x )

namespace optix {

// Store a list of strings paired with the time the string was added to the list,
// relative to when the Trace was created When the Trace is destroyed, the list is
// streamed to the log.
class Trace
{
  public:
    Trace( const char* location, bool doDelta = false )
        : m_doDelta( doDelta )
    {
        m_start = m_prev = corelib::getTimerTick();
        m_location       = location;
    }

    virtual ~Trace()
    {
        corelib::timerTick now         = corelib::getTimerTick();
        double             prev_delta  = corelib::getDeltaSeconds( m_prev, now ) * 1000.0;
        double             total_delta = corelib::getDeltaSeconds( m_start, now ) * 1000.0;
        std::ostream&      logstream   = lprint_stream;
        logstream << "Trace:\n";
        for( std::list<std::string>::const_iterator it = m_markers.begin(); it != m_markers.end(); ++it )
        {
            logstream << *it;
        }
        if( m_doDelta )
            logstream << m_location << ", Trace destroyed, " << prev_delta << ", ms, " << total_delta << ", ms " << std::endl;
        else
            logstream << m_location << ", Trace destroyed, " << total_delta << ", ms" << std::endl;
    }

    void marker( const char* location )
    {
        std::stringstream  str;
        corelib::timerTick prev = m_doDelta ? m_prev : m_start;
        corelib::timerTick now  = corelib::getTimerTick();
        m_prev                  = now;
        str << m_location << ", " << location << ", " << corelib::getDeltaSeconds( prev, now ) * 1000.0 << ", ms" << std::endl;
        m_markers.push_back( str.str() );
    }

    void marker( const std::string& location ) { marker( location.c_str() ); }

    void reset()
    {
        m_start = corelib::getTimerTick();
        std::stringstream str;
        str << "RESET" << std::endl;
        m_markers.push_back( str.str() );
    }

  protected:
    std::list<std::string> m_markers;
    corelib::timerTick     m_start, m_prev;
    std::string            m_location;
    bool                   m_doDelta;  // Prints the delta time from the last marker instead of the start
};

}  // end namespace optix
