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

#include <Context/ProfileMapping.h>

using namespace optix;


ProfileMapping::ProfileMapping( int numCounters, int numEvents, int numTimers )
{
    m_counters.resize( numCounters );
    m_events.resize( numEvents );
    m_timers.resize( numTimers );
}

ProfileMapping::~ProfileMapping()
{
}

void ProfileMapping::setCounter( int counter, const std::string& name )
{
    Counter& c = m_counters[counter];
    c.name     = name;
}

void ProfileMapping::setEvent( int event, const std::string& name )
{
    Event& e = m_events[event];
    e.name   = name;
}

void ProfileMapping::setTimer( int timer, const std::string& name, CanonicalProgramID cpid )
{
    Timer& t = m_timers[timer];
    t.kind   = Program;
    t.name   = name;
    t.cpid   = cpid;
}

void ProfileMapping::setTimer( int timer, const std::string& name )
{
    Timer& t = m_timers[timer];
    t.kind   = Generic;
    t.name   = name;
    t.cpid   = 0;
}

ProfileMapping::Timer::Timer()
    : kind( Invalid )
{
}
