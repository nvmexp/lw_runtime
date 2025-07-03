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

#include <FrontEnd/Canonical/CanonicalProgramID.h>
#include <Objects/SemanticType.h>

#include <corelib/misc/Concepts.h>

#include <string>
#include <vector>


namespace optix {
//
// Note: ProfileMapping is created by the profile manager so that it
// can size device-side buffers appropriately.
//
class ProfileMapping : private corelib::NonCopyable
{
  public:
    ~ProfileMapping();

    enum TimerKind
    {
        Program,
        Generic,
        Invalid
    };
    void setCounter( int counter, const std::string& name );
    void setEvent( int event, const std::string& name );
    void setTimer( int timer, const std::string& name, CanonicalProgramID cpid );
    void setTimer( int timer, const std::string& name );

  private:
    friend class ProfileManager;
    ProfileMapping( int numCounters, int numEvents, int numTimers );

    struct Counter
    {
        std::string name;
    };
    struct Event
    {
        std::string name;
    };
    struct Timer
    {
        TimerKind          kind;
        std::string        name;
        CanonicalProgramID cpid;
        Timer();
    };

    std::vector<Counter> m_counters;
    std::vector<Event>   m_events;
    std::vector<Timer>   m_timers;
};

}  // end namespace optix
