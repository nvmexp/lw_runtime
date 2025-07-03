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

#include <lwca.h>

namespace optix {
namespace lwca {

class Stream;

class Event
{
  public:
    Event();
    explicit Event( LWevent event );

    // Get the low-level event
    LWevent       get();
    const LWevent get() const;

    // Creates an event.
    static Event create( unsigned int Flags = LW_EVENT_DEFAULT, LWresult* returnResult = nullptr );

    // Destroys an event.
    void destroy( LWresult* returnResult = nullptr );

    // Computes the elapsed time between two events.
    static float elapsedTime( const Event& start, const Event& end, LWresult* returnResult = nullptr );

    // Queries an event's status.  Returns true if the event is ready.
    bool query( LWresult* returnResult = nullptr ) const;

    // Records an event.
    void record( const Stream& stream, LWresult* returnResult = nullptr );

    // Waits for an event to complete.
    void synchronize( LWresult* returnResult = nullptr ) const;

  private:
    LWevent m_event;
};

}  // namespace lwca
}  // namespace optix
