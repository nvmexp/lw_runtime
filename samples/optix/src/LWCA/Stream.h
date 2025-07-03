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
class Event;

class Stream
{
  public:
    Stream();
    explicit Stream( LWstream stream );

    // Get the low-level stream
    LWstream       get();
    const LWstream get() const;

    // Create a stream.
    static Stream create( unsigned int Flags = LW_STREAM_DEFAULT, LWresult* returnResult = nullptr );

    // Create a stream with the given priority.
    static Stream createWithPriority( unsigned int flags, int priority, LWresult* returnResult = nullptr );

    // Destroys a stream.
    void destroy( LWresult* returnResult = nullptr );

    // Query the flags of a given stream.
    unsigned int getFlags( LWresult* returnResult = nullptr );

    // Query the priority of a given stream.
    int getPriority( LWresult* returnResult = nullptr );

    // Determine status of a compute stream.  Return true if the stream has completed all operations.
    bool query( LWresult* returnResult = nullptr );

    // Wait until a stream's tasks are completed.
    void synchronize( LWresult* returnResult = nullptr );

    // Make a compute stream wait on an event.
    void waitEvent( const Event& event, unsigned int Flags, LWresult* returnResult = nullptr ) const;

  private:
    LWstream m_stream;
};

}  // namespace lwca
}  // namespace optix
