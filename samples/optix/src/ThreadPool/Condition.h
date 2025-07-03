// Copyright (c) 2018, LWPU CORPORATION.
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

#include <condition_variable>
#include <mutex>

namespace optix {

/// Conditions allow threads to signal an event and to wait for such a signal, respectively.
///
/// This class is similar to std::condition_variable. The differences are (a) the corresponding
/// mutex is part of the class and hidden from the user, and (b) spurious wakeups are handled
/// internally.
class Condition
{
  public:
    /// Constructor
    Condition()
        : m_signaled( false )
    {
    }

    /// Waits for the condition to be signaled.
    ///
    /// If the condition is already signaled at this time the call will return immediately.
    void wait();

    /// Waits for the condition to be signaled until a given timeout.
    ///
    /// If the condition is already signaled at this time the call will return immediately.
    ///
    /// \param timeout    Maximum time period (in seconds) to wait for the condition to be signaled.
    /// \return           \c true if the timeout was hit, and \c false if the condition was
    ///                   signaled.
    bool waitFor( double timeout );

    /// Signals the condition.
    ///
    /// This will wake up one thread waiting for the condition. It does not matter if the call to
    /// #signal() or #wait() comes first.
    ///
    /// \note If there are two or more calls to #signal() without a call to #wait() in between (and
    /// no outstanding #wait() call), all calls to #signal() except the first one are ignored, i.e.,
    /// calls to #signal() do not increment some counter, but just set a flag.
    void signal();

    /// Resets the condition.
    ///
    /// This will undo the effect of a #signal() call if there was no outstanding #wait() call.
    void reset();

  private:
    std::condition_variable m_conditiolwariable;
    std::mutex              m_mutex;
    bool                    m_signaled;
};

}  // namespace optix
