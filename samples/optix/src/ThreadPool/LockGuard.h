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

#include <prodlib/misc/TimeViz.h>

namespace optix {

/// Similar to std::lock_guard, but with TimeViz support.
template <typename T>
class LockGuard
{
  public:
    LockGuard( T& mutex )
        : m_mutex( mutex )
    {
        TIMEVIZ_SCOPE( "Waiting for lock" );
        m_mutex.lock();
    }
    ~LockGuard() { m_mutex.unlock(); }
    LockGuard( const LockGuard& ) = delete;
    LockGuard& operator=( const LockGuard& ) = delete;

  private:
    T& m_mutex;
};

/// Similar to LockGuard, except that locking and unlocking are reversed.
template <typename T>
class ReverseLockGuard
{
  public:
    ReverseLockGuard( T& mutex )
        : m_mutex( mutex )
    {
        m_mutex.unlock();
    }
    ~ReverseLockGuard()
    {
        TIMEVIZ_SCOPE( "Waiting for lock" );
        m_mutex.lock();
    }
    ReverseLockGuard( const ReverseLockGuard& ) = delete;
    ReverseLockGuard& operator=( const ReverseLockGuard& ) = delete;

  private:
    T& m_mutex;
};

}  // namespace optix
