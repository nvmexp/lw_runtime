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

#include <ThreadPool/Condition.h>

#include <chrono>

namespace optix {

void Condition::wait()
{
    std::unique_lock<std::mutex> guard( m_mutex );
    while( !m_signaled )
        m_conditiolwariable.wait( guard );
    m_signaled = false;
}

bool Condition::waitFor( double timeout )
{
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::chrono::system_clock::duration   duration(
        std::chrono::duration_cast<std::chrono::system_clock::duration>( std::chrono::duration<double>( timeout ) ) );
    std::chrono::system_clock::time_point then = now + duration;

    std::unique_lock<std::mutex> guard( m_mutex );
    while( !m_signaled )
        if( m_conditiolwariable.wait_until( guard, then ) == std::cv_status::timeout )
            return true;

    m_signaled = false;
    return false;
}

void Condition::signal()
{
    std::unique_lock<std::mutex> guard( m_mutex );
    m_signaled = true;
    m_conditiolwariable.notify_one();
}

void Condition::reset()
{
    std::unique_lock<std::mutex> guard( m_mutex );
    m_signaled = false;
}

}  // namespace optix
