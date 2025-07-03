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

/// \file
/// \brief Implementation for WorkerThread.

#include <ThreadPool/Job.h>
#include <ThreadPool/ThreadPool.h>
#include <ThreadPool/WorkerThread.h>

#include <prodlib/exceptions/Assert.h>

namespace optix {

WorkerThread::WorkerThread( ThreadPool* thread_pool )
    : m_threadPool( thread_pool )
    , m_state( THREAD_STARTING )
    , m_shutdown( false )
    , m_threadId( 0 )
{
    m_threadPool->increaseThreadStateCounter( m_state );
}

WorkerThread::~WorkerThread()
{
    RT_ASSERT_NOTHROW( m_state == THREAD_SHUTDOWN, "" );
    m_threadPool->decreaseThreadStateCounter( m_state );
}

void WorkerThread::start()
{
    RT_ASSERT( m_state == THREAD_STARTING );
    m_thread = new std::thread( &WorkerThread::run, this );
    m_startCondition.wait();
    RT_ASSERT( m_state == THREAD_SLEEPING );
}

void WorkerThread::shutdown()
{
    RT_ASSERT( m_state == THREAD_SLEEPING || m_state == THREAD_IDLE || m_state == THREAD_RUNNING || m_state == THREAD_SUSPENDED );
    m_shutdown = true;
    m_condition.signal();
    m_thread->join();
    delete m_thread;
}

void WorkerThread::wakeUp()
{
    RT_ASSERT( m_state == THREAD_SLEEPING );
    m_condition.signal();
}

void WorkerThread::setState( ThreadState state )
{
    m_threadPool->decreaseThreadStateCounter( m_state );
    m_state = state;
    m_threadPool->increaseThreadStateCounter( m_state );
}

void WorkerThread::run()
{
    m_threadId = std::hash<std::thread::id>{}( std::this_thread::get_id() );

    RT_ASSERT( m_state == THREAD_STARTING );
    setState( THREAD_SLEEPING );
    m_startCondition.signal();

    m_condition.wait();
    while( !m_shutdown )
    {
        RT_ASSERT( m_state == THREAD_SLEEPING );
        setState( THREAD_IDLE );
        processJobs();
        RT_ASSERT( m_state == THREAD_SLEEPING );
        m_condition.wait();
    }

    RT_ASSERT( m_state == THREAD_SLEEPING );
    setState( THREAD_SHUTDOWN );

    m_threadId = 0;
}

void WorkerThread::processJobs()
{
    bool jobDone;
    do
    {
        RT_ASSERT( m_state == THREAD_IDLE );
        jobDone = processJob();
        RT_ASSERT( m_state == THREAD_IDLE || m_state == THREAD_SLEEPING );
    } while( jobDone );
    RT_ASSERT( m_state == THREAD_SLEEPING );
}

bool WorkerThread::processJob()
{
    // printf( "Entering processJob()\n");
    std::shared_ptr<Job> job( m_threadPool->getNextJob( this ) );
    if( !job )
    {
        // printf( "Leaving processJob() (got no job)\n");
        RT_ASSERT( m_state == THREAD_SLEEPING );
        return false;
    }

    RT_ASSERT( m_state == THREAD_IDLE );
    setState( THREAD_RUNNING );
    // printf( "Exelwting job %p ... \n", job);
    job->execute();
    // printf( "Exelwting job %p done\n", job);
    RT_ASSERT( m_state == THREAD_RUNNING );
    setState( THREAD_IDLE );

    m_threadPool->jobExelwtionFinished( this, job );
    // printf( "Leaving processJob() (exelwted a job)\n");
    return true;
}

}  // namespace optix
