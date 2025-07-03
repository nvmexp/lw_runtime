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
/// \brief Implementation for ThreadPool. Utility classes ConditionJob and ResumeJob.

#include <ThreadPool/Condition.h>
#include <ThreadPool/Job.h>
#include <ThreadPool/ThreadPool.h>

#include <prodlib/exceptions/Assert.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cstdio>
#include <thread>
#include <utility>

/// Enables additional debug output for the thread pool
// #define THREAD_POOL_VERBOSE

namespace optix {

/// Wrapper for Job that allows to wait for #execute() to finish.
///
/// The additional method #wait() waits for the signal sent at the end of #execute(). If the
/// wrapped jobs supports parallel calls to #execute() the signal is sent by the thread that
/// leaves #execute() last.
class ConditionJob : public Job
{
  public:
    ConditionJob( const std::shared_ptr<Job>& job )
        : m_job( job )
        , m_parallelIlwocations( 0 )
    {
    }
    ConditionJob( const ConditionJob& ) = delete;
    ConditionJob& operator=( const ConditionJob& ) = delete;
    virtual ~ConditionJob() {}

    float    getCpuLoad() const { return m_job->getCpuLoad(); }
    float    getGpuLoad() const { return m_job->getGpuLoad(); }
    Priority getPriority() const { return m_job->getPriority(); }
    void     preExelwte() { m_job->preExelwte(); }
    void     execute()
    {
        ++m_parallelIlwocations;
        m_job->execute();
        if( --m_parallelIlwocations == 0 )
            m_condition.signal();
    }
    bool isRemainingWorkSplittable() { return m_job->isRemainingWorkSplittable(); }
    void wait() { m_condition.wait(); }

  private:
    std::shared_ptr<Job>      m_job;
    Condition                 m_condition;
    std::atomic<unsigned int> m_parallelIlwocations;
};

/// A resume job is used to allocate the resources needed to resume a lwrrently suspended job via
/// the regular job queue. This is achieved by a special case inside the thread pool specific to
/// this class.
class ResumeJob : public Job
{
  public:
    ResumeJob( float cpuLoad, float gpuLoad, Priority priority )
        : m_cpuLoad( cpuLoad )
        , m_gpuLoad( gpuLoad )
        , m_priority( priority )
    {
    }
    ResumeJob( const ResumeJob& ) = delete;
    ResumeJob& operator=( const ResumeJob& ) = delete;
    virtual ~ResumeJob() {}

    float    getCpuLoad() const { return m_cpuLoad; }
    float    getGpuLoad() const { return m_gpuLoad; }
    Priority getPriority() const { return m_priority; }
    void     preExelwte() {}
    void     execute() { m_condition.signal(); }
    bool     isRemainingWorkSplittable() { return false; }
    void     wait() { m_condition.wait(); }

  private:
    Condition m_condition;
    float     m_cpuLoad;
    float     m_gpuLoad;
    Priority  m_priority;
};

/// For now, we do \em not permit jobs that do not create any CPU load at all (for jobs that do not
/// do much work, RPC calls), etc. So the number of threads \em is bound in terms of the number of
/// CPUs.
float ThreadPool::s_minCpuLoad = 1.0f;

/// A positive number might create problems if the system has no GPUs.
float ThreadPool::s_minGpuLoad = 0.0f;

ThreadPool::ThreadPool( float cpuLoadLimit, float gpuLoadLimit, size_t nrOfWorkerThreads )
    : m_cpuLoadLimit( cpuLoadLimit )
    , m_gpuLoadLimit( gpuLoadLimit )
    , m_lwrrentCpuLoad( 0.0f )
    , m_lwrrentGpuLoad( 0.0f )
    , m_shutdown( false )
    , m_numWorkerThreads( nrOfWorkerThreads )
{
    m_threadStateCounter[THREAD_STARTING]  = 0;
    m_threadStateCounter[THREAD_SLEEPING]  = 0;
    m_threadStateCounter[THREAD_IDLE]      = 0;
    m_threadStateCounter[THREAD_RUNNING]   = 0;
    m_threadStateCounter[THREAD_SUSPENDED] = 0;
    m_threadStateCounter[THREAD_SHUTDOWN]  = 0;
}

ThreadPool::~ThreadPool()
{
    std::lock_guard<std::mutex> guard( m_lock );
    m_shutdown = true;

    // Wait until job queue is empty.
    while( !m_jobQueue.empty() )
    {
        m_lock.unlock();
        std::this_thread::sleep_for( std::chrono::duration<double>( 0.01 ) );
        m_lock.lock();
    }

    // Wait for all worker threads to terminate.
    //
    // Do not hold m_lock because idle threads might call one last time into getNextJob() and
    // wait for m_lock. Accessing m_allThreads without holding m_lock should be fine since at
    // this time no new threads should be created anymore and m_allThreads should remain constant.
    //
    // If this approach is not sufficient one could use a more elaborate scheme were worker threads
    // are told to shutdown (without blocking) and wait in a loop similar as above for confirmations
    // from all threads that they left their main loop.
    m_lock.unlock();
    for( size_t i = 0; i < m_allThreads.size(); ++i )
        m_allThreads[i]->shutdown();
    m_lock.lock();

    RT_ASSERT_NOTHROW( m_jobQueue.empty(), "" );
    RT_ASSERT_NOTHROW( m_runningJobs.empty(), "" );
    RT_ASSERT_NOTHROW( m_suspendedJobs.empty(), "" );

    RT_ASSERT_NOTHROW( m_threadStateCounter[THREAD_STARTING] == 0, "" );
    RT_ASSERT_NOTHROW( m_threadStateCounter[THREAD_SLEEPING] == 0, "" );
    RT_ASSERT_NOTHROW( m_threadStateCounter[THREAD_IDLE] == 0, "" );
    RT_ASSERT_NOTHROW( m_threadStateCounter[THREAD_RUNNING] == 0, "" );
    RT_ASSERT_NOTHROW( m_threadStateCounter[THREAD_SUSPENDED] == 0, "" );
    RT_ASSERT_NOTHROW( m_threadStateCounter[THREAD_SHUTDOWN] == m_allThreads.size(), "" );

    for( size_t i = 0; i < m_allThreads.size(); ++i )
        delete m_allThreads[i];
    m_allThreads.clear();
    m_sleepingThreads.clear();

    RT_ASSERT_NOTHROW( m_threadStateCounter[THREAD_SHUTDOWN] == 0, "" );
}

void ThreadPool::createWorkerThreads()
{
    if( !m_allThreads.empty() )
        return;

    for( size_t i = 0; i < m_numWorkerThreads; ++i )
        createWorkerThread();

    RT_ASSERT( m_threadStateCounter[THREAD_SLEEPING] == m_numWorkerThreads );
}

bool ThreadPool::setCpuLoadLimit( float limit )
{
    if( limit < 1.0f )
        return false;

    std::lock_guard<std::mutex> guard( m_lock );
    m_cpuLoadLimit = limit;
    return true;
}

bool ThreadPool::setGpuLoadLimit( float limit )
{
    if( limit < 1.0f )
        return false;

    std::lock_guard<std::mutex> guard( m_lock );
    m_gpuLoadLimit = limit;
    return true;
}

void ThreadPool::submitJob( const std::shared_ptr<Job>& job )
{
#ifdef THREAD_POOL_DEBUG
    // Check that jobs are not submitted from suspended worker threads (to prevent misuse of the
    // suspend/resume feature).
    float         cpuLoad;
    float         gpuLoad;
    Priority      priority;
    unsigned long threadId;
    bool          suspendedWorkerThread = getLwrrentJobData( cpuLoad, gpuLoad, priority, threadId, true );
    RT_ASSERT( !suspendedWorkerThread );
#endif  // THREAD_POOL_DEBUG

    submitJobInternal( job, /*logAsynchronous*/ true );
}

void ThreadPool::submitJobAndWait( const std::shared_ptr<Job>& job )
{
#ifdef THREAD_POOL_DEBUG
    // Check that jobs are not submitted from suspended worker threads (to prevent misuse of the
    // suspend/resume feature).
    float         cpuLoad;
    float         gpuLoad;
    Priority      priority;
    unsigned long threadId;
    bool          suspendedWorkerThread = getLwrrentJobData( cpuLoad, gpuLoad, priority, threadId, true );
    RT_ASSERT( !suspendedWorkerThread );
#endif  // THREAD_POOL_DEBUG

    // Submit new job ...
    std::shared_ptr<ConditionJob> wrapped_job( std::make_shared<ConditionJob>( job ) );
    submitJobInternal( wrapped_job, /*logAsynchronous*/ false );

    // ... before the current job (if any) is suspended, such that child jobs have higher priority
    // than the jobs lwrrently in the queue.
    suspendLwrrentJob();
    wrapped_job->wait();
    resumeLwrrentJob();
}

bool ThreadPool::removeJob( const std::shared_ptr<Job>& job )
{
    std::lock_guard<std::mutex> guard( m_lock );

    JobQueue::iterator itMap    = m_jobQueue.begin();
    JobQueue::iterator itMapEnd = m_jobQueue.end();
    while( itMap != itMapEnd )
    {

        JobList::iterator itList    = itMap->second.begin();
        JobList::iterator itListEnd = itMap->second.end();
        while( itList != itListEnd )
        {

            if( *itList == job )
            {
                itMap->second.erase( itList );
                if( itMap->second.empty() )
                    m_jobQueue.erase( itMap );
                return true;
            }

            ++itList;
        }
        ++itMap;
    }

    return false;
}

void ThreadPool::suspendLwrrentJob()
{
    suspendLwrrentJobInternal( /*onlyForHigherPriority*/ false );
}

void ThreadPool::resumeLwrrentJob()
{
    resumeLwrrentJobInternal();
}

void ThreadPool::yield()
{
    bool success = suspendLwrrentJobInternal( /*onlyForHigherPriority*/ true );
    if( success )
        resumeLwrrentJobInternal();
}

std::shared_ptr<Job> ThreadPool::getNextJob( WorkerThread* thread )
{
    std::lock_guard<std::mutex> guard( m_lock );

    RT_ASSERT( thread->getState() == THREAD_IDLE );

    JobQueue::iterator itMap    = m_jobQueue.begin();
    JobQueue::iterator itMapEnd = m_jobQueue.end();
    JobList::iterator  itList;
    JobList::iterator  itListEnd;
#ifdef THREAD_POOL_VERBOSE
    size_t k = 0;
#endif  // THREAD_POOL_VERBOSE

    std::shared_ptr<Job> job;
    float                requested_cpuLoad;
    float                requested_gpuLoad;

    // find first job whose resource request fits the load limits
    while( itMap != itMapEnd )
    {

        itList    = itMap->second.begin();
        itListEnd = itMap->second.end();
        while( itList != itListEnd )
        {

            job               = *itList;
            requested_cpuLoad = job->getCpuLoad();
            requested_gpuLoad = job->getGpuLoad();
            adjustLoad( requested_cpuLoad, requested_gpuLoad );
            if( jobFitsLoadLimits( requested_cpuLoad, requested_gpuLoad ) )
                break;
#ifdef THREAD_POOL_VERBOSE
            char buffer[1024];
            snprintf( buffer, sizeof( buffer ) - 1,
                      "Delaying job %p, queue index %zu, CPU load %.1f / %.1f / %.1f, "
                      "GPU load %.1f / %.1f / %.1f, priority %d\n",
                      job.get(), k, requested_cpuLoad, m_lwrrentCpuLoad, m_cpuLoadLimit, requested_gpuLoad,
                      m_lwrrentGpuLoad, m_gpuLoadLimit, (int)job->getPriority() );
            ltemp << buffer;
#endif  // THREAD_POOL_VERBOSE
            ++itList;
#ifdef THREAD_POOL_VERBOSE
            ++k;
#endif  // THREAD_POOL_VERBOSE
            job = nullptr;
        }
        if( job )
            break;
        ++itMap;
    }

    if( itMap == itMapEnd )
    {
        RT_ASSERT( !job );
        thread->setState( THREAD_SLEEPING );
        std::pair<SleepingThreads::iterator, bool> result = m_sleepingThreads.insert( thread );
        RT_ASSERT( result.second );
        (void)result;
        return nullptr;
    }

#ifdef THREAD_POOL_VERBOSE
    char buffer[1024];
    snprintf( buffer, sizeof( buffer ) - 1,
              "Exelwting job %p, queue index %zu, CPU load %.1f/%.1f/%.1f, GPU load %.1f/%.1f/%.1f, priority %d\n",
              job.get(), k, requested_cpuLoad, m_lwrrentCpuLoad, m_cpuLoadLimit, requested_gpuLoad, m_lwrrentGpuLoad,
              m_gpuLoadLimit, (int)job->getPriority() );
    ltemp << buffer;
#endif  // THREAD_POOL_VERBOSE

    // notify job about upcoming execute() call (might affect isRemainingWorkSplittable() below)
    job->preExelwte();

    // remove job from queue if the job does no want more parallel calls
    bool dequeue_job = !job->isRemainingWorkSplittable();
    if( dequeue_job )
    {
        itMap->second.erase( itList );
        if( itMap->second.empty() )
            m_jobQueue.erase( itMap );
    }

    // adjust current load
    m_lwrrentCpuLoad += requested_cpuLoad;
    m_lwrrentGpuLoad += requested_gpuLoad;

    // wake up another worker thread for jobs that want more parallel calls (after removing this
    // thread from the set of sleeping threads)
    if( !dequeue_job )
        wakeUpWorkerThread();

    // map thread to the job
    unsigned long threadId = thread->getThreadId();
    RT_ASSERT( m_runningJobs.find( threadId ) == m_runningJobs.end() );
    m_runningJobs[threadId] = job;

    return job;
}

void ThreadPool::jobExelwtionFinished( WorkerThread* thread, const std::shared_ptr<Job>& job )
{
    std::lock_guard<std::mutex> guard( m_lock );

    RT_ASSERT( thread->getState() == THREAD_IDLE );

    float requested_cpuLoad = job->getCpuLoad();
    float requested_gpuLoad = job->getGpuLoad();
#ifdef THREAD_POOL_VERBOSE
    char buffer[1024];
    snprintf( buffer, sizeof( buffer ) - 1,
              "Finished job %p, queue index n/a, CPU load %.1f/%.1f/%.1f, GPU load %.1f/%.1f/%.1f, priority %d\n",
              job.get(), requested_cpuLoad, m_lwrrentCpuLoad, m_cpuLoadLimit, requested_gpuLoad, m_lwrrentGpuLoad,
              m_gpuLoadLimit, (int)job->getPriority() );
    ltemp << buffer;
#endif  // THREAD_POOL_VERBOSE

    // adjust current load except for resume jobs
    if( !dynamic_cast<ResumeJob*>( job.get() ) )
    {
        adjustLoad( requested_cpuLoad, requested_gpuLoad );
        m_lwrrentCpuLoad -= requested_cpuLoad;
        m_lwrrentGpuLoad -= requested_gpuLoad;
    }

    // unmap job from thread
    unsigned long    threadId = thread->getThreadId();
    JobMap::iterator it       = m_runningJobs.find( threadId );
    RT_ASSERT( it != m_runningJobs.end() );
    m_runningJobs.erase( it );
}

void ThreadPool::dumpLoad() const
{
    char buffer[1024];
    snprintf( buffer, sizeof( buffer ) - 1, "Current CPU load: %.1f/%.1f, GPU load: %.1f/%.1f\n", m_lwrrentCpuLoad,
              m_cpuLoadLimit, m_lwrrentGpuLoad, m_gpuLoadLimit );
    ltemp << buffer;
}

void ThreadPool::dumpThreadStateCounters() const
{
    unsigned int starting  = m_threadStateCounter[THREAD_STARTING];
    unsigned int idle      = m_threadStateCounter[THREAD_IDLE];
    unsigned int sleeping  = m_threadStateCounter[THREAD_SLEEPING];
    unsigned int running   = m_threadStateCounter[THREAD_RUNNING];
    unsigned int suspended = m_threadStateCounter[THREAD_SUSPENDED];
    unsigned int shutdown  = m_threadStateCounter[THREAD_SHUTDOWN];

    char buffer[1024];
    snprintf( buffer, sizeof( buffer ) - 1,
              "Worker thread states: %u starting, %u sleeping, %u idle, %u running, %u suspended, %u shutdown, %u "
              "total\n",
              starting, sleeping, idle, running, suspended, shutdown, starting + sleeping + idle + running + suspended + shutdown );
    ltemp << buffer;
}

void ThreadPool::submitJobInternal( const std::shared_ptr<Job>& job, bool logAsynchronous )
{
    std::lock_guard<std::mutex> guard( m_lock );

    createWorkerThreads();

    // Submitting new jobs while another thread ilwokes the destructor is an error.
    RT_ASSERT( !m_shutdown );
    if( m_shutdown )
        return;

    // Put top-level jobs at the end of the job queue, put child jobs and resume jobs at the
    // beginning of the queue.
    unsigned long threadId  = std::hash<std::thread::id>{}( std::this_thread::get_id() );
    bool          resumeJob = m_runningJobs.find( threadId ) != m_runningJobs.end();
    bool          child_job = m_suspendedJobs.find( threadId ) != m_suspendedJobs.end();

    Priority priority = job->getPriority();

    if( child_job || resumeJob )
        m_jobQueue[priority].push_front( job );
    else
        m_jobQueue[priority].push_back( job );

    // Check whether the job could be exelwted immediately.
    float requested_cpuLoad = job->getCpuLoad();
    float requested_gpuLoad = job->getGpuLoad();
    adjustLoad( requested_cpuLoad, requested_gpuLoad );
    if( !jobFitsLoadLimits( requested_cpuLoad, requested_gpuLoad ) )
    {
#ifdef THREAD_POOL_VERBOSE
        char buffer[1024];
        snprintf( buffer, sizeof( buffer ) - 1,
                  "Submitted job %p, CPU load %.1f/%.1f/%.1f, GPU load %.1f/%.1f/%.1f, priority %d, %s%sjob, exelwtion "
                  "delayed\n",
                  job.get(), requested_cpuLoad, m_lwrrentCpuLoad, m_cpuLoadLimit, requested_gpuLoad, m_lwrrentGpuLoad,
                  m_gpuLoadLimit, (int)priority, resumeJob ? "resume " : ( child_job ? "child " : "top-level " ),
                  resumeJob ? "" : ( logAsynchronous ? "asynchronous " : "synchronous " ) );
        ltemp << buffer;
#endif  // THREAD_POOL_VERBOSE
        return;
    }

// Wake up some worker thread to process some job (not necessarily the one just submitted).
#ifdef THREAD_POOL_VERBOSE
    char buffer[1024];
    snprintf( buffer, sizeof( buffer ) - 1,
              "Submitted job %p, CPU load %.1f/%.1f/%.1f, GPU load %.1f/%.1f/%.1f, priority %d, %s%sjob, waking up "
              "thread\n",
              job.get(), requested_cpuLoad, m_lwrrentCpuLoad, m_cpuLoadLimit, requested_gpuLoad, m_lwrrentGpuLoad,
              m_gpuLoadLimit, (int)priority, resumeJob ? "resume " : ( child_job ? "child " : "top-level " ),
              resumeJob ? "" : ( logAsynchronous ? "asynchronous " : "synchronous " ) );
    ltemp << buffer;
#endif  // THREAD_POOL_VERBOSE
    wakeUpWorkerThread();
}

bool ThreadPool::suspendLwrrentJobInternal( bool onlyForHigherPriority )
{
    std::lock_guard<std::mutex> guard( m_lock );

    // check whether we actually suspend if the flag is set, part 1
    if( onlyForHigherPriority && m_jobQueue.empty() )
        return false;

    // get job and thread properties (and check whether this a running worker thread at all)
    float         cpuLoad;
    float         gpuLoad;
    Priority      priority;
    unsigned long threadId;
    bool          runningWorkerThread = getLwrrentJobDataLocked( cpuLoad, gpuLoad, priority, threadId, false );
    if( !runningWorkerThread )
    {
#ifdef THREAD_POOL_DEBUG
        // detect nested suspend calls
        bool suspendedWorkerThread = getLwrrentJobDataLocked( cpuLoad, gpuLoad, priority, threadId, true );
        RT_ASSERT( !suspendedWorkerThread );
#endif  // THREAD_POOL_DEBUG
        return false;
    }

    // check whether we actually suspend if the flag is set, part 2
    if( onlyForHigherPriority )
    {
        JobQueue::iterator it = m_jobQueue.begin();
        if( it->first >= priority )
            return false;
    }

    --m_threadStateCounter[THREAD_RUNNING];
    ++m_threadStateCounter[THREAD_SUSPENDED];

    // adjust current load
    m_lwrrentCpuLoad -= cpuLoad;
    m_lwrrentGpuLoad -= gpuLoad;

    // move job from map of running threads to map of suspended threads
    JobMap::iterator it = m_runningJobs.find( threadId );
    RT_ASSERT( it != m_runningJobs.end() );
    RT_ASSERT( m_suspendedJobs.find( threadId ) == m_suspendedJobs.end() );
    m_suspendedJobs[threadId] = it->second;
    m_runningJobs.erase( it );

    // wake up some worker thread if there are jobs in the queue
    if( !m_jobQueue.empty() )
        wakeUpWorkerThread();

    return true;
}

void ThreadPool::resumeLwrrentJobInternal()
{
    // get job and thread properties (and check whether this a suspended worker thread at all)
    float         cpuLoad;
    float         gpuLoad;
    Priority      priority;
    unsigned long threadId;
    bool          suspendedWorkerThread = getLwrrentJobData( cpuLoad, gpuLoad, priority, threadId, true );
    if( !suspendedWorkerThread )
    {
#ifdef THREAD_POOL_DEBUG
        // detect nested resume calls
        bool runningWorkerThread = getLwrrentJobData( cpuLoad, gpuLoad, priority, threadId, false );
        RT_ASSERT( !runningWorkerThread );
#endif  // THREAD_POOL_DEBUG
        return;
    }

    // wait for the required resources
    std::shared_ptr<ResumeJob> resumeJob( std::make_shared<ResumeJob>( cpuLoad, gpuLoad, priority ) );
    submitJobInternal( resumeJob, /*logAsynchronous*/ false );
    resumeJob->wait();

    // No need to adjust the current load: when resumeJob is finished the current load is \em not
    // decreased (the whole purpose of resumeJob is to grab the required resources for the job to
    // be resumed via the regular job queue).

    --m_threadStateCounter[THREAD_SUSPENDED];
    ++m_threadStateCounter[THREAD_RUNNING];

    // move job from map of suspended threads to map of running threads
    std::lock_guard<std::mutex> guard( m_lock );
    JobMap::iterator            it = m_suspendedJobs.find( threadId );
    RT_ASSERT( it != m_suspendedJobs.end() );
    RT_ASSERT( m_runningJobs.find( threadId ) == m_runningJobs.end() );
    m_runningJobs[threadId] = it->second;
    m_suspendedJobs.erase( it );
}

void ThreadPool::createWorkerThread()
{
    // Attempts to create a worker thread while another thread ilwokes the destructor is an error.
    // (The destructor uses m_allThreads without lock.)
    RT_ASSERT( !m_shutdown );
    if( m_shutdown )
        return;

    // The caller is supposed to hold m_lock.
    WorkerThread* thread = new WorkerThread( this );
    thread->start();
    m_allThreads.push_back( thread );
    RT_ASSERT( thread->getState() == THREAD_SLEEPING );
    m_sleepingThreads.insert( thread );
}

void ThreadPool::wakeUpWorkerThread()
{
    // The caller is supposed to hold m_lock.
    bool m_sleepingThreadsWasEmpty = m_sleepingThreads.empty();
    if( m_sleepingThreadsWasEmpty )
        createWorkerThread();
    WorkerThread* thread = *m_sleepingThreads.begin();
    RT_ASSERT( thread->getState() == THREAD_SLEEPING );
    thread->wakeUp();

    // remove thread from the set of sleeping threads
    size_t result = m_sleepingThreads.erase( thread );
    RT_ASSERT( result == 1 );
    (void)result;
}

bool ThreadPool::jobFitsLoadLimits( float cpuLoad, float gpuLoad ) const
{
    // The caller is supposed to hold m_lock.

    // Clip requested resources against limits to avoid delaying forever jobs with unsatisfiable
    // requirements. Such jobs will only be exelwted if the current load is 0.0, ignoring the limit.
    if( cpuLoad > m_cpuLoadLimit )
        cpuLoad = m_cpuLoadLimit;
    if( gpuLoad > m_gpuLoadLimit )
        gpuLoad = m_gpuLoadLimit;

    return m_lwrrentCpuLoad + cpuLoad <= m_cpuLoadLimit * 1.001 && m_lwrrentGpuLoad + gpuLoad <= m_gpuLoadLimit * 1.001;
}

bool ThreadPool::getLwrrentJobData( float& cpuLoad, float& gpuLoad, Priority& priority, unsigned long& threadId, bool suspended ) const
{
    std::lock_guard<std::mutex> guard( m_lock );
    return getLwrrentJobDataLocked( cpuLoad, gpuLoad, priority, threadId, suspended );
}

bool ThreadPool::getLwrrentJobDataLocked( float& cpuLoad, float& gpuLoad, Priority& priority, unsigned long& threadId, bool suspended ) const
{
    // The caller is supposed to hold m_lock.
    threadId = std::hash<std::thread::id>{}( std::this_thread::get_id() );
    JobMap::const_iterator it;
    if( suspended )
    {
        it = m_suspendedJobs.find( threadId );
        if( it == m_suspendedJobs.end() )
        {
            cpuLoad  = 0.0;
            gpuLoad  = 0.0;
            priority = 0;
            return false;
        }
    }
    else
    {
        it = m_runningJobs.find( threadId );
        if( it == m_runningJobs.end() )
        {
            cpuLoad  = 0.0;
            gpuLoad  = 0.0;
            priority = 0;
            return false;
        }
    }

    cpuLoad = it->second->getCpuLoad();
    gpuLoad = it->second->getGpuLoad();
    adjustLoad( cpuLoad, gpuLoad );
    priority = it->second->getPriority();
    return true;
}

void ThreadPool::adjustLoad( float& cpuLoad, float& gpuLoad )
{
    cpuLoad = cpuLoad == 0.0f ? 0.0f : std::max( cpuLoad, s_minCpuLoad );
    gpuLoad = gpuLoad == 0.0f ? 0.0f : std::max( gpuLoad, s_minGpuLoad );
    if( cpuLoad == 0.0f && gpuLoad == 0.0f )
        cpuLoad = s_minCpuLoad;
}

}  // namespace optix
