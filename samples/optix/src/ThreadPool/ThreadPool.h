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
/// \brief Header for ThreadPool.

#pragma once

#include <atomic>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

#include <ThreadPool/WorkerThread.h>  // for ThreadState

#ifndef NDEBUG
/// Enables additional assertions for the thread pool
#define THREAD_POOL_DEBUG
#endif

namespace optix {

typedef signed char Priority;

class Job;

/// The thread pool can be used to execute work synchronously or asynchronously in other threads.
///
/// Work is represented by instances of the Job class. The thread pool manages a queue of jobs to
/// be exelwted in a FIFO order. It also manages a set of worker threads which continuously pick
/// jobs from the job queue and execute them, or sleep if there is lwrrently no job to be exelwted.
///
/// The maximum amount of parallelism is controlled via load limits for the thread pool. Each job
/// announces the load it causes per worker thread (typically 1.0). The thread pool exelwtes a job
/// only if the addition of its load does not exceed the configured load limit. Loads and load
/// limits exist separately for CPU and GPU, even though the GPU load and GPU load limit is hardly
/// used. This mechanism allows indirectly to control the number of worker threads.
///
/// The thread pool supports jobs that accept multiple worker threads (so-called fragmented jobs).
/// Worker threads will be assigned to the first job in the queue as long as this job indicates that
/// it accepts more worker threads via #Job::isRemainingWorkSplittable() (and the load limit is
/// not yet reached). If a job does not accept more worker threads, it is removed from the queue and
/// the next job is considered. In other words, the thread pool attempts to minimize the runtime of
/// jobs by assigning as many worker threads as possible, instead of running as many jobs as
/// possible in parallel.
///
/// The thread pool also supports nested jobs, i.e., a parent job submitting new jobs, so-called
/// child jobs. This requires some care if the parent job waits for the completion of the child
/// jobs.
///
/// Last but not least the thread pool supports job priorities which can be used to influence the
/// position of the job in the queue.
class ThreadPool
{
  public:
    /// Constructor
    ///
    /// Creates the given number of worker threads and starts them.
    ///
    /// \param cpuLoadLimit        The initial limit for the CPU load.
    /// \param gpuLoadLimit        The initial limit for the GPU load.
    /// \param nrOfWorkerThreads   The number of worker threads to create upfront. Note that the
    ///                            thread pool might start additional worker threads as needed.
    ThreadPool( float cpuLoadLimit = 1.0, float gpuLoadLimit = 1.0, size_t nrOfWorkerThreads = 1 );

    /// The thread pool is non-copyable.
    ThreadPool( const ThreadPool& ) = delete;
    ThreadPool& operator=( const ThreadPool& ) = delete;

    /// Destructor
    ///
    /// Shuts down the worker threads and destroys them.
    ~ThreadPool();

    /// \name Load limits
    //@{

    /// Sets the CPU load limit.
    ///
    /// If it is reduced it might take some time until the current CPU load obeys the limit.
    ///
    /// Returns \c true in case of success (iff \c limit is greater to or equal to 1.0).
    bool setCpuLoadLimit( float limit );

    /// Sets the GPU load limit.
    ///
    /// If it is reduced it might take some time until the current GPU load obeys the limit.
    ///
    /// Returns \c true in case of success (iff \c limit is greater to or equal to 1.0).
    bool setGpuLoadLimit( float limit );

    /// Returns the CPU load limit.
    float getCpuLoad_limit()
    {
        std::lock_guard<std::mutex> guard( m_lock );
        return m_cpuLoadLimit;
    }

    /// Returns the GPU load limit.
    float getGpuLoad_limit()
    {
        std::lock_guard<std::mutex> guard( m_lock );
        return m_gpuLoadLimit;
    }

    /// Set the number of worker threads.  This should be done before createWorkerThreads is called.
    void setNumWorkerThreads( int n )
    {
        std::lock_guard<std::mutex> guard( m_lock );
        m_numWorkerThreads = static_cast<size_t>( n );
    }

    //@}
    /// \name Jobs
    //@{

    /// Submits a job for asynchronous exelwtion.
    ///
    /// The job is exelwted asynchronously and the method returns immediately. Jobs may not be
    /// submitted from suspended worker threads.
    ///
    /// If the submitted job is a child job, i.e., the job is submitted from a job lwrrently being
    /// exelwted (the parent job), and the parent job later waits for the completion of the child
    /// job, special care needs to taken to avoid dead locks, see #suspendLwrrentJob() and
    /// #resumeLwrrentJob() for details. If the parent job does nothing else except waiting for
    /// completion of the child job, consider using #submitJobAndWait() which is less error-prone.
    ///
    /// \see #submitJobAndWait() for synchronous exelwtion
    void submitJob( const std::shared_ptr<Job>& job );

    /// Submits a job for synchronous exelwtion.
    ///
    /// The job is exelwted synchronously, i.e., the method blocks until the job has been exelwted.
    /// Jobs may not be submitted from suspended worker threads.
    ///
    /// \see #submit() for asynchronous exelwtion
    void submitJobAndWait( const std::shared_ptr<Job>& job );

    /// Removes a job from the job queue and indicates success/failure.
    ///
    /// \note Removing an already submitted job should not be done imprudently. This method is
    ///       expensive in the sense that it needs to search the entire job queue to find the job.
    ///       In addition, nothing is known about the exelwtion status (see return value), and
    ///       lwrrently running jobs are not interrupted.
    ///
    /// \return Indicates whether the job was found in the queue (and therefore, removed from the
    ///         queue). The return value \true does \em not indicate that the job has not already
    ///         been started. The return value \false does \em not indicate that the job has
    ///         already been finished.
    bool removeJob( const std::shared_ptr<Job>& job );

    /// Notifies the thread pool that this worker thread suspends job exelwtion (because it is going
    /// to wait for some event).
    ///
    /// The thread pool does not do anything except that it decreases the current load values
    /// accordingly. The method returns immediately if not being called from a worker thread.
    ///
    /// Usage of this method is mandatory if a child job is exelwted asynchronously from within
    /// a parent job, and the parent job waits for the child job to complete. Usage of this method
    /// is strongly recommended (but not mandatory) if the parent jobs waits for some event
    /// unrelated to child jobs. Usage of this method is not necessary for synchronous exelwtion
    /// of child jobs.
    ///
    /// Failure to use this method when mandatory might lead to dead locks. Failure to use this
    /// method when recommended might lead to reduced performance.
    ///
    /// Example:
    /// \code
    /// Condition condition;
    /// ...
    /// thread_pool->suspendLwrrentJob();
    /// condition.wait();
    /// thread_pool->resumeLwrrentJob();
    /// \endcode
    ///
    /// This method needs to be used in conjunction with #resumeLwrrentJob().
    void suspendLwrrentJob();

    /// Notifies the thread pool that this worker thread resumes job exelwtion (because it waited
    /// for some event that now happened).
    ///
    /// The thread pool does not do anything except that it increases the current load values
    /// accordingly. This method blocks if the current load values and limits do not permit instant
    /// resuming of the job. The method returns immediately if not being called from a worker
    /// thread.
    ///
    /// \see #suspendLwrrentJob() for further details
    void resumeLwrrentJob();

    /// Notifies the thread pool that this worker thread is willing to give up its resources for a
    /// while in favor of other jobs.
    ///
    /// Yielding is similar to calling #suspendLwrrentJob() followed by #resumeLwrrentJob(), but
    /// it takes job priorities into account and is more efficient if there is no job of higher
    /// priority in the job queue.
    void yield();

    //@}
    /// \name Methods to be used by worker threads only
    //@{

    /// Returns the next job to be exelwted by \p thread.
    ///
    /// Returns \c NULL if there is no job to be exelwted, or no job whose load requirements fit the
    /// gap between load limits and current load values.
    ///
    /// Otherwise, the job is returned and removed from the job queue, the current load values are
    /// increased according to the job's requirements, and the worker thread is removed from the set
    /// of sleeping worker threads.
    std::shared_ptr<Job> getNextJob( WorkerThread* thread );

    /// Notifies the thread pool that exelwtion of a job has finished.
    ///
    /// The current load values are decreased according to the job's requirements, and the worker
    /// thread is added again to the set of sleeping worker threads.
    void jobExelwtionFinished( WorkerThread* thread, const std::shared_ptr<Job>& job );

    /// Increases the thread state counter for \p state.
    void increaseThreadStateCounter( ThreadState state ) { ++m_threadStateCounter[state]; }

    /// Decreases the thread state counter for \p state.
    void decreaseThreadStateCounter( ThreadState state ) { --m_threadStateCounter[state]; }

    //@}
    /// \name Methods to be used for debugging only.
    //@{

    /// Returns the number of worker threads.
    ///
    /// \note This value is for debugging purposes only. It is not meaningful without holding
    ///       the corresponding lock. Do not base any scheduling decisions on this value.
    size_t getNumberOfWorkerThreads() const
    {
        std::lock_guard<std::mutex> guard( m_lock );
        return m_allThreads.size();
    }

    /// Returns a particular thread state counter.
    ///
    /// \note These values are for debugging purposes only. Modifications are atomic, but not
    ///       locked. Temporarily the values do \em not add up to the correct value. Do not base
    ///       any scheduling decisions on these values.
    unsigned int get_threadStateCounter( ThreadState state ) const { return m_threadStateCounter[state]; }

    /// Returns the current CPU load.
    ///
    /// \note This value is for debugging purposes only. It is not meaningful without holding
    ///       the corresponding lock. Do not base any scheduling decisions on this value.
    float getLwrrentCpuLoad() const
    {
        std::lock_guard<std::mutex> guard( m_lock );
        return m_lwrrentCpuLoad;
    }

    /// Returns the current GPU load.
    ///
    /// \note This value is for debugging purposes only. It is not meaningful without holding
    ///       the corresponding lock. Do not base any scheduling decisions on this value.
    float getLwrrentGpuLoad() const
    {
        std::lock_guard<std::mutex> guard( m_lock );
        return m_lwrrentGpuLoad;
    }

    /// Dumps the current CPU/GPU load/limits to ltemp.
    void dumpLoad() const;

    /// Dumps the thread state counters to ltemp.
    void dumpThreadStateCounters() const;

    //@}

  private:
    /// Submits a job, i.e., puts it into the job queue.
    ///
    /// The job is exelwted asynchronously and the method returns immediately. Used by #submitJob()
    /// and #submitJobAndWait() to do the actual work.
    ///
    /// \param job                The job to be exelwted.
    /// \param logAsynchronous    Indicates whether the jobs is exelwted synchronously or
    ///                           asynchronously. This value is passed for log output only!
    void submitJobInternal( const std::shared_ptr<Job>& job, bool logAsynchronous );

    /// Notifies the thread pool that this worker thread suspends job exelwtion (because it is going
    /// to wait for some event).
    ///
    /// See #suspendLwrrentJob() for details. The additional flag and return value is used by
    /// #yield().
    ///
    /// \param onlyForHigherPriority
    ///                     Indicates whether exelwtion should be suspended only if there is job of
    ///                     higher priority in the queue. #suspendLwrrentJob() calls this method
    ///                     passing \c false; #yield() calls this method passing \c true.
    /// \return             Indicates whether the job exelwtion was actually suspended (might return
    ///                     \c false if \p onlyForHigherPriority was set and there was no such
    ///                     job, or if job exelwtion for this thread was already suspended).
    bool suspendLwrrentJobInternal( bool onlyForHigherPriority );

    /// Notifies the thread pool that this worker thread resumes job exelwtion (because it waited
    /// for some event that now happened).
    ///
    /// See #resumeLwrrentJob() for details.
    void resumeLwrrentJobInternal();

    /// Creates a pool of worker threads lazily.
    ///
    /// The callers needs to hold m_lock.
    void createWorkerThreads();

    /// Creates a new worker thread and adds it to m_allThreads and m_sleepingThreads.
    ///
    /// The callers needs to hold m_lock.
    void createWorkerThread();

    /// Wakes up a sleeping worker thread.
    ///
    /// If there are no sleeping worker threads, a new thread will be created.
    ///
    /// The callers needs to hold m_lock.
    void wakeUpWorkerThread();

    /// Indicates whether a job with given CPU/GPU loads can be exelwted given the current CPU/GPU
    /// load and the CPU/GPU load limits.
    ///
    /// The callers needs to hold m_lock.
    bool jobFitsLoadLimits( float cpuLoad, float gpuLoad ) const;

    /// Returns some job data for the job assigned to a running or suspended worker thread.
    ///
    /// Acquires m_lock and calls #getLwrrentJobDataLocked().
    bool getLwrrentJobData( float& cpuLoad, float& gpuLoad, Priority& priority, unsigned long& threadId, bool suspended ) const;

    /// Returns some job data for the job assigned to a running or suspended worker thread.
    ///
    /// The callers needs to hold m_lock.
    ///
    /// \param[out] cpuLoad   The CPU load of the job, or 0.0 if the calling thread is not a
    ///                       running/suspended worker thread.
    /// \param[out] gpuLoad   The GPU load of the job, or 0.0 if the calling thread is not a
    ///                       running/suspended worker thread.
    /// \param[out] priority  The priority of the job.
    /// \param[out] threadId  The thread ID.
    /// \param suspended      Indicates whether thread is supposed to be running or suspended.
    /// \return               \true if the calling thread is a running/suspended worker thread,
    ///                       and \false otherwise
    bool getLwrrentJobDataLocked( float& cpuLoad, float& gpuLoad, Priority& priority, unsigned long& threadId, bool suspended ) const;

    /// Clips the values against #s_minCpuLoad and #s_minGpuLoad.
    ///
    /// This is done to protect against malicious job implementations that return negative values
    /// which would cause a huge number of threads to be spawned.
    static void adjustLoad( float& cpuLoad, float& gpuLoad );

    /// The configured CPU load limit.
    float m_cpuLoadLimit;
    /// The configured GPU load limit.
    float m_gpuLoadLimit;
    /// The current CPU load. Protected by m_lock.
    float m_lwrrentCpuLoad;
    /// The current GPU load. Protected by m_lock.
    float m_lwrrentGpuLoad;
    /// The smallest CPU load a job can cause.
    static float s_minCpuLoad;
    /// The smallest GPU load a job can cause.
    static float s_minGpuLoad;

    /// The type of the vector of all worker threads.
    typedef std::vector<WorkerThread*> AllThreads;
    /// The vector of all worker threads. Protected by m_lock.
    AllThreads m_allThreads;

    /// The type of the set of sleeping worker threads.
    typedef std::set<WorkerThread*> SleepingThreads;
    /// The set of sleeping worker threads. Protected by m_lock.
    SleepingThreads m_sleepingThreads;

    /// The type of the job list. One job list is used for each priority.
    typedef std::list<std::shared_ptr<Job>> JobList;
    /// The type of the job queue. Maps priorities to job lists.
    typedef std::map<int, JobList> JobQueue;
    /// The job queue. Protected by m_lock.
    ///
    /// Actually, this is a map of lists instead of a queue such that we can efficiently insert and
    /// remove jobs at the required position (insertion at the front and back per priority, removal
    /// at any position with given iterator).
    ///
    /// Ilwariant: there is no empty list as map element (otherwise we cannot efficiently check
    /// whether the job queue is empty).
    JobQueue m_jobQueue;

    /// The type of the maps below.
    typedef std::map<unsigned long, std::shared_ptr<Job>> JobMap;
    /// The map that holds for each running worker thread the job that it is exelwting.
    /// Protected by m_lock.
    JobMap m_runningJobs;
    /// The map that holds for each suspended worker thread the job that it is exelwting.
    /// Protected by m_lock.
    JobMap m_suspendedJobs;

    /// The lock that protects the current loads, the vector of all worker threads, the set of
    /// sleeping worker threads, the job queue, and the maps for running and suspended jobs.
    mutable std::mutex m_lock;

    /// The thread state counters.
    ///
    /// These values are updated by the worker threads themselves, not by the thread pool.
    ///
    /// \note These values are for debugging purposes only. Modifications are atomic, but not
    ///       locked. Temporarily the values do \em not add up to the correct value. Do not base
    ///       any scheduling decisions on these values.
    std::atomic<unsigned int> m_threadStateCounter[N_THREAD_STATES];

    /// Used by the destructor to block submitting of new jobs.
    bool m_shutdown;

    size_t m_numWorkerThreads;
};

}  // namespace optix
