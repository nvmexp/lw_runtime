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
/// \brief Header for WorkerThread.

#pragma once

#include <thread>

#include "ThreadPool/Condition.h"

namespace optix {

class ThreadPool;

/// The various states for worker threads.
///
/// The transition diagram is as follows:
///
/// STARTING
///     |
///     V
/// SLEEPING -> IDLE -> RUNNING -> SUSPENDED
///     |    <-      <-         <-
///     V
/// SHUTDOWN
///
/// See also the note for WorkerThread::m_state and THREAD_SUSPENDED.
enum ThreadState
{
    THREAD_STARTING,
    THREAD_SLEEPING,
    THREAD_IDLE,
    THREAD_RUNNING,
    THREAD_SUSPENDED,
    THREAD_SHUTDOWN,
    N_THREAD_STATES
};

/// A worker thread used by the thread pool.
///
/// Worker threads are created on-demand by the thread pool.
class WorkerThread
{
  public:
    /// Constructor.
    ///
    /// Sets the thread state to THREAD_STARTING.
    WorkerThread( ThreadPool* thread_pool );

    /// Worker threads are non-copyable.
    WorkerThread( const WorkerThread& ) = delete;
    WorkerThread& operator=( const WorkerThread& ) = delete;

    /// Destructor.
    ///
    /// Expects the thread state to be THREAD_SHUTDOWN.
    virtual ~WorkerThread();

    /// Starts the thread.
    ///
    /// Blocks until the run() method is actually exelwted and sets the state from THREAD_STARTING
    /// to THREAD_SLEEPING.
    void start();

    /// Shuts the thread down.
    ///
    /// Blocks until the run() methods ends and set the state to THREAD_SHUTDOWN.
    void shutdown();

    /// Wakes the tread up from sleeping state.
    ///
    /// Signals the condition variable used by run() between processJobs() calls.
    void wakeUp();

    /// Returns the thread ID or 0 if the thread is not running.
    unsigned long getThreadId() const { return m_threadId; }

    /// Sets the thread state.
    ///
    /// Takes care of decrementing the counter for the old state and incrementing the counter for
    /// the new state.
    void setState( ThreadState state );

    /// Returns the threat state.
    ///
    /// Note that suspended threads still return THREAD_RUNNING instead of THREAD_SUSPENDED here
    /// because the pointer to the thread is not available during suspend/resume. This is just a
    /// cosmetic annoyance but has no bad effects. The thread state counters in the thread pool
    /// are updated correctly, though.
    ThreadState getState() const { return m_state; }

  private:
    /// The main method of the worker thread.
    ///
    /// Calls processJobs() in a loop until shutdown is initiated. Waits for the condition variable
    /// after each invocation.
    void run();

    /// Processes jobs (if possible).
    ///
    /// Calls processJob() in a loop as long as this method is successful (or shutdown was
    /// initiated).
    void processJobs();

    /// Processes a job (if possible).
    ///
    /// Attempts to get a job that can be exelwted from the work queue. If successful, exelwtes it
    /// and returns \c true. Otherwise (no job, or does not fit load limits) returns \c false.
    bool processJob();

    /// The underlying thread.
    ///
    /// Ilwokes run().
    std::thread* m_thread;

    /// The thread pool this worker thread belongs to.
    ThreadPool* m_threadPool;

    /// The state of the worker thread.
    ///
    /// Note that suspended threads still use THREAD_RUNNING instead of THREAD_SUSPENDED here
    /// because the pointer to the thread is not available during suspend/resume. This is just a
    /// cosmetic annoyance but has no bad effects. The thread state counters in the thread pool
    /// are updated correctly, though.
    ThreadState m_state;

    /// If set, the thread will go into state THREAD_SHUTDOWN instead of THREAD_IDLE when woken
    /// up from state THREAD_SLEEPING.
    bool m_shutdown;

    /// Used to wake up the thread when in state THREAD_SLEEPING.
    Condition m_condition;

    /// Used to synchronize thread startup between start() and run().
    Condition m_startCondition;

    /// The thread ID.
    unsigned long m_threadId;
};

}  // namespace optix
