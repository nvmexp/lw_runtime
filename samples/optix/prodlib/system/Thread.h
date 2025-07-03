// Copyright LWPU Corporation 2010
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

#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/ValueOutOfRange.h>
#include <prodlib/system/System.h>

#ifdef _WIN32
#define THREAD_API_WIN
#endif

#ifdef THREAD_API_WIN
#define WIN32_LEAN_AND_MEAN
#include <functional>
#include <windows.h>
#else  // POSIX
#include <functional>
#include <pthread.h>
#include <semaphore.h>
#endif

#include <algorithm>
#include <stdexcept>
#include <vector>


namespace prodlib {

// ----------------------------------------------------------------------------
// ThreadId
// ----------------------------------------------------------------------------

class ThreadId
{
  public:
    // Constructor initializes with the thread id of the calling thread.
    ThreadId()
    {
#ifndef THREAD_API_WIN
        m_thread_id = pthread_self();
#else
        m_thread_id = GetLwrrentThreadId();
#endif
    }

    bool operator==( const ThreadId& other ) const
    {
#ifndef THREAD_API_WIN
        return pthread_equal( m_thread_id, other.m_thread_id ) != 0;
#else
        return m_thread_id == other.m_thread_id;
#endif
    }

    bool operator!=( const ThreadId& other ) const { return !operator==( other ); }

    bool operator<( const ThreadId& other ) const { return this->as_uint() < other.as_uint(); }


    // Return the numerical representation of the thread id.
    // Note that POSIX does not require pthread_t to be an arithmetic type.
    unsigned long long as_uint() const { return (unsigned long long)m_thread_id; }

  private:
#ifndef THREAD_API_WIN
    pthread_t m_thread_id;
#else
    DWORD               m_thread_id;
#endif
};

// ----------------------------------------------------------------------------
//  Thread
// ----------------------------------------------------------------------------

class Thread
{
  public:
    Thread();
    ~Thread();

    // create() starts the code functionPtr points to as a new thread.
    // argumentPtr must point to a block of memory that stores input and output values.
    // The thread function must know how to interpret data in this block.
    //
    // create() returns true if the thread could be successfully started.
    // It return false if the OS signals an error or if a thread has already been created.
    // It is neccessary to wait for a thread to exit using wait() before a new thread can
    // be created.
    bool create( void ( *functionPtr )( void* ), void* argumentPtr, bool createSuspended = false );

    // wait() performs a blocking wait until a running thread has ended.
    //
    // wait() returns true once a thread has ended. It returns false if the OS signals an
    // error or if the thread was already not running anymore, i.e., if wait() had been
    // called before but no new thread has been started.
    bool wait();

    // isRunning() returns true if a thread has been started using create() and wait()
    // has not been called. It returns false after wait() has been called, or if no thread
    // has been created.
    bool isRunning() { return m_running; }

    bool resume();
    bool suspend();

    // The Mutex class guards access to a critical area.
    // Use lock() before entering a critical area and unlock() after leaving.
    // If one thread has successfully performed lock(), others wait on this call until
    // unlock() is called by a thread leaving the critical area.
    //
    // lock() and unlock() return false if the OS signals an error, otherwise true.
    //
    // In case it is desired that a mutex is automatically unlocked when a thread leaves
    // the critical area (e.g., because of an exception) use the Lock class.
    class Mutex
    {
      public:
        Mutex();
        ~Mutex();

        bool lock();
        bool unlock();

      private:
#ifdef THREAD_API_WIN
        CRITICAL_SECTION m_critical_section;
#else  // POSIX
        pthread_mutex_t m_mutex;
#endif
    };

    // Lock is a container that automatically locks the provided mutex object.
    // The mutex is automatically unlocked upon destruction of the Lock object.
    class Lock
    {
      public:
        Lock( Mutex& mutex )
            : m_mutex( mutex )
        {
            m_mutex.lock();
        }
        ~Lock() { m_mutex.unlock(); }

      private:
        Mutex& m_mutex;
    };

    // A container that encapsulates a semaphore object
    //
    // A semaphore is an object which counts and tracks units of a specific resource
    // After setting an initialValue (the initial available units of that specific resource) and
    // a maximum of available units, you can use waitForAcquire() to request access to a unit of that
    // resource (thread-safe) and after your work is finished you can call releaseAcquired() to release
    // that resource unit and make it available to other threads
    class Semaphore
    {
      public:
        Semaphore( long initialValue, long maxValue );
        ~Semaphore();

        bool waitForAcquire();
        bool releaseAcquired();

      private:
#ifdef THREAD_API_WIN
        long   m_count;
        HANDLE m_handle;
#else
        sem_t           sem;
#endif
    };


    // ----- only internal data beyond this point -----

    // public, because Call is also used by non-member functions
    struct Call
    {
        void ( *m_functionPtr )( void* );
        void* m_argumentPtr;
    };

  private:
    Call m_call;
    bool m_running;

#ifdef THREAD_API_WIN
    DWORD  m_id;
    HANDLE m_handle;
#else  // POSIX
    pthread_t           m_thread;
#endif
};

// ----------------------------------------------------------------------------
//  ThreadPool
// ----------------------------------------------------------------------------

// The ThreadPool class manages a set of threads and their arguments.
//
// When instancing a thread pool, a class Args must be suplied, which encapsulates
// all input/output parameters of the actual thread function.
//
// Args must implement two static functions:
//   startFunction : This is the thread entry function.
//   killFunctions : This function is called if a thread should terminate. It
//                   is responsible to signal the thread function to wrap up.
//
//   struct Args
//   {
//     ... // argument variables
//
//     // Thread entry function
//     static void startFunction(Args *argumentPtr);
//
//     // Thread kill function
//     static void killFunction(Args *argumentPtr);
//   };

template <class Args>
class ThreadPool
{
  public:
    static const unsigned int THREAD_LIMIT;

    enum Error
    {
        FAILURE = 0,
        SUCCESS = 1
    };

    inline ThreadPool();

    // Set the maximum number of conlwrrently running threads.
    inline void setMaxThreads( unsigned int n );
    // Get the maximum number of conlwrrently running threads.
    inline unsigned int getMaxThreads();
    // Get the number of free thread slots.
    inline unsigned int getFreeThreads();

    // Create a new thread with arguments args, returns the index of the thread.
    inline Error createThread( unsigned int& index, const Args& args );
    // Blocking wait for a single thread to terminate. Must be called to free a slot.
    inline Error waitThread( unsigned int index );
    // Wait for all threads to terminate.
    inline Error syncThreads();
    // Calls Args::killFunction() for all threads.
    inline Error killThreads();

    // Get access to the thread handle.
    inline Thread& getThread( unsigned int index );
    // Get access to the thread arguments.
    inline Args& getArgs( unsigned int index );

  private:
    std::vector<Thread> m_threads;
    std::vector<Args>   m_args;

    unsigned int m_max_threads;
    unsigned int m_num_threads;
};

// ----------------------------------------------------------------------------

template <class Args>
const unsigned int ThreadPool<Args>::THREAD_LIMIT = 64u;

// ----------------------------------------------------------------------------

template <class Args>
ThreadPool<Args>::ThreadPool()
    : m_max_threads( 0 )
    , m_num_threads( 0 )
{
    m_threads.resize( THREAD_LIMIT );
    m_args.resize( THREAD_LIMIT );
}

// ----------------------------------------------------------------------------

template <class Args>
void ThreadPool<Args>::setMaxThreads( unsigned int n )
{
    m_max_threads = std::min( n, THREAD_LIMIT );
}

// ----------------------------------------------------------------------------

template <class Args>
unsigned int ThreadPool<Args>::getMaxThreads()
{
    return m_max_threads;
}

// ----------------------------------------------------------------------------

template <class Args>
unsigned int ThreadPool<Args>::getFreeThreads()
{
    // Number or running threads can be larger than max_threads after shrinking it.
    return ( m_num_threads > m_max_threads ) ? 0 : m_max_threads - m_num_threads;
}

// ----------------------------------------------------------------------------

template <class Args>
typename ThreadPool<Args>::Error ThreadPool<Args>::createThread( unsigned int& index, const Args& args )
{
    // Number or running threads can be larger than max_threads after shrinking it.
    if( m_num_threads >= m_max_threads )
        return FAILURE;

    // Find a free slot
    for( unsigned int i = 0; i < (unsigned int)m_threads.size(); i++ )
        if( !m_threads[i].isRunning() )
        {
            m_args[i] = args;
            index     = i;

            // Start new thread with given function/arguments.
            if( m_threads[i].create( ( void ( * )( void* ) )( m_args[i].startFunction ), &m_args[i] ) )
            {
                m_num_threads++;
                return SUCCESS;
            }
            else
                return FAILURE;
        }

    // Should not reach this point.
    return FAILURE;
}

// ----------------------------------------------------------------------------

template <class Args>
typename ThreadPool<Args>::Error ThreadPool<Args>::waitThread( unsigned int index )
{
    RT_ASSERT( (size_t)index < m_threads.size() );
    if( m_threads[index].wait() )
    {
        m_num_threads--;
        return SUCCESS;
    }
    else
        return FAILURE;
}

// ----------------------------------------------------------------------------

template <class Args>
typename ThreadPool<Args>::Error ThreadPool<Args>::syncThreads()
{
    Error result = SUCCESS;

    // Call wait on all running threads.
    for( unsigned int i = 0; i < (unsigned int)m_threads.size(); i++ )
        if( m_threads[i].isRunning() && ( waitThread( i ) == FAILURE ) )
            result = FAILURE;

    return result;
}

// ----------------------------------------------------------------------------

template <class Args>
typename ThreadPool<Args>::Error ThreadPool<Args>::killThreads()
{
    Error result = SUCCESS;

    // Send all running threads a "kill" signal.
    for( unsigned int i = 0; i < (unsigned int)m_threads.size(); i++ )
        if( m_threads[i].isRunning() )
        {
            // Signal the thread function to finish computation.
            m_args[i].killFunction( &m_args[i] );

            // Then wait for the OS to shut down the thread.
            if( waitThread( i ) == FAILURE )
                result = FAILURE;
        }

    return result;
}

// ----------------------------------------------------------------------------

template <class Args>
Thread& ThreadPool<Args>::getThread( unsigned int index )
{
    RT_ASSERT( (size_t)index < m_threads.size() );
    return m_threads[index];
}

// ----------------------------------------------------------------------------

template <class Args>
Args& ThreadPool<Args>::getArgs( unsigned int index )
{
    RT_ASSERT( (size_t)index < m_args.size() );
    return m_args[index];
}

// ----------------------------------------------------------------------------

// These two classes make it possible for a user to just launch work() with the number of devices
// he wants. All other parameters are already bound to the WorkUnit

class WorkUnitTraits
{
  public:
    virtual void work( int tID ) = 0;
};


class WorkUnit : public WorkUnitTraits
{
    std::function<void( int )> kernB;

  public:
    template <class C>
    WorkUnit( C* obj, void ( C::*f )( int, void* ), void* args )
    {
        kernB = std::bind( f, obj, std::placeholders::_1, args );
    }

    inline void work( int tID ) override { kernB( tID ); }
};

// ----------------------------------------------------------------------------
//  PThreadPool aka PersistentThreadPool
// ----------------------------------------------------------------------------

// The PersistentThreadPool is a thread pool of persistent threads (i.e. threads that
// are never destroyed for the entire duration of the application).
// Each thread is available to perform a unit of work dispatched by the work() function
// in a fashion similar to a LWCA thread: let's suppose we want a Linear buffer to be
// uploaded in a multi-threaded way with the PersistentThreadPool:
//
// 1) declare a payload structure which contains the arguments of an upload function
// which, in a LWCA-kernel fashion, will be called by a number of threads conlwrrently:
//
//  struct LinearUploadPayload {
//          LinearUploadPayload(const ::std::vector<Device*>* devices)
//          : m_devices(devices) {
//          }
//        const ::std::vector<Device*>* m_devices;
//      };
//
// 2) Create a "host-kernel" function that will be called by all threads
//
//    void Linear::uploadDevices(int threadID, void* in_payload)
//    {
//      LinearUploadPayload* payload = reinterpret_cast<LinearUploadPayload*>(in_payload);
//      const Linear* linear = this;
//      const std::vector<Device*>& devices = *payload->m_devices;
//      ...
//
//  3) Add some work units and call the work() function. To synchronize with all threads exelwtion call wait()
//
//          PersistentThreadPool& tp = this->m_bufferStorage->getBuffer()->getContext()->getPersistentThreadPool();
//          LinearUploadPayload payload(&devices);
//          tp.add_work(new WorkUnit(this, &Linear::uploadDevices, (void*)&payload));
//          tp.work((unsigned int)devices.size());
//          tp.wait();
//
//  Done, the uploadDevices() function will be called by devices.size() threads that will conlwrrently
//  upload data to the devices and then get back to sleep as soon as their workload is finished.
//
template <class Args>
class PThreadPool
{
  public:
    inline PThreadPool( unsigned int maxThreadCount = 0 );  // if maxThreadCount is zero, the maximum number of threads is set to the number of host CPU cores
    ~PThreadPool();

    void work( WorkUnitTraits* unit, size_t threadsNum );
    void workAsync( WorkUnitTraits* unit, size_t threadsNum );
    void wait();

  private:
    size_t          m_launchedThreads;
    bool            m_sigterm;  // This flag allows the persistent thread pool to stop all the threads launched
    WorkUnitTraits* m_ptpJob;
    friend void Args::startFunction( Args* args );  // Only thread static functions can access this class's private members
    bool         m_do_work_in_progress;
    unsigned int m_maxThreadCount;  // Maximum threads allowed on the system

    // This semaphore counts how many threads we need to wait for if we want all of them to be finished
    Thread::Semaphore* m_ThreadsToWaitForSemaphore;

    // The threads this pool owns
    std::vector<std::pair<Thread, Args>> m_threads;
    // A vector of semaphores, one per each thread used to put asleep/wake up that thread
    std::vector<Thread::Semaphore*> m_threadWorkSemaphores;
    // The index in the job queue each thread is assigned to
    std::vector<size_t> m_threadJobIndex;

    inline void threadAcquireWorkSemaphore( size_t tID ) { m_threadWorkSemaphores[tID]->waitForAcquire(); }
    inline void increaseNumberOfSleepingThreads() { m_ThreadsToWaitForSemaphore->releaseAcquired(); }
    inline bool getSigtermFlag() { return m_sigterm; };
    inline void changePoolSize( size_t minimumThreadsNeeded );
};

// ----------------------------------------------------------------------------

template <class Args>
PThreadPool<Args>::PThreadPool( unsigned int maxThreadCount )
    : m_sigterm( false )
    , m_do_work_in_progress( false )
    , m_ThreadsToWaitForSemaphore( nullptr )
{
    // No need to initialize any other variable because they're all 0-sized
    if( maxThreadCount == 0 )
        m_maxThreadCount = prodlib::getNumberOfCPUCores();
    else
        m_maxThreadCount = maxThreadCount;
}

template <class Args>
PThreadPool<Args>::~PThreadPool()
{
    for( unsigned int i = 0; i < m_threadWorkSemaphores.size(); i++ )
        delete m_threadWorkSemaphores[i];
    delete m_ThreadsToWaitForSemaphore;
}

// ----------------------------------------------------------------------------

// if threadsNum == 0, no thread is launched and all the jobs are immediately
// performed by the main thread (BLOCKING). If threadsNum == 2, two threads with id 0 and 1 are launched, etc...
template <class Args>
void PThreadPool<Args>::workAsync( WorkUnitTraits* unit, size_t threadsNum )
{
    if( m_do_work_in_progress )
        return;  // There's already some work going on

    m_ptpJob = unit;

    if( threadsNum == 0 )
    {
        // No thread needed, just call the functions
        m_launchedThreads = 0;
        m_ptpJob->work( 0 );
        return;
    }

    // check threadsNum and possibly create some more, also re-create semaphore m_ThreadsToWaitForSemaphore
    // If there are not enough threads, increase our persistent thread pool dimension
    if( threadsNum > m_threads.size() )
    {
        changePoolSize( threadsNum );
    }

    // Launch threads
    m_do_work_in_progress = true;
    m_launchedThreads     = threadsNum;
    for( unsigned int i = 0; i < threadsNum; i++ )
    {
        // Wake up threads to do their work
        m_threadWorkSemaphores[i]->releaseAcquired();
    }
}

template <class Args>
void PThreadPool<Args>::wait()
{
    if( m_launchedThreads == 0 )
        return;

    // Wait for completion before returning..
    unsigned long m_threadsReturnedToSleep = 0;
    do
    {
        if( m_ThreadsToWaitForSemaphore->waitForAcquire() )
        {
            // Acquired, check if all threads have returned to their sleeping state
            m_threadsReturnedToSleep++;
            if( m_threadsReturnedToSleep == m_launchedThreads )
            {
                // Done
                m_launchedThreads = 0;
                break;
            }
        }
    } while( true );
    m_do_work_in_progress = false;
}

template <class Args>
void PThreadPool<Args>::changePoolSize( size_t minimumThreadsNeeded )
{
    if( minimumThreadsNeeded > m_maxThreadCount )
        throw ValueOutOfRange( RT_EXCEPTION_INFO, "Number of threads exceeding maximum allowance" );
    size_t oldNum = m_threads.size();
    m_threads.resize( minimumThreadsNeeded );
    m_threadWorkSemaphores.resize( m_threads.size(), 0 );
    delete m_ThreadsToWaitForSemaphore;
    m_ThreadsToWaitForSemaphore = new Thread::Semaphore( 0, (long)m_threads.size() );  // Start semaphore with no threads launched
    for( size_t i = oldNum; i < m_threads.size(); i++ )
    {  // Regenerate semaphores only for new threads
        m_threadWorkSemaphores[i]         = new Thread::Semaphore( 0, 1 );
        m_threads[i].second.threadPoolPtr = this;
        m_threads[i].second.thread_id     = (unsigned int)i;
        m_threads[i].first.create( ( void ( * )( void* ) ) & Args::startFunction, &m_threads[i].second, false );
    }
}

template <class Args>
void PThreadPool<Args>::work( WorkUnitTraits* unit, size_t threadsNum )
{
    // A synchronous call with just one thread means: do all the work in the main thread and don't bother waking up threads
    if( threadsNum == 1 )
        workAsync( unit, 0 );
    else
        workAsync( unit, threadsNum );
    wait();
}

// ----------------------------------------------------------------------------

// A WorkerThread is a structure that encapsulates all the arguments of one of the threads
// belonging to the PersistentThreadPool class. Each thread's ID and the address of the
// owner pool are stored into this structure. The thread's main function belongs to this
// as well.
//
struct WorkerThread
{
    WorkerThread();
    WorkerThread( PThreadPool<WorkerThread>* ptr );

    // Entry function for multi-threaded build calls.
    static void startFunction( WorkerThread* argumentPtr );

    unsigned int               thread_id;      // %tid
    PThreadPool<WorkerThread>* threadPoolPtr;  // A pointer to our thread pool
    size_t                     m_jobIndex;     // The index of the job we have to perform at our next cycle
};
typedef PThreadPool<WorkerThread> PersistentThreadPool;
}  // end namespace prodlib
