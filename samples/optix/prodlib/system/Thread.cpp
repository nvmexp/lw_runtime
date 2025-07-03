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

// Super simple thread wrapper

#include <prodlib/system/Thread.h>

#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/BasicException.h>
#include <prodlib/exceptions/Exception.h>

using namespace prodlib;


#ifdef THREAD_API_WIN
static DWORD WINAPI winEntry( LPVOID call )
#else  // POSIX
static void* ptEntry( void* call )
#endif
{
    Thread::Call* c = (Thread::Call*)call;
    c->m_functionPtr( c->m_argumentPtr );

    return 0;
}

// ----------------------------------------------------------------------------

Thread::Thread()
    : m_running( false )
#ifdef THREAD_API_WIN
    , m_id( 0 )
    , m_handle( 0 )
#endif
{
}

// ----------------------------------------------------------------------------

Thread::~Thread()
{
    // Don't kill threads here
}

// ----------------------------------------------------------------------------

bool Thread::create( void ( *functionPtr )( void* ), void* argumentPtr, bool createSuspended )
{
    RT_ASSERT( functionPtr != 0 && argumentPtr != 0 );
    bool returlwalue;

    // Return an error if we have an already running thread.
    if( m_running )
        return false;

    m_call.m_functionPtr = functionPtr;
    m_call.m_argumentPtr = argumentPtr;

#ifdef THREAD_API_WIN
    m_handle = CreateThread( NULL,                                    // default security attributes
                             0,                                       // use default stack size
                             &winEntry,                               // thread function name
                             &m_call,                                 // argument to thread function
                             createSuspended ? CREATE_SUSPENDED : 0,  // create suspended or default creation flag
                             &m_id                                    // returns the thread identifier
                             );
    returlwalue = ( m_handle != NULL );
    if( returlwalue )
        m_running = createSuspended ? false : true;

#else  // POSIX
    pthread_attr_t attr;
    pthread_attr_init( &attr );
    pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_JOINABLE );

    const int result = pthread_create( &m_thread,  // returns the thread identifier
                                       &attr,      // attributes
                                       &ptEntry,   // thread function name
                                       &m_call     // argument to thread function
                                       );

    pthread_attr_destroy( &attr );
    m_running   = true;
    returlwalue = ( result == 0 );
#endif

    return returlwalue;
}

// ----------------------------------------------------------------------------

bool Thread::resume()
{
    // Return an error if we have an already running thread.
    if( m_running )
        return false;

#ifdef THREAD_API_WIN
    m_running = true;
    ResumeThread( m_handle );
#else  // POSIX
    pthread_attr_t attr;
    pthread_attr_init( &attr );
    pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_JOINABLE );

    const int result = pthread_create( &m_thread,  // returns the thread identifier
                                       &attr,      // attributes
                                       &ptEntry,   // thread function name
                                       &m_call     // argument to thread function
                                       );

    pthread_attr_destroy( &attr );
    m_running = ( result == 0 );
#endif

    return m_running;
}

// ----------------------------------------------------------------------------

bool Thread::suspend()
{
    // Return an error if we have an already suspended thread.
    if( !m_running )
        return false;

#ifdef THREAD_API_WIN
    m_running = false;
    SuspendThread( m_handle );
#else  // POSIX
    pthread_attr_t attr;
    pthread_attr_init( &attr );
    pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_JOINABLE );

    const int result = pthread_create( &m_thread,  // returns the thread identifier
                                       &attr,      // attributes
                                       &ptEntry,   // thread function name
                                       &m_call     // argument to thread function
                                       );

    pthread_attr_destroy( &attr );
    m_running        = ( result == 0 );
#endif

    return m_running;
}

// ----------------------------------------------------------------------------

bool Thread::wait()
{
    // Return an error if this thread is not running;
    if( !m_running )
        return false;

    // Mark thread a invalid.
    m_running = false;

#ifdef THREAD_API_WIN
    const DWORD result = WaitForSingleObject( m_handle,  // handle to thread
                                              INFINITE   // no time-out interval
                                              );
    CloseHandle( m_handle );
    return ( result == WAIT_OBJECT_0 );
#else  // POSIX
    const int result = pthread_join( m_thread, NULL );
    return ( result == 0 );
#endif
}

// ----------------------------------------------------------------------------

Thread::Mutex::Mutex()
{
#ifdef THREAD_API_WIN
    InitializeCriticalSection( &m_critical_section );
#else  // POSIX
    pthread_mutexattr_t attr;
    int                 result = pthread_mutexattr_init( &attr );
    if( result != 0 )
        throw prodlib::UnknownError( RT_EXCEPTION_INFO, "Cannot init new mutex attribute." );
    result = pthread_mutexattr_settype( &attr, PTHREAD_MUTEX_RELWRSIVE );
    if( result != 0 )
        throw prodlib::UnknownError( RT_EXCEPTION_INFO, "Cannot set mutex attribute." );
    result = pthread_mutex_init( &m_mutex, &attr );
    if( result != 0 )
        throw prodlib::UnknownError( RT_EXCEPTION_INFO, "Cannot create new Mutex." );
    result = pthread_mutexattr_destroy( &attr );
    if( result != 0 )
        throw prodlib::UnknownError( RT_EXCEPTION_INFO, "Cannot destroy mutex attribute." );
#endif
}

// ----------------------------------------------------------------------------

Thread::Mutex::~Mutex()
{
#ifdef THREAD_API_WIN
    DeleteCriticalSection( &m_critical_section );
#else  // POSIX
    int result = pthread_mutex_destroy( &m_mutex );
    RT_ASSERT_NOTHROW( result == 0, "Cannot destroy mutex." );
    (void)result;
#endif
}

// ----------------------------------------------------------------------------

bool Thread::Mutex::lock()
{
#ifdef THREAD_API_WIN
    EnterCriticalSection( &m_critical_section );
    return true;
#else  // POSIX
    const int result = pthread_mutex_lock( &m_mutex );
    return ( result == 0 );
#endif
}

// ----------------------------------------------------------------------------

bool Thread::Mutex::unlock()
{
#ifdef THREAD_API_WIN
    LeaveCriticalSection( &m_critical_section );
    return true;
#else  // POSIX
    const int result = pthread_mutex_unlock( &m_mutex );
    return ( result == 0 );
#endif
}

// ----------------------------------------------------------------------------

Thread::Semaphore::Semaphore( long initialValue, long maxValue )
#ifdef THREAD_API_WIN
    : m_count( initialValue )
#endif
{
#ifdef THREAD_API_WIN
    m_handle = CreateSemaphore( NULL,          // default security attributes
                                initialValue,  // initial count
                                maxValue,      // maximum count
                                NULL );        // unnamed semaphore
#else
    sem_init( &sem, 0, initialValue );
#endif
}

Thread::Semaphore::~Semaphore()
{
#ifdef THREAD_API_WIN
    CloseHandle( m_handle );
#else
    sem_destroy( &sem );
#endif
}

// ----------------------------------------------------------------------------

bool Thread::Semaphore::waitForAcquire()
{
#ifdef THREAD_API_WIN
    DWORD dwWaitResult = WaitForSingleObject( m_handle,    // handle to semaphore
                                              INFINITE );  //infinite time-out interval

    switch( dwWaitResult )
    {
        // The semaphore object was signaled.
        case WAIT_OBJECT_0:
        {
            m_count--;
            return true;
        }
        break;

        default:
            return false;
            break;
    }
#else
    if( !sem_wait( &sem ) )
        return true;
    else
        return false;
#endif
}

// ----------------------------------------------------------------------------

bool Thread::Semaphore::releaseAcquired()
{
#ifdef THREAD_API_WIN
    m_count++;                        // This needs to be done before releasing the semaphore
    if( !ReleaseSemaphore( m_handle,  // handle to semaphore
                           1,         // increase count by one
                           NULL ) )   // not interested in previous count
    {
        // Failed, revert count
        m_count--;
        return false;
    }
    else
        return true;
#else
    if( !sem_post( &sem ) )
        return true;
    else
        return false;
#endif
}

// ----------------------------------------------------------------------------


WorkerThread::WorkerThread()
    : thread_id( -1 )
    , threadPoolPtr( NULL )
{
}

WorkerThread::WorkerThread( PThreadPool<WorkerThread>* ptr )
    : thread_id( -1 )
    , threadPoolPtr( ptr )
{
}

void WorkerThread::startFunction( WorkerThread* args )
{
    do
    {
        // Try to acquire the semaphore that signals that we have a unit of work
        args->threadPoolPtr->threadAcquireWorkSemaphore( args->thread_id );

        // Check if it's sigterm time
        if( args->threadPoolPtr->getSigtermFlag() )
            break;

        // Call the job stored in the PersistentThreadPool with our threadId
        args->threadPoolPtr->m_ptpJob->work( args->thread_id );

        // Go to sleep, I did all the jobs in the queue
        args->threadPoolPtr->increaseNumberOfSleepingThreads();
    } while( true );
}
