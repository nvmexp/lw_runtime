/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2015-2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
/*
 *  Module name              : stdThreadsPOSIX.c
 *
 *  Description              :
 *
 */

/*----------------------------- Includes ------------------------------------*/

#include "stdThreads.h"
#include "stdLocal.h"
#include "stdList.h"

// Use Mach semaphores on Mac, because Apple don't properly implement unnamed POSIX
// semaphores (sem_init() and sem_destroy() are bogus stub functions.)
#if defined(STD_OS_Darwin)
#include <mach/semaphore.h>
#include <mach/task.h>
#include <mach/mach_init.h> /* mach_task_self in semaphores */
#endif

#include "threadsMessageDefs.h"

/*--------------------------------- Types -----------------------------------*/

#define MIN_STACK_SIZE     100000

#if defined(STD_OS_Darwin)
typedef semaphore_t      _sem;
typedef kern_return_t    _sem_result;
static const _sem_result SEM_RESULT_SUCCESS = KERN_SUCCESS;
#else
typedef sem_t            _sem;
typedef int              _sem_result;
static const _sem_result SEM_RESULT_SUCCESS = 0;
#endif

#if defined(STD_OS_Hos)
// HOS only supports FIFO and OTHER policies. The remaining SHED_* symbols are undefined.
#define SCHED_RR SCHED_OTHER
#endif

struct stdThread {
    stdThreadContext     context;
    stdThreadEntry_t     entry;
    Pointer              arg;
    Bool                 joinable;
    pthread_t            pThread;
    pthread_cond_t       sleepCond;
    pthread_mutex_t      sleepMutex;
    _sem                 suspendSem;
    _sem                *termSignal;

    stdThread_t          next,prev;
};

/*--------------- POSIX/Mach semaphore compatibility functions --------------*/

static inline _sem_result _sem_init(_sem *sem, int count)
{
    _sem_result result =
#if defined(STD_OS_Darwin)
	semaphore_create(mach_task_self(), sem, SYNC_POLICY_FIFO, count);
#else
    sem_init(sem, 0, count);
#endif

    return result;
}

static inline _sem_result _sem_wait(_sem *sem)
{
    _sem_result result;

#if defined(STD_OS_Darwin)
    do {
        result = semaphore_wait(*sem);
    } while (result == KERN_ABORTED);
#else
    do {
        result = sem_wait(sem);
    } while (result == -1 && errno == EINTR);
#endif

    return result;
}

static inline _sem_result _sem_destroy(_sem *sem)
{
    _sem_result result =
#if defined(STD_OS_Darwin)
    semaphore_destroy(mach_task_self(), *sem);
#else
    sem_destroy(sem);
#endif

    return result;
}

static inline _sem_result _sem_post(_sem *sem)
{
    _sem_result result =
#if defined(STD_OS_Darwin)
    semaphore_signal(*sem);
#else
    sem_post(sem);
#endif

    return result;
}

/*----------------------------- Global Module State -------------------------*/

static Int               maxPOSIXPriority;
static Int               minPOSIXPriority;
static Int               nrofPriorities;
static pthread_key_t     taskKey;
static pthread_mutex_t   globalMutex;

struct stdThread sentStart,sentEnd;

#define EE3   ((Int64)  1000       )
#define EE9   ((Int64)  1000000000 )


/*----------------------------- Priority Colwersion -------------------------*/

#define std_TO_POSIX(i)\
        ( ((i)<0) ? ( (i) + minPOSIXPriority + nrofPriorities ) \
        : ( (i) + minPOSIXPriority ) )

/*----------------------- Initialization/Termination ------------------------*/

    #define threadsInitialized()  (sentStart.next != Nil)

    #define assertInitialized() if (!threadsInitialized()) { stdThreadsInit(); }

    static void destroyThread( stdThread_t thread );


    Bool stdThreadsInit(void)
    {
        if (!threadsInitialized()) {
            pthread_mutexattr_t  attr;

            pthread_key_create (&taskKey, (void(*)(void*))destroyThread);

            pthread_mutexattr_init    (&attr);
            pthread_mutexattr_settype (&attr, PTHREAD_MUTEX_RELWRSIVE);
            pthread_mutex_init(&globalMutex, &attr);

            pthread_mutexattr_destroy (&attr);

            maxPOSIXPriority = sched_get_priority_max( SCHED_RR );
            minPOSIXPriority = sched_get_priority_min( SCHED_RR );
            nrofPriorities   = maxPOSIXPriority - minPOSIXPriority + 1;

            sentStart.next= &sentEnd;
            sentEnd  .prev= &sentStart;
        }

        return True;
    }


   /*
    * Explicit termination of threads library:
    * wait until all threads have terminated, and
    * then free all data structures.
    */
    void stdThreadsTerm(void)
    {
        if (threadsInitialized()) {
            _sem termSignal;
            _sem_init(&termSignal, 0);

            stdThreadDetach();

            stdGlobalEnter();

            while (sentStart.next != &sentEnd) {
                sentStart.next->termSignal = &termSignal;

                stdGlobalExit ();
                _sem_wait(&termSignal);
                stdGlobalEnter();
            }

            stdGlobalExit();

            _sem_destroy          (&termSignal);
            pthread_mutex_destroy (&globalMutex);
            pthread_key_delete    (taskKey);

            stdMEMCLEAR(&sentStart);
            stdMEMCLEAR(&sentEnd);
        }
    }


   /*
    * Version for use in destructor below.
    * Functions called from DllMain cannot (b)lock,
    * so just destroy all data and prey that all the
    * threads have terminated.
    */
    static void stdThreadsTermX(void)
    {
        if (threadsInitialized()) {
            while (sentStart.next != &sentEnd) {
                destroyThread( sentStart.next );
            }

            pthread_mutex_destroy (&globalMutex);
            pthread_key_delete    (taskKey);

            stdMEMCLEAR(&sentStart);
            stdMEMCLEAR(&sentEnd);
        }
    }


   /*
    * Module initialization before anyone
    * sees any of its functions:
    */
    extern "C" { Bool threadsDoCleanupHandlersOnTermination = True; }
    extern "C" { Bool threadsDoCleanupOnTermination         = True; }

    class InitTerm {
      public :
       InitTerm() { stdThreadsInit(); }

       ~InitTerm() { Terminate(); }

       void Terminate() {
           if (threadsDoCleanupHandlersOnTermination) {
               stdProcessCleanupHandlers();
               threadsDoCleanupHandlersOnTermination = false;
           }
           if (threadsDoCleanupOnTermination) {
               stdThreadsTermX();
               threadsDoCleanupOnTermination = false;
           }
       }


    } ___stdThreads_initialized___;

    // explicit destroy() API for destroying the thread, and associated memory, purposefully before CRT deinit.
    // otherwise, memory leak will be reported by the driver.
    // Refer following comment - borrowed from Windows couterpart of this file (stdThreadsWin32.cpp).
    /*
     * Bug: 200475130 - destruction of __stdThreads_initialized___, and thus the deletion of dynamic memory
     * that it holds doesn't happen until after the call in the D3D driver to tear down the CRT. This is too
     * late and causes it to appear that the driver leaks memory. So we add a function that the driver can call
     * to free this memory just before doing the memory leak check.
     *
     */
    void destroyStdThreads()
    {
        ___stdThreads_initialized___.Terminate();
    }

/*-------------------------- Local Functions --------------------------------*/

static stdThread_t allocateThread(void)
{
    stdThread_t result;

    assertInitialized();

   /* don't use stdNEW for this,
    * because that macro needs the current thread
    * context, which we are creating here:
    */
    result= (stdThread_t)malloc(sizeof *result);

    stdCHECK( result, (threadsMsgMemoryOverFlow) );

    stdMEMCLEAR(result);

    if (result)  {
        pthread_cond_init  (&result->sleepCond,  Nil);
        pthread_mutex_init (&result->sleepMutex, Nil);
        _sem_init          (&result->suspendSem, 0);

        stdGlobalEnter();

            result->next=  sentStart.next;
            result->prev= &sentStart;

            sentStart.next->prev = result;
            sentStart.next       = result;

        stdGlobalExit();
    }

    return result;
}


static void destroyThread( stdThread_t thread )
{
    if (thread) {
        _sem *termSignal;

        stdGlobalEnter();

        termSignal         = thread->termSignal;

        thread->next->prev = thread->prev;
        thread->prev->next = thread->next;

        stdGlobalExit();

        pthread_cond_destroy  (&thread->sleepCond );
        pthread_mutex_destroy (&thread->sleepMutex);
        _sem_destroy          (&thread->suspendSem);

        if (!thread->joinable) {
            free (thread);
        }

        if (termSignal) {
            _sem_post(termSignal);
        }
    }
}


/*
 *  Function       : Detach the current thread from stdThreads library
 *                   This function can be called at any time by
 *                   threads whether or not they ever had
 *                   anything to do with this library.
 *                   Afterwards, the current thread is .not. prohibited
 *                   from making any further calls to the stdThreads library:
 *                   doing this will cause to automatically reattach.
 */
void stdThreadDetach(void)
{
    if (threadsInitialized()) {
        stdThread_t self= (stdThread_t)pthread_getspecific (taskKey);

        if (self) {
            destroyThread(self);
            pthread_setspecific(taskKey,Nil);
        }
    }
}



/*---------------------------- General Functions ----------------------------*/

    static inline stdThread_t lwrrentThread(void)
    {
        stdThread_t result = threadsInitialized() ? (stdThread_t)pthread_getspecific (taskKey) : (stdThread_t)Nil;

        if (!result) {
            result= allocateThread();
            pthread_setspecific (taskKey, result);
        }

        return result;
    }


/*
 * Function        : Get exelwtion context of current thread.
 * Parameters      :
 * Function Result : Requested context
 */
stdThreadContext_t stdGetThreadContext()
{
    return &lwrrentThread()->context;
}



/*
 * Function        : Obtain handle to current thread.
 * Parameters      :
 * Function Result : Lwrrently exelwting thread.
 */
stdThread_t stdThreadSelf()
{
    return lwrrentThread();
}



/*
 * Function        : This function terminates the current process, including all threads
 *                   that are part of the process.
 * Parameters      : status (I) Return status.
 * Function Result :
 */
void stdExit (Int status)
{
    stdEXIT(status);
}


/*
 * Function        : Obtain number of thread priorities supported by
 *                   the current Threads framework.
 *                   Priorities run from 0 to N-1, where N is the value
 *                   returned by this function.
 *
 *                   Note: without consulting this function, only
 *                         one (1) priority value may be assumed.
 *
 * Function Result : Retrieved number of thread priorities.
 */
uInt stdNrofThreadPriorities()
{
    return nrofPriorities;
}



/*--------------------------------- Threads -----------------------------------*/

    static void *wrapper (void *arg)
    {
        stdThread_t  thread= (stdThread_t)arg;

        pthread_setspecific (taskKey, thread);

        thread->entry (thread->arg);

        stdThreadExit();

        return Nil;
    }

/*
 * Function        : Create new thread for exelwtion of the specified entry function.
 *                   The created thread starts exelwting the specified function
 *                   with the specified argument; the thread is automatically
 *                   destroyed upon termination of this function, or when it
 *                   performs a call to function stdThreadExit().
 * Parameters      : entry     (I) Function to execute by the created thread.
 *                   arg       (I) Argument to pass to 'entry'.
 *                   prio      (I) Thread's exelwtion priority.
 *                   stackSize (I) Size of thread's stack in bytes.
 *                   joinable  (I) Whether this thread can be joined.
 *                   name      (I) Name of thread, used in info prints
 * Function Result : Created thread, or Nil in case of failure.
 */
stdThread_t
    _stdThreadCreate (
        stdThreadEntry_t   entry,
        Pointer            arg,
        Int                prio,
        Int                stackSize,
        Bool               joinable,
        cString            name
    )
{
#if !defined(STD_OS_Hos)
    pthread_attr_t      attr;
    stdThread_t         result;
    pthread_t           pthread;

    result= allocateThread();
    if (!result) { return Nil; }

    result->entry    = entry;
    result->arg      = arg;
    result->joinable = joinable;

    pthread_attr_init           (&attr);
    pthread_attr_setstacksize   (&attr, stdMAX( stackSize, MIN_STACK_SIZE ) );

#if defined(STD_OS_Hos)
    stdASSERT( joinable, ("Detached threads are not supported on HOS") );
#else
    if (!joinable) {
        pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED             );
    }
#endif

  #if defined(STD_OS_FAMILY_Unix) || defined(STD_OS_Darwin)
    if ( geteuid() == 0 )
  #endif
    {
        struct sched_param   param;
        param.sched_priority = std_TO_POSIX (prio);
        param.sched_priority = stdMIN( param.sched_priority, maxPOSIXPriority );
        param.sched_priority = stdMAX( param.sched_priority, minPOSIXPriority );

#if !defined(__ANDROID__)
        pthread_attr_setinheritsched (&attr, PTHREAD_EXPLICIT_SCHED);
#endif
        pthread_attr_setschedpolicy  (&attr, SCHED_RR );
        pthread_attr_setschedparam   (&attr, &param);
    }

    if ( pthread_create (&pthread, &attr, wrapper, result) ) {
        destroyThread(result);
        result= Nil;
    } else {
        result->pThread = pthread;
    }

    pthread_attr_destroy(&attr);

    return result;
#else
    // Spawning threads is not supported on HOS drivers
    return Nil;
#endif
}



/*
 * Function        : Stop exelwtion of current thread, and delete it.
 * Parameters      :
 * Function Result :
 */
void stdThreadExit ()
{
   /*
    * This is not necessary here,
    * because destroyThread is specified as
    * delete function for the taskKey under
    * which the thread descriptor is stored:
    *
    destroyThread( stdThreadSelf() );
    */
#if !defined(STD_OS_Hos)
    pthread_exit (Nil);
#endif
}



/*
 * Function        : Delay exelwtion of current thread for specified duration.
 * Parameters      : nanosecs  (I) Minimum amount of time to sleep (in nanoseconds).
 * Function Result :
 */
    static void colwertToAbsoluteTime( uInt64 nanosecs, struct timespec *timeSpec)
    {
        struct timeval now;

        gettimeofday(&now,0);

        timeSpec->tv_sec   = now.tv_sec        + nanosecs / EE9;
        timeSpec->tv_nsec  = now.tv_usec * EE3 + nanosecs % EE9;
        timeSpec->tv_sec  += timeSpec->tv_nsec / EE9;
        timeSpec->tv_nsec  = timeSpec->tv_nsec % EE9;
    }

void stdThreadSleep (uInt64 nanosecs)
{
    stdThread_t     self;
    struct timespec timeSpec;

    self= stdThreadSelf();

    colwertToAbsoluteTime(nanosecs, &timeSpec);

    pthread_mutex_lock     (&self->sleepMutex);

    pthread_cond_timedwait (&self->sleepCond, &self->sleepMutex, &timeSpec);

    pthread_mutex_unlock   (&self->sleepMutex);
}



/*
 * Function        : Suspend lwrrently exelwted thread.
 * Parameters      :
 * Function Result :
 */
void stdThreadSuspend ()
{
    stdThread_t  self= stdThreadSelf();

	_sem_wait(&self->suspendSem);
}


/*
 * Function        : Resume specified thread.
 * Parameters      : thread (I) Thread to resume
 * Function Result :
 */
void stdThreadResume ( stdThread_t thread )
{
    _sem_post(&thread->suspendSem);
}


/*
 * Function        : Kill the thread
 * Parameters      : thread (I) Thread to be killed
 *                   sig Signal to be sent to thread
 * Function Result :
 */
void STD_CDECL stdThreadKill( stdThread_t thread , int sig )
{
#if !defined(STD_OS_Hos)
    pthread_kill(thread->pThread, sig);
#endif
}


/*
 * Function        : Wait for a thread to finish
 * Parameters      : thread (I) Thread to wait for
 * Function Result :
 */
void stdThreadJoin ( stdThread_t thread )
{
    stdASSERT(thread->joinable, ("Joining non-joinable thread"));
    pthread_join(thread->pThread, Nil);
    free(thread); // thread completion cleans up the rest of the structure
}



/*------------------------------- Semaphores --------------------------------*/

/*
 * Function        : Create new semaphore
 * Parameters      : count     (I) Initial semaphore count.
 * Function Result : Created Semaphore, or Nil in case of failure.
 */
stdSem_t stdSemCreate (Int count)
{
    stdSem_t result;

    stdNEW(result);

    if (result==Nil) { return Nil; }

    if (!stdSemInit(result,count)) {
        stdFREE(result);
        result= Nil;
    }


    return result;
}

/*
 * Function        : Delete semaphore.
 * Parameters      : sem  (I) Semaphore to delete.
 * Function Result :
 */
void stdSemDelete (stdSem_t sem)
{
    stdSemTerm(sem);
    stdFREE (sem);
}



/*
 * Function        : Initialize semaphore in preallocated memory.
 * Parameters      : sem       (O) Semaphore space to initialize
 *                   count     (I) Initial semaphore count.
 * Function Result : True iff. initialization succeeded.
 */
Bool stdSemInit (struct stdSem *sem, Int count)
{
    return _sem_init(&sem->sem, count) == SEM_RESULT_SUCCESS;
}

/*
 * Function        : Delete semaphore in preallocated memory.
 * Parameters      : sem  (O Semaphore to delete.
 * Function Result :
 */
void stdSemTerm (struct stdSem *sem)
{
	_sem_destroy(&sem->sem);
}



/*
 * Function        : Acquire semaphore or wait.
 *                   If the semaphore's count is lwrrently equal to zero,
 *                   then wait until this count increases.
 * Parameters      : sem (I) semaphore to acquire
 * Function Result :
 */
void stdSemP (stdSem_t sem)
{
    _sem_wait(&sem->sem);
}


// We don't implement this function for the Mac because there's no equivalent of POSIX
// semaphores' sem_trywait() function in the Mach semaphores that we use for the Mac.
// Luckily, this function doesn't seem to be used in current code. If it turns out
// that we do need it, we'll either have to figure out how to implement similar
// "non-blocking wait" functionality with Mach semaphores, or move to POSIX *named*
// semaphores on the Mac.
#if !defined(STD_OS_Darwin)
/*
 * Function        : Acquire semaphor or fail.
 *                   Try to acquire a semaphore, and return
 *                   immediate failure if the semaphore's
 *                   count is lwrrently equal to zero.
 * Parameters      : sem (I) Semaphore to acquire.
 * Function Result : True iff. semaphore could be aquired
 */
Bool stdSemTryP (stdSem_t sem)
{
    int status;

    do {
        status = sem_trywait((sem_t*)&sem->sem);
    } while (status == -1 && errno == EINTR);

    return status == 0;
}
#endif



/*
 * Function        : Release semaphore.
 * Parameters      : sem (I) Semaphore to release.
 * Function Result :
 */
void stdSemV (stdSem_t sem)
{
    _sem_post(&sem->sem);
}



/*--------------------------------- Mutexes ---------------------------------*/


/*
 * Function        : Create new mutex.
 *                   When supported by the underlying operating system,
 *                   the mutex will be created in priority inheritance mode.
 * Parameters      :
 * Function Result : created mutex, or Nil in case of failure.
 */
stdMutex_t stdMutexCreate ()
{
    stdMutex_t result;

    stdNEW(result);
    if (result == Nil) { return Nil; }

    if (!stdMutexInit(result)) {
        stdFREE(result);
        result= Nil;
    }

    return result;
}

/*
 * Function        : Delete mutex.
 * Parameters      : mutex  (I) Mutex to delete.
 * Function Result :
 */
void stdMutexDelete (stdMutex_t mutex)
{
    pthread_mutex_destroy ((pthread_mutex_t*)&mutex->mutex);

    stdFREE(mutex);
}



/*
 * Function        : Initialize mutex in preallocated memory.
 *                   When supported by the underlying operating system,
 *                   the mutex will be initialized in priority inheritance mode.
 * Parameters      : mutex  (O) Mutex space to initialize
 * Function Result : True iff. initialization succeeded.
 */
Bool stdMutexInit (struct stdMutex *mutex)
{
    Bool                 result;
    pthread_mutexattr_t  attr;

    pthread_mutexattr_init    (&attr);
    pthread_mutexattr_settype (&attr, PTHREAD_MUTEX_RELWRSIVE);
    result= pthread_mutex_init((pthread_mutex_t*)&mutex->mutex, &attr) == 0;

    pthread_mutexattr_destroy (&attr);

    return result;
}

/*
 * Function        : Delete mutex in preallocated memory.
 * Parameters      : mutex  (O) Mutex to delete.
 * Function Result :
 */
void stdMutexTerm (struct stdMutex *mutex)
{
    pthread_mutex_destroy ((pthread_mutex_t*)&mutex->mutex);
}



/*
 * Function        : Acquire mutex.
 * Parameters      : mutex   (I) Mutex to acquire.
 * Function Result :
 */
void stdMutexEnter (stdMutex_t mutex)
{
    pthread_mutex_lock ((pthread_mutex_t*)&mutex->mutex);
}



/*
 * Function        : Release mutex.
 * Parameters      : mutex   (I) Mutex to release.
 * Function Result :
 */
void stdMutexExit (stdMutex_t mutex)
{
    pthread_mutex_unlock ((pthread_mutex_t*)&mutex->mutex);
}



/*---------------------------- Atomicity Providers --------------------------*/

#if defined __KERNEL__
stdAtomic_t _stdAtomicCreate()
{
    stdAtomic_t result;
    stdNEW(result);
    stdAtomicInit(result);
    return result;
}
#endif

/*--------------------------- Process- Global Mutex -------------------------*/

/*
 * Function        : Enter process- global mutex.
 *                   This function enters the one global mutex that
 *                   is provided by the std.
 *                   This mutex should be used with care, in order to
 *                   avoid it becoming a global bottleneck. It is intended
 *                   to provide exclusion in initialization phases in which
 *                   software components were not yet able to create their
 *                   own mutexes.
 * Parameters      :
 * Function Result :
 */
void stdGlobalEnter()
{
    assertInitialized();

    pthread_mutex_lock (&globalMutex);
}



/*
 * Function        : Exit process- global mutex.
 * Parameters      :
 * Function Result :
 */
void stdGlobalExit()
{
    pthread_mutex_unlock (&globalMutex);
}



/*---------------------------------- Time -----------------------------------*/

/*
 * Function        : Obtain current time
 *                   Return current time in nanoseconds since unspecified time origin.
 * Parameters      :
 * Function Result : Current time
 */
uInt64 stdTimeNow (void)
{
    struct timeval now;

    gettimeofday(&now,0);

    return  (now.tv_usec * EE3)
          + (now.tv_sec  * EE9);
}



/*----------------------------- Condition Variables -------------------------*/

struct stdCondVar {
    stdList_t        waitQueue;
    struct stdMutex  mutex;
    stdThread_t      owner;
    uInt             count;
};



/*
 * Function        : Create new condition variable.
 * Parameters      :
 * Function Result : Created condition variable
 */
stdCondVar_t stdCondCreate ()
{
    stdCondVar_t result;

    stdNEW(result);

    if ( result && stdMutexInit(&result->mutex) ) {
        return result;
    } else {
        stdFREE(result);
        return Nil;
    }
}



/*
 * Function        : Delete condition variable.
 * Parameters      : condition  (I) Condition variable to delete.
 * Function Result :
 */
void stdCondDelete (stdCondVar_t condition)
{
    stdMutexTerm (&condition->mutex);

    stdFREE(condition);
}



/*
 * Function        : Acquire condition variable.
 * Parameters      : condition   (I) Condition variable to acquire.
 * Function Result :
 */
void stdCondEnter (stdCondVar_t condition)
{
    stdMutexEnter( &condition->mutex );
    condition->owner= stdThreadSelf();
    condition->count++;
}



/*
 * Function        : Release condition variable.
 * Parameters      : condition   (I) Condition variable to release.
 * Function Result :
 */
void stdCondExit (stdCondVar_t condition)
{
    stdASSERT( stdThreadSelf() == condition->owner, ("stdCondExit: not owner") );

    if (!--condition->count) { condition->owner= Nil; }
    stdMutexExit( &condition->mutex );
}



/*
 * Function        : Wait until condition variable broadcast or signalled.
 * Parameters      : condition   (I) Condition variable to wait on.
 * Function Result :
 */
void stdCondWait (stdCondVar_t condition)
{
    stdListRec  q;
    stdThread_t self= stdThreadSelf();

    stdASSERT( self == condition->owner, ("stdCondWait: not owner"       ) );
    stdASSERT( condition->count == 1,    ("stdCondWait: held relwrsively") );

    q.head = self;
    q.tail = condition->waitQueue;
    condition->waitQueue= &q;

    stdCondExit( condition );

	_sem_wait(&self->suspendSem);
}



/*
 * Function        : Release all waiters on condition variable.
 * Parameters      : condition   (I) Condition to release.
 * Function Result :
 */
void stdCondBroadcast (stdCondVar_t condition)
{
    stdList_t waiters= Nil;

    stdASSERT( stdThreadSelf() == condition->owner, ("stdCondBroadcast: not owner") );

    stdSWAP(waiters, condition->waitQueue, stdList_t);

    while (waiters) {
        stdThread_t waiter= (stdThread_t)waiters->head;
        waiters= waiters->tail;
        _sem_post(&waiter->suspendSem);
    }

    stdCondExit( condition );
}



/*-------------------------------- Barriers ---------------------------------*/


struct stdBarrier {
    uInt       count;
    Bool       locked;
    stdSem_t   preLock;
    stdSem_t   postLock;
};


/*
 * Function        : Create new barrier.
 * Parameters      : count      (I) Amount of threads that
 *                                  will use the barrier
 * Function Result : Created barrier
 */
stdBarrier_t stdBarrierCreate( uInt count )
{
    stdBarrier_t result;

    stdNEW(result);

    result->count    = count;
    result->locked   = False;
    result->preLock  = stdSemCreate(1);
    result->postLock = stdSemCreate(0);

    return result;
}


/*
 * Function        : Delete barrier.
 * Parameters      : barrier    (I) Barrier to delete.
 * Function Result :
 */
void stdBarrierDelete ( stdBarrier_t barrier )
{
    stdSemDelete(barrier->preLock);
    stdSemDelete(barrier->postLock);

    stdFREE(barrier);
}


/*
 * Function        : Wait until N threads (including the one making
 *                   the call to this function) have arrived at the barrier,
 *                   where N is the count value by which the barrier
 *                   was created. After that, all N threads will be released.
 * Parameters      : barrier    (I) Barrier to wait on
 * Function Result :
 */
void stdBarrierWait( stdBarrier_t barrier )
{
    Bool locked;

    stdSemP( barrier->preLock  );
    locked = barrier->locked;
    stdSemV( barrier->postLock );

    if (!locked) {
        uInt i;

        barrier->locked= True;

        for (i=0; i<barrier->count-1; i++) {
            stdSemV( barrier->preLock  );
        }

        for (i=0; i< barrier->count; i++) {
            stdSemP( barrier->postLock );
        }

        barrier->locked= False;

        stdSemV( barrier->preLock  );
    }

}
