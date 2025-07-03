/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2006-2020, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : stdThreads.h
 *
 *  Description              :
 *
 */

#include "stdLocal.h"
#ifndef stdThreads_INCLUDED
#define stdThreads_INCLUDED

/*------------------------------- Includes ----------------------------------*/

#include "stdTypes.h"
#include "stdMessages.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------- Types -----------------------------------*/

/*
 * Function        : Thread entry function.
 *                   This type specifies the entry functions for tasks
 *                   created by stdThreadCreate.
 * Parameters      : data (I) Generic data argument to this function,
 *                            passed to stdThreadCreate.
 * Function Result :
 */

typedef void (STD_CDECL *stdThreadEntry_t) (Pointer data);

typedef struct stdThread         *stdThread_t;
typedef struct stdSem            *stdSem_t;
typedef struct stdMutex          *stdMutex_t;
typedef struct stdAtomic         *stdAtomic_t;
typedef struct stdCondVar        *stdCondVar_t;
typedef struct stdBarrier        *stdBarrier_t;


/*
 * Time in nano seconds:
 */
typedef uInt64 stdTime_t;


/*-------------------------------- Constants --------------------------------*/

/*
 * Time constants in nano seconds:
 */

#define stdMICRO_SECOND      ((stdTime_t)       1000)
#define stdMILLI_SECOND      ((stdTime_t)    1000000)
#define stdSECOND            ((stdTime_t) 1000000000)


/*---------------------------- General Functions ----------------------------*/

/*
 * Function        : Get exelwtion context of current thread.
 * Parameters      :
 * Function Result : Requested context
 */
stdThreadContext_t STD_CDECL stdGetThreadContext(void);



/*
 * Function        : This function terminates the current process, including all threads
 *                   that are part of the process.
 * Parameters      : status (I) Return status.
 * Function Result :
 */
void STD_CDECL stdExit (Int status);



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
uInt STD_CDECL stdNrofThreadPriorities(void);



/*--------------------------------- Threads ---------------------------------*/

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
stdThread_t STD_CDECL
    _stdThreadCreate (
        stdThreadEntry_t   entry,
        Pointer            arg,
        Int                prio,
        Int                stackSize,
        Bool               joinable,
        cString             name
    );

#define stdThreadCreate(entry,arg,prio,stackSize)  \
            _stdThreadCreate(entry,arg,prio,stackSize,False,#entry)

#define stdThreadCreateJoinable(entry,arg,prio,stackSize)  \
            _stdThreadCreate(entry,arg,prio,stackSize,True, #entry)


/*
 * Function        : Stop exelwtion of current thread, and delete it.
 * Parameters      :
 * Function Result :
 */
void STD_CDECL stdThreadExit (void);



/*
 * Function        : Obtain handle to current thread.
 * Parameters      :
 * Function Result : Lwrrently exelwting thread.
 */
stdThread_t STD_CDECL stdThreadSelf(void);



/*
 * Function        : Delay exelwtion of current thread for specified duration.
 * Parameters      : nanosecs  (I) Minimum amount of time to sleep (in nanoseconds).
 * Function Result :
 */
void STD_CDECL stdThreadSleep (stdTime_t nanosecs);



/*
 * Function        : Suspend lwrrently exelwted thread.
 * Parameters      :
 * Function Result :
 */
void STD_CDECL stdThreadSuspend (void);



/*
 * Function        : Resume specified thread.
 * Parameters      : thread (I) Thread to resume
 * Function Result :
 */
void STD_CDECL stdThreadResume ( stdThread_t thread );



/*
 * Function        : Kill the thread
 * Parameters      : thread (I) Thread to be killed
 *                   sig Signal to be sent to thread (ignored on Windows)
 * Function Result :
 */
void STD_CDECL stdThreadKill( stdThread_t thread , int sig );



/*
 * Function        : Wait for a thread to finish
 * Parameters      : thread (I) Thread to wait for
 * Function Result :
 */
void STD_CDECL stdThreadJoin( stdThread_t thread );



/*
 *  Function       : Detach the current thread from stdThreads library
 *                   This function can be called at any time by
 *                   threads whether or not they ever had
 *                   anything to do with this library.
 *                   Afterwards, the current thread is .not. prohibited
 *                   from making any further calls to the stdThreads library:
 *                   doing this will cause to automatically reattach.
 */
void STD_CDECL stdThreadDetach(void);


/*------------------------------- Semaphores --------------------------------*/

/*
 * Function        : Create new semaphore
 * Parameters      : count     (I) Initial semaphore count.
 * Function Result : Created Semaphore, or Nil in case of failure.
 */
stdSem_t STD_CDECL stdSemCreate (Int count);

/*
 * Function        : Delete semaphore.
 * Parameters      : sem  (I) Semaphore to delete.
 * Function Result :
 */
void STD_CDECL stdSemDelete (stdSem_t sem);



/*
 * Function        : Initialize semaphore in preallocated memory.
 * Parameters      : sem       (O) Semaphore space to initialize
 *                   count     (I) Initial semaphore count.
 * Function Result : True iff. initialization succeeded.
 */
Bool STD_CDECL stdSemInit (struct stdSem *sem, Int count);

/*
 * Function        : Delete semaphore in preallocated memory.
 * Parameters      : sem  (O Semaphore to delete.
 * Function Result :
 */
void STD_CDECL stdSemTerm (struct stdSem *sem);



/*
 * Function        : Acquire semaphore or wait.
 *                   If the semaphore's count is lwrrently equal to zero,
 *                   then wait until this count increases.
 * Parameters      : sem (I) semaphore to acquire
 * Function Result :
 */
void STD_CDECL stdSemP (stdSem_t sem);



// There's no Mach semaphore equivalent for POSIX semaphores' sem_trywait() function,
// which this function needs. Luckily, it doesn't seem to be used in current code,
// so we just omit this function. If it turns out we do need this function, we may
// have to move to POSIX *named* semaphores on the Mac, instead of Mach semaphores.
#if !defined(__APPLE__)
/*
 * Function        : Acquire semaphor or fail.
 *                   Try to acquire a semaphore, and return
 *                   immediate failure if the semaphore's
 *                   count is lwrrently equal to zero.
 * Parameters      : sem (I) Semaphore to acquire.
 * Function Result : True iff. semaphore could be aquired
 */
Bool STD_CDECL stdSemTryP (stdSem_t sem);
#endif



/*
 * Function        : Release semaphore.
 * Parameters      : sem (I) Semaphore to release.
 * Function Result :
 */
void STD_CDECL stdSemV (stdSem_t sem);


/*--------------------------------- Mutexes ---------------------------------*/

/*
 * Function        : Create new mutex.
 *                   When supported by the underlying operating system,
 *                   the mutex will be created in priority inheritance mode.
 * Parameters      :
 * Function Result : created mutex, or Nil in case of failure.
 */
stdMutex_t STD_CDECL stdMutexCreate (void);

/*
 * Function        : Delete mutex.
 * Parameters      : mutex  (I) Mutex to delete.
 * Function Result :
 */
void STD_CDECL stdMutexDelete (stdMutex_t mutex);



/*
 * Function        : Initialize mutex in preallocated memory.
 *                   When supported by the underlying operating system,
 *                   the mutex will be initialized in priority inheritance mode.
 * Parameters      : mutex  (O) Mutex space to initialize
 * Function Result : True iff. initialization succeeded.
 */
Bool STD_CDECL stdMutexInit (struct stdMutex *mutex);

/*
 * Function        : Delete mutex in preallocated memory.
 * Parameters      : mutex  (O) Mutex to delete.
 * Function Result :
 */
void STD_CDECL stdMutexTerm (struct stdMutex *mutex);



/*
 * Function        : Acquire mutex.
 * Parameters      : mutex   (I) Mutex to acquire.
 * Function Result :
 */
void STD_CDECL stdMutexEnter (stdMutex_t mutex);



/*
 * Function        : Release mutex.
 * Parameters      : mutex   (I) Mutex to release.
 * Function Result :
 */
void STD_CDECL stdMutexExit (stdMutex_t mutex);


/*---------------------------- Atomicity Providers --------------------------*/

/*
 * Function        : Create new atomicity provider.
 *                   This object has similar usage as mutexes, namely
 *                   locking critical regions, but with the restriction
 *                   that it cannot be used around code regions that
 *                   might block, or might unblock other threads.
 *                   Typically, the regions for which an atomic can be used
 *                   are updating counter variables, or push or pop a list.
 *                   Depending on the used kernel, such atomics might have
 *                   a cheaper implementation, or may not do anything at all.
 * Parameters      :
 * Function Result : created atomicity provider, or Nil in case of failure.
 */
stdAtomic_t STD_CDECL stdAtomicCreate (void);

/*
 * Function        : Delete atomicity provider.
 * Parameters      : atomic  (I) Atomicity provider to delete.
 * Function Result :
 */
void STD_CDECL stdAtomicDelete (stdAtomic_t atomic);



/*
 * Function        : Initialize atomicity provider in preallocated memory.
 *                   When supported by the underlying operating system,
 *                   the atomicity provider will be initialized in priority inheritance mode.
 * Parameters      : atomic  (O) Atomicity provider space to initialize
 * Function Result : True iff. initialization succeeded.
 */
Bool STD_CDECL stdAtomicInit (struct stdAtomic *atomic);

/*
 * Function        : Delete atomicity provider in preallocated memory.
 * Parameters      : atomic  (O) Atomicity provider to delete.
 * Function Result :
 */
void STD_CDECL stdAtomicTerm (struct stdAtomic *atomic);



/*
 * Function        : Acquire atomicity provider.
 * Parameters      : atomic   (I) Atomicity provider to acquire.
 * Function Result :
 */
void STD_CDECL stdAtomicEnter (stdAtomic_t atomic);



/*
 * Function        : Release atomicity provider.
 * Parameters      : atomic   (I) Atomicity provider to release.
 * Function Result :
 */
void STD_CDECL stdAtomicExit (stdAtomic_t atomic);


/*---------------------------------- Time -----------------------------------*/

/*
 * Function        : Return current time in nanoseconds since unspecified time origin.
 * Parameters      :
 * Function Result : Current time.
 */

stdTime_t STD_CDECL stdTimeNow (void);


/*----------------------------- Condition Variables -------------------------*/

/*
 * Function        : Create new condition variable.
 * Parameters      :
 * Function Result : Created condition variable
 */
stdCondVar_t STD_CDECL stdCondCreate(void);


/*
 * Function        : Delete condition variable.
 * Parameters      : condition  (I) Condition variable to delete.
 * Function Result :
 */
void STD_CDECL stdCondDelete (stdCondVar_t condition);


/*
 * Function        : Acquire condition variable.
 * Parameters      : condition  (I) Condition variable to acquire.
 * Function Result :
 */
void STD_CDECL stdCondEnter (stdCondVar_t condition);


/*
 * Function        : Release condition variable.
 * Parameters      : condition  (I) Condition variable to release.
 * Function Result :
 */
void STD_CDECL stdCondExit (stdCondVar_t condition);


/*
 * Function        : Wait until condition variable broadcast or signalled.
 * Parameters      : condition  (I) Condition variable to wait on.
 * Function Result :
 */
void STD_CDECL stdCondWait (stdCondVar_t condition);


/*
 * Function        : Release all waiters on condition variable.
 * Parameters      : condition  (I) Condition variable to release.
 * Function Result :
 */
void STD_CDECL stdCondBroadcast (stdCondVar_t condition);



/*-------------------------------- Barriers ---------------------------------*/


/*
 * Function        : Create new barrier.
 * Parameters      : count      (I) Amount of threads that
 *                                  will use the barrier
 * Function Result : Created barrier
 */
stdBarrier_t STD_CDECL stdBarrierCreate( uInt count );


/*
 * Function        : Delete barrier.
 * Parameters      : barrier    (I) Barrier to delete.
 * Function Result :
 */
void STD_CDECL stdBarrierDelete ( stdBarrier_t barrier );


/*
 * Function        : Wait until N threads (including the one making
 *                   the call to this function) have arrived at the barrier,
 *                   where N is the count value by which the barrier
 *                   was created. After that, all N threads will be released.
 * Parameters      : barrier    (I) Barrier to wait on
 * Function Result :
 */
void STD_CDECL stdBarrierWait( stdBarrier_t barrier );


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
void STD_CDECL stdGlobalEnter(void);


/*
 * Function        : Exit process- global mutex.
 * Parameters      :
 * Function Result :
 */
void STD_CDECL stdGlobalExit(void);


/*----------------------------- Module Cleanup ------------------------------*/

/*
 * Explicit termination of threads library:
 * wait until all threads have terminated, and
 * then free all data structures.
 */
void STD_CDECL stdThreadsTerm(void);

/*-------------------------- Platform Specific ------------------------------*/

#if defined __KERNEL__

  struct stdAtomic { uInt32 flags; };
  #define stdAtomicCreate          _stdAtomicCreate
  #define stdAtomicInit(a)          (a)->flags= 0;
  #define stdAtomicTerm(a)
  #define stdAtomicEnter(a)         stdDisableInterrupts( (a)->flags )
  #define stdAtomicExit(a)          stdRestoreInterrupts( (a)->flags )

  stdAtomic_t _stdAtomicCreate();

  #define stdAtomicDelete(a) stdFREE(a)


#elif defined(USE_SIMPLE_KERNEL) && (USE_SIMPLE_KERNEL > 0)
 /*
  * The simple kernel in non-kernel mode
  * is non- premptive, user mode,
  * hence atomic services are not needed.
  */
  struct stdAtomic { uInt dummy; };
  #define stdAtomicCreate()         Nil
  #define stdAtomicDelete(a)
  #define stdAtomicInit(a)          True
  #define stdAtomicTerm(a)
  #define stdAtomicEnter(a)
  #define stdAtomicExit(a)


#else
 /*
  * When all else fails, then revert
  * to full mutexes.
  */
  #define stdAtomic         stdMutex
  #define stdAtomic_t       stdMutex_t
  #define stdAtomicCreate   stdMutexCreate
  #define stdAtomicDelete   stdMutexDelete
  #define stdAtomicInit     stdMutexInit
  #define stdAtomicTerm     stdMutexTerm
  #define stdAtomicEnter    stdMutexEnter
  #define stdAtomicExit     stdMutexExit
#endif


/*
struct stdMutex {
    Pointer  mutex;
};

 * This assumes that the underlying
 * kernel is either pthreads, or
 * simple_kernel (which provides a
 * pthread view dedicated for threadsPOSIX
 * implementation).
 */

#if (defined(STD_OS_FAMILY_Unix) || defined(STD_OS_Darwin) || defined(STD_OS_POSIX) || (defined(USE_SIMPLE_KERNEL) && USE_SIMPLE_KERNEL > 0))

struct stdSem {
#if defined(STD_OS_Darwin)
    semaphore_t      sem;
#else
    sem_t            sem;
#endif
};

struct stdMutex {
    pthread_mutex_t  mutex;
};

#else

struct stdSem {
    //CRITICAL_SECTION m_cs;     // used to arbitrate between waiters
    //HANDLE m_semaEvent;        // auto-reset event, used to signal a single waiter
    //volatile long m_semaCount; // count of resources

    Pointer           sem;
};

struct stdMutex {
    CRITICAL_SECTION  mutex;
};

#endif

#ifdef __cplusplus
}
#endif

#endif
