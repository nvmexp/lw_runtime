/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2016-2020, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : stdThreadsWin32.c
 *
 *  Description              :
 *
 */
/*------------------------------- Includes ----------------------------------*/

#include "stdThreads.h"
#include "stdLocal.h"
#include "stdList.h"
#include <windows.h>

#include "threadsMessageDefs.h"

/*--------------------------------- Types -----------------------------------*/

#define MIN_STACK_SIZE     100000

struct stdThread {
    stdThreadContext     context;
    stdThreadEntry_t     entry;
    Pointer              arg;
    Bool                 joinable;
    DWORD                dwThreadId;
    HANDLE               hThread;
    HANDLE               sleepMutex;
    HANDLE               suspendSem;
     
    HANDLE              *termSignal;
    
    stdThread_t          next,prev;
};

/*----------------------------- Global Module State -------------------------*/

static CRITICAL_SECTION  globalCriticalSection;
static DWORD             taskKey;

struct stdThread sentStart,sentEnd;

/*----------------------- Initialization/Termination ------------------------*/

    #define threadsInitialized()  (sentStart.next != Nil)
    
    #define assertInitialized() if (!threadsInitialized()) { stdThreadsInit(); }
    
    static void destroyThread( stdThread_t thread );
    
    

    Bool STD_CDECL stdThreadsInit(void)
    {
        if (!threadsInitialized()) {
            taskKey = TlsAlloc();

            InitializeCriticalSection(&globalCriticalSection);

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
    void STD_CDECL stdThreadsTerm(void)
    {
        if (threadsInitialized()) {
            HANDLE termSignal = CreateSemaphore(Nil, 0, 0xFFFFFFF, Nil);

            stdThreadDetach();

            stdGlobalEnter();

            while (sentStart.next != &sentEnd) {
                sentStart.next->termSignal = &termSignal;

                stdGlobalExit ();
                WaitForSingleObject(termSignal, INFINITE);
                stdGlobalEnter();
            }

            stdGlobalExit();

            CloseHandle(termSignal);
            DeleteCriticalSection(&globalCriticalSection);
            TlsFree (taskKey);

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

            DeleteCriticalSection(&globalCriticalSection);
            TlsFree (taskKey);

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
       
      /*
       * Bug: 200475130 - destruction of __stdThreads_initialized___, and thus the deletion of dynamic memory 
       * that it holds doesn't happen until after the call in the D3D driver to tear down the CRT. This is too 
       * late and causes it to appear that the driver leaks memory. So we add a function that the driver can call 
       * to free this memory just before doing the memory leak check.
       *  
       */
            
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

    
    void destroyStdThreads() 
    {
        ___stdThreads_initialized___.Terminate();
    }

/*-------------------------- Local Functions --------------------------------*/

static stdThread_t allocateThread()
{
    stdThread_t   result;

    assertInitialized();

   /* don't use stdNEW for this, 
    * because that macro needs the current thread
    * context, which we are creating here:
    */
    result= (stdThread_t)malloc(sizeof *result); 

    stdCHECK( result, (threadsMsgMemoryOverFlow) );
    
    stdMEMCLEAR(result);

    if (result)  { 
        result->sleepMutex = CreateWaitableTimer(Nil, FALSE, Nil);
        result->suspendSem = CreateSemaphore(Nil, 0, 0xFFFFFFF, Nil);

        stdGlobalEnter();
        
            result->next=  sentStart.next;
            result->prev= &sentStart;

            sentStart.next->prev      = result;
            sentStart.next            = result;
                        
        stdGlobalExit();
    }

    return result;
}


static void destroyThread( stdThread_t thread )
{
    if (thread) {
        HANDLE *termSignal;

        stdGlobalEnter();

            termSignal        = thread->termSignal;

            thread->next->prev= thread->prev;
            thread->prev->next= thread->next;

        stdGlobalExit();

        CloseHandle( thread->sleepMutex );
        CloseHandle( thread->suspendSem );

        if (!thread->joinable) {
            free (thread);
        }

        if (termSignal) {
            ReleaseSemaphore(*termSignal, 1, Nil);
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
void STD_CDECL stdThreadDetach(void) 
{
    if (threadsInitialized()) {
        stdThread_t self= (stdThread_t)TlsGetValue (taskKey);

        if (self) {
            destroyThread(self);
            TlsSetValue(taskKey,Nil);
        }
    }
}

/*---------------------------- General Functions ----------------------------*/

    static inline stdThread_t lwrrentThread()
    {
        stdThread_t result = threadsInitialized() ? (stdThread_t)TlsGetValue (taskKey) : Nil;

        if (!result) {
            result= allocateThread();
            TlsSetValue (taskKey, result);
        }

        return result;
    }


/*
 * Function        : Get exelwtion context of current thread.
 * Parameters      : 
 * Function Result : Requested context
 */
stdThreadContext_t STD_CDECL stdGetThreadContext()
{
    return &lwrrentThread()->context;
}



/*
 * Function        : Obtain handle to current thread.
 * Parameters      : 
 * Function Result : Lwrrently exelwting thread.
 */
stdThread_t STD_CDECL stdThreadSelf()
{
    return lwrrentThread();
}



/*
 * Function        : This function terminates the current process, including all threads
 *                   that are part of the process.
 * Parameters      : status (I) Return status.
 * Function Result : 
 */
void STD_CDECL stdExit (Int status)
{
    stdEXIT(status);
}

/*--------------------------------- Threads ---------------------------------*/

    static DWORD WINAPI wrapper(LPVOID arg)
    {
        stdThread_t thread = (stdThread_t)arg;

        TlsSetValue(taskKey, thread);

        thread->entry (thread->arg);

        stdThreadExit();

        return 0;
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
stdThread_t STD_CDECL _stdThreadCreate ( stdThreadEntry_t entry, Pointer arg,
                               Int prio, Int stackSize, Bool joinable, cString name)
{
    stdThread_t result;

    result = allocateThread();
    if (!result) { return Nil; }

    result->entry    = entry;
    result->arg      = arg;
    result->joinable = joinable;

    result->hThread = CreateThread( 
        Nil,                        // default security attributes 
        stackSize,                   // use default stack size  
        wrapper,                     // thread function 
        result,                      // argument to thread function 
        0,                           // use default creation flags 
        &result->dwThreadId);        // returns the thread identifier 

    if (result->hThread == Nil) {
        destroyThread(result);
        return (Nil);
    }

    return (result);
}


/*
 * Function        : Stop exelwtion of current thread, and delete it.
 * Parameters      :
 * Function Result : 
 */
void STD_CDECL stdThreadExit ()
{
    destroyThread( stdThreadSelf() );

    ExitThread(0);
}



/*
 * Function        : Delay exelwtion of current thread for specified duration.
 * Parameters      : nanosecs  (I) Minimum amount of time to sleep (in nanoseconds).
 * Function Result : 
 */
void STD_CDECL stdThreadSleep (stdTime_t nanosecs)
{
    stdThread_t self      = stdThreadSelf();
    uInt64      millisecs = nanosecs/1000000;
    
    while (millisecs) {
        uInt64 delta= stdMIN( 0x10000000, millisecs );
        
        WaitForSingleObject(self->sleepMutex, delta);
        millisecs -= delta;
    }
}



/*
 * Function        : Suspend lwrrently exelwted thread.
 * Parameters      : 
 * Function Result : 
 */
void STD_CDECL stdThreadSuspend ()
{
    stdThread_t  self= stdThreadSelf();
    WaitForSingleObject(self->suspendSem, INFINITE);
}



/*
 * Function        : Resume specified thread.
 * Parameters      : thread (I) Thread to resume
 * Function Result : 
 */
void STD_CDECL stdThreadResume ( stdThread_t thread )
{
    ReleaseSemaphore(thread->suspendSem, 1, Nil);
}


/*
 * Function        : Kill the thread
 * Parameters      : thread (I) Thread to be killed
 *                   sig Signal to be sent to thread (ignored on Windows)
 * Function Result :
 */
void STD_CDECL stdThreadKill( stdThread_t thread , int sig )
{
    TerminateThread(thread->hThread, 0);
}


/*
 * Function        : Wait for a thread to finish
 * Parameters      : thread (I) Thread to wait for
 * Function Result : 
 */
void STD_CDECL stdThreadJoin( stdThread_t thread )
{
    stdASSERT(thread->joinable, ("Joining non-joinable thread"));
    WaitForSingleObject(thread->hThread, INFINITE);
    free(thread); // thread completion cleans up the rest of the structure
}

uInt STD_CDECL stdNrofThreadPriorities()
{
    return 0;
}

/*------------------------------- Semaphores --------------------------------*/

/*
 * Function        : Create new semaphore
 * Parameters      : count     (I) Initial semaphore count.
 * Function Result : Created Semaphore, or Nil in case of failure.
 */
stdSem_t STD_CDECL stdSemCreate (Int count)
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
void STD_CDECL stdSemDelete (stdSem_t sem)
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
Bool STD_CDECL stdSemInit (struct stdSem *sem, Int count)
{
    //InitializeCriticalSection(&m_cs);
    //m_semaEvent = CreateEventA(NULL, /* bManualReset */ FALSE, /* bInitialState */ FALSE, NULL);
    //m_semaCount = initialCount;

    sem->sem = (Pointer)CreateSemaphore(Nil, count, 0xFFFFFFF, Nil);
    if (sem->sem == Nil) { return (False); }

    return (True);
}

/*
 * Function        : Delete semaphore in preallocated memory.
 * Parameters      : sem  (O Semaphore to delete.
 * Function Result : 
 */
void STD_CDECL stdSemTerm (struct stdSem *sem)
{
    CloseHandle((HANDLE)sem->sem);
}



/*
 * Function        : Acquire semaphore or wait.
 *                   If the semaphore's count is lwrrently equal to zero,
 *                   then wait until this count increases.
 * Parameters      : sem (I) semaphore to acquire
 * Function Result : 
 */
void STD_CDECL stdSemP (stdSem_t sem)
{
    //EnterCriticalSection(&m_cs);
    // at most one waiter enters here
    //long newSemaCount = _InterlockedCompareExchangeAdd(&m_semaCount, -1) - 1;
    //if (newSemaCount < 0){ WaitForSingleObject(m_semaEvent, INFINITE); }
    //LeaveCriticalSection(&m_cs);

    WaitForSingleObject((HANDLE)sem->sem, INFINITE);
}



/*
 * Function        : Acquire semaphor or fail.
 *                   Try to acquire a semaphore, and return
 *                   immediate failure if the semaphore's
 *                   count is lwrrently equal to zero.
 * Parameters      : sem (I) Semaphore to acquire.
 * Function Result : True iff. semaphore could be aquired
 */
Bool STD_CDECL stdSemTryP (stdSem_t sem)
{
    //long prevCount = _InterlockedExchangeAdd(&m_semaCount, delta);
    //if (prevCount < 0) { SetEvent(m_semaEvent); }    
    DWORD ret = WaitForSingleObject((HANDLE)sem->sem, 0);
    if (ret == WAIT_OBJECT_0) {
        return (True);
    }
    return (False);
}



/*
 * Function        : Release semaphore.
 * Parameters      : sem (I) Semaphore to release.
 * Function Result : 
 */
void STD_CDECL stdSemV (stdSem_t sem)
{
    ReleaseSemaphore((HANDLE)sem->sem, 1, Nil);
}


/*--------------------------------- Mutexes ---------------------------------*/


/*
 * Function        : Create new mutex.
 *                   When supported by the underlying operating system,
 *                   the mutex will be created in priority inheritance mode.
 * Parameters      : 
 * Function Result : created mutex, or Nil in case of failure.
 */
stdMutex_t STD_CDECL stdMutexCreate ()
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
void STD_CDECL stdMutexDelete (stdMutex_t mutex)
{
    stdMutexTerm(mutex);
    stdFREE(mutex);
}



/*
 * Function        : Initialize mutex in preallocated memory.
 *                   When supported by the underlying operating system,
 *                   the mutex will be initialized in priority inheritance mode.
 * Parameters      : mutex  (O) Mutex space to initialize
 * Function Result : True iff. initialization succeeded.
 */
Bool STD_CDECL stdMutexInit (struct stdMutex *mutex)
{
    InitializeCriticalSection(&mutex->mutex);
    return (True);
}

/*
 * Function        : Delete mutex in preallocated memory.
 * Parameters      : mutex  (O) Mutex to delete.
 * Function Result : 
 */
void STD_CDECL stdMutexTerm (struct stdMutex *mutex)
{
    DeleteCriticalSection(&mutex->mutex);
}



/*
 * Function        : Acquire mutex.
 * Parameters      : mutex   (I) Mutex to acquire.
 * Function Result : 
 */
void STD_CDECL stdMutexEnter (stdMutex_t mutex)
{
    EnterCriticalSection(&mutex->mutex);
}



/*
 * Function        : Release mutex.
 * Parameters      : mutex   (I) Mutex to release.
 * Function Result : 
 */
void STD_CDECL stdMutexExit (stdMutex_t mutex)
{
    LeaveCriticalSection(&mutex->mutex);
}


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
void STD_CDECL stdGlobalEnter()
{
    assertInitialized();

    EnterCriticalSection(&globalCriticalSection);
}


/*
 * Function        : Exit process- global mutex.
 * Parameters      : 
 * Function Result : 
 */
void STD_CDECL stdGlobalExit()
{
    LeaveCriticalSection(&globalCriticalSection);
}


/*---------------------------------- Time -----------------------------------*/

/*
 * Function        : Obtain current time
 *                   Return current time in nanoseconds since unspecified time origin.
 * Parameters      :
 * Function Result : Current time
 */
uInt64 STD_CDECL stdTimeNow (void)
{
    //SYSTEMTIME ns;
    //FILETIME   nf;
    //GetSystemTime(&ns);
    //SystemTimeToFileTime(&ns, &nf);
    //GetSystemTimePreciseAsFileTime(&nf); 
    //printf("low: %ld, ", nf.dwLowDateTime);
    //printf("high: %ld.\n", nf.dwHighDateTime);


    //return ((uInt64) nf.dwLowDateTime);


    LARGE_INTEGER t;
    LARGE_INTEGER f;

    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (t.QuadPart / (f.QuadPart / 1e9));
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
stdCondVar_t STD_CDECL stdCondCreate ()
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
void STD_CDECL stdCondDelete (stdCondVar_t condition)
{
    stdMutexTerm (&condition->mutex);
    
    stdFREE(condition); 
}



/* 
 * Function        : Acquire condition variable.
 * Parameters      : condition   (I) Condition variable to acquire.
 * Function Result :
 */        
void STD_CDECL stdCondEnter (stdCondVar_t condition)
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
void STD_CDECL stdCondExit (stdCondVar_t condition)
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
void STD_CDECL stdCondWait (stdCondVar_t condition)
{
    stdListRec  q;
    stdThread_t self= stdThreadSelf();

    stdASSERT( self == condition->owner, ("stdCondExit: not owner"       ) );
    stdASSERT( condition->count == 1,    ("stdCondWait: held relwrsively") );

    q.head = self;
    q.tail = condition->waitQueue;
    condition->waitQueue= &q;

    stdCondExit( condition );

    WaitForSingleObject(self->suspendSem, INFINITE);
}



/* 
 * Function        : Release all waiters on condition variable.
 * Parameters      : condition   (I) Condition to release.
 * Function Result :
 */        
void STD_CDECL stdCondBroadcast (stdCondVar_t condition)
{
    stdList_t waiters= Nil;

    stdASSERT( stdThreadSelf() == condition->owner, ("stdCondExit: not owner") );

    stdSWAP(waiters, condition->waitQueue, stdList_t);

    while (waiters) {
        stdThread_t waiter= (stdThread_t)waiters->head;
        waiters= waiters->tail;
        ReleaseSemaphore(waiter->suspendSem, 1, Nil);
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
stdBarrier_t STD_CDECL stdBarrierCreate( uInt count )
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
void STD_CDECL stdBarrierDelete ( stdBarrier_t barrier )
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
void STD_CDECL stdBarrierWait( stdBarrier_t barrier )
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

