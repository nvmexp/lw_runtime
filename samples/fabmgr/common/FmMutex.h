/*
 *  Copyright 2018-2019 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
 #ifndef FmMutex_H
#define FmMutex_H

#include "lwos.h"

/* API Status codes */
typedef enum FmMutexSt
{
    FM_MUTEX_ST_OK	         =  0, /* OK */
    FM_MUTEX_ST_LOCKEDBYOTHER  =  1, /* Another thread has the lock */
    FM_MUTEX_ST_LOCKEDBYME     = -2, /* My thread already owns the lock */
    FM_MUTEX_ST_TIMEOUT        = -3, /* Tried to wait for the lock but didn't get it before our 
                                          timeout expired */
    FM_MUTEX_ST_NOTLOCKED      = -4, /* Mutex is lwrrently not locked */
    FM_MUTEX_ST_ERROR			 = -5  /* Generic, unspecified error */
} FmMutexReturn_t;

/* OS-dependent NULL Thread ID. This is a pthread_t for POSIX */
#define FM_MUTEX_TID_NULL ((pthread_t)0)

typedef struct fm_mutex_locker_t
{
    const char *file;	       /* Pointer to the filename that locked this. */
    int   line;			       /* Line of code in leaf this was locked from */
    int   unused;              /* padding to 8-byte alignment */
    long long whenLockedUsec;  /* usec since 1970 of when this sem was locked */
    LWOSthreadId ownerTid;     /* Thread id of the current locker. FM_MUTEX_TID_NULL if no one */
} fm_mutex_locker_t, *fm_mutex_locker_p;


/* FmMutex class. Instantiate this class to get a mutex */
class FmMutex
{
public:
    /*************************************************************************/
    /* Constructor
     *
     * timeoutMs IN: How long we should wait when locking this mutex before 
     *               giving up and returning FM_MUTEX_ST_TIMEOUT. 
     *               0 = never timeout. This is slightly faster because no timing
     *               information is recorded in this case
     *               
     */
    FmMutex(int timeoutMs);
    
    /*************************************************************************/
    /**
     * Destructor
     */
    ~FmMutex();

    /*************************************************************************/
    /**
     * Lock this mutex
     *
     * complainMe  IN: Whether or not to complain if the mutex is already locked
     *                 by my thread.
     * file        IN: Should be __FILE__ or some other heap-allocated pointer to
     *                 a source code line.
     * line        IN: Should be __LINE__
     *
     * RETURNS: FM_MUTEX_ST_OK if OK
     *          FM_MUTEX_ST_LOCKEDBYME if mutex was locked by me already
     *          FM_MUTEX_ST_? enum on error
     */
    FmMutexReturn_t Lock(int complainMe, const char *file, int line);

    /* Colwenience macros for using Lock() on a FmMutex pointer */
    #define fm_mutex_lock(m) m->Lock(1,__FILE__,__LINE__)
    #define fm_mutex_lock_me(m) m->Lock(0,__FILE__,__LINE__)

    /*************************************************************************/
    /**
     * Unlock this mutex
     *
     * file IN: Should be __FILE__ or some other heap-allocated pointer to
     *          a source code line.
     * line IN: Should be __LINE__
     *   
     *   RETURNS: 0 if OK
     *            FM_MUTEX_ST_? enum on error
     */
    FmMutexReturn_t Unlock(const char *file, int line);

    #define fm_mutex_lock(m) m->Unlock(__FILE__,__LINE__)
    
    /*************************************************************************/
    /*
     * Query the current state of this mutex
     *
     * RETURNS: FM_MUTEX_ST_? enum state of the mutex
    */
    FmMutexReturn_t Poll(void);

    /*************************************************************************/
    /* Enable or disable debug logging spew from this mutex
     *
     * Default is disabled
     *
     */
    void EnableDebugLogging(bool enabled);

    /*************************************************************************/
    /* Wait on a condition variable using this mutex as the underlying mutex
     *
     * Since pthread_cond_timedwait() unlocks and relocks the underlying mutex,
     * we must update internal locking state accordingly.
     *
     * The mutex will still be locked by the calling thread after this call.
     *
     * cv        IN: lwos condition to wait on
     * timeoutMs IN: How long to wait on this condition in ms. 
     *
     * RETURNS: FM_MUTEX_ST_OK if the condition was signalled. 
     *          FM_MUTEX_ST_TIMEOUT if timeoutMs elapsed before the condition 
                                      was signalled
     */
    FmMutexReturn_t CondWait(lwosCV *cv, unsigned int timeoutMs);

    /*************************************************************************/
    /*
     * Get the number of times this mutex has ever been locked. This is useful for
     * detecting places that lock a mutex too many times. Note that this value
     * only counts successful locks. Relwrsive locks where the mutex was already locked
     * don't increment this counter.
     *
     * RETURNS: Number of times this mutex was locked.
     */
    long long GetLockCount(void);

private:
    
    /*************************************************************************/

    /* OS Handle to the mutex */
    long long m_timeoutUsec; /* How long to wait in usec before timing out. 0=never timeout */
    int	m_handleInit;        /* Is handle/critSec is initialized? */
    bool m_debugLogging;     /* Should we log verbose debug logs? true=yes */
    long long m_lockCount;   /* Number of times this mutex has been locked. This doesn't count relwrsive locks */
#ifndef LW_UNIX
    #error "Windows not supported at this time"
#else //Linux
    pthread_mutex_t	m_pthreadMutex;
#endif

    fm_mutex_locker_t m_locker; /* Information about the locker of this mutex */

    /*************************************************************************/
};

/**
 * RAII style locking mechanism.  Meant to be similar to C++11 lock_guard.
 */
 class FmLockGuard
 {
 public:
    FmLockGuard(FmMutex *mutex)
    {
        m_mutex = mutex;
        /* Use relwrsive version of lock. The destructor will handle this properly */
        m_mutexReturn = fm_mutex_lock_me(m_mutex);
    }
 
    ~FmLockGuard()
    {
        if(m_mutexReturn == FM_MUTEX_ST_OK)
            fm_mutex_lock(m_mutex);
    }
 
private:
    FmMutex *m_mutex;
    FmMutexReturn_t m_mutexReturn;
 };

#endif //FmMutex_H

