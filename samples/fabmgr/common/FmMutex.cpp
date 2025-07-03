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
 #include "FmMutex.h"
#include "fm_log.h"
#include "timelib.h"

/*****************************************************************************/
FmMutex::FmMutex(int timeoutMs)
{
    int st;

    m_debugLogging = false;
    m_timeoutUsec = (long long)timeoutMs * 1000;
    memset(&m_locker, 0, sizeof(m_locker));
    m_locker.ownerTid = FM_MUTEX_TID_NULL;
    m_lockCount = 0;

    st = pthread_mutex_init(&m_pthreadMutex, NULL);
    if(st != 0)
    {
        FM_LOG_ERROR("mutex initialization failed with error %d\n", st);
        return;
    }
    
    m_handleInit= 1;

    if(m_debugLogging)
        FM_LOG_DEBUG("Mutex %p allocated", this);
}

/*****************************************************************************/
FmMutex::~FmMutex(void)
{
    int st;

    st = pthread_mutex_destroy(&m_pthreadMutex);
    if(st)
    {
        FM_LOG_ERROR("mutex destroy failed with error %d", st);
    }
    m_handleInit = 0;

    if(m_debugLogging)
        FM_LOG_DEBUG("Mutex %p destroyed", this);
}

/*****************************************************************************/
FmMutexReturn_t FmMutex::Poll(void)
{
    LWOSthreadId myTid;

    myTid = lwosGetLwrrentThreadId();

    /* Locked mutex? */
    if(m_locker.ownerTid == myTid)
        return FM_MUTEX_ST_LOCKEDBYME;
    else if(m_locker.ownerTid != FM_MUTEX_TID_NULL)
        return FM_MUTEX_ST_LOCKEDBYOTHER;

    return FM_MUTEX_ST_NOTLOCKED;
}

/*****************************************************************************/
FmMutexReturn_t FmMutex::Unlock(const char *file, int line)
{
    LWOSthreadId myTid;

    myTid = lwosGetLwrrentThreadId();

    if(m_locker.ownerTid == FM_MUTEX_TID_NULL)
    {
        FM_LOG_ERROR("unlock mutex failed as provided mutex object is not in locked state\n");
        return FM_MUTEX_ST_NOTLOCKED;
    }
    else if(m_locker.ownerTid != myTid)
    {
        FM_LOG_ERROR("unlock mutex failed as provided mutex object is locked by different thread\n");
        return FM_MUTEX_ST_LOCKEDBYOTHER;
    }

    /* Clear locker info */
    memset(&m_locker, 0, sizeof(m_locker));

    int st = pthread_mutex_unlock(&m_pthreadMutex);
    if(st != 0)
    {
        FM_LOG_ERROR("unlock mutex request failed with error %d for mutex %p\n", st, this);
        return FM_MUTEX_ST_ERROR;
    }

    if(m_debugLogging)
    {
        FM_LOG_DEBUG("Mutex %p unlocked by tid %lld from %s[%d]", 
                    this, (long long)myTid, file, line);
    }
    return FM_MUTEX_ST_OK;
}

/*****************************************************************************/
FmMutexReturn_t FmMutex::Lock(int complainMe, const char *file, int line)
{
    LWOSthreadId myTid;
    FmMutexReturn_t retSt = FM_MUTEX_ST_OK;
    timelib64_t diff;
    int st;

    myTid = lwosGetLwrrentThreadId();

    if(m_locker.ownerTid == myTid)
    {
        if(complainMe)
        {
            FM_LOG_ERROR("lock mutex request failed as provided mutex object is already in locked state\n");
        }
        return FM_MUTEX_ST_LOCKEDBYME;
    }

    /* Try and get the lock */
    if(!m_timeoutUsec)
    {
        /* No timeout. Just lock it */
        st = pthread_mutex_lock(&m_pthreadMutex);
        if(st)
        {
            FM_LOG_ERROR("mutex unlock request failed with error %d", st);
            return FM_MUTEX_ST_ERROR;
        }

        /* Got lock */
        retSt = FM_MUTEX_ST_OK;
    }
    else
    {
        /* Lock with a timeout */
        int st;
        timespec ts;
        clock_gettime(CLOCK_REALTIME , &ts);
        ts.tv_sec += m_timeoutUsec / 1000000;
        ts.tv_nsec += (m_timeoutUsec % 1000000) * 1000;
        if(ts.tv_nsec >= 1000000000)
        {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000;
        }
        st = pthread_mutex_timedlock(&m_pthreadMutex, &ts);
        switch(st)
        {
            case 0:
                retSt = FM_MUTEX_ST_OK;
                break;

            case ETIMEDOUT:
                retSt = FM_MUTEX_ST_TIMEOUT;
                break;

            default:
                FM_LOG_ERROR("got unexpected status %d from mutex timed lock, remaining timeout %lld\n",
                            st, m_timeoutUsec);
                return FM_MUTEX_ST_ERROR;
        }
    }

    /* Handle the mutex statuses */
    switch(retSt)
    {
        case FM_MUTEX_ST_OK:
            break; /* Keep going */
        
        case FM_MUTEX_ST_TIMEOUT:
        {
            timelib64_t now = timelib_usecSince1970();
            diff = now - m_locker.whenLockedUsec;
            FM_LOG_DEBUG("Mutex timeout by tid %lld %s[%d] owned by tid "
                        "%lld %s[%d] for %lld usec\n", 
                        (long long)myTid, file, line, (long long)m_locker.ownerTid, m_locker.file,
                         m_locker.line, (long long)diff);
            return retSt;
        }
        
        default:
            FM_LOG_ERROR("got unexpected status %d\n for mutex timed lock", (int)retSt);
            return retSt;
    }

    /* Finally got lock Populate info */
    m_locker.ownerTid = myTid;
    m_locker.line = line;
    m_locker.file = file;
    m_lockCount++;
    /* Only spend time retrieving the timestamp if we allow lock timeouts */
    if(m_timeoutUsec)
        m_locker.whenLockedUsec = timelib_usecSince1970();

    if(m_debugLogging)
    {
        FM_LOG_DEBUG("Mutex %p locked by tid %lld %s[%d] lockCount %lld\n", 
                    this, m_locker.ownerTid, m_locker.file, m_locker.line, m_lockCount);
    }

    return FM_MUTEX_ST_OK;
}

/*****************************************************************************/
void FmMutex::EnableDebugLogging(bool enabled)
{
    m_debugLogging = enabled;
}

/*****************************************************************************/
FmMutexReturn_t FmMutex::CondWait(lwosCV *cv, unsigned int timeoutMs)
{
    FmMutexReturn_t lockSt, retSt;
    fm_mutex_locker_t backupLocker;

    /* Make sure the mutex is actually locked by us. This also checks the mutex
       status */
    lockSt = Lock(0, __FILE__, __LINE__);
    if(lockSt != FM_MUTEX_ST_OK && lockSt != FM_MUTEX_ST_LOCKEDBYME)
    {
        FM_LOG_ERROR("conditional wait on mutex %p returned unexpected status %d", 
                    this, lockSt);
        return FM_MUTEX_ST_ERROR;
    }

    /* Back up the owner and clear it since lwosCondWait will unlock the mutex */
    backupLocker = m_locker;
    memset(&m_locker, 0, sizeof(m_locker));
    m_locker.ownerTid = FM_MUTEX_TID_NULL;

    if(!timeoutMs)
        timeoutMs = LWOS_INFINITE_TIMEOUT;

    int st = lwosCondWait(cv, (LWOSCriticalSection *)&m_pthreadMutex, timeoutMs);

    /* We now have the lock again. Restore the locker info */
    m_locker = backupLocker;
    m_lockCount++; /* We technically unlocked and locked again */

    switch(st)
    {
        case LWOS_SUCCESS:
            retSt = FM_MUTEX_ST_OK;
            break;

        case LWOS_TIMEOUT:
            retSt = FM_MUTEX_ST_TIMEOUT;
            break;
        
        default:
            retSt = FM_MUTEX_ST_ERROR;
            break;
    }

    if(m_debugLogging)
    {
        FM_LOG_DEBUG("CondWait finished on mutex %p. retSt %d, st %d", this, retSt, st);
    }

    return retSt;
}

#if (LWOSCriticalSection != pthread_mutex_t)
#error "This function assumes that LWOSCriticalSection == pthread_mutex_t"
#endif

/*****************************************************************************/
long long FmMutex::GetLockCount(void)
{
    return m_lockCount;
}

/*****************************************************************************/

