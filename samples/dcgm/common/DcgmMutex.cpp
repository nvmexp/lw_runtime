#include "DcgmMutex.h"
#include "logging.h"
#include "timelib.h"

/*****************************************************************************/
DcgmMutex::DcgmMutex(int timeoutMs)
{
    int st;

    m_debugLogging = false;
    m_timeoutUsec = (long long)timeoutMs * 1000;
    memset(&m_locker, 0, sizeof(m_locker));
    m_locker.ownerTid = DCGM_MUTEX_TID_NULL;
    m_lockCount = 0;

    st = pthread_mutex_init(&m_pthreadMutex, NULL);
    if(st != 0)
    {
        PRINT_ERROR("%d", "pthread_mutex_init returned %d\n", st);
        return;
    }
    
    m_handleInit= 1;

    if(m_debugLogging)
        PRINT_DEBUG("%p", "Mutex %p allocated", this);
}

/*****************************************************************************/
DcgmMutex::~DcgmMutex(void)
{
    int st;

    st = pthread_mutex_destroy(&m_pthreadMutex);
    if(st)
    {
        PRINT_ERROR("%d", "pthread_mutex_destroy returned %d", st);
    }
    m_handleInit = 0;

    if(m_debugLogging)
        PRINT_DEBUG("%p", "Mutex %p destroyed", this);
}

/*****************************************************************************/
dcgmMutexReturn_t DcgmMutex::Poll(void)
{
    LWOSthreadId myTid;

    myTid = lwosGetLwrrentThreadId();

    /* Locked mutex? */
    if(m_locker.ownerTid == myTid)
        return DCGM_MUTEX_ST_LOCKEDBYME;
    else if(m_locker.ownerTid != DCGM_MUTEX_TID_NULL)
        return DCGM_MUTEX_ST_LOCKEDBYOTHER;

    return DCGM_MUTEX_ST_NOTLOCKED;
}

/*****************************************************************************/
dcgmMutexReturn_t DcgmMutex::Unlock(const char *file, int line)
{
    LWOSthreadId myTid;

    myTid = lwosGetLwrrentThreadId();

    if(m_locker.ownerTid == DCGM_MUTEX_TID_NULL)
    {
        PRINT_ERROR("%s %d", "%s[%d] passed in an unlocked mutex to Unlock\n", 
                    file, line);
        return DCGM_MUTEX_ST_NOTLOCKED;
    }
    else if(m_locker.ownerTid != myTid)
    {
        PRINT_ERROR("%s %d %lld %s %d", "%s[%d] passed in locked by tid %lld %s[%d]\n", 
                    file, line, (long long)m_locker.ownerTid, m_locker.file, m_locker.line);
        return DCGM_MUTEX_ST_LOCKEDBYOTHER;
    }

    /* Clear locker info */
    memset(&m_locker, 0, sizeof(m_locker));

    int st = pthread_mutex_unlock(&m_pthreadMutex);
    if(st != 0)
    {
        PRINT_ERROR("%d %p", "pthread_mutex_unlock returned %d for mutex %p\n", st, this);
        return DCGM_MUTEX_ST_ERROR;
    }

    if(m_debugLogging)
    {
        PRINT_DEBUG("%p %lld %s %d", "Mutex %p unlocked by tid %lld from %s[%d]", 
                    this, (long long)myTid, file, line);
    }
    return DCGM_MUTEX_ST_OK;
}

/*****************************************************************************/
dcgmMutexReturn_t DcgmMutex::Lock(int complainMe, const char *file, int line)
{
    LWOSthreadId myTid;
    dcgmMutexReturn_t retSt = DCGM_MUTEX_ST_OK;
    timelib64_t diff;
    int st;

    myTid = lwosGetLwrrentThreadId();

    if(m_locker.ownerTid == myTid)
    {
        if(complainMe)
        {
            PRINT_ERROR("%s %d %s %d", "%s[%d] mutex already locked by me from %s[%d]\n", 
                        file, line, m_locker.file, m_locker.line);
        }
        return DCGM_MUTEX_ST_LOCKEDBYME;
    }

    /* Try and get the lock */
    if(!m_timeoutUsec)
    {
        /* No timeout. Just lock it */
        st = pthread_mutex_lock(&m_pthreadMutex);
        if(st)
        {
            PRINT_ERROR("%d", "pthread_mutex_lock returned %d", st);
            return DCGM_MUTEX_ST_ERROR;
        }

        /* Got lock */
        retSt = DCGM_MUTEX_ST_OK;
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
                retSt = DCGM_MUTEX_ST_OK;
                break;

            case ETIMEDOUT:
                retSt = DCGM_MUTEX_ST_TIMEOUT;
                break;

            default:
                PRINT_ERROR("%d %lld", "Got unexpected st %d from pthread_mutex_timedlock() of timeout %lld\n",
                            st, m_timeoutUsec);
                return DCGM_MUTEX_ST_ERROR;
        }
    }

    /* Handle the mutex statuses */
    switch(retSt)
    {
        case DCGM_MUTEX_ST_OK:
            break; /* Keep going */
        
        case DCGM_MUTEX_ST_TIMEOUT:
        {
            timelib64_t now = timelib_usecSince1970();
            diff = now - m_locker.whenLockedUsec;
            PRINT_ERROR("%lld %s %d %lld %s %d %lld", "Mutex timeout by tid %lld %s[%d] owned by tid "
                        "%lld %s[%d] for %lld usec\n", 
                        (long long)myTid, file, line, (long long)m_locker.ownerTid, m_locker.file,
                         m_locker.line, (long long)diff);
            return retSt;
        }
        
        default:
            PRINT_ERROR("%d", "Unexpected retSt %d\n", (int)retSt);
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
        PRINT_DEBUG("%p %lld %s %d %lld", "Mutex %p locked by tid %lld %s[%d] lockCount %lld\n", 
                    this, m_locker.ownerTid, m_locker.file, m_locker.line, m_lockCount);
    }

    return DCGM_MUTEX_ST_OK;
}

/*****************************************************************************/
void DcgmMutex::EnableDebugLogging(bool enabled)
{
    m_debugLogging = enabled;
}

/*****************************************************************************/
dcgmMutexReturn_t DcgmMutex::CondWait(lwosCV *cv, unsigned int timeoutMs)
{
    dcgmMutexReturn_t lockSt, retSt;
    dcgm_mutex_locker_t backupLocker;

    /* Make sure the mutex is actually locked by us. This also checks the mutex
       status */
    lockSt = Lock(0, __FILE__, __LINE__);
    if(lockSt != DCGM_MUTEX_ST_OK && lockSt != DCGM_MUTEX_ST_LOCKEDBYME)
    {
        PRINT_ERROR("%p %d", "CondWait of mutex %p call to Lock() returned unexpected %d", 
                    this, lockSt);
        return DCGM_MUTEX_ST_ERROR;
    }

    /* Back up the owner and clear it since lwosCondWait will unlock the mutex */
    backupLocker = m_locker;
    memset(&m_locker, 0, sizeof(m_locker));
    m_locker.ownerTid = DCGM_MUTEX_TID_NULL;

    if(!timeoutMs)
        timeoutMs = LWOS_INFINITE_TIMEOUT;

    int st = lwosCondWait(cv, (LWOSCriticalSection *)&m_pthreadMutex, timeoutMs);

    /* We now have the lock again. Restore the locker info */
    m_locker = backupLocker;
    m_lockCount++; /* We technically unlocked and locked again */

    switch(st)
    {
        case LWOS_SUCCESS:
            retSt = DCGM_MUTEX_ST_OK;
            break;

        case LWOS_TIMEOUT:
            retSt = DCGM_MUTEX_ST_TIMEOUT;
            break;
        
        default:
            retSt = DCGM_MUTEX_ST_ERROR;
            break;
    }

    if(m_debugLogging)
    {
        PRINT_DEBUG("%p %d %d", "CondWait finished on mutex %p. retSt %d, st %d", this, retSt, st);
    }

    return retSt;
}

#if (LWOSCriticalSection != pthread_mutex_t)
#error "This function assumes that LWOSCriticalSection == pthread_mutex_t"
#endif

/*****************************************************************************/
long long DcgmMutex::GetLockCount(void)
{
    return m_lockCount;
}

/*****************************************************************************/

