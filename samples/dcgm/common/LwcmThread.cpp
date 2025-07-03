

#include "LwcmThread.h"
#include <stdio.h>
#include "logging.h"

#ifdef __linux__
#include <unistd.h> //usleep()
#include <sys/syscall.h> //syscall()
#endif

/*****************************************************************************/
/*
    Helper static function to pass to CreateThread
*/
#ifndef __linux__
DWORD dcgmthread_starter(void *parm)
#else
void *dcgmthread_starter(void *parm)
#endif
{
    LwcmThread *dcgmThread = (LwcmThread *)parm;
    dcgmThread->RunInternal();
    return 0;
}

/*****************************************************************************/
LwcmThread::LwcmThread(bool sendSignalOnStop)
{
    m_sendSignalOnStop = sendSignalOnStop;
    resetStatusFlags();

#ifdef __linux__
    m_pthread = 0;
#else
    m_handle = NULL;
    m_threadId = 0;
#endif
}

/*****************************************************************************/
LwcmThread::~LwcmThread()
{
}

void LwcmThread::resetStatusFlags()
{
    m_shouldStop = 0;
    m_hasStarted = 0;
    m_hasRun = 0;
    m_hasExited = 0;

#ifdef __linux__
    m_alreadyJoined = false;
#endif
}

/*****************************************************************************/
int LwcmThread::Start()
#ifndef __linux__
{
    if(m_handle != NULL || m_threadId != 0)
    {
        PRINT_ERROR("%p %u", "Can't start thread. Already running as handle %p tid %u\n",
            (void *)m_handle, m_threadId);
        return -100;
    }

    m_handle = CreateThread(0, 0, (LPTHREAD_START_ROUTINE)dcgmthread_starter,
                            this, 0, &m_threadId);
    if(m_handle == ILWALID_HANDLE_VALUE)
    {
        DWORD gle = GetLastError();
        PRINT_ERROR("%u", "Unable to CreateThread. gle=%u\n", gle);
        return -200;
    }

    PRINT_INFO("%u %p", "Created thread ID %u handle %p\n", m_threadId, (void *)m_handle);

    return 0;
}
#else
{
    if(!m_hasExited)
    {
        if(m_hasRun)
        {
            PRINT_ERROR("%u", "Can't start thread. Already running as handle %u\n",
                        (unsigned int)m_pthread);
            return -100;
        }
        else if(m_hasStarted)
        {
            PRINT_ERROR("", "Can't start thread. Thread is already about to start running\n");
            return -101;
        }
    }

    /* Reset the status flags before we start the thread since the thread will set m_hasRun and
       may even do it before pthread_create returns */
    resetStatusFlags();

    int st = pthread_create(&m_pthread, 0, dcgmthread_starter,
                            this);
    if(st)
    {
        m_pthread = 0;
        PRINT_ERROR("%d", "Unable to pthread_create. errno=%d\n", st);
        return -200;
    }

    m_hasStarted = 1;

    PRINT_INFO("%u", "Created thread ID %u\n", (unsigned int)m_pthread);
    return 0;

}
#endif

/*****************************************************************************/
void LwcmThread::Stop()
{
    m_shouldStop = 1;

    /* Wake this thread up */
    if(m_hasStarted && m_hasRun && !m_hasExited && m_sendSignalOnStop)
        SendSignal(LWCM_THREAD_SIGNUM);
    
    OnStop();
}

/*****************************************************************************/
void LwcmThread::Kill()
#ifndef __linux__
{
    if(m_handle == ILWALID_HANDLE_VALUE || m_hasExited)
    {
        /* Nothing to do */
        return;
    }

    BOOL st = TerminateThread(m_handle, 0);

    PRINT_INFO("%d", "Terminated thread id %d\n", m_threadId);

    m_handle = ILWALID_HANDLE_VALUE;
    m_threadId = 0;
}
#else
{
    if(!m_hasStarted || m_hasExited)
    {
        /* Nothing to do */
        return;
    }

    int st = pthread_cancel(m_pthread);
    if(st == 0 || st == ESRCH)
        return; /* Thread terminated */

    PRINT_WARNING("%u %d", "pthread_cancel(%u) returned %d\n", (unsigned int)m_pthread, st);
}
#endif

/*****************************************************************************/
void LwcmThread::SendSignal(int signum)
{
    PRINT_DEBUG("%u %d", "Signalling thread %u with signum %d", 
                (unsigned int)m_pthread, signum);
    pthread_kill(m_pthread, signum);
}

/*****************************************************************************/
int LwcmThread::Wait(int timeoutMs)
#ifndef __linux__
{
    if(m_handle == ILWALID_HANDLE_VALUE || m_threadId == 0 || m_hasExited)
    {
        /* Already terminated */
        return 0;
    }

    DWORD millis = timeoutMs;
    if(timeoutMs == 0)
        millis = INFINITE;

    DWORD st = WaitForSingleObject(m_handle, millis);
    if(st == WAIT_TIMEOUT)
        return 1;
    else if(st == ERROR_SUCCESS)
        return 0;
    else if(st != WAIT_OBJECT_0)
        return 0;

    DWORD gle = GetLastError();
    PRINT_WARNING("%u %u", "Got unknown status code %u from WaitForSingleObject gle=%u\n",
                  st, gle);

    if(m_handle == ILWALID_HANDLE_VALUE || m_threadId == 0)
        return 0; /* I guess it stopped */
    else
        return 1;
}
#else
{
    void *retVal;

    // thread has not been started, therefore it cannot be running
    if (!m_hasStarted)
    {
        return 0;
    }

    /* Infinite timeout? */
    if(timeoutMs == 0)
    {
        /* Does this thread exist yet? */
        while(!m_hasRun)
        {
            usleep(10000);
        }

        /* Calling pthread_join a second time results in undefined behavior */
        if (m_alreadyJoined == false)
        {
            int st = pthread_join(m_pthread, &retVal);
            m_alreadyJoined = true;
            if(st)
            {
                PRINT_ERROR("%p %d", "pthread_join(%p) returned st %d\n", (void *)m_pthread, st);
                return 1;
            }

            return 0; /* Thread is gone */
        }
        return 0;
    }
    else
    {
        /* Does this thread exist yet? */
        while(!m_hasRun && timeoutMs > 0)
        {
            /* Sleep for small intervals until we exhaust our timeout */
            usleep(10000);
            timeoutMs -= 10;
        }

        if(timeoutMs < 0)
            return 1; /* Hasn't started yet. I guess it's running */

#if 1
        while(!m_hasExited && timeoutMs > 0)
        {
            usleep(10000);
            timeoutMs -= 10;
        }

        if(!m_hasExited)
            return 1; /* Running still */
        else
        {
            /* Calling pthread_join a second time results in undefined behavior */
            if (m_alreadyJoined == false)
            {
                int st = pthread_join(m_pthread, &retVal);
                m_alreadyJoined = true;
                if(st)
                    PRINT_ERROR("%p %d", "pthread_join(%p) returned st %d\n", (void *)m_pthread, st);
            }
            return 0;
        }

#else /* This code won't work until we upgrade to glibc 2.2.3 */
        timespec ts;
        clock_gettime(CLOCK_REALTIME , &ts);
        ts.tv_sec += timeoutMs / 1000;
        ts.tv_nsec += (timeoutMs % 1000) * 1000000;
        if(ts.tv_nsec >= 1000000000)
        {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000;
        }

        int st = pthread_timedjoin_np(m_pthread, &retVal, &ts);
        if(st == ETIMEDOUT)
            return 1;
        else if(!st)
            return 0; /* Thread was gone */

        PRINT_ERROR("%u %d", "pthread_timedjoin_np(%u) returned st %d\n", m_pthread, st);
        return 1;
#endif
    }
}
#endif

/*****************************************************************************/
int LwcmThread::StopAndWait(int timeoutMs)
{
    int st;

#ifndef __linux__
    if(m_handle == ILWALID_HANDLE_VALUE || m_threadId == 0)
#else
    if(!m_hasStarted)
#endif
    {
        /* Already terminated */
        return 0;
    }

    Stop();

    st = Wait(timeoutMs);
    return st;
}

/*****************************************************************************/
void LwcmThread::RunInternal(void)
#ifndef __linux__
{
    m_hasRun = 1;
    
    PRINT_DEBUG("%d", "Thread id %d running\n", m_threadId);
    run();
    PRINT_DEBUG("%d", "Thread id %d stopped\n", m_threadId);

    m_threadId = 0;
    m_hasExited = 1;
}
#else
{
    m_hasRun = 1;

    PRINT_DEBUG("%u", "Thread handle %u running\n", (unsigned int)m_pthread);
    run();
    PRINT_DEBUG("%u",  "Thread id %u stopped\n", (unsigned int)m_pthread);

    m_hasExited = 1;
}
#endif

/*****************************************************************************/
int LwcmThread::ShouldStop(void)
{
    return m_shouldStop; // || main_should_stop; /* If we later add a global variable for dcgm should stop, reference it here */
}

/*****************************************************************************/
unsigned long LwcmThread::GetTid()
{
#ifdef __linux__
    pid_t tid = syscall(SYS_gettid);
#else
    DWORD tid = GetLwrrentThreadId();
#endif

    return (unsigned long)tid;
}

/*****************************************************************************/
void LwcmThread::Sleep(long long howLongUsec)
{
    if(ShouldStop())
        return; /* Return immediately if we're supposed to shut down */
    if(howLongUsec < 0)
        return; /* Bad value */

#ifndef __linux__
    SleepEx((DWORD)howLongUsec / 1000, 1);
#else
    usleep(howLongUsec);
#endif
}

/*****************************************************************************/
int LwcmThread::HasRun()
{
    return m_hasRun;
}

/*****************************************************************************/
int LwcmThread::HasExited()
{
    return m_hasExited;
}

/*****************************************************************************/
static void lwcm_thread_signal_handler(int signum)
{
    /* Do nothing, especially not IO */
}

/*****************************************************************************/
void LwcmThread::InstallSignalHandler(void)
{
    struct sigaction lwrrentSigHandler;

    /* See if this process already has a signal handler */
    int ret = sigaction(LWCM_THREAD_SIGNUM, NULL, &lwrrentSigHandler);
    if(ret < 0)
    {
        PRINT_ERROR("%d", "Got st %d from sigaction", ret);
        return;
    }
    if(lwrrentSigHandler.sa_handler != SIG_DFL && lwrrentSigHandler.sa_handler != SIG_IGN)
    {
        PRINT_INFO("%d", "Signal %d is already handled. Nothing to do.",
                   LWCM_THREAD_SIGNUM);
        return;
    }

    /* Install our handler */
    struct sigaction newSigHandler;
    sigemptyset(&newSigHandler.sa_mask);
    newSigHandler.sa_flags = 0;
    newSigHandler.sa_handler = lwcm_thread_signal_handler;

    ret = sigaction(LWCM_THREAD_SIGNUM, &newSigHandler, NULL);
    if(ret < 0)
    {
        PRINT_ERROR("%d", "Got error %d from sigaction while adding our signal handler.",
                    ret);
    }
}

/*****************************************************************************/




