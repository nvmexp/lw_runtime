/*
 *  Copyright 2018-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#include "FmThread.h"
#include <stdio.h>
#include <cstdint>
#include "fm_log.h"


/*****************************************************************************/
/*
    Helper static function to pass to CreateThread
*/
#ifndef __linux__
DWORD fmthread_starter(void *parm)
#else
int fmthread_starter(void *parm)
#endif
{
    FmThread *tempFmThread = (FmThread *)parm;
    tempFmThread->RunInternal();
    return 0;
}

/*****************************************************************************/
FmThread::FmThread(bool sendSignalOnStop)
{
    resetStatusFlags();

#ifdef __linux__
    m_pthread = NULL;
	m_sendSignalOnStop = sendSignalOnStop;
#else
    m_handle = NULL;
    m_threadId = 0;
#endif
}

/*****************************************************************************/
FmThread::~FmThread()
{
}

void FmThread::resetStatusFlags()
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
int FmThread::Start()
#ifndef __linux__
{
    if(m_handle != NULL || m_threadId != 0)
    {
        FM_LOG_ERROR("Can't start thread. Already running as handle %p tid %u\n",
            (void *)m_handle, m_threadId);
        return -100;
    }

    m_handle = CreateThread(0, 0, (LPTHREAD_START_ROUTINE)fmthread_starter,
                            this, 0, &m_threadId);
    if(m_handle == ILWALID_HANDLE_VALUE)
    {
        DWORD gle = GetLastError();
        FM_LOG_ERROR("Unable to CreateThread. gle=%u\n", gle);
        return -200;
    }

    FM_LOG_INFO("Created thread ID %u handle %p\n", m_threadId, (void *)m_handle);

    return 0;
}
#else
{
    if(!m_hasExited)
    {
        if(m_hasRun)
        {
            FM_LOG_ERROR("Can't start thread. Already running as handle %lu\n",
                        reinterpret_cast<std::uintptr_t>(m_pthread));
            return -100;
        }
        else if(m_hasStarted)
        {
            FM_LOG_ERROR("Can't start thread. Thread is already about to start running\n");
            return -101;
        }
    }

    /* Reset the status flags before we start the thread since the thread will set m_hasRun and
       may even do it before pthread_create returns */
    resetStatusFlags();

    int st = lwosThreadCreate(&m_pthread, fmthread_starter,
                            this);
    if(st)
    {
        m_pthread = 0;
        FM_LOG_ERROR("Unable to pthread_create. errno=%d\n", st);
        return -200;
    }

    m_hasStarted = 1;

    FM_LOG_DEBUG("Created thread ID %lu\n", reinterpret_cast<std::uintptr_t>(m_pthread));
    return 0;

}
#endif

/*****************************************************************************/
void FmThread::Stop()
{
    m_shouldStop = 1;


#if defined(__linux__) && !defined(LW_MODS) 
    /* Wake this thread up */
    if(m_hasStarted && m_hasRun && !m_hasExited && m_sendSignalOnStop)
        SendSignal(FM_THREAD_SIGNUM);
#endif
}

/*****************************************************************************/
void FmThread::Kill()
#ifndef __linux__
{
    if(m_handle == ILWALID_HANDLE_VALUE || m_hasExited)
    {
        /* Nothing to do */
        return;
    }

    BOOL st = TerminateThread(m_handle, 0);

    FM_LOG_INFO("Terminated thread id %d\n", m_threadId);

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

#ifndef LW_MODS
    int st = lwosThreadCancel(m_pthread);
    if(st == 0 || st == ESRCH)
        return; /* Thread terminated */

    FM_LOG_DEBUG("pthread_cancel(%lu) returned %d\n", reinterpret_cast<std::uintptr_t>(m_pthread), st);
#else
    FM_LOG_DEBUG("Thread kill unspported on MODS\n");
#endif
}
#endif

/*****************************************************************************/
void FmThread::SendSignal(int signum)
{
#if defined(__linux__) && !defined(LW_MODS) 
    FM_LOG_DEBUG("Signalling thread %lu with signum %d", 
                reinterpret_cast<std::uintptr_t>(m_pthread), signum);
    lwosThreadSignal(m_pthread, signum);
#endif
}

/*****************************************************************************/
int FmThread::Wait(int timeoutMs)
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
    FM_LOG_WARNING("Got unknown status code %u from WaitForSingleObject gle=%u\n",
                  st, gle);

    if(m_handle == ILWALID_HANDLE_VALUE || m_threadId == 0)
        return 0; /* I guess it stopped */
    else
        return 1;
}
#else
{
    int retVal;

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
            lwosSleep(10);
        }

        /* Calling pthread_join a second time results in undefined behavior */
        if (m_alreadyJoined == false)
        {
            lwosThreadJoin(m_pthread, &retVal);
            m_alreadyJoined = true;
            if(retVal)
            {
                FM_LOG_ERROR("pthread_join(%p) returned st %d\n", (void *)m_pthread, retVal);
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
            lwosSleep(10);
            timeoutMs -= 10;
        }

        if(timeoutMs < 0)
            return 1; /* Hasn't started yet. I guess it's running */

#if 1
        while(!m_hasExited && timeoutMs > 0)
        {
            lwosSleep(10);
            timeoutMs -= 10;
        }

        if(!m_hasExited)
            return 1; /* Running still */
        else
        {
            /* Calling pthread_join a second time results in undefined behavior */
            if (m_alreadyJoined == false)
            {
                lwosThreadJoin(m_pthread, &retVal);
                m_alreadyJoined = true;
                if(retVal)
                    FM_LOG_ERROR("pthread_join(%p) returned st %d\n", (void *)m_pthread, retVal);
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

        FM_LOG_ERROR("pthread_timedjoin_np(%u) returned st %d\n", m_pthread, st);
        return 1;
#endif
    }
}
#endif

/*****************************************************************************/
int FmThread::StopAndWait(int timeoutMs)
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
void FmThread::RunInternal(void)
#ifndef __linux__
{
    m_hasRun = 1;
    
    FM_LOG_DEBUG("Thread id %d running\n", m_threadId);
    run();
    FM_LOG_DEBUG("Thread id %d stopped\n", m_threadId);

    m_threadId = 0;
    m_hasExited = 1;
}
#else
{
    m_hasRun = 1;

    FM_LOG_DEBUG("Thread handle %lu running\n", reinterpret_cast<std::uintptr_t>(m_pthread));
    run();
    FM_LOG_DEBUG("Thread id %lu stopped\n", reinterpret_cast<std::uintptr_t>(m_pthread));

    m_hasExited = 1;
}
#endif

/*****************************************************************************/
int FmThread::ShouldStop(void)
{
    return m_shouldStop; // || main_should_stop; 
    /* If we later add a global variable for FM should stop, reference it here */
}

/*****************************************************************************/
unsigned long FmThread::GetTid()
{
#ifdef __linux__
    LWOSthreadId tid = lwosGetLwrrentThreadId();
#else
    DWORD tid = GetLwrrentThreadId();
#endif

    return (unsigned long)tid;
}

/*****************************************************************************/
void FmThread::Sleep(long long howLongUsec)
{
    if(ShouldStop())
        return; /* Return immediately if we're supposed to shut down */
    if(howLongUsec < 0)
        return; /* Bad value */

#ifndef __linux__
    SleepEx((DWORD)howLongUsec / 1000, 1);
#else
    lwosSleep(howLongUsec / 1000);
#endif
}

/*****************************************************************************/
int FmThread::HasRun()
{
    return m_hasRun;
}

/*****************************************************************************/
int FmThread::HasExited()
{
    return m_hasExited;
}

/*****************************************************************************/
static void fm_thread_signal_handler(int signum)
{
    /* Do nothing, especially not IO */
}

/*****************************************************************************/
void FmThread::InstallSignalHandler(void)
{

#if defined(__linux__) && !defined(LW_MODS) 
    struct sigaction lwrrentSigHandler;

    /* See if this process already has a signal handler */
    int ret = sigaction(FM_THREAD_SIGNUM, NULL, &lwrrentSigHandler);
    if(ret < 0)
    {
        FM_LOG_ERROR("Got st %d from sigaction", ret);
        return;
    }
    if(lwrrentSigHandler.sa_handler != SIG_DFL && lwrrentSigHandler.sa_handler != SIG_IGN)
    {
        FM_LOG_INFO("Signal %d is already handled. Nothing to do.",
                   FM_THREAD_SIGNUM);
        return;
    }

    /* Install our handler */
    struct sigaction newSigHandler;
    sigemptyset(&newSigHandler.sa_mask);
    newSigHandler.sa_flags = 0;
    newSigHandler.sa_handler = fm_thread_signal_handler;

    ret = sigaction(FM_THREAD_SIGNUM, &newSigHandler, NULL);
    if(ret < 0)
    {
        FM_LOG_ERROR("Got error %d from sigaction while adding our signal handler.",
                    ret);
    }
#endif
}
