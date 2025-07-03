
#include "DcgmTaskRunner.h"
#include "logging.h"
#include <stdexcept>

/*****************************************************************************/
DcgmTaskRunner::DcgmTaskRunner()
{
    m_mutex = new DcgmMutex(0);
    m_debugLogging = false;
    lwosCondCreate(&m_condition);
}

/*****************************************************************************/
DcgmTaskRunner::~DcgmTaskRunner()
{
    /* wait for the worker. LwcmThread::StopAndWait() will call LwcmThread::Stop(), 
       which will trigger OnStop() to wake up our thread */
    StopAndWait(60000);

    delete(m_mutex);
    lwosCondDestroy(&m_condition);
}

/*****************************************************************************/
void DcgmTaskRunner::OnStop()
{
    /* Wake up the worker */
    dcgm_mutex_lock(m_mutex);
    lwosCondSignal(&m_condition);
    dcgm_mutex_unlock(m_mutex);
}

/*****************************************************************************/
void DcgmTaskRunner::SetDebugLogging(bool enabled)
{
    m_debugLogging = enabled;

    if(m_debugLogging)
        PRINT_INFO("%p", "Debug logging enabled for DcgmTaskRunner %p", this);
}

/*****************************************************************************/
dcgmReturn_t DcgmTaskRunner::QueueTask(DcgmTask *task)
{
    if(!task)
        return DCGM_ST_BADPARAM;
    
    PRINT_DEBUG("%p %p", "DcgmTaskRunner %p queuing task %p", this, task);

    dcgm_mutex_lock(m_mutex);
    m_taskQueue.push(task);
    lwosCondSignal(&m_condition);
    dcgm_mutex_unlock(m_mutex);

    if(m_debugLogging)
        PRINT_DEBUG("%p %p", "DcgmTaskRunner %p Task %p was queued", this, task);
    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmTaskRunner::run()
{
    /* Top of loop is assumed to have the lock. The code below will handle unlocks and locks */
    dcgm_mutex_lock(m_mutex);
    
    while(1)
    {    
        if(m_taskQueue.size() < 1)
        {
            dcgm_mutex_unlock(m_mutex);
            /* We check ShouldStop() here so that the queue is consumed before we exit.
               Otherwise we could be left with memory leaked items on m_taskQueue and threads
               blocked on their results. */
            if(ShouldStop())
            {
                PRINT_DEBUG("%p", "DcgmTaskRunner %p requested to stop. Exiting update loop.", this);
                break;
            }

            if(m_debugLogging)
                PRINT_INFO("%p", "DcgmTaskRunner %p before RunOnce().", this);

            /* Give the worker thread some idle cycles. Note that we're unlocked here
               so we don't have the lock while in RunOnce() */
            unsigned int sleepMsec = RunOnce();
            if(!sleepMsec)
                sleepMsec = 60000;

            /* Lock, check the queue again, and possibly wait for more items */
            dcgm_mutex_lock(m_mutex);

            if(m_taskQueue.size() > 0)
            {
                if(m_debugLogging)
                    PRINT_INFO("%p", "DcgmTaskRunner %p A task was queued while we were in RunOnce(). Restarting loop.", this);
                continue; /* Restart with the lock. The top of the loop needs it held */
            }

            if(ShouldStop())
            {
                dcgm_mutex_unlock(m_mutex);
                PRINT_DEBUG("%p", "DcgmTaskRunner %p requested to stop. Exiting update loop.", this);
                break;
            }

            if(m_debugLogging)
                PRINT_INFO("%p %u", "DcgmTaskRunner %p planning to wait %u msec.", this, sleepMsec);

            m_mutex->CondWait(&m_condition, sleepMsec);
            /* Restart loop with lock held */

            if(m_debugLogging)
                PRINT_INFO("%p", "DcgmTaskRunner %p woke up after sleep.", this);
            continue;
        }

        DcgmTask *taskPtr = m_taskQueue.front();
        m_taskQueue.pop(); /* Remove the first element */
        dcgm_mutex_unlock(m_mutex); /* Unlock while running callbacks */

        PRINT_DEBUG("%p", "Processing task %p", taskPtr);
        taskPtr->Process();

        /* Restart loop with lock held */
        dcgm_mutex_lock(m_mutex);
    }
}

/*****************************************************************************/
