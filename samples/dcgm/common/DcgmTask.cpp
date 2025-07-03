
#include "DcgmMutex.h"
#include "DcgmTask.h"

/*****************************************************************************/
DcgmTask::DcgmTask()
{
    m_mutex = new DcgmMutex(0);
    lwosCondCreate(&m_condition);
    m_isCompleted = false;
}

/*****************************************************************************/
DcgmTask::~DcgmTask()
{
    m_isCompleted = true;
    lwosCondDestroy(&m_condition);
    delete(m_mutex);
}

/*****************************************************************************/
void DcgmTask::Wait(void)
{
    /* Note that we're assuming that m_mutex is not locked by us when we enter
       this call. CondWait() will acquire the mutex, and we'll unlock it afterwards */

    while(!m_isCompleted)
    {
        dcgmMutexReturn_t mutexSt = m_mutex->CondWait(&m_condition, 0);
        /* CondWait() will leave the mutex locked if it succeeds */
        if(mutexSt == DCGM_MUTEX_ST_OK)
            dcgm_mutex_unlock(m_mutex);

    }
}

/*****************************************************************************/
void DcgmTask::MarkDoneAndNotify(void)
{
    dcgm_mutex_lock(m_mutex);
    m_isCompleted = true;
    lwosCondSignal(&m_condition);
    dcgm_mutex_unlock(m_mutex);
}

/*****************************************************************************/
