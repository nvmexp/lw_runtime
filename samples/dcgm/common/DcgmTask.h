#pragma once

#include "lwos.h"

/* Base class for an asynchronous task to be run by a DCGM worker thread */

class DcgmTask
{
public:
    DcgmTask();
    virtual ~DcgmTask();

    /**************************************************************************
     * Method to be implemented by the child class that processes this request
     * in a separate thread
     */
    virtual void Process() = 0;

    /**************************************************************************
     * Mark this request as done, waking anyone blocked in Wait(). This should
     * be called from within Process() in the child class of this.
     * You cannot use this object after you call MarkDoneAndNotify(), as the caller
     * is free to delete this object after that.
     */
    void MarkDoneAndNotify();

    /**************************************************************************
     * Wait for this request to be completed by a child thread. Once the child
     * thread calls MarkDoneAndNotify() from its Process() method, this method
     * will return.
     */
    void Wait();

private:
    DcgmMutex *m_mutex; /* Mutex used in conjunction with m_condition 
                           since POSIX requires it */
    lwosCV m_condition; /* Condition used to signal Wait that we have
                           left pending state. */
    bool m_isCompleted; /* Has this request completed or not? This is the predicate
                           for m_condition */
};

