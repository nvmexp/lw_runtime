#pragma once

/* Worker thread to process and signal DcgmTask objects */

#include "dcgm_structs.h"
#include "LwcmThread.h"
#include "DcgmMutex.h"
#include "DcgmTask.h"
#include <queue>

class DcgmTaskRunner : public LwcmThread
{
public:
    /*************************************************************************/
    /* Constructor and destructor */
    DcgmTaskRunner();
    virtual ~DcgmTaskRunner();

    /*************************************************************************/
    /* Virtual method inherited from LwcmThread that is the main() for this thread */
    void run();

    
    /*************************************************************************/
    /*
     * Enable (true) or disable (false) debug logging in this instance
     */
    void SetDebugLogging(bool enabled);

    /*************************************************************************/
    /*
     * Queue a task to be run by this task runner's thread. It's still up to the caller
     * to Wait() on this task and delete it once Wait() returns. 
     * 
     * Returns: DCGM_ST_OK on success.
     */
    dcgmReturn_t QueueTask(DcgmTask *task);

    /*************************************************************************/
    /*
     * Give this worker thread the opportunity to do some work when the work
     * queue is empty. 
     * 
     * Returns: Minimum ms before we should call this function again. This will
     *          be how long we block on QueueTask() being called again.
     *          Returning 0 = Don't care when we get called back. 
     */
    virtual unsigned int RunOnce() = 0;

    /*************************************************************************/
    /*
     * Virtual method of LwcmThread that is called when LwcmThread::Stop() is
     * called. We use this to signal m_condition to wake up our worker so it
     * will shut down.
     */
    void OnStop();

    /*************************************************************************/
    /* Constructor and destructor */

private:
    DcgmMutex *m_mutex;                 /* Mutex used to provide consistency */
    std::queue<DcgmTask *>m_taskQueue;  /* Actual queue of pointer to task items */
    lwosCV m_condition; /* Condition used to signal that there are items pending on m_taskQueue */
    bool m_debugLogging; /* Boolean as to whether (true) or not (false) we should do debug 
                            logging from this class. Change with SetDebugLogging() */
    
    /*************************************************************************/
};

