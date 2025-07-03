#ifndef LWVSTHREAD_H
#define LWVSTHREAD_H

#include "pthread.h"
#include "errno.h"
#include "DcgmMutex.h"

class LwvsThread
{

private:
    int         m_shouldStop; /* Has our thread been signalled to quit or not? */
    int         m_hasExited;  /* Has our thread finished running yet? */
    DcgmMutex   m_mutex;      /* Mutex for controlling conlwrrent access to internal variables */
    pthread_t   m_pthread;
    int         m_hasRun; /* Has this thread run yet? Is needed for slow-starting posix threads */


protected:
    static DcgmMutex m_sync_mutex; /* Synchronization mutex for use by subclasses to control access to global data */

public:

    /*************************************************************************/
    /*
     * Constructor
     */

    LwvsThread(void);

    /*************************************************************************/
    /*
     * Destructor (virtual to satisfy ancient compiler)
     */
    virtual ~LwvsThread();


    /*************************************************************************/
    /*
     * Static method for getting the current thread's thread ID on linux/windows
     *
     */
    static unsigned long GetTid();


    /*************************************************************************/
    /*
     * Signal this thread that it should stop at its earliest colwenience
     */
    void Stop();


    /*************************************************************************/
    /*
     * Terminate this thread using OS calls to stop it in its tracks
     */
    void Kill();

    /*************************************************************************/
    /*
     * Spawn the separate thread and call its run() method
     *
     * RETURNS: 0 if OK
     *         !0 on error
     */
    int Start();

    /*************************************************************************/
    /*
     * Wait for this thread to exit. Stop() will be called on this
     * thread before waiting to ask the thread to stop
     *
     * timeoutMs IN: Milliseconds to wait for the thread to stop before returning
     *               0=forever
     *
     *  RETURNS: 0 if the thread is no longer running
     *            1 if the thread is still running after timeoutMs
     */
    int StopAndWait(int timeoutMs);

    /*************************************************************************/
    int Wait(int timeoutMs);
    /*
     * Wait for this thread to exit. Call StopAndWait() if you actually want to
     * signal the thread to quit
     *
     * timeoutMs IN: Milliseconds to wait for the thread to stop before returning
     *               0=forever
     *
     * RETURNS: 0 if the thread is no longer running
     *          1 if the thread is still running after timeoutMs
     */

    /*************************************************************************/
    /*
     * Call this method from within your run callback to see if you have
     * been signaled to stop or not
     *
     *  0 = No. Keep running
     * !0 = Yes. Stop running
     */
    int ShouldStop();

    /*************************************************************************/
    /*
     * Implement this virtual method within your class to be run
     * from the separate thread that is created.
     *
     * RETURNS: Nothing.
     */
    virtual void run(void) = 0;

    /*************************************************************************/
    /*
     * Internal method that only need to be public to satisfy C++
     */
    void RunInternal();

    /*************************************************************************/
    /*
     * Sleep for a specified time
     *
     * How long to sleep in microseconds (at least). This will return
     * immediately if ShouldStop() is true
     */
    void Sleep(long long howLongUsec);

    /*************************************************************************/
};


#endif /* LWVSTHREAD_H_ */
