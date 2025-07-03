#include "LwcmClientCallbackQueue.h"
#include <iostream>
#include "LwcmSettings.h"

/*****************************************************************************/
LwcmClientCallbackQueue::LwcmClientCallbackQueue()
{
    mStarted = false;
}

/*****************************************************************************/
LwcmClientCallbackQueue::~LwcmClientCallbackQueue()
{
    int st = StopAndWait(30000);
    if(st)
    {
        PRINT_ERROR("", "Killing LwcmClientCallbackQueue worker thread after waiting for it to stop");
        Kill();
    }
}

/*****************************************************************************/
dcgmReturn_t LwcmClientCallbackQueue::PushFunctionPtr(fpRecvUpdates fp, dcgmPolicyCallbackResponse_t callbackResponse)
{
    funcData_t newFunc;
    newFunc.func = fp;
    newFunc.callbackResponse = callbackResponse;

    mFunctionList.PushBack(newFunc);

    // only start the thread if it isn't already running
    if (!mStarted)
    {
        DEBUG_STDERR("Starting clientside callback loop");
        Start();
        mStarted = true;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
void LwcmClientCallbackQueue::run()
{
    while(!ShouldStop())
    {
        if (mFunctionList.Size())
        {
            DEBUG_STDERR("A function is in the callback queue, calling it");
            funcData_t function = mFunctionList.Front();
            // funcData_t is likely to change and propagate here
            mFunctionList.PopFront();
            function.func(&function.callbackResponse);
            Sleep(1000);
        }
    }
}
