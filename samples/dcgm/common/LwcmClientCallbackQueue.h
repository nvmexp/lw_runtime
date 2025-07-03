/*
 * File: LwcmClientCallbackQueue.h
 */

#ifndef LWCMCLIENTCALLBACKQUEUE_H
#define LWCMCLIENTCALLBACKQUEUE_H

#include "LwcmThread.h"
#include "LwcmThreadSafeSTL.h"
#include "dcgm_structs.h"

class LwcmClientCallbackQueue : public LwcmThread
{
public:
    LwcmClientCallbackQueue();
    ~LwcmClientCallbackQueue();
    dcgmReturn_t PushFunctionPtr(fpRecvUpdates fp, dcgmPolicyCallbackResponse_t callbackResponse);


private:
    void run();
    typedef struct funcData_st {
        fpRecvUpdates func;
        dcgmPolicyCallbackResponse_t callbackResponse;
    } funcData_t;

    bool mStarted;
    LwcmThreadSafeList <funcData_t> mFunctionList;
};

#endif /* LWCMCLIENTCALLBACKQUEUE_H */

