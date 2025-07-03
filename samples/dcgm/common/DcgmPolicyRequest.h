#ifndef DCGMPOLICYREQUEST_H
#define	DCGMPOLICYREQUEST_H

#include "LwcmRequest.h"
#include <iostream>
#include "dcgm_structs.h"
#include "LwcmProtobuf.h"
#include "LwcmClientCallbackQueue.h"
using namespace std;

class DcgmPolicyRequest : public LwcmRequest {
public:
    DcgmPolicyRequest(fpRecvUpdates beginCB, fpRecvUpdates finishCB);
    virtual ~DcgmPolicyRequest();
    int ProcessMessage(LwcmMessage *msg);
private:
    bool mIsAckRecvd;
    fpRecvUpdates mBeginCB;
    fpRecvUpdates mFinishCB;
};

#endif	/* DCGMPOLICYREQUEST_H */

