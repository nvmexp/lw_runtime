/* 
 * File:   DcgmPolicyRequest.cpp
 */

#include "DcgmPolicyRequest.h"
#include "dcgm_structs.h"
#include "LwcmSettings.h"
#include "LwcmProtocol.h"

/*****************************************************************************/
DcgmPolicyRequest::DcgmPolicyRequest(fpRecvUpdates beginCB, fpRecvUpdates finishCB) : LwcmRequest(0) 
{
    mIsAckRecvd = false;
    mBeginCB = beginCB;
    mFinishCB = finishCB;
}

/*****************************************************************************/
DcgmPolicyRequest::~DcgmPolicyRequest() 
{
    
}

/*****************************************************************************/
int DcgmPolicyRequest::ProcessMessage(LwcmMessage *msg)
{
    if(!msg)
        return DCGM_ST_BADPARAM;

    Lock();
    /* The first response is the policy manager confirming that we've watched
       it. Further responses will be policy violations */
    msg->SwapHeader(); /* Make sure it's in little endian */
    dcgm_message_header_t* header = msg->GetMessageHdr();
    switch(header->msgType)
    {
        case DCGM_MSG_PROTO_REQUEST:
        case DCGM_MSG_PROTO_RESPONSE:
            /* Request/response messages complete the initial request */
            if(!mIsAckRecvd)
            {
                m_status = DCGM_ST_OK;
                mIsAckRecvd = true;
                m_messages.push_back(msg);
                lwosCondBroadcast(&m_condition); /* The waiting thread will wake up and read the messages */
            }
            else
            {
                PRINT_ERROR("", "Ignoring unexpected duplicate ACK");
                delete msg;
            }
            Unlock();
            return DCGM_ST_OK;

        case DCGM_MSG_POLICY_NOTIFY:
            /* This #if is here because we can either ignore updates before the initial request
               is ACKed or we can process them right away. The default behavior is to process
               callbacks right away, meaning that if there is a policy violation, we are likely 
               to call our callbacks from within dcgmPolicyRegister because it sets up the watches,
               calls UpdateAllFields(), and gets instant notifications of FV updates. Leaving the
               alternative here in case lwstomers complain too much :) */
#if 1
            break; /* Code handled below */
#else /* Notify only after ACK */
            if(mIsAckRecvd)
                break; /* Code handled below */
            else
            {
                PRINT_DEBUG("%p", "Request %p ignoring notification before the initial request was ACKed", this);
                delete msg;
            }
            Unlock();
#endif
            return DCGM_ST_OK;

        default:
            PRINT_ERROR("%u", "Unexpected msgType %u received.", header->msgType);
            Unlock();
            delete msg;
            return DCGM_ST_OK; /* Returning an error here doesn't affect anything we want to affect */
    }

    /* We should only be here if we got a policy notification */
    dcgm_msg_policy_notify_t *policy = (dcgm_msg_policy_notify_t *)msg->GetContent();

    /* Make local copies of the callbacks so we can safely unlock. I don't want this code to deadlock if someone
       removes this object from one of the callbacks */
    fpRecvUpdates beginCB = mBeginCB;
    fpRecvUpdates finishCB = mFinishCB;

    Unlock();

    /* Call the callbacks if they're present */
    if(policy->begin && beginCB)
        beginCB(&policy->response);
    if(!policy->begin && finishCB)
        finishCB(&policy->response);
    
    delete msg;
    return DCGM_ST_OK;
}


