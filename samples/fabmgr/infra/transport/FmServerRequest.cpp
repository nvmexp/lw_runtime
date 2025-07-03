/* 
 * File:   FmServerRequest.cpp
 */

#include "FmServerRequest.h"
#include "FMErrorCodesInternal.h"

FmServerRequest::FmServerRequest(fm_request_id_t requestId) : FmRequest (requestId) {
	mRequestStatus = 0;
}

FmServerRequest::~FmServerRequest() {
}

int FmServerRequest::ProcessMessage(FmSocketMessage* msg)
{
    if(!msg)
        return FM_INT_ST_BADPARAM;
    
    Lock();
    m_messages.push_back(msg);
    Unlock();
    return FM_INT_ST_OK;
}
