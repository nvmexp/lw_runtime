/* 
 * File:   LwcmServerRequest.cpp
 */

#include "LwcmServerRequest.h"
#include "dcgm_structs.h"

LwcmServerRequest::LwcmServerRequest(dcgm_request_id_t requestId) : LwcmRequest (requestId) {
}

LwcmServerRequest::~LwcmServerRequest() {
}

int LwcmServerRequest::ProcessMessage(LwcmMessage* msg)
{
    if(!msg)
        return DCGM_ST_BADPARAM;
    
    Lock();
    m_messages.push_back(msg);
    Unlock();
    return DCGM_ST_OK;
}
