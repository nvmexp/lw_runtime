/* 
 * File:   LwcmServerRequest.h
 */

#ifndef LWCMSERVERREQUEST_H
#define	LWCMSERVERREQUEST_H

#include <iostream>
#include "LwcmRequest.h"

using namespace std;

class LwcmServerRequest : public LwcmRequest {
public:
    LwcmServerRequest(dcgm_request_id_t requestId);
    virtual ~LwcmServerRequest();

    /*****************************************************************************
     * This message pushes the received message to Lwcm Request Container
     *****************************************************************************/
    int ProcessMessage(LwcmMessage *msg);

    /*****************************************************************************
     * Set Status of the request
     *****************************************************************************/    
    int SetStatus(int status);
    
    /*****************************************************************************
     * Get Status of the request in process
     *****************************************************************************/
    int GetStatus();

private:
    int mRequestStatus;                 /* Set status to in progress or complete */ 
};

#endif	/* LWCMSERVERREQUEST_H */

