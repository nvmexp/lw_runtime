/* 
 * File:   FmServerRequest.h
 */

#ifndef FMSERVEREQUEST_H
#define	FMSERVEREQUEST_H

#include <iostream>
#include "FmRequest.h"

using namespace std;

class FmServerRequest : public FmRequest {
public:
    FmServerRequest(fm_request_id_t requestId);
    virtual ~FmServerRequest();

    /*****************************************************************************
     * This message pushes the received message to Fm Request Container
     *****************************************************************************/
    int ProcessMessage(FmSocketMessage *msg);

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

#endif	/* FMSERVEREQUEST_H */

