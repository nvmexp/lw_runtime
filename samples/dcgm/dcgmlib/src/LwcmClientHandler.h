/* 
 * File:   LwcmClientHandler.h
 */

#ifndef LWCMCLIENTHANDLER_H
#define	LWCMCLIENTHANDLER_H

#include <iostream>
#include "LwcmProtobuf.h"
#include "LwcmConnection.h"
#include "dcgm_structs.h"


class LwcmClientListener;
class LwcmClientConnection;
class LwcmClientCallbackQueue;

class LwcmClientHandler {
public:

    /*****************************************************************************
     * Constructor/destructor
     *****************************************************************************/
    LwcmClientHandler();
    virtual ~LwcmClientHandler();
    
    /*****************************************************************************
     * This method is used to get Connection to the host engine corresponding to 
     * the IP address
     *****************************************************************************/
    dcgmReturn_t GetConnHandleForHostEngine(char *identifier, dcgmHandle_t *pLwcmHandle,
                                   unsigned int timeoutMs, bool addressIsUnixSocket);
    
    /*****************************************************************************
     * This method is used to close connection with the Host Engine
     *****************************************************************************/
    void CloseConnForHostEngine(dcgmHandle_t pConnHandle);
    
    /*****************************************************************************
     * This method is used to return reference to connection handler
     *****************************************************************************/
    LwcmConnectionHandler * GetConnectionHandler();
    
    /*****************************************************************************
     * This method is used to exchange protobuf encoded commands with the Host Engine
     * Used to achieve Blocking functionality
     *****************************************************************************/
    dcgmReturn_t ExchangeMsgBlocking(dcgmHandle_t connHandle, LwcmProtobuf *pEncodedObj,
                 LwcmProtobuf *pDecodeObj, vector<lwcm::Command *> *pRecvdCmds, unsigned int timeout = 60000);

    /*****************************************************************************
     * This method is used to exchange protobuf encoded commands with the Host Engine
     * Used to achieve Async functionality
     *
     * request IN: Custom request handler that you have allocated with new. Do NOT
     *             reference this once you have passed it into this call
     * 
     *****************************************************************************/    
    dcgmReturn_t ExchangeMsgAsync(dcgmHandle_t connHandle, LwcmProtobuf *pEncodedObj,
                     LwcmProtobuf *pDecodeObj, vector<lwcm::Command *> *pRecvdCmds, 
                     LwcmRequest *request, dcgm_request_id_t *pRequestId);
    
private:

    int tryConnectingToHostEngine(char identifier[],
                                  unsigned int portNumber,
                                  dcgmHandle_t* pLwcmHandle,
                                  bool addressIsUnixSocket,
                                  int connectionTimeoutMs);

    LwcmClientListener *mpClientBase;
    LwcmClientCallbackQueue *mpClientCQ;
    LwcmConnectionHandler   * mpConnectionHandler;
};

#endif	/* LWCMCLIENTHANDLER_H */
