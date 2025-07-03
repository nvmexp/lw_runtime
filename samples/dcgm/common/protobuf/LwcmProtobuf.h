/* 
 * File:   LwcmProtobuf.h
 */

#ifndef LWCMPROTOBUF_H
#define	LWCMPROTOBUF_H

#include <iostream>
#include <vector>    
#include "LwcmProtocol.h"
#include "lwcm.pb.h"
#include "dcgm_client_internal.h"

using namespace std;

class LwcmProtobuf {
public:
    LwcmProtobuf();
    virtual ~LwcmProtobuf();
    
    /*****************************************************************************
     * Add command to the protobuf message to be sent over the network
     * Returns !NULL on Success
     *          NULL on Error
     *****************************************************************************/
    lwcm::Command* AddCommand(unsigned int cmdType, unsigned int opMode, int id, int status);
    
    /*****************************************************************************
     This method returns the encoded message to be sent over socket* 
     *****************************************************************************/
    int GetEncodedMessage(char **pLwcmMessage, unsigned int *pMsgLength);

    /*****************************************************************************
     * Parse the received protobuf message. This method gets reference to all the
     * commands in the message.
     *****************************************************************************/
    int ParseRecvdMessage(char *buf, int length, vector<lwcm::Command *> *pCommands);
    
    /*****************************************************************************
     * This method is used to get reference to all the commands in the protobuf message
     *****************************************************************************/
    int GetAllCommands(vector<lwcm::Command *> *pCommands);
        
protected:
    lwcm::Msg *mpProtoMsg;                  /* Google protobuf format message */ 
    char *mpEncodedMessage;                 /* Encoded message is stored in this buffer */
    int mMsgType;                           /* Represents one of Request, Response or Notify */
};

#endif	/* LWCMPROTOBUF_H */
