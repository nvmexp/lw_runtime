/* 
 * File:   LwcmProtobuf.cpp
 */

#include "LwcmProtobuf.h"
#include "logging.h"
#include "LwcmSettings.h"

/*****************************************************************************
 * Constructor for LWCM Protobuf
 *****************************************************************************/
LwcmProtobuf::LwcmProtobuf() 
{
    mpProtoMsg = new lwcm::Msg;
    mpEncodedMessage = NULL;
}

/*****************************************************************************
 * Destructor for LWCM Protobuf
 *****************************************************************************/
LwcmProtobuf::~LwcmProtobuf() 
{
    delete mpProtoMsg;
    mpProtoMsg = NULL;
    if (mpEncodedMessage) {
        delete [] mpEncodedMessage;
        mpEncodedMessage = NULL;
    }
}

/*****************************************************************************
 This method returns the encoded message to be sent over socket* 
 *****************************************************************************/
int LwcmProtobuf::GetEncodedMessage(char **pLwcmMessage, unsigned int *pMsgLength)
{
    unsigned int length;
    length = mpProtoMsg->ByteSize();
    mpEncodedMessage = new char [length];
    mpProtoMsg->SerializeToArray(mpEncodedMessage, length);
    *pLwcmMessage = mpEncodedMessage;
    *pMsgLength = length;
    return DCGM_ST_OK;
}

/*****************************************************************************
 * Parse the received protobuf message.
 *****************************************************************************/
int LwcmProtobuf::ParseRecvdMessage(char *buf, int bufLength, vector<lwcm::Command *> *pCommands)
{
    unsigned int numCmds, j;
    
    
    if ((NULL == buf) || (bufLength <= 0)) {
        return -1;
    }
    
    if (true != mpProtoMsg->ParseFromArray(buf, bufLength)) {
        PRINT_ERROR("", "Failed to parse protobuf message");
        return -1;
    }

    numCmds = mpProtoMsg->cmd_size();
    if (numCmds <= 0) {
        PRINT_ERROR("", "Invalid number of commands in the protobuf message");
        return -1;
    }
    
    for (j = 0; j < numCmds; j++) {
        const lwcm::Command &cmdMsg = mpProtoMsg->cmd(j);
        pCommands->push_back((lwcm::Command *)&cmdMsg);      /* Store reference to the command in protobuf message */
    }
    
    return 0;
}

int LwcmProtobuf::GetAllCommands(vector<lwcm::Command*>* pCommands)
{
    unsigned int numCmds, j;
    
    numCmds = mpProtoMsg->cmd_size();
    if (numCmds <= 0) {
        PRINT_ERROR("", "Invalid number of commands in the protobuf message");
        return -1;
    }
    
    for (j = 0; j < numCmds; j++) {
        const lwcm::Command &cmdMsg = mpProtoMsg->cmd(j);
        pCommands->push_back((lwcm::Command *)&cmdMsg);      /* Store reference to the command in protobuf message */
    }
    
    return 0;    
}

/*****************************************************************************
 * Add command to the protobuf message to be sent over the network
 *****************************************************************************/
lwcm::Command* LwcmProtobuf::AddCommand(unsigned int cmdType, unsigned int opMode, int id, int status)
{
    lwcm::Command *pCmd;

    pCmd = mpProtoMsg->add_cmd();
    if (NULL == pCmd) {
        return NULL;
    }
    
    pCmd->set_cmdtype((lwcm::CmdType)cmdType);
    pCmd->set_opmode((lwcm::CmdOperationMode)opMode);
    pCmd->set_id(id);
    pCmd->set_status(status);    
    return pCmd;    
}
