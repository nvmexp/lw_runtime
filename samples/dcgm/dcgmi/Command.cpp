/* 
 * File:   Command.cpp
 */

#include "Command.h"
#include <sstream>
#include <iostream>
#include <stdexcept>
#include "dcgm_structs.h"
#include "dcgm_agent.h"
#include <string.h>
#include "dcgmi_common.h"

/*****************************************************************************/
Command::Command() : mHostName(), mLwcmHandle(0), mSilent(false), mPersistAfterDisconnect(0) {
}

/*****************************************************************************/
Command::~Command() {
    dcgmReturn_t result;

    if (mLwcmHandle) {
        // Disconnect
        result = dcgmDisconnect(mLwcmHandle);
        if (DCGM_ST_OK != result) {
            std::cout << "Error: unable to close connection to specified host: " << mHostName << std::endl;
        }
        mLwcmHandle = 0;
    }
}

/*****************************************************************************/
dcgmReturn_t Command::Connect(void)
{
    dcgmReturn_t result;
    dcgmConnectV2Params_t connectParams;
    const char *hostNameStr = mHostName.c_str();
    bool isUnixSocketAddress = false;
    
    /* For now, do a global init of DCGM on the start of a command. We can change this later to
     * only connect to the remote host engine from within the command object
     */

    result = dcgmInit();
    if(DCGM_ST_OK != result) 
    {
        if (mSilent == false)
            std::cout << "Error: unable to initialize DCGM" << std::endl;
        return result;
    }

    hostNameStr = dcgmi_parse_hostname_string(hostNameStr, &isUnixSocketAddress, !mSilent);
    if(!hostNameStr)
        return DCGM_ST_BADPARAM; /* Don't need to print here. The function above already did */ 

    memset(&connectParams, 0, sizeof(connectParams));
    connectParams.version = dcgmConnectV2Params_version;
    connectParams.persistAfterDisconnect = mPersistAfterDisconnect;
    connectParams.addressIsUnixSocket = isUnixSocketAddress ? 1 : 0;
    connectParams.timeoutMs = 0; /* Use default timeout */

    result = dcgmConnect_v2((char *)hostNameStr, &connectParams, &mLwcmHandle);
    if (DCGM_ST_OK != result) 
    {
        if (mSilent == false)
            std::cout << "Error: unable to establish a connection to the specified host: " << mHostName << std::endl;
        return result;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
int Command::Execute(void)
{
    dcgmReturn_t result;
    
    result = Connect();
    if (DCGM_ST_OK != result) {
        return DCGM_ST_CONNECTION_NOT_VALID;
    }
    
    return DCGM_ST_OK;
}

/*****************************************************************************/
void Command::SetPersistAfterDisconnect(unsigned int persistAfterDisconnect)
{
    mPersistAfterDisconnect = persistAfterDisconnect;
}

/*****************************************************************************/
