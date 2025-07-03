/* 
 * File:   Command.h
 */

#ifndef COMMAND_H
#define	COMMAND_H

#include <sstream>
#include <iostream>
#include <stdexcept>
#include "dcgm_structs.h"

class Command
{
public:
    Command();

    virtual ~Command();

    /*****************************************************************************
     * Connect to the host name
     *****************************************************************************/
    dcgmReturn_t Connect();

    /*****************************************************************************
    * persistAfterDisconnect: Should the host engine persist the watches created
    *                         by this connection after the connection goes away?
    *                         1=yes. 0=no (default). 
    *****************************************************************************/
    void SetPersistAfterDisconnect(unsigned int persistAfterDisconnect);
    
    /*****************************************************************************
     * Disconnect or cleanup for the connection
     *****************************************************************************/
    dcgmReturn_t Disconnect();
    
    /*****************************************************************************
     * Execute command on the Host Engine 
     * Abstract function and should be implemented by the derived class 
     *****************************************************************************/
    virtual int Execute();
    
protected:
    std::string mHostName;
    dcgmHandle_t mLwcmHandle;
    bool         mJson;
    bool         mSilent;
    unsigned int mPersistAfterDisconnect; /* Should the host engine persist the watches created
                                             by this connection after the connection goes away?
                                             1=yes. 0=no (default). */
};


#endif	/* COMMAND_H */
