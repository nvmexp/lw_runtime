/* 
 * File:   LwcmConnection.h
 */

#ifndef LWCMCONNECTION_H
#define	LWCMCONNECTION_H

#include "LwcmProtocol.h"
#include <event2/event.h>
#include <event2/bufferevent.h>
#include <event2/buffer.h>
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <string.h>
#include "LwcmThread.h"
#include "LwcmRequest.h"
#include "dcgm_structs_internal.h"
#include <iostream>
#include <map>
#include <vector>

using namespace std;

enum ReadState {
    DCGM_CONNECTION_READ_HDR = 0,
    DCGM_CONNECTION_READ_CONTENT = 1,
};

enum ConnectionState {
    DCGM_CONNECTION_UNKNOWN         =  0,
    DCGM_CONNECTION_PENDING         =  1, /* Are in progress trying to connect */
    DCGM_CONNECTION_ACTIVE          =  2, /* Connection has been established */
    DCGM_CONNECTION_MARK_TO_CLOSE   =  3, /* Connection should be closed */
    DCGM_CONNECTION_CLOSED          =  4, /* The connection is closed */
};

class LwcmConnection;

class LwcmConnectionHandler
{
public:
    /*****************************************************************************
     * Constructor
     *****************************************************************************/
    LwcmConnectionHandler();
    
    /*****************************************************************************
     * Destructor
     *****************************************************************************/
    virtual ~LwcmConnectionHandler();

    /*****************************************************************************
     * Checks if the connection handler is empty
     *****************************************************************************/
    bool IsEmpty();
    
    /*****************************************************************************
     * This method is used to add to the connection table
     * Returns:
     * 0  On Success
     * <0 On any error
     *****************************************************************************/
    int AddToConnectionTable(LwcmConnection* pConnection, dcgm_connection_id_t *pConnectionId);
    
    /*****************************************************************************
     * This method is used to remove entry from the connection table
     * 0  On Success
     * <0 On any error
     *****************************************************************************/
    int RemoveFromConnectionTable(dcgm_connection_id_t connectionId);    
    
    /*****************************************************************************
     * This method is used to get connection entry corresponding to the connection
     * id.
     * Returns:
     * Pointer to Connection: On Success
     * NULL: On Failure
     *****************************************************************************/
    LwcmConnection * GetConnectionEntry(dcgm_connection_id_t connectionId);
    
    /*****************************************************************************
     * This method is used to remove all the entries maintained by the connection
     * handler
     *****************************************************************************/
    void Cleanup(void);

    /*****************************************************************************
     * This method is used to close and remove all the connection entries 
     * maintained by the connection handler
     *****************************************************************************/
    void CloseAndCleanup(void);
    
private:
    /*****************************************************************************
     * This method is used to get next connection Identifier
     *****************************************************************************/
     dcgm_connection_id_t GetNextConnectionId();

    int Lock();
    int UnLock();    
    
    LWOSCriticalSection mLock;          /* Lock used for accessing connection table */
    dcgm_connection_id_t mConnectionId;			/* Connection Identifier sequence */
    std::map<dcgm_connection_id_t, LwcmConnection*> mConnectionTable;   /* Maintains mapping between connection id and connection pointer */
};


class LwcmConnection {
public:
    static const size_t MAX_PENDING_REQUESTS_PER_CONN = 1000; /* Maximum number of requests that can
                                                                 be pending per connection object at a time.
                                                                 This is used to prevent a malicious actor
                                                                 from creating a bunch of random request IDs */

    LwcmConnection(LwcmConnectionHandler *pConnectionHandler);
    virtual ~LwcmConnection();

    /*****************************************************************************
     * This method is a soft destructor and frees all resources used by this instance
     *****************************************************************************/
    void Cleanup();
    
    /*****************************************************************************
     * This message is used to update the map with request ID and the corresponding
     * request
     * Returns 0 on Success
     *         <0 on Error
     *****************************************************************************/
    int AddRequest(dcgm_request_id_t requestId, LwcmRequest *pLwcmRequest);
    
    /*****************************************************************************
     * This method is used to remove the entry corresponding to request ID.
     * Must be called when the request is completed.
     * Returns 0 on Success
     *         <0 on Error
     *****************************************************************************/
    int RemoveRequest(dcgm_request_id_t requestId);
    
    /*****************************************************************************
     * This method is used to remove all request entries on this connection
     * Returns 0 on Success
     *         <0 on Error
     *****************************************************************************/
    int RemoveAllRequests();

    /*****************************************************************************
     * Remove this connection from the connection table
     * Returns Nothing
     *****************************************************************************/
    void RemoveFromConnectionTable();

    /*****************************************************************************
     * This method is used to get Client request entry corresponding to the 
     * request id.
     * Returns Entry on Success
     *         NULL if the entry is not found
     *****************************************************************************/
    LwcmRequest * GetRequest(dcgm_request_id_t requestId);
    
    /*****************************************************************************
     * This method is used to get the next request ID used as part of message header
     *****************************************************************************/
     dcgm_request_id_t GetNextRequestId();
    
    /*****************************************************************************
     * This method is used set the reference to current recvd message handled for the
     * connection
     *****************************************************************************/
    void SetLwrrentHandledMessage(LwcmMessage *pLwcmMessage);
    
    /*****************************************************************************
     * This method is used to get reference to current recvd message handled by
     * the connection
     *****************************************************************************/
    LwcmMessage* GetLwrrentHandledMessage();
    
    /*****************************************************************************
     * This method is used to increase reference count by 1
       Derived classes can override default implementation if ref counting is not
       needed like reusing the same object for connection/disconnection cycle
     *****************************************************************************/
    virtual void IncrReference();
    
    /*****************************************************************************
     * This method is used to decrease the connection reference count by 1
     * Deletes the connection if the reference count reaches 0
       Derived classes can override default implementation if ref counting is not
       needed like reusing the same object for connection/disconnection cycle
     *****************************************************************************/
    virtual void DecrReference();
    
    /*****************************************************************************
     * This method is used to check if the connection is still active and in healthy state
     *****************************************************************************/
    bool IsConnectionActive();    
    
    /*****************************************************************************
    * This message is used to get the read state for the connection
    * Returns read state such as LWCM_CONNECTION_READ_HDR.
    *****************************************************************************/
    ReadState GetReadState();

    /*****************************************************************************
     * This message is used to set the read state for the connection
     *****************************************************************************/
    void SetReadState(ReadState state);
    
    /*****************************************************************************
     * This method is used to set the connection state
     *****************************************************************************/
    void SetConnectionState(ConnectionState state);
    
    /*****************************************************************************
     * This method is used to get the connection state
     *****************************************************************************/
    ConnectionState GetConnectionState();

    /*****************************************************************************
     * This method is used to propagate connection changes to derived class
       The default implementation will do nothing
     *****************************************************************************/
    virtual void SignalConnStateChange(ConnectionState state) { };
    
    /*****************************************************************************
     * This method is used to set the connection ID for identifying the connection
     *****************************************************************************/
    void SetConnectionId(dcgm_connection_id_t connectionId);
    
    /*****************************************************************************
     * This method is used to get connection ID corresponding to the connection
     *****************************************************************************/
     dcgm_connection_id_t GetConnectionId();

    /*****************************************************************************
     * This method is used to mark the request as completed
     *****************************************************************************/
    int CompleteRequest(dcgm_request_id_t requestId);
    
    /*****************************************************************************
    * This method is used to set all pending requests to a certain status and
    * notify anyone who is waiting on them to wake up.
    *
    * Pass DCGM_ST_? enum as the status
    *
    * Returns 0 on success.
    *
    *****************************************************************************/
    int SetAllRequestsStatus(int status);

    /*****************************************************************************
     * This method is used to set output buffer to be sent to the destination
     *****************************************************************************/
    virtual int SetOutputBuffer(LwcmMessage *pMsg) = 0;
    
    /*****************************************************************************
     * This method is used to disable any further notifications for the connection
     *****************************************************************************/
    virtual void DisableConnectionNotifications() = 0;

    /*****************************************************************************/
    /* Close this connection, ilwalidating any pending requests and closing
     * the socket handle.
     *****************************************************************************/
    void Close(void);
    
protected:

    /*****************************************************************************
     * Methods to Lock and Unlock the Request Table
     *****************************************************************************/
    int Lock();
    int UnLock();

    LwcmConnectionHandler *mpConnectionHandler;  /* Reference to connection Handler */
    dcgm_connection_id_t mConnectionId; /* Represents Connection ID */
    ReadState mReadState; /* Represents if the connection is expecting header or contents */
    ConnectionState mConnectionState; /* Connection state */
    int mRefCount; /* Number of times the connection is referenced in pending requests */
    map<unsigned int, LwcmRequest *> mRequestTable; /* Request Table */
    LwcmMessage *mpLwrrentRecvdMessage; /* Reference to received message. Allocated here and freed when the message is processed */
    dcgm_request_id_t mRequestId; /* Request ID */

    int mMutexesInitialized; /* Has mRequestTableLock been initialized yet */
    LWOSCriticalSection mRequestTableLock; /* Lock used for accessing Request map table */
};

#endif	/* LWCMCONNECTION_H */
