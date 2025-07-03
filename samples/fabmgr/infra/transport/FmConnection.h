/** 
 * @file   FmConnection.h
 * 
 * contains classes for describing a connection and for associated handlers for 
 * messages received on this connection
 */

#ifndef FMCONNECTION_H
#define	FMCONNECTION_H

#include "FmSocketMessage.h"
#include <event2/event.h>
#include <event2/bufferevent.h>
#include <event2/buffer.h>
#ifdef __linux__
#include <sys/socket.h>
#include <netinet/tcp.h>
#else
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#endif
#include <string.h>
#include "FmThread.h"
#include "FmRequest.h"
#include "FMCommonTypes.h"
#include <iostream>
#include <map>
#include <vector>

using namespace std;

enum ReadState {
    FM_CONNECTION_READ_HDR = 0,         /**< Connection is expecting to read a header */
    FM_CONNECTION_READ_CONTENT = 1,     /**< Connection is expecting to read content */
};

enum ConnectionState {
    FM_CONNECTION_UNKNOWN         =  0, /**< Connection in unknown state */
    FM_CONNECTION_PENDING         =  1, /**< Connection is being established */
    FM_CONNECTION_ACTIVE          =  2, /**< Connection has been established */
    FM_CONNECTION_MARK_TO_CLOSE   =  3, /**< Connection should be closed */
    FM_CONNECTION_CLOSED          =  4, /**< Connection is closed */
};

class FmConnection;

/**
 * Each client end of a connection and each Server socket has a unique connection 
 * handler. When multiple clients connect to the same server port, the server side 
 * shares a single connection handler for all clients
 */
class FmConnectionHandler
{
public:
    /*****************************************************************************
     * Constructor
     *****************************************************************************/
    FmConnectionHandler();
    
    /*****************************************************************************
     * Destructor
     *****************************************************************************/
    virtual ~FmConnectionHandler();

    /*****************************************************************************
     * Checks if the connection handler is empty
     *
     * @returns true if connection handler is empty
     *          false if connection handler is not empty
     *****************************************************************************/
    bool IsEmpty();
    
    /*****************************************************************************
     * This method is used to add to the connection table
     * @param[in]       pConnection     pointer to connection to add
     * @param[in,out]   pConnectionId   connecton id is written to this pointer
     *
     * @returns     0  On Success
     *              <0 On any error
     *****************************************************************************/
    int AddToConnectionTable(FmConnection* pConnection, fm_connection_id_t *pConnectionId);
    
    /*****************************************************************************
     * This method is used to remove entry from the connection table
     * 
     * @param[in] connection ID
     * @returns 0  On Success
     *          <0 On any error
     *****************************************************************************/
    int RemoveFromConnectionTable(fm_connection_id_t connectionId);    
    
    /*****************************************************************************
     * This method is used to get connection entry corresponding to the connection
     * id.
     * 
     * @param[in] connection ID 
     * 
     * @returns     Pointer to Connection: On Success
     *              NULL: On Failure
     *****************************************************************************/
    FmConnection * GetConnectionEntry(fm_connection_id_t connectionId);
    
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
    fm_connection_id_t GetNextConnectionId();

    /** Lock connection table */
    int Lock();
    /** Unlock connection table */
    int UnLock();    
    
    LWOSCriticalSection mLock;                  /**< Lock used for accessing connection table */
    fm_connection_id_t mConnectionId;			/**< Connection Identifier sequence number*/

    /** Maintains mapping between connection id and connection pointer */
    std::map<fm_connection_id_t, FmConnection*> mConnectionTable;   
};

/**
 * An FmConnection represents one end of an established connection. i.e. one end of a 5-tuple
 * For server sockets all connections to the same server socket share an FmConnectionHandler
 */
class FmConnection {
public:
    
    /*****************************************************************************
     * Constructor
     *
     * @param[in]  pConnectionHandler connection handler to use for messages received 
     *                                  on this connection
     *****************************************************************************/
    FmConnection(FmConnectionHandler *pConnectionHandler);
    virtual ~FmConnection();

    /*****************************************************************************
     * This method is a soft destructor and frees all resources used by this instance
     *****************************************************************************/
    void Cleanup();
    
    /*****************************************************************************
     * Add request ID to the list of pending requests being tracked
     * @param[in]   requestId ID of request to track
     * @param[in]   pFmRequest pointer to the tracked request
     *
     * @returns     0 on Success
     *              <0 on Error
     *****************************************************************************/
    int AddRequest(fm_request_id_t requestId, FmRequest *pFmRequest);
    
    /*****************************************************************************
     * This method is used to remove the entry corresponding to request ID.
     * Must be called when the request is completed.
     * @param[in]   requestId ID of request to stop tracking
     *
     * @returns 0 on Success
     *          <0 on Error
     *****************************************************************************/
    int RemoveRequest(fm_request_id_t requestId);
    
    /*****************************************************************************
     * This method is used to remove all request entries on this connection
     *
     * @returns     0 on Success
     *          <0 on Error
     *****************************************************************************/
    int RemoveAllRequests();

    /*****************************************************************************
     * Remove this connection from the connection table
     *
     *****************************************************************************/
    void RemoveFromConnectionTable();

    /*****************************************************************************
     * This method is used to get Client request entry corresponding to the 
     * request id.
     * @param[in]   requestId ID of request for which 
     *
     * @returns     Entry on Success
     *              NULL if the entry is not found
     *****************************************************************************/
    FmRequest * GetRequest(fm_request_id_t requestId);
    
    /*****************************************************************************
     * This method is used to get the next request ID used as part of message header
     *
     * @returns     next available request ID
     *****************************************************************************/
     fm_request_id_t GetNextRequestId();
    
    /*****************************************************************************
     * This method is used set the reference to current recvd message handled for the
     * connection
     * 
     * @param[in]   pFmMessage  pointer to current received message
     *****************************************************************************/
    void SetLwrrentHandledMessage(FmSocketMessage *pFmMessage);
    
    /*****************************************************************************
     * This method is used to get reference to current recvd message handled by
     * the connection
     *
     * @returns     pointer to current received message
     *****************************************************************************/
    FmSocketMessage* GetLwrrentHandledMessage();
    
    /*****************************************************************************
     * This method is used to increase reference count by 1
     * Derived classes can override default implementation if ref counting is not
     * needed like reusing the same object for connection/disconnection cycle
     *****************************************************************************/
    virtual void IncrReference();
    
    /*****************************************************************************
     * This method is used to decrease the connection reference count by 1
     * Deletes the connection if the reference count reaches 0
     * Derived classes can override default implementation if ref counting is not
     * needed like reusing the same object for connection/disconnection cycle
     *****************************************************************************/
    virtual void DecrReference();
    
    /*****************************************************************************
     * This method is used to check if the connection is still active and in healthy state
     *
     * @returns true if connection is active, false otherwise
     *****************************************************************************/
    bool IsConnectionActive();    
    
    /*****************************************************************************
    * This message is used to get the read state for the connection
    *
    * @returns read state of connection such as FM_CONNECTION_READ_HDR.
    *****************************************************************************/
    ReadState GetReadState();

    /*****************************************************************************
     * This message is used to set the read state for the connection
     *
     * @param[in]   state value to which to set read state of connection
     *****************************************************************************/
    void SetReadState(ReadState state);
    
    /*****************************************************************************
     * This method is used to set the connection state
     *
     * @param[in]   state value to set for state of connection
     *****************************************************************************/
    void SetConnectionState(ConnectionState state);
    
    /*****************************************************************************
     * This method is used to get the connection state
     *
     * @returns     state of this connection
     *****************************************************************************/
    ConnectionState GetConnectionState();

    /*****************************************************************************
     * This method is used to propagate connection changes to derived class
     * The default implementation will do nothing
     *
     * @param[in]   state   state to set for this connection
     *****************************************************************************/
    virtual void SignalConnStateChange(ConnectionState state) { };
    
    /*****************************************************************************
     * This method is used to set the connection ID for identifying the connection
     *****************************************************************************/
    void SetConnectionId(fm_connection_id_t connectionId);
    
    /*****************************************************************************
     * This method is used to get connection ID corresponding to the connection
     *
     * @returns connection ID for this connection
     *****************************************************************************/
     fm_connection_id_t GetConnectionId();

    /*****************************************************************************
     * This method is used to mark the request as completed
     *
     * @param   requestId   request ID of request to mark as completed
     *****************************************************************************/
    int CompleteRequest(fm_request_id_t requestId);
    
    /*****************************************************************************
    * This method is used to set all pending requests to a certain status and
    * notify anyone who is waiting on them to wake up.
    *
    * @param[in]  status FM_ST_? enum as the status
    *
    * @returns  0 always
    *
    *****************************************************************************/
    int SetAllRequestsStatus(int status);

    /*****************************************************************************
     * This method is used to set output buffer to be sent to the destination
     *
     * @param[in]   pMsg    pointer to message to set as output buffer
     *****************************************************************************/
    virtual int SetOutputBuffer(FmSocketMessage *pMsg) = 0;
    
    /*****************************************************************************
     * This method is used to disable any further notifications for the connection
     *****************************************************************************/
    virtual void DisableConnectionNotifications() = 0;

    /*****************************************************************************
     * Close this connection, ilwalidating any pending requests and closing
     * the socket handle.
     *****************************************************************************/
    void Close(void);
    
protected:

    /** Lock Request Table */
    int Lock();
    /** UnLock Request Table */
    int UnLock();

    FmConnectionHandler *mpConnectionHandler;   /**< Reference to connection Handler */
    fm_connection_id_t  mConnectionId;          /**< Connection ID for this connection*/
    ReadState           mReadState;             /**< Represents if the connection is expecting to read header/contents*/
    ConnectionState     mConnectionState;       /**< Connection state */
    int                 mRefCount;              /**< Number of times this connection is referenced */

    map<unsigned int, FmRequest *> mRequestTable; /**< Table of pending requests on this connection*/
    
    FmSocketMessage *mpLwrrentRecvdMessage;     /**< Reference to received message. Allocated here and freed 
                                                     when the message is processed */
    fm_request_id_t mRequestId;                 /**< Last used Request ID */

    LWOSCriticalSection mRequestTableLock;      /**< Lock used for accessing Request map table */
};

#endif	/* FMCONNECTION_H */
