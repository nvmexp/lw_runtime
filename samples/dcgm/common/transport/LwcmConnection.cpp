/* 
 * File:   LwcmConnection.cpp
 */

#include "LwcmConnection.h"
#include "LwcmSettings.h"
#include "logging.h"

/*****************************************************************************/
LwcmConnectionHandler::LwcmConnectionHandler()
{
    mConnectionId = DCGM_CONNECTION_ID_NONE + 1; /* This should start with connectionId 1 */
    lwosInitializeCriticalSection(&mLock);
}

/*****************************************************************************/
LwcmConnectionHandler::~LwcmConnectionHandler()
{
    map<dcgm_connection_id_t, LwcmConnection*>::iterator it;
    map<dcgm_connection_id_t, LwcmConnection*>::iterator lwr;
    
    /* This destructor will be ilwoked last in the sequence so that all the
       modules have removed the bindings with any of the pending connections */
    
    if (mConnectionTable.size()) {
        PRINT_DEBUG("%zu", "Number of entries with connection Handler at destruction: %zu",  mConnectionTable.size());
    }
    
    /* Disable all the existing connections to receive any further notification over the network */
    it = mConnectionTable.begin();
    while (it != mConnectionTable.end()) {
        LwcmConnection *pConnection;
        pConnection = (*it).second;
        if (pConnection)
            pConnection->DisableConnectionNotifications();
        
        it++;
    }

    CloseAndCleanup();

    lwosDeleteCriticalSection(&mLock);
}

/*****************************************************************************/
int LwcmConnectionHandler::AddToConnectionTable(LwcmConnection* pConnection, dcgm_connection_id_t *pConnectionId)
{
    dcgm_connection_id_t connectionId;
    Lock();
    
    connectionId = GetNextConnectionId();

    if (mConnectionTable.count(connectionId)) {
        DEBUG_STDERR("Entry corresponding to " << connectionId << "already exists");
        UnLock();
        return -1;
    }
    
    mConnectionTable.insert(make_pair(connectionId, pConnection));
    pConnection->SetConnectionId(connectionId);
    pConnection->IncrReference();
    *pConnectionId = connectionId;
    UnLock();
    return 0;    
}

/*****************************************************************************/
void LwcmConnectionHandler::CloseAndCleanup(void)
{
    std::map<dcgm_connection_id_t, LwcmConnection*>::iterator it;

    /* If there is any entry in the map then reduce the reference count so 
       that the memory is freed for the connection */    
    Lock();
    
    while(mConnectionTable.size()) 
    {
        it = mConnectionTable.begin();
        LwcmConnection *pConnection;
        pConnection = (*it).second;
        if (pConnection)
        {
            pConnection->Close();
            pConnection->DecrReference();
        }
        mConnectionTable.erase(it);
    }

    UnLock();
}

/*****************************************************************************/
int LwcmConnectionHandler::RemoveFromConnectionTable(dcgm_connection_id_t connectionId)
{
    map<dcgm_connection_id_t, LwcmConnection*>::iterator it;
    
    Lock();
    it = mConnectionTable.find(connectionId);
    if (it != mConnectionTable.end()) {
        it->second->DecrReference();
        mConnectionTable.erase(it);
    } else {
        UnLock();
        return -1;
    }
    
    UnLock();
    return 0;      
}

/*****************************************************************************/
LwcmConnection * LwcmConnectionHandler::GetConnectionEntry(dcgm_connection_id_t connectionId)
{
    LwcmConnection *pConnection = NULL;
    map<dcgm_connection_id_t, LwcmConnection*>::iterator it;
    
    Lock();    
    it = mConnectionTable.find(connectionId);
    if (it != mConnectionTable.end()) {
        pConnection = (*it).second;
        pConnection->IncrReference();
    } else {
        pConnection = NULL;
    }

    UnLock();
    return pConnection;    
}

/*****************************************************************************/
bool LwcmConnectionHandler::IsEmpty()
{
    Lock();
    if (mConnectionTable.empty()) {
        UnLock();
        return true;
    } else {
        UnLock();
        return false;
    }
}


/*****************************************************************************/
int LwcmConnectionHandler::Lock()
{
    lwosEnterCriticalSection(&mLock);
    return DCGM_ST_OK;    
}

/*****************************************************************************/
int LwcmConnectionHandler::UnLock()
{
    lwosLeaveCriticalSection(&mLock);
    return DCGM_ST_OK;    
}

/*****************************************************************************/
dcgm_connection_id_t LwcmConnectionHandler::GetNextConnectionId()
{
    dcgm_connection_id_t newId = lwosInterlockedIncrement(&mConnectionId);

    /* Don't allocate a connection as id DCGM_CONNECTION_ID_NONE. In practice,
       this will only happen after 2^32 connections */
    if(newId == DCGM_CONNECTION_ID_NONE)
        newId = lwosInterlockedIncrement(&mConnectionId);

    return newId;
}

/*****************************************************************************/
LwcmConnection::LwcmConnection(LwcmConnectionHandler *pConnectionHandler) {
    mRefCount = 0;
    mRequestId = 0;
    mConnectionId = DCGM_CONNECTION_ID_NONE;
    mReadState = DCGM_CONNECTION_READ_HDR;
    mConnectionState = DCGM_CONNECTION_UNKNOWN;
    mpConnectionHandler = pConnectionHandler;
    mpLwrrentRecvdMessage = 0;
    // lwosInitializeCriticalSection(&gLockRefernceCount);

    lwosInitializeCriticalSection(&mRequestTableLock);
    mMutexesInitialized = 1;
}

/*****************************************************************************/
LwcmConnection::~LwcmConnection() {
    Cleanup();

    lwosDeleteCriticalSection(&mRequestTableLock);
    mMutexesInitialized = 0;
}

/*****************************************************************************/
void LwcmConnection::Cleanup() {
    /**
     * Free up the Request table if there is any pending
     */
    map<dcgm_request_id_t,  LwcmRequest *> ::iterator itr = mRequestTable.begin();
    while (itr != mRequestTable.end()) {
        LwcmRequest * pLwcmRequest = (*itr).second;
        if (NULL != pLwcmRequest) {
            delete pLwcmRequest;
            pLwcmRequest = NULL;
        }
        mRequestTable.erase(itr++);
    }
}

/*****************************************************************************
 * This message is used to update the map with request ID and the corresponding
 * request
 * Returns 0 on Success
 *         <0 on Error
 *****************************************************************************/
int LwcmConnection::AddRequest(dcgm_request_id_t requestId, LwcmRequest *pLwcmRequest) {
    Lock();

    if(mRequestTable.size() > MAX_PENDING_REQUESTS_PER_CONN)
    {
        PRINT_ERROR("%zu %u", "Hit maximum outstanding request count of %zu for connectionId %u", 
                    MAX_PENDING_REQUESTS_PER_CONN, GetConnectionId());
        UnLock();
        return -1;
    }

    if (mRequestTable.count(requestId)) 
    {
        PRINT_ERROR("%u %u", "Entry corresponding to requestId %u already exists for connectionId %u", 
                    requestId, GetConnectionId());
        UnLock();
        return -1;
    }
    
    mRequestTable.insert(make_pair(requestId, pLwcmRequest));

    UnLock();
    return 0;    
    
}

/*****************************************************************************
 * This method is used to remove the entry corresponding to request ID.
 * Must be called when the request is completed.
 * Returns 0 on Success
 *         <0 on Error
 *****************************************************************************/
int LwcmConnection::RemoveRequest(dcgm_request_id_t requestId) {
    map<dcgm_request_id_t, LwcmRequest *>::iterator it;
    
    Lock();
    it = mRequestTable.find(requestId);
    if (it != mRequestTable.end()) {
        mRequestTable.erase(it);
    } else {
        UnLock();
        return -1;
    }
    
    UnLock();
    return 0;
}

/*****************************************************************************
 * This method is used to remove all request entries on this connection
 * Returns 0 on Success
 *         <0 on Error
 *****************************************************************************/
int LwcmConnection::RemoveAllRequests() {

    Lock();
    map<dcgm_request_id_t, LwcmRequest *> ::iterator itr = mRequestTable.begin();
    while (itr != mRequestTable.end()) {
        LwcmRequest * pLwcmRequest = (*itr).second;
        if (NULL != pLwcmRequest) {
            delete pLwcmRequest;
            pLwcmRequest = NULL;
        }
        mRequestTable.erase(itr++);
    }

    UnLock();
    return 0;
}

/*****************************************************************************
 * This method is used to get Client request entry corresponding to the 
 * request id.
 * Returns Entry on Success
 *         NULL if the entry is not found
 *****************************************************************************/
LwcmRequest * LwcmConnection::GetRequest(dcgm_request_id_t requestId) {
    map<dcgm_request_id_t, LwcmRequest *>::iterator it;
    
    Lock();
    it = mRequestTable.find(requestId);
    if (it != mRequestTable.end()) {
        UnLock();
        return (*it).second;
    } else {
        UnLock();
        return NULL;
    }
}


/*****************************************************************************/
dcgm_request_id_t LwcmConnection::GetNextRequestId()
{
    return lwosInterlockedIncrement(&mRequestId);
}

/*****************************************************************************/
int LwcmConnection::CompleteRequest(dcgm_request_id_t requestId)
{
    LwcmRequest *pLwcmRequest;

    pLwcmRequest = GetRequest(requestId);
    if (pLwcmRequest) {
        delete pLwcmRequest;
        pLwcmRequest = NULL;
    }

    (void)RemoveRequest(requestId);

    return 0;    
}
/*****************************************************************************/
int LwcmConnection::SetAllRequestsStatus(int status)
{
    Lock();

    map<dcgm_request_id_t,  LwcmRequest *> ::iterator itr;

    for (itr = mRequestTable.begin(); itr != mRequestTable.end(); itr++)
    {
        LwcmRequest * pLwcmRequest = (*itr).second;
        if(!pLwcmRequest)
            continue;

        pLwcmRequest->SetStatus(status);
    }

    UnLock();

    return 0;
}


/*****************************************************************************/
void LwcmConnection::Close(void)
{
    if(mConnectionState >= DCGM_CONNECTION_MARK_TO_CLOSE)
    {
        PRINT_DEBUG("%p %u", "Ignoring Close() for connection %p already in state %u", 
                    this, mConnectionState);
        return;
    } 

    SetConnectionState(DCGM_CONNECTION_MARK_TO_CLOSE);
    SetAllRequestsStatus(DCGM_ST_CONNECTION_NOT_VALID);
}


/*****************************************************************************
 * This message is used to get the read state for the connection
 * Returns read state such as DCGM_CONNECTION_READ_HDR.
 *****************************************************************************/
ReadState LwcmConnection::GetReadState() 
{
    return mReadState;
}

/*****************************************************************************
 * This message is used to set the read state for the connection
 *****************************************************************************/
void LwcmConnection::SetReadState(ReadState state) 
{
    mReadState = state;
}


/*****************************************************************************
 * This method is used set the reference to current recvd message handled for the
 * connection
 *****************************************************************************/
void LwcmConnection::SetLwrrentHandledMessage(LwcmMessage *pLwcmMessage) {
    mpLwrrentRecvdMessage = pLwcmMessage;
}

/*****************************************************************************
 * This method is used to get reference to current recvd message handled by
 * the connection
 *****************************************************************************/
LwcmMessage* LwcmConnection::GetLwrrentHandledMessage() {
    return mpLwrrentRecvdMessage;
}

/*****************************************************************************
 * This method is used to increase reference count by 1
 *****************************************************************************/
void LwcmConnection::IncrReference() {
    int newRefCount = __sync_add_and_fetch(&mRefCount, 1);
    PRINT_DEBUG("%d","++Connection Reference Count : %d", newRefCount);
}

/*****************************************************************************
 * This method is used to decrease the connection reference count by 1
 * Deletes the connection if the reference count reaches 0
 *****************************************************************************/
void LwcmConnection::DecrReference() {

    int newRefCount = __sync_sub_and_fetch(&mRefCount, 1);

    PRINT_DEBUG("%d", "--Connection Reference Count : %d", mRefCount);

    if (0 == newRefCount)
    {
        LwcmConnection* pThis = const_cast<LwcmConnection*> (this);
        PRINT_DEBUG("%p", "LwcmConnection destructor : %p", pThis);
        delete pThis;
        pThis = NULL;
    }
}

/*****************************************************************************
 * This method is used to check if the connection is still active and in healthy state
 *****************************************************************************/
bool LwcmConnection::IsConnectionActive() {
    
    /* Note : The condition is termed as active only when DCGM_CONNECTION_ACTIVE bit is set 
       and no other bit is set. This is the reason the comparison is performed with DCGM_CONNECTION_ACTIVE
       to ensure only active bit is set */

    if (GetConnectionState() == DCGM_CONNECTION_ACTIVE) {
        return true;
    } else {
        return false;
    }    
}

/*****************************************************************************
 * This method is used to set the connection state
 *****************************************************************************/
void LwcmConnection::SetConnectionState(ConnectionState state) {
    mConnectionState = state;
    SignalConnStateChange(state);
}

/*****************************************************************************
 * This method is used to get the connection state 
 *****************************************************************************/
ConnectionState LwcmConnection::GetConnectionState() {
    ConnectionState state;
    state = mConnectionState;
    return state;
}

/*****************************************************************************/
void LwcmConnection::SetConnectionId(dcgm_connection_id_t connectionId)
{
    mConnectionId = connectionId;
}

/*****************************************************************************/
dcgm_connection_id_t LwcmConnection::GetConnectionId()
{
    return mConnectionId;
}

/*****************************************************************************/
void LwcmConnection::RemoveFromConnectionTable()
{
    if(mpConnectionHandler)
    {
        mpConnectionHandler->RemoveFromConnectionTable(mConnectionId);
        /* NOTE: Don't do anything after this. This causes a DecrReference and "this" may
                 now be invalid */
    }
    else
        PRINT_ERROR("%p", "Null mpConnectionHandler on %p", mpConnectionHandler);
}


#if 0
/*****************************************************************************
 * This method is used to Set the Output Buffer to be sent back to the 
 * client
 *****************************************************************************/
int LwcmConnection::SetOutputBuffer(void *buf, int length) {
    if (!IsConnectionActive()) {
        return -1;
    }
    
    bufferevent_lock(mpBufEv);
    
    /* Update libevent output buffer with the data to be sent to the client */
    evbuffer_add(mpOutputBuffer, buf, length);
    
    if (bufferevent_write_buffer(mpBufEv, mpOutputBuffer)) {
        PRINT_ERROR("", "Failed to write message to event buffer");
        DEBUG_STDERR("Failed to write message to event buffer");
        bufferevent_unlock(mpBufEv);
        return -2;
    }
    
    bufferevent_unlock(mpBufEv);
    return 0;    
}
#endif

/*****************************************************************************
 Acquire Lock
 *****************************************************************************/
int LwcmConnection::Lock(void)
{
    lwosEnterCriticalSection(&mRequestTableLock);
    return DCGM_ST_OK;
}

/*****************************************************************************
 Release Lock
 *****************************************************************************/
int LwcmConnection::UnLock(void)
{
    lwosLeaveCriticalSection(&mRequestTableLock);
    return DCGM_ST_OK;
}
