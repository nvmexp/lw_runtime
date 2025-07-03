/* 
 * File:   FmConnection.cpp
 */

#include "FmConnection.h"
#include "FMCommonTypes.h"
#include "fm_log.h"

LWOSCriticalSection gLockRefernceCount;


/*****************************************************************************/
FmConnectionHandler::FmConnectionHandler()
{
    mConnectionId = FM_CONNECTION_ID_START; 
    lwosInitializeCriticalSection(&mLock);
}

/*****************************************************************************/
FmConnectionHandler::~FmConnectionHandler()
{
    map<fm_connection_id_t, FmConnection*>::iterator it;
    map<fm_connection_id_t, FmConnection*>::iterator lwr;
    
    /* This destructor will be ilwoked last in the sequence so that all the
       modules have removed the bindings with any of the pending connections */
    
    if (mConnectionTable.size()) {
        FM_LOG_DEBUG("Number of entries with connection Handler at destruction: %zu",  mConnectionTable.size());
    }
    
    /* Disable all the existing connections to receive any further notification over the network */
    it = mConnectionTable.begin();
    while (it != mConnectionTable.end()) {
        FmConnection *pConnection;
        pConnection = (*it).second;
        if (pConnection)
            pConnection->DisableConnectionNotifications();
        
        it++;
    }

    CloseAndCleanup();

    lwosDeleteCriticalSection(&mLock);
}

/*****************************************************************************/
int FmConnectionHandler::AddToConnectionTable(FmConnection* pConnection, fm_connection_id_t *pConnectionId)
{
    fm_connection_id_t connectionId;
    Lock();
    
    connectionId = GetNextConnectionId();

    if (mConnectionTable.count(connectionId)) {
        
        FM_LOG_ERROR("connection handler: specified connection id %d already exists in connection table", connectionId);
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
void FmConnectionHandler::CloseAndCleanup(void)
{
    std::map<fm_connection_id_t, FmConnection*>::iterator it;

    /* If there is any entry in the map then reduce the reference count so 
       that the memory is freed for the connection */    
    Lock();
    
    while(mConnectionTable.size()) 
    {
        it = mConnectionTable.begin();
        FmConnection *pConnection;
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
int FmConnectionHandler::RemoveFromConnectionTable(fm_connection_id_t connectionId)
{
    map<fm_connection_id_t, FmConnection*>::iterator it;
    
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
FmConnection * FmConnectionHandler::GetConnectionEntry(fm_connection_id_t connectionId)
{
    FmConnection *pConnection = NULL;
    map<fm_connection_id_t, FmConnection*>::iterator it;
    
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
bool FmConnectionHandler::IsEmpty()
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
int FmConnectionHandler::Lock()
{
    lwosEnterCriticalSection(&mLock);
    return FM_INT_ST_OK;    
}

/*****************************************************************************/
int FmConnectionHandler::UnLock()
{
    lwosLeaveCriticalSection(&mLock);
    return FM_INT_ST_OK;    
}

/*****************************************************************************/
fm_connection_id_t FmConnectionHandler::GetNextConnectionId()
{
    fm_connection_id_t newId = lwosInterlockedIncrement(&mConnectionId);

    /* Don't allocate a connection as id FM_CONNECTION_ID_NONE. In practice,
       this will only happen after 2^32 connections */
    if(newId == FM_CONNECTION_ID_NONE)
        newId = lwosInterlockedIncrement(&mConnectionId);

    return newId;
}

/*****************************************************************************/
FmConnection::FmConnection(FmConnectionHandler *pConnectionHandler) {
    mRefCount = 0;
    mRequestId = 0;
    mConnectionId = FM_CONNECTION_ID_NONE;
    mReadState = FM_CONNECTION_READ_HDR;
    mConnectionState = FM_CONNECTION_UNKNOWN;
    mpConnectionHandler = pConnectionHandler;
    mpLwrrentRecvdMessage = 0;
    // lwosInitializeCriticalSection(&gLockRefernceCount);

    lwosInitializeCriticalSection(&mRequestTableLock);
}

/*****************************************************************************/
FmConnection::~FmConnection() {
    Cleanup();

    lwosDeleteCriticalSection(&mRequestTableLock);
}

/*****************************************************************************/
void FmConnection::Cleanup() {
    /**
     * Free up the Request table if there is any pending
     */
    map<fm_request_id_t,  FmRequest *> ::iterator itr = mRequestTable.begin();
    while (itr != mRequestTable.end()) {
        FmRequest * pFmRequest = (*itr).second;
        if (NULL != pFmRequest) {
            delete pFmRequest;
            pFmRequest = NULL;
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
int FmConnection::AddRequest(fm_request_id_t requestId, FmRequest *pFmRequest) {
    Lock();

    if (mRequestTable.count(requestId)) {
        FM_LOG_ERROR("entry corresponding to requsted id %d already exists in request table", requestId);
        UnLock();
        return -1;
    }
    
    mRequestTable.insert(make_pair(requestId, pFmRequest));

    UnLock();
    return 0;    
    
}

/*****************************************************************************
 * This method is used to remove the entry corresponding to request ID.
 * Must be called when the request is completed.
 * Returns 0 on Success
 *         <0 on Error
 *****************************************************************************/
int FmConnection::RemoveRequest(fm_request_id_t requestId) {
    map<fm_request_id_t, FmRequest *>::iterator it;
    
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
int FmConnection::RemoveAllRequests() {

    Lock();
    map<fm_request_id_t, FmRequest *> ::iterator itr = mRequestTable.begin();
    while (itr != mRequestTable.end()) {
        FmRequest * pFmRequest = (*itr).second;
        if (NULL != pFmRequest) {
            delete pFmRequest;
            pFmRequest = NULL;
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
FmRequest * FmConnection::GetRequest(fm_request_id_t requestId) {
    map<fm_request_id_t, FmRequest *>::iterator it;
    
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
fm_request_id_t FmConnection::GetNextRequestId()
{
    return lwosInterlockedIncrement(&mRequestId);
}

/*****************************************************************************/
int FmConnection::CompleteRequest(fm_request_id_t requestId)
{
    FmRequest *pFmRequest;

    pFmRequest = GetRequest(requestId);
    if (pFmRequest) {
        delete pFmRequest;
        pFmRequest = NULL;
    }

    (void)RemoveRequest(requestId);

    return 0;    
}
/*****************************************************************************/
int FmConnection::SetAllRequestsStatus(int status)
{
    Lock();

    map<fm_request_id_t,  FmRequest *> ::iterator itr;

    for (itr = mRequestTable.begin(); itr != mRequestTable.end(); itr++)
    {
        FmRequest * pFmRequest = (*itr).second;
        if(!pFmRequest)
            continue;

        pFmRequest->SetStatus(status);
    }

    UnLock();

    return 0;
}


/*****************************************************************************/
void FmConnection::Close(void)
{
    if(mConnectionState >= FM_CONNECTION_MARK_TO_CLOSE)
    {
        FM_LOG_DEBUG("Ignoring Close() for connection %p already in state %u", 
                    this, mConnectionState);
        return;
    } 

    SetConnectionState(FM_CONNECTION_MARK_TO_CLOSE);
    SetAllRequestsStatus(FM_INT_ST_CONNECTION_NOT_VALID);
}


/*****************************************************************************
 * This message is used to get the read state for the connection
 * Returns read state such as FM_CONNECTION_READ_HDR.
 *****************************************************************************/
ReadState FmConnection::GetReadState() 
{
    return mReadState;
}

/*****************************************************************************
 * This message is used to set the read state for the connection
 *****************************************************************************/
void FmConnection::SetReadState(ReadState state) 
{
    mReadState = state;
}


/*****************************************************************************
 * This method is used set the reference to current recvd message handled for the
 * connection
 *****************************************************************************/
void FmConnection::SetLwrrentHandledMessage(FmSocketMessage *pFmMessage) {
    mpLwrrentRecvdMessage = pFmMessage;
}

/*****************************************************************************
 * This method is used to get reference to current recvd message handled by
 * the connection
 *****************************************************************************/
FmSocketMessage* FmConnection::GetLwrrentHandledMessage() {
    return mpLwrrentRecvdMessage;
}

/*****************************************************************************
 * This method is used to increase reference count by 1
 *****************************************************************************/
void FmConnection::IncrReference() {
    lwosEnterCriticalSection(&gLockRefernceCount);
    mRefCount++;
    //FM_LOG_DEBUG("++Connection Reference Count : %d", mRefCount);
    lwosLeaveCriticalSection(&gLockRefernceCount);
}

/*****************************************************************************
 * This method is used to decrease the connection reference count by 1
 * Deletes the connection if the reference count reaches 0
 *****************************************************************************/
void FmConnection::DecrReference() {
    lwosEnterCriticalSection(&gLockRefernceCount);

    if (mRefCount) {
        mRefCount--;
        //FM_LOG_DEBUG("--Connection Reference Count : %d", mRefCount);
    }

    lwosLeaveCriticalSection(&gLockRefernceCount);

    if (0 == mRefCount)
    {
        FmConnection* pThis = const_cast<FmConnection*> (this);
        FM_LOG_DEBUG("FmConnection destructor : %p", pThis);
        delete pThis;
        pThis = NULL;
    }
}

/*****************************************************************************
 * This method is used to check if the connection is still active and in healthy state
 *****************************************************************************/
bool FmConnection::IsConnectionActive() {
    
    /* Note : The condition is termed as active only when FM_CONNECTION_ACTIVE bit is set 
       and no other bit is set. This is the reason the comparison is performed with FM_CONNECTION_ACTIVE
       to ensure only active bit is set */

    if (GetConnectionState() == FM_CONNECTION_ACTIVE) {
        return true;
    } else {
        return false;
    }    
}

/*****************************************************************************
 * This method is used to set the connection state
 *****************************************************************************/
void FmConnection::SetConnectionState(ConnectionState state) {
    mConnectionState = state;
    SignalConnStateChange(state);
}

/*****************************************************************************
 * This method is used to get the connection state 
 *****************************************************************************/
ConnectionState FmConnection::GetConnectionState() {
    ConnectionState state;
    state = mConnectionState;
    return state;
}

/*****************************************************************************/
void FmConnection::SetConnectionId(fm_connection_id_t connectionId)
{
    mConnectionId = connectionId;
}

/*****************************************************************************/
fm_connection_id_t FmConnection::GetConnectionId()
{
    return mConnectionId;
}

/*****************************************************************************/
void FmConnection::RemoveFromConnectionTable()
{
    if(mpConnectionHandler)
    {
        mpConnectionHandler->RemoveFromConnectionTable(mConnectionId);
        /* NOTE: Don't do anything after this. This causes a DecrReference and "this" may
                 now be invalid */
    }
}


#if 0
/*****************************************************************************
 * This method is used to Set the Output Buffer to be sent back to the 
 * client
 *****************************************************************************/
int FmConnection::SetOutputBuffer(void *buf, int length) {
    if (!IsConnectionActive()) {
        return -1;
    }
    
    bufferevent_lock(mpBufEv);
    
    /* Update libevent output buffer with the data to be sent to the client */
    evbuffer_add(mpOutputBuffer, buf, length);
    
    if (bufferevent_write_buffer(mpBufEv, mpOutputBuffer)) {
        FM_LOG_ERROR("Failed to write message to event buffer");
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
int FmConnection::Lock(void)
{
    lwosEnterCriticalSection(&mRequestTableLock);
    return FM_INT_ST_OK;
}

/*****************************************************************************
 Release Lock
 *****************************************************************************/
int FmConnection::UnLock(void)
{
    lwosLeaveCriticalSection(&mRequestTableLock);
    return FM_INT_ST_OK;
}
