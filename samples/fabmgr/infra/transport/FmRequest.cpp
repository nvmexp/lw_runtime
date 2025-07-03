#include "FmRequest.h"
#include "timelib.h"
#include "fm_log.h"
#include "FMErrorCodesInternal.h"


/*****************************************************************************/
FmRequest::FmRequest(fm_request_id_t requestId)
{
	m_status = FM_INT_ST_PENDING;
	m_messages.clear();
	m_requestId = requestId;

	lwosInitializeCriticalSection(&m_mutex);
    lwosCondCreate(&m_condition);
}

/*****************************************************************************/
fm_request_id_t FmRequest::GetRequestId(void)
{
	return m_requestId;
}

/*****************************************************************************/
void FmRequest::SetRequestId(fm_request_id_t requestId)
{
	m_requestId = requestId;
}

/*****************************************************************************/
FmRequest::~FmRequest()
{
	std::vector<FmSocketMessage *>::iterator messagesIt;
	FmSocketMessage * msg;

	Lock();

	/* Free any messages left */
	for (messagesIt = m_messages.begin(); messagesIt != m_messages.end(); messagesIt++)
	{
		msg = *messagesIt;
		if(msg)
			delete(msg);
	}
	m_messages.clear();

	m_status = FM_INT_ST_UNINITIALIZED;
	Unlock();

	lwosDeleteCriticalSection(&m_mutex);
    lwosCondDestroy(&m_condition);
}

/*****************************************************************************/
void FmRequest::Lock()
{
	lwosEnterCriticalSection(&m_mutex);
}

/*****************************************************************************/
void FmRequest::Unlock()
{
	lwosLeaveCriticalSection(&m_mutex);
}

/*****************************************************************************/
int FmRequest::Wait(int timeoutMs)
{
	int st = 0;
	int retSt = FM_INT_ST_OK;

	if(m_status != FM_INT_ST_PENDING)
	{
	    FM_LOG_DEBUG("FmRequest %p already in state %d", this, m_status);
		return FM_INT_ST_OK; /* Already out of pending state */
	}

	/* Technically, we could sleep n x timeoutMs times here if we get n spurious
	 * wake-ups. We can deal with this corner case later
	 */
	Lock();
	while(m_status == FM_INT_ST_PENDING)
	{
		st = lwosCondWait(&m_condition, &m_mutex, timeoutMs);
		if(st == LWOS_TIMEOUT)
		{
			retSt = FM_INT_ST_TIMEOUT;
			break;
		}

		/* Were we signalled with another status like FM_ST_CONNECTION_NOT_VALID? */
		if(m_status != FM_INT_ST_PENDING)
		{
		    retSt = m_status;
		    break;
		}
	}
	Unlock();

	FM_LOG_DEBUG("FmRequest %p wait complete. m_status %d, retSt %d", this, m_status, retSt);

	return retSt;
}

/*****************************************************************************/
int FmRequest::MessageCount()
{
	int count;

	/* Don't take any chances */
	Lock();
	count = (int)m_messages.size();
	Unlock();
	return count;
}

/*****************************************************************************/
int FmRequest::ProcessMessage(FmSocketMessage *msg)
{
	if(!msg)
		return FM_INT_ST_BADPARAM;

	FM_LOG_DEBUG("FmRequest::ProcessMessage msg %p FmRequest %p", msg, this);

	Lock();
	m_status = FM_INT_ST_OK;
	m_messages.push_back(msg);
	lwosCondBroadcast(&m_condition);
	Unlock();

	return FM_INT_ST_OK;
}

/*****************************************************************************/
int FmRequest::SetStatus(int status)
{
    FM_LOG_DEBUG("FmRequest::SetStatus FmRequest %p, status %d", this, status);

    Lock();
    m_status = status;
    lwosCondBroadcast(&m_condition);
    Unlock();
    return FM_INT_ST_OK;
}

/*****************************************************************************/
FmSocketMessage *FmRequest::GetNextMessage()
{
	FmSocketMessage *retVal = 0;
	std::vector<FmSocketMessage *>::iterator it;

	Lock();

	/* Intentionally always get first element since we're trying to pop off the front */


	for(it = m_messages.begin(); it != m_messages.end();)
	{
	    retVal = *it;
	    if(!retVal)
	    {
   	        FM_LOG_ERROR("failed to get next fabric manager message from socket message queue %p handler", this);
	        it = m_messages.erase(it);
	        continue; /* Keep trying */
	    }

	    /* Have a value message. Return it */
	    m_messages.erase(it);

	    Unlock();

	    //FM_LOG_DEBUG("FmRequest::GetNextMessage %p found message %p", this, retVal);
	    return retVal;
	}

	Unlock();
	//FM_LOG_DEBUG("FmRequest::GetNextMessage %p found no messages", this);
	return NULL;
}

/*****************************************************************************/
