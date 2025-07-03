
#include "dcgm_structs.h"
#include "LwcmRequest.h"
#include "timelib.h"
#include "logging.h"

/*****************************************************************************/
LwcmRequest::LwcmRequest(dcgm_request_id_t requestId)
{
	m_status = DCGM_ST_PENDING;
	m_messages.clear();
	m_requestId = requestId;

	lwosInitializeCriticalSection(&m_mutex);
    lwosCondCreate(&m_condition);

    PRINT_DEBUG("%p %d", "LwcmRequest %p, requestId %d created", this, m_requestId);
}

/*****************************************************************************/
dcgm_request_id_t LwcmRequest::GetRequestId(void)
{
	return m_requestId;
}

/*****************************************************************************/
void LwcmRequest::SetRequestId(dcgm_request_id_t requestId)
{
	m_requestId = requestId;
}

/*****************************************************************************/
LwcmRequest::~LwcmRequest()
{
	std::vector<LwcmMessage *>::iterator messagesIt;
	LwcmMessage * msg;

	Lock();

	/* Free any messages left */
	for (messagesIt = m_messages.begin(); messagesIt != m_messages.end(); messagesIt++)
	{
		msg = *messagesIt;
		if(msg)
			delete(msg);
	}
	m_messages.clear();

	m_status = DCGM_ST_UNINITIALIZED;
	Unlock();

	lwosDeleteCriticalSection(&m_mutex);
    lwosCondDestroy(&m_condition);

    PRINT_DEBUG("%p", "LwcmRequest %p destructed", this);
}

/*****************************************************************************/
void LwcmRequest::Lock()
{
	lwosEnterCriticalSection(&m_mutex);
}

/*****************************************************************************/
void LwcmRequest::Unlock()
{
	lwosLeaveCriticalSection(&m_mutex);
}

/*****************************************************************************/
int LwcmRequest::Wait(int timeoutMs)
{
	int st = 0;
	int retSt = DCGM_ST_OK;

	if(m_status != DCGM_ST_PENDING)
	{
	    PRINT_DEBUG("%p %d", "LwcmRequest %p already in state %d", this, m_status);
		return DCGM_ST_OK; /* Already out of pending state */
	}

	/* Technically, we could sleep n x timeoutMs times here if we get n spurious
	 * wake-ups. We can deal with this corner case later
	 */
	Lock();
	while(m_status == DCGM_ST_PENDING)
	{
		st = lwosCondWait(&m_condition, &m_mutex, timeoutMs);
		if(st == LWOS_TIMEOUT)
		{
			retSt = DCGM_ST_TIMEOUT;
			break;
		}

		/* Were we signalled with another status like DCGM_ST_CONNECTION_NOT_VALID? */
		if(m_status != DCGM_ST_PENDING)
		{
		    retSt = m_status;
		    break;
		}
	}
	Unlock();

	PRINT_DEBUG("%p %d %d", "LwcmRequest %p wait complete. m_status %d, retSt %d", this, m_status, retSt);

	return retSt;
}

/*****************************************************************************/
int LwcmRequest::MessageCount()
{
	int count;

	/* Don't take any chances */
	Lock();
	count = (int)m_messages.size();
	Unlock();
	return count;
}

/*****************************************************************************/
int LwcmRequest::ProcessMessage(LwcmMessage *msg)
{
	if(!msg)
		return DCGM_ST_BADPARAM;

	PRINT_DEBUG("%p %p", "LwcmRequest::ProcessMessage msg %p LwcmRequest %p", msg, this);

	Lock();
	m_status = DCGM_ST_OK;
	m_messages.push_back(msg);
	lwosCondBroadcast(&m_condition);
	Unlock();

	return DCGM_ST_OK;
}

/*****************************************************************************/
int LwcmRequest::SetStatus(int status)
{
    PRINT_DEBUG("%p %d", "LwcmRequest::SetStatus LwcmRequest %p, status %d", this, status);

    Lock();
    m_status = status;
    lwosCondBroadcast(&m_condition);
    Unlock();
    return DCGM_ST_OK;
}

/*****************************************************************************/
LwcmMessage *LwcmRequest::GetNextMessage()
{
	LwcmMessage *retVal = 0;
	std::vector<LwcmMessage *>::iterator it;

	Lock();

	/* Intentionally always get first element since we're trying to pop off the front */


	for(it = m_messages.begin(); it != m_messages.end(); it++)
	{
	    retVal = *it;
	    if(!retVal)
	    {
	        PRINT_ERROR("%p", "LwcmRequest::GetNextMessage %p got null message", this);
	        m_messages.erase(it);
	        continue; /* Keep trying */
	    }

	    /* Have a value message. Return it */
	    m_messages.erase(it);

	    Unlock();

	    PRINT_DEBUG("%p %p", "LwcmRequest::GetNextMessage %p found message %p", this, retVal);
	    return retVal;
	}

	Unlock();
	PRINT_DEBUG("%p", "LwcmRequest::GetNextMessage %p found no messages", this);
	return NULL;
}

/*****************************************************************************/
