#ifndef FMREQUEST_H
#define FMREQUEST_H

#include "FmSocketMessage.h"
#include <vector>
#include "lwos.h"

/*****************************************************************************/
class FmRequest
{
public:
	FmRequest(fm_request_id_t requestId);
	virtual ~FmRequest();

	/*************************************************************************/
	/* Accessors for m_requestId */
	void SetRequestId(fm_request_id_t requestId);
	fm_request_id_t GetRequestId();

	/*************************************************************************/
	/* Overridable method for processing an incoming message.
	 *
	 * Calling this method yields ownership of msg to this object.
	 *
	 * Returns FM_INT_ST_SUCCESS on success.
	 *         Other FM_INT_ST_? status code on error
	 */
	virtual int ProcessMessage(FmSocketMessage *msg);

	/*************************************************************************/
	/*
	 * Wait for object to leave pending state
	 *
	 * Returns FM_INT_ST_OK if left pending state
	 *         FM_INT_ST_TIMEOUT if waited timeoutMs milliseconds
	 *
	 **/
	int Wait(int timeoutMs);

	/*************************************************************************/
	/*
	 * Change the status of this request object and broadcast any changes to
	 * its associated waiters. This can cause a Wait() in another thread to
	 * wake up
	 *
	 * status should be a FM_INT_ST_? enum.
	 *
	 */
	int SetStatus(int status);

	/*************************************************************************/
	/*
	 * Return number of responses ready to process
	 *
	 */
	int MessageCount();

	/*************************************************************************/
	/*
	 * Get next message to process. Messages are returned in the order they
	 * were passed to ProcessMessage
	 *
	 * Returns Message pointer if found
	 *         NULL if not
	 */
	FmSocketMessage *GetNextMessage();

	/*************************************************************************/
protected:
	int m_status;		/* Status of this request. A FM_INT_ST_? constant */
	fm_request_id_t m_requestId;    /* Request identifier of this request. Should be unique
	                                     across all requests of this connection */
	std::vector<FmSocketMessage *>m_messages; /* Vector of responses in the order they
	                                         were received we will expect one for
	                                         now, but there could be more in the
	                                         future */

	LWOSCriticalSection m_mutex; /* We need a lock around this because we may be
	                                updating this out of band after the initial
	                                ack of our data has completed */
	lwosCV m_condition;          /* Condition used to signal Wait that we have
	                                left pending state. */

	/*************************************************************************/
	/* Used for protecting the internal state of this class */
	void Lock();
	void Unlock();

	/*************************************************************************/
};

/*****************************************************************************/

#endif //FMREQUEST_H
