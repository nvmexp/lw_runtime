/*
 * Copyright (c) 2007-2012 Niels Provos and Nick Mathewson
 * Copyright (c) 2002-2006 Niels Provos <provos@citi.umich.edu>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <sys/types.h>

#include "event2/event-config.h"

#ifdef _EVENT_HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _EVENT_HAVE_STDARG_H
#include <stdarg.h>
#endif
#ifdef _EVENT_HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#ifdef _EVENT_HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif
#ifdef _EVENT_HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef _EVENT_HAVE_NETINET_IN6_H
#include <netinet/in6.h>
#endif

#include "event2/util.h"
#include "event2/bufferevent.h"
#include "event2/buffer.h"
#include "event2/bufferevent_struct.h"
#include "event2/bufferevent_compat.h"
#include "event2/event.h"
#include "log-internal.h"
#include "mm-internal.h"
#include "bufferevent-internal.h"
#include "util-internal.h"
#ifdef WIN32
#include "iocp-internal.h"
#endif

/* prototypes */
static int be_socket_enable(struct bufferevent *, short);
static int be_socket_disable(struct bufferevent *, short);
static void be_socket_destruct(struct bufferevent *);
static int be_socket_adj_timeouts(struct bufferevent *);
static int be_socket_flush(struct bufferevent *, short, enum bufferevent_flush_mode);
static int be_socket_ctrl(struct bufferevent *, enum bufferevent_ctrl_op, union bufferevent_ctrl_data *);

static void be_socket_setfd(struct bufferevent *, evutil_socket_t);

const struct bufferevent_ops bufferevent_ops_socket = {
	"socket",
	evutil_offsetof(struct bufferevent_private, bev),
	be_socket_enable,
	be_socket_disable,
	be_socket_destruct,
	be_socket_adj_timeouts,
	be_socket_flush,
	be_socket_ctrl,
};

#define be_socket_add(ev, t)			\
	_bufferevent_add_event((ev), (t))

static void
bufferevent_socket_outbuf_cb(struct evbuffer *buf,
    const struct evbuffer_cb_info *cbinfo,
    void *arg)
{
	struct bufferevent *bufev = arg;
	struct bufferevent_private *bufev_p =
	    EVUTIL_UPCAST(bufev, struct bufferevent_private, bev);

	if (cbinfo->n_added &&
	    (bufev->enabled & EV_WRITE) &&
	    !event_pending(&bufev->ev_write, EV_WRITE, NULL) &&
	    !bufev_p->write_suspended) {
		/* Somebody added data to the buffer, and we would like to
		 * write, and we were not writing.  So, start writing. */
		if (be_socket_add(&bufev->ev_write, &bufev->timeout_write) == -1) {
		    /* Should we log this? */
		}
	}
}

static void
bufferevent_readcb(evutil_socket_t fd, short event, void *arg)
{
	struct bufferevent *bufev = arg;
	struct bufferevent_private *bufev_p =
	    EVUTIL_UPCAST(bufev, struct bufferevent_private, bev);
	struct evbuffer *input;
	int res = 0;
	short what = BEV_EVENT_READING;
	ev_ssize_t howmuch = -1, readmax=-1;

	_bufferevent_incref_and_lock(bufev);

	if (event == EV_TIMEOUT) {
		/* Note that we only check for event==EV_TIMEOUT. If
		 * event==EV_TIMEOUT|EV_READ, we can safely ignore the
		 * timeout, since a read has oclwrred */
		what |= BEV_EVENT_TIMEOUT;
		goto error;
	}

	input = bufev->input;

	/*
	 * If we have a high watermark configured then we don't want to
	 * read more data than would make us reach the watermark.
	 */
	if (bufev->wm_read.high != 0) {
		howmuch = bufev->wm_read.high - evbuffer_get_length(input);
		/* we somehow lowered the watermark, stop reading */
		if (howmuch <= 0) {
			bufferevent_wm_suspend_read(bufev);
			goto done;
		}
	}
	readmax = _bufferevent_get_read_max(bufev_p);
	if (howmuch < 0 || howmuch > readmax) /* The use of -1 for "unlimited"
					       * uglifies this code. XXXX */
		howmuch = readmax;
	if (bufev_p->read_suspended)
		goto done;

	evbuffer_unfreeze(input, 0);
	res = evbuffer_read(input, fd, (int)howmuch); /* XXXX evbuffer_read would do better to take and return ev_ssize_t */
	evbuffer_freeze(input, 0);

	if (res == -1) {
		int err = evutil_socket_geterror(fd);
		if (EVUTIL_ERR_RW_RETRIABLE(err))
			goto reschedule;
		/* error case */
		what |= BEV_EVENT_ERROR;
	} else if (res == 0) {
		/* eof case */
		what |= BEV_EVENT_EOF;
	}

	if (res <= 0)
		goto error;

	_bufferevent_decrement_read_buckets(bufev_p, res);

	/* Ilwoke the user callback - must always be called last */
	if (evbuffer_get_length(input) >= bufev->wm_read.low)
		_bufferevent_run_readcb(bufev);

	goto done;

 reschedule:
	goto done;

 error:
	bufferevent_disable(bufev, EV_READ);
	_bufferevent_run_eventcb(bufev, what);

 done:
	_bufferevent_decref_and_unlock(bufev);
}

static void
bufferevent_writecb(evutil_socket_t fd, short event, void *arg)
{
	struct bufferevent *bufev = arg;
	struct bufferevent_private *bufev_p =
	    EVUTIL_UPCAST(bufev, struct bufferevent_private, bev);
	int res = 0;
	short what = BEV_EVENT_WRITING;
	int connected = 0;
	ev_ssize_t atmost = -1;

	_bufferevent_incref_and_lock(bufev);

	if (event == EV_TIMEOUT) {
		/* Note that we only check for event==EV_TIMEOUT. If
		 * event==EV_TIMEOUT|EV_WRITE, we can safely ignore the
		 * timeout, since a read has oclwrred */
		what |= BEV_EVENT_TIMEOUT;
		goto error;
	}
	if (bufev_p->connecting) {
		int c = evutil_socket_finished_connecting(fd);
		/* we need to fake the error if the connection was refused
		 * immediately - usually connection to localhost on BSD */
		if (bufev_p->connection_refused) {
		  bufev_p->connection_refused = 0;
		  c = -1;
		}

		if (c == 0)
			goto done;

		bufev_p->connecting = 0;
		if (c < 0) {
			event_del(&bufev->ev_write);
			event_del(&bufev->ev_read);
			_bufferevent_run_eventcb(bufev, BEV_EVENT_ERROR);
			goto done;
		} else {
			connected = 1;
#ifdef WIN32
			if (BEV_IS_ASYNC(bufev)) {
				event_del(&bufev->ev_write);
				bufferevent_async_set_connected(bufev);
				_bufferevent_run_eventcb(bufev,
						BEV_EVENT_CONNECTED);
				goto done;
			}
#endif
			_bufferevent_run_eventcb(bufev,
					BEV_EVENT_CONNECTED);
			if (!(bufev->enabled & EV_WRITE) ||
			    bufev_p->write_suspended) {
				event_del(&bufev->ev_write);
				goto done;
			}
		}
	}

	atmost = _bufferevent_get_write_max(bufev_p);

	if (bufev_p->write_suspended)
		goto done;

	if (evbuffer_get_length(bufev->output)) {
		evbuffer_unfreeze(bufev->output, 1);
		res = evbuffer_write_atmost(bufev->output, fd, atmost);
		evbuffer_freeze(bufev->output, 1);
		if (res == -1) {
			int err = evutil_socket_geterror(fd);
			if (EVUTIL_ERR_RW_RETRIABLE(err))
				goto reschedule;
			what |= BEV_EVENT_ERROR;
		} else if (res == 0) {
			/* eof case
			   XXXX Actually, a 0 on write doesn't indicate
			   an EOF. An ECONNRESET might be more typical.
			 */
			what |= BEV_EVENT_EOF;
		}
		if (res <= 0)
			goto error;

		_bufferevent_decrement_write_buckets(bufev_p, res);
	}

	if (evbuffer_get_length(bufev->output) == 0) {
		event_del(&bufev->ev_write);
	}

	/*
	 * Ilwoke the user callback if our buffer is drained or below the
	 * low watermark.
	 */
	if ((res || !connected) &&
	    evbuffer_get_length(bufev->output) <= bufev->wm_write.low) {
		_bufferevent_run_writecb(bufev);
	}

	goto done;

 reschedule:
	if (evbuffer_get_length(bufev->output) == 0) {
		event_del(&bufev->ev_write);
	}
	goto done;

 error:
	bufferevent_disable(bufev, EV_WRITE);
	_bufferevent_run_eventcb(bufev, what);

 done:
	_bufferevent_decref_and_unlock(bufev);
}

struct bufferevent *
bufferevent_socket_new(struct event_base *base, evutil_socket_t fd,
    int options)
{
	struct bufferevent_private *bufev_p;
	struct bufferevent *bufev;

#ifdef WIN32
	if (base && event_base_get_iocp(base))
		return bufferevent_async_new(base, fd, options);
#endif

	if ((bufev_p = mm_calloc(1, sizeof(struct bufferevent_private)))== NULL)
		return NULL;

	if (bufferevent_init_common(bufev_p, base, &bufferevent_ops_socket,
				    options) < 0) {
		mm_free(bufev_p);
		return NULL;
	}
	bufev = &bufev_p->bev;
	evbuffer_set_flags(bufev->output, EVBUFFER_FLAG_DRAINS_TO_FD);

	event_assign(&bufev->ev_read, bufev->ev_base, fd,
	    EV_READ|EV_PERSIST, bufferevent_readcb, bufev);
	event_assign(&bufev->ev_write, bufev->ev_base, fd,
	    EV_WRITE|EV_PERSIST, bufferevent_writecb, bufev);

	evbuffer_add_cb(bufev->output, bufferevent_socket_outbuf_cb, bufev);

	evbuffer_freeze(bufev->input, 0);
	evbuffer_freeze(bufev->output, 1);

	return bufev;
}

int
bufferevent_socket_connect(struct bufferevent *bev,
    struct sockaddr *sa, int socklen)
{
	struct bufferevent_private *bufev_p =
	    EVUTIL_UPCAST(bev, struct bufferevent_private, bev);

	evutil_socket_t fd;
	int r = 0;
	int result=-1;
	int ownfd = 0;

	_bufferevent_incref_and_lock(bev);

	if (!bufev_p)
		goto done;

	fd = bufferevent_getfd(bev);
	if (fd < 0) {
		if (!sa)
			goto done;
		fd = socket(sa->sa_family, SOCK_STREAM, 0);
		if (fd < 0)
			goto done;
		if (evutil_make_socket_nonblocking(fd)<0)
			goto done;
		ownfd = 1;
	}
	if (sa) {
#ifdef WIN32
		if (bufferevent_async_can_connect(bev)) {
			bufferevent_setfd(bev, fd);
			r = bufferevent_async_connect(bev, fd, sa, socklen);
			if (r < 0)
				goto freesock;
			bufev_p->connecting = 1;
			result = 0;
			goto done;
		} else
#endif
		r = evutil_socket_connect(&fd, sa, socklen);
		if (r < 0)
			goto freesock;
	}
#ifdef WIN32
	/* ConnectEx() isn't always around, even when IOCP is enabled.
	 * Here, we borrow the socket object's write handler to fall back
	 * on a non-blocking connect() when ConnectEx() is unavailable. */
	if (BEV_IS_ASYNC(bev)) {
		event_assign(&bev->ev_write, bev->ev_base, fd,
		    EV_WRITE|EV_PERSIST, bufferevent_writecb, bev);
	}
#endif
	bufferevent_setfd(bev, fd);
	if (r == 0) {
		if (! be_socket_enable(bev, EV_WRITE)) {
			bufev_p->connecting = 1;
			result = 0;
			goto done;
		}
	} else if (r == 1) {
		/* The connect succeeded already. How very BSD of it. */
		result = 0;
		bufev_p->connecting = 1;
		event_active(&bev->ev_write, EV_WRITE, 1);
	} else {
		/* The connect failed already.  How very BSD of it. */
		bufev_p->connection_refused = 1;
		bufev_p->connecting = 1;
		result = 0;
		event_active(&bev->ev_write, EV_WRITE, 1);
	}

	goto done;

freesock:
	_bufferevent_run_eventcb(bev, BEV_EVENT_ERROR);
	if (ownfd)
		evutil_closesocket(fd);
	/* do something about the error? */
done:
	_bufferevent_decref_and_unlock(bev);
	return result;
}

static void
bufferevent_connect_getaddrinfo_cb(int result, struct evutil_addrinfo *ai,
    void *arg)
{
	struct bufferevent *bev = arg;
	struct bufferevent_private *bev_p =
	    EVUTIL_UPCAST(bev, struct bufferevent_private, bev);
	int r;
	BEV_LOCK(bev);

	bufferevent_unsuspend_write(bev, BEV_SUSPEND_LOOKUP);
	bufferevent_unsuspend_read(bev, BEV_SUSPEND_LOOKUP);

	if (result != 0) {
		bev_p->dns_error = result;
		_bufferevent_run_eventcb(bev, BEV_EVENT_ERROR);
		_bufferevent_decref_and_unlock(bev);
		if (ai)
			evutil_freeaddrinfo(ai);
		return;
	}

	/* XXX use the other addrinfos? */
	/* XXX use this return value */
	r = bufferevent_socket_connect(bev, ai->ai_addr, (int)ai->ai_addrlen);
	(void)r;
	_bufferevent_decref_and_unlock(bev);
	evutil_freeaddrinfo(ai);
}

int
bufferevent_socket_connect_hostname(struct bufferevent *bev,
    struct evdns_base *evdns_base, int family, const char *hostname, int port)
{
	char portbuf[10];
	struct evutil_addrinfo hint;
	int err;
	struct bufferevent_private *bev_p =
	    EVUTIL_UPCAST(bev, struct bufferevent_private, bev);

	if (family != AF_INET && family != AF_INET6 && family != AF_UNSPEC)
		return -1;
	if (port < 1 || port > 65535)
		return -1;

	BEV_LOCK(bev);
	bev_p->dns_error = 0;
	BEV_UNLOCK(bev);

	evutil_snprintf(portbuf, sizeof(portbuf), "%d", port);

	memset(&hint, 0, sizeof(hint));
	hint.ai_family = family;
	hint.ai_protocol = IPPROTO_TCP;
	hint.ai_socktype = SOCK_STREAM;

	bufferevent_suspend_write(bev, BEV_SUSPEND_LOOKUP);
	bufferevent_suspend_read(bev, BEV_SUSPEND_LOOKUP);

	bufferevent_incref(bev);
	err = evutil_getaddrinfo_async(evdns_base, hostname, portbuf,
	    &hint, bufferevent_connect_getaddrinfo_cb, bev);

	if (err == 0) {
		return 0;
	} else {
		bufferevent_unsuspend_write(bev, BEV_SUSPEND_LOOKUP);
		bufferevent_unsuspend_read(bev, BEV_SUSPEND_LOOKUP);
		return -1;
	}
}

int
bufferevent_socket_get_dns_error(struct bufferevent *bev)
{
	int rv;
	struct bufferevent_private *bev_p =
	    EVUTIL_UPCAST(bev, struct bufferevent_private, bev);

	BEV_LOCK(bev);
	rv = bev_p->dns_error;
	BEV_UNLOCK(bev);

	return rv;
}

/*
 * Create a new buffered event object.
 *
 * The read callback is ilwoked whenever we read new data.
 * The write callback is ilwoked whenever the output buffer is drained.
 * The error callback is ilwoked on a write/read error or on EOF.
 *
 * Both read and write callbacks maybe NULL.  The error callback is not
 * allowed to be NULL and have to be provided always.
 */

struct bufferevent *
bufferevent_new(evutil_socket_t fd,
    bufferevent_data_cb readcb, bufferevent_data_cb writecb,
    bufferevent_event_cb eventcb, void *cbarg)
{
	struct bufferevent *bufev;

	if (!(bufev = bufferevent_socket_new(NULL, fd, 0)))
		return NULL;

	bufferevent_setcb(bufev, readcb, writecb, eventcb, cbarg);

	return bufev;
}


static int
be_socket_enable(struct bufferevent *bufev, short event)
{
	if (event & EV_READ) {
		if (be_socket_add(&bufev->ev_read,&bufev->timeout_read) == -1)
			return -1;
	}
	if (event & EV_WRITE) {
		if (be_socket_add(&bufev->ev_write,&bufev->timeout_write) == -1)
			return -1;
	}
	return 0;
}

static int
be_socket_disable(struct bufferevent *bufev, short event)
{
	struct bufferevent_private *bufev_p =
	    EVUTIL_UPCAST(bufev, struct bufferevent_private, bev);
	if (event & EV_READ) {
		if (event_del(&bufev->ev_read) == -1)
			return -1;
	}
	/* Don't actually disable the write if we are trying to connect. */
	if ((event & EV_WRITE) && ! bufev_p->connecting) {
		if (event_del(&bufev->ev_write) == -1)
			return -1;
	}
	return 0;
}

static void
be_socket_destruct(struct bufferevent *bufev)
{
	struct bufferevent_private *bufev_p =
	    EVUTIL_UPCAST(bufev, struct bufferevent_private, bev);
	evutil_socket_t fd;
	EVUTIL_ASSERT(bufev->be_ops == &bufferevent_ops_socket);

	fd = event_get_fd(&bufev->ev_read);

	event_del(&bufev->ev_read);
	event_del(&bufev->ev_write);

	if ((bufev_p->options & BEV_OPT_CLOSE_ON_FREE) && fd >= 0)
		EVUTIL_CLOSESOCKET(fd);
}

static int
be_socket_adj_timeouts(struct bufferevent *bufev)
{
	int r = 0;
	if (event_pending(&bufev->ev_read, EV_READ, NULL))
		if (be_socket_add(&bufev->ev_read, &bufev->timeout_read) < 0)
			r = -1;
	if (event_pending(&bufev->ev_write, EV_WRITE, NULL)) {
		if (be_socket_add(&bufev->ev_write, &bufev->timeout_write) < 0)
			r = -1;
	}
	return r;
}

static int
be_socket_flush(struct bufferevent *bev, short iotype,
    enum bufferevent_flush_mode mode)
{
	return 0;
}


static void
be_socket_setfd(struct bufferevent *bufev, evutil_socket_t fd)
{
	BEV_LOCK(bufev);
	EVUTIL_ASSERT(bufev->be_ops == &bufferevent_ops_socket);

	event_del(&bufev->ev_read);
	event_del(&bufev->ev_write);

	event_assign(&bufev->ev_read, bufev->ev_base, fd,
	    EV_READ|EV_PERSIST, bufferevent_readcb, bufev);
	event_assign(&bufev->ev_write, bufev->ev_base, fd,
	    EV_WRITE|EV_PERSIST, bufferevent_writecb, bufev);

	if (fd >= 0)
		bufferevent_enable(bufev, bufev->enabled);

	BEV_UNLOCK(bufev);
}

/* XXXX Should non-socket bufferevents support this? */
int
bufferevent_priority_set(struct bufferevent *bufev, int priority)
{
	int r = -1;

	BEV_LOCK(bufev);
	if (bufev->be_ops != &bufferevent_ops_socket)
		goto done;

	if (event_priority_set(&bufev->ev_read, priority) == -1)
		goto done;
	if (event_priority_set(&bufev->ev_write, priority) == -1)
		goto done;

	r = 0;
done:
	BEV_UNLOCK(bufev);
	return r;
}

/* XXXX Should non-socket bufferevents support this? */
int
bufferevent_base_set(struct event_base *base, struct bufferevent *bufev)
{
	int res = -1;

	BEV_LOCK(bufev);
	if (bufev->be_ops != &bufferevent_ops_socket)
		goto done;

	bufev->ev_base = base;

	res = event_base_set(base, &bufev->ev_read);
	if (res == -1)
		goto done;

	res = event_base_set(base, &bufev->ev_write);
done:
	BEV_UNLOCK(bufev);
	return res;
}

static int
be_socket_ctrl(struct bufferevent *bev, enum bufferevent_ctrl_op op,
    union bufferevent_ctrl_data *data)
{
	switch (op) {
	case BEV_CTRL_SET_FD:
		be_socket_setfd(bev, data->fd);
		return 0;
	case BEV_CTRL_GET_FD:
		data->fd = event_get_fd(&bev->ev_read);
		return 0;
	case BEV_CTRL_GET_UNDERLYING:
	case BEV_CTRL_CANCEL_ALL:
	default:
		return -1;
	}
}
