/*
 * Copyright 2001-2007 Niels Provos <provos@citi.umich.edu>
 * Copyright 2007-2012 Niels Provos and Nick Mathewson
 *
 * This header file contains definitions for dealing with HTTP requests
 * that are internal to libevent.  As user of the library, you should not
 * need to know about these.
 */

#ifndef _HTTP_INTERNAL_H_
#define _HTTP_INTERNAL_H_

#include "event2/event_struct.h"
#include "util-internal.h"
#include "defer-internal.h"

#define HTTP_CONNECT_TIMEOUT	45
#define HTTP_WRITE_TIMEOUT	50
#define HTTP_READ_TIMEOUT	50

#define HTTP_PREFIX		"http://"
#define HTTP_DEFAULTPORT	80

enum message_read_status {
	ALL_DATA_READ = 1,
	MORE_DATA_EXPECTED = 0,
	DATA_CORRUPTED = -1,
	REQUEST_CANCELED = -2,
	DATA_TOO_LONG = -3
};

enum evhttp_connection_error {
	EVCON_HTTP_TIMEOUT,
	EVCON_HTTP_EOF,
	EVCON_HTTP_ILWALID_HEADER,
	EVCON_HTTP_BUFFER_ERROR,
	EVCON_HTTP_REQUEST_CANCEL
};

struct evbuffer;
struct addrinfo;
struct evhttp_request;

/* Indicates an unknown request method. */
#define _EVHTTP_REQ_UNKNOWN (1<<15)

enum evhttp_connection_state {
	EVCON_DISCONNECTED,	/**< not lwrrently connected not trying either*/
	EVCON_CONNECTING,	/**< tries to lwrrently connect */
	EVCON_IDLE,		/**< connection is established */
	EVCON_READING_FIRSTLINE,/**< reading Request-Line (incoming conn) or
				 **< Status-Line (outgoing conn) */
	EVCON_READING_HEADERS,	/**< reading request/response headers */
	EVCON_READING_BODY,	/**< reading request/response body */
	EVCON_READING_TRAILER,	/**< reading request/response chunked trailer */
	EVCON_WRITING		/**< writing request/response headers/body */
};

struct event_base;

/* A client or server connection. */
struct evhttp_connection {
	/* we use this tailq only if this connection was created for an http
	 * server */
	TAILQ_ENTRY(evhttp_connection) next;

	evutil_socket_t fd;
	struct bufferevent *bufev;

	struct event retry_ev;		/* for retrying connects */

	char *bind_address;		/* address to use for binding the src */
	u_short bind_port;		/* local port for binding the src */

	char *address;			/* address to connect to */
	u_short port;

	size_t max_headers_size;
	ev_uint64_t max_body_size;

	int flags;
#define EVHTTP_CON_INCOMING	0x0001	/* only one request on it ever */
#define EVHTTP_CON_OUTGOING	0x0002  /* multiple requests possible */
#define EVHTTP_CON_CLOSEDETECT  0x0004  /* detecting if persistent close */

	int timeout;			/* timeout in seconds for events */
	int retry_cnt;			/* retry count */
	int retry_max;			/* maximum number of retries */

	enum evhttp_connection_state state;

	/* for server connections, the http server they are connected with */
	struct evhttp *http_server;

	TAILQ_HEAD(evcon_requestq, evhttp_request) requests;

	void (*cb)(struct evhttp_connection *, void *);
	void *cb_arg;

	void (*closecb)(struct evhttp_connection *, void *);
	void *closecb_arg;

	struct deferred_cb read_more_deferred_cb;

	struct event_base *base;
	struct evdns_base *dns_base;
};

/* A callback for an http server */
struct evhttp_cb {
	TAILQ_ENTRY(evhttp_cb) next;

	char *what;

	void (*cb)(struct evhttp_request *req, void *);
	void *cbarg;
};

/* both the http server as well as the rpc system need to queue connections */
TAILQ_HEAD(evconq, evhttp_connection);

/* each bound socket is stored in one of these */
struct evhttp_bound_socket {
	TAILQ_ENTRY(evhttp_bound_socket) next;

	struct evconnlistener *listener;
};

/* server alias list item. */
struct evhttp_server_alias {
	TAILQ_ENTRY(evhttp_server_alias) next;

	char *alias; /* the server alias. */
};

struct evhttp {
	/* Next vhost, if this is a vhost. */
	TAILQ_ENTRY(evhttp) next_vhost;

	/* All listeners for this host */
	TAILQ_HEAD(boundq, evhttp_bound_socket) sockets;

	TAILQ_HEAD(httpcbq, evhttp_cb) callbacks;

	/* All live connections on this host. */
	struct evconq connections;

	TAILQ_HEAD(vhostsq, evhttp) virtualhosts;

	TAILQ_HEAD(aliasq, evhttp_server_alias) aliases;

	/* NULL if this server is not a vhost */
	char *vhost_pattern;

	int timeout;

	size_t default_max_headers_size;
	ev_uint64_t default_max_body_size;

	/* Bitmask of all HTTP methods that we accept and pass to user
	 * callbacks. */
	ev_uint16_t allowed_methods;

	/* Fallback callback if all the other callbacks for this connection
	   don't match. */
	void (*gencb)(struct evhttp_request *req, void *);
	void *gencbarg;

	struct event_base *base;
};

/* XXX most of these functions could be static. */

/* resets the connection; can be reused for more requests */
void evhttp_connection_reset(struct evhttp_connection *);

/* connects if necessary */
int evhttp_connection_connect(struct evhttp_connection *);

/* notifies the current request that it failed; resets connection */
void evhttp_connection_fail(struct evhttp_connection *,
    enum evhttp_connection_error error);

enum message_read_status;

enum message_read_status evhttp_parse_firstline(struct evhttp_request *, struct evbuffer*);
enum message_read_status evhttp_parse_headers(struct evhttp_request *, struct evbuffer*);

void evhttp_start_read(struct evhttp_connection *);

/* response sending HTML the data in the buffer */
void evhttp_response_code(struct evhttp_request *, int, const char *);
void evhttp_send_page(struct evhttp_request *, struct evbuffer *);

#endif /* _HTTP_H */
