/*
 * Copyright (c) 2009-2012 Niels Provos and Nick Mathewson
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
#ifndef _EVENT2_BUFFEREVENT_SSL_H_
#define _EVENT2_BUFFEREVENT_SSL_H_

/** @file event2/bufferevent_ssl.h

    OpenSSL support for bufferevents.
 */

#include <event2/event-config.h>
#include <event2/bufferevent.h>
#include <event2/util.h>

#ifdef __cplusplus
extern "C" {
#endif

/* This is what openssl's SSL objects are underneath. */
struct ssl_st;

/**
   The state of an SSL object to be used when creating a new
   SSL bufferevent.
 */
enum bufferevent_ssl_state {
	BUFFEREVENT_SSL_OPEN = 0,
	BUFFEREVENT_SSL_CONNECTING = 1,
	BUFFEREVENT_SSL_ACCEPTING = 2
};

#if defined(_EVENT_HAVE_OPENSSL) || defined(_EVENT_IN_DOXYGEN)
/**
   Create a new SSL bufferevent to send its data over another bufferevent.

   @param base An event_base to use to detect reading and writing.  It
      must also be the base for the underlying bufferevent.
   @param underlying A socket to use for this SSL
   @param ssl A SSL* object from openssl.
   @param state The current state of the SSL connection
   @param options One or more bufferevent_options
   @return A new bufferevent on success, or NULL on failure
*/
struct bufferevent *
bufferevent_openssl_filter_new(struct event_base *base,
    struct bufferevent *underlying,
    struct ssl_st *ssl,
    enum bufferevent_ssl_state state,
    int options);

/**
   Create a new SSL bufferevent to send its data over an SSL * on a socket.

   @param base An event_base to use to detect reading and writing
   @param fd A socket to use for this SSL
   @param ssl A SSL* object from openssl.
   @param state The current state of the SSL connection
   @param options One or more bufferevent_options
   @return A new bufferevent on success, or NULL on failure.
*/
struct bufferevent *
bufferevent_openssl_socket_new(struct event_base *base,
    evutil_socket_t fd,
    struct ssl_st *ssl,
    enum bufferevent_ssl_state state,
    int options);

/** Return the underlying openssl SSL * object for an SSL bufferevent. */
struct ssl_st *
bufferevent_openssl_get_ssl(struct bufferevent *bufev);

/** Tells a bufferevent to begin SSL renegotiation. */
int bufferevent_ssl_renegotiate(struct bufferevent *bev);

/** Return the most recent OpenSSL error reported on an SSL bufferevent. */
unsigned long bufferevent_get_openssl_error(struct bufferevent *bev);

#endif

#ifdef __cplusplus
}
#endif

#endif /* _EVENT2_BUFFEREVENT_SSL_H_ */
