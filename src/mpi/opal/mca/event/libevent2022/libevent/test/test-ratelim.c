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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#ifdef WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
# ifdef _XOPEN_SOURCE_EXTENDED
#  include <arpa/inet.h>
# endif
#endif
#include <signal.h>

#include "event2/bufferevent.h"
#include "event2/buffer.h"
#include "event2/event.h"
#include "event2/util.h"
#include "event2/listener.h"
#include "event2/thread.h"

#include "../util-internal.h"

static int cfg_verbose = 0;
static int cfg_help = 0;

static int cfg_n_connections = 30;
static int cfg_duration = 5;
static int cfg_connlimit = 0;
static int cfg_grouplimit = 0;
static int cfg_tick_msec = 1000;
static int cfg_min_share = -1;

static int cfg_connlimit_tolerance = -1;
static int cfg_grouplimit_tolerance = -1;
static int cfg_stddev_tolerance = -1;

#ifdef _WIN32
static int cfg_enable_iocp = 0;
#endif

static struct timeval cfg_tick = { 0, 500*1000 };

static struct ev_token_bucket_cfg *conn_bucket_cfg = NULL;
static struct ev_token_bucket_cfg *group_bucket_cfg = NULL;
struct bufferevent_rate_limit_group *ratelim_group = NULL;
static double seconds_per_tick = 0.0;

struct client_state {
	size_t queued;
	ev_uint64_t received;
};

static int n_echo_conns_open = 0;

static void
loud_writecb(struct bufferevent *bev, void *ctx)
{
	struct client_state *cs = ctx;
	struct evbuffer *output = bufferevent_get_output(bev);
	char buf[1024];
#ifdef WIN32
	int r = rand() % 256;
#else
	int r = random() % 256;
#endif
	memset(buf, r, sizeof(buf));
	while (evbuffer_get_length(output) < 8192) {
		evbuffer_add(output, buf, sizeof(buf));
		cs->queued += sizeof(buf);
	}
}

static void
discard_readcb(struct bufferevent *bev, void *ctx)
{
	struct client_state *cs = ctx;
	struct evbuffer *input = bufferevent_get_input(bev);
	size_t len = evbuffer_get_length(input);
	evbuffer_drain(input, len);
	cs->received += len;
}

static void
write_on_connectedcb(struct bufferevent *bev, short what, void *ctx)
{
	if (what & BEV_EVENT_CONNECTED) {
		loud_writecb(bev, ctx);
		/* XXXX this shouldn't be needed. */
		bufferevent_enable(bev, EV_READ|EV_WRITE);
	}
}

static void
echo_readcb(struct bufferevent *bev, void *ctx)
{
	struct evbuffer *input = bufferevent_get_input(bev);
	struct evbuffer *output = bufferevent_get_output(bev);

	evbuffer_add_buffer(output, input);
	if (evbuffer_get_length(output) > 1024000)
		bufferevent_disable(bev, EV_READ);
}

static void
echo_writecb(struct bufferevent *bev, void *ctx)
{
	struct evbuffer *output = bufferevent_get_output(bev);
	if (evbuffer_get_length(output) < 512000)
		bufferevent_enable(bev, EV_READ);
}

static void
echo_eventcb(struct bufferevent *bev, short what, void *ctx)
{
	if (what & (BEV_EVENT_EOF|BEV_EVENT_ERROR)) {
		--n_echo_conns_open;
		bufferevent_free(bev);
	}
}

static void
echo_listenercb(struct evconnlistener *listener, evutil_socket_t newsock,
    struct sockaddr *sourceaddr, int socklen, void *ctx)
{
	struct event_base *base = ctx;
	int flags = BEV_OPT_CLOSE_ON_FREE|BEV_OPT_THREADSAFE;
	struct bufferevent *bev;

	bev = bufferevent_socket_new(base, newsock, flags);
	bufferevent_setcb(bev, echo_readcb, echo_writecb, echo_eventcb, NULL);
	if (conn_bucket_cfg)
		bufferevent_set_rate_limit(bev, conn_bucket_cfg);
	if (ratelim_group)
		bufferevent_add_to_rate_limit_group(bev, ratelim_group);
	++n_echo_conns_open;
	bufferevent_enable(bev, EV_READ|EV_WRITE);
}

static int
test_ratelimiting(void)
{
	struct event_base *base;
	struct sockaddr_in sin;
	struct evconnlistener *listener;

	struct sockaddr_storage ss;
	ev_socklen_t slen;

	struct bufferevent **bevs;
	struct client_state *states;
	struct bufferevent_rate_limit_group *group = NULL;

	int i;

	struct timeval tv;

	ev_uint64_t total_received;
	double total_sq_persec, total_persec;
	double variance;
	double expected_total_persec = -1.0, expected_avg_persec = -1.0;
	int ok = 1;
	struct event_config *base_cfg;

	memset(&sin, 0, sizeof(sin));
	sin.sin_family = AF_INET;
	sin.sin_addr.s_addr = htonl(0x7f000001); /* 127.0.0.1 */
	sin.sin_port = 0; /* unspecified port */

	if (0)
		event_enable_debug_mode();

	base_cfg = event_config_new();

#ifdef _WIN32
	if (cfg_enable_iocp) {
		evthread_use_windows_threads();
		event_config_set_flag(base_cfg, EVENT_BASE_FLAG_STARTUP_IOCP);
	}
#endif

	base = event_base_new_with_config(base_cfg);
	event_config_free(base_cfg);

	listener = evconnlistener_new_bind(base, echo_listenercb, base,
	    LEV_OPT_CLOSE_ON_FREE|LEV_OPT_REUSEABLE, -1,
	    (struct sockaddr *)&sin, sizeof(sin));

	slen = sizeof(ss);
	if (getsockname(evconnlistener_get_fd(listener), (struct sockaddr *)&ss,
		&slen) < 0) {
		perror("getsockname");
		return 1;
	}

	if (cfg_connlimit > 0) {
		conn_bucket_cfg = ev_token_bucket_cfg_new(
			cfg_connlimit, cfg_connlimit * 4,
			cfg_connlimit, cfg_connlimit * 4,
			&cfg_tick);
		assert(conn_bucket_cfg);
	}

	if (cfg_grouplimit > 0) {
		group_bucket_cfg = ev_token_bucket_cfg_new(
			cfg_grouplimit, cfg_grouplimit * 4,
			cfg_grouplimit, cfg_grouplimit * 4,
			&cfg_tick);
		group = ratelim_group = bufferevent_rate_limit_group_new(
			base, group_bucket_cfg);
		expected_total_persec = cfg_grouplimit;
		expected_avg_persec = cfg_grouplimit / cfg_n_connections;
		if (cfg_connlimit > 0 && expected_avg_persec > cfg_connlimit)
			expected_avg_persec = cfg_connlimit;
		if (cfg_min_share >= 0)
			bufferevent_rate_limit_group_set_min_share(
				ratelim_group, cfg_min_share);
	}

	if (expected_avg_persec < 0 && cfg_connlimit > 0)
		expected_avg_persec = cfg_connlimit;

	if (expected_avg_persec > 0)
		expected_avg_persec /= seconds_per_tick;
	if (expected_total_persec > 0)
		expected_total_persec /= seconds_per_tick;

	bevs = calloc(cfg_n_connections, sizeof(struct bufferevent *));
	states = calloc(cfg_n_connections, sizeof(struct client_state));

	for (i = 0; i < cfg_n_connections; ++i) {
		bevs[i] = bufferevent_socket_new(base, -1,
		    BEV_OPT_CLOSE_ON_FREE|BEV_OPT_THREADSAFE);
		assert(bevs[i]);
		bufferevent_setcb(bevs[i], discard_readcb, loud_writecb,
		    write_on_connectedcb, &states[i]);
		bufferevent_enable(bevs[i], EV_READ|EV_WRITE);
		bufferevent_socket_connect(bevs[i], (struct sockaddr *)&ss,
		    slen);
	}

	tv.tv_sec = cfg_duration - 1;
	tv.tv_usec = 995000;

	event_base_loopexit(base, &tv);

	event_base_dispatch(base);

	ratelim_group = NULL; /* So no more responders get added */

	for (i = 0; i < cfg_n_connections; ++i) {
		bufferevent_free(bevs[i]);
	}
	evconnlistener_free(listener);

	/* Make sure no new echo_conns get added to the group. */
	ratelim_group = NULL;

	/* This should get _everybody_ freed */
	while (n_echo_conns_open) {
		printf("waiting for %d conns\n", n_echo_conns_open);
		tv.tv_sec = 0;
		tv.tv_usec = 300000;
		event_base_loopexit(base, &tv);
		event_base_dispatch(base);
	}

	if (group)
		bufferevent_rate_limit_group_free(group);

	total_received = 0;
	total_persec = 0.0;
	total_sq_persec = 0.0;
	for (i=0; i < cfg_n_connections; ++i) {
		double persec = states[i].received;
		persec /= cfg_duration;
		total_received += states[i].received;
		total_persec += persec;
		total_sq_persec += persec*persec;
		printf("%d: %f per second\n", i+1, persec);
	}
	printf("   total: %f per second\n",
	    ((double)total_received)/cfg_duration);
	if (expected_total_persec > 0) {
		double diff = expected_total_persec -
		    ((double)total_received/cfg_duration);
		printf("  [Off by %lf]\n", diff);
		if (cfg_grouplimit_tolerance > 0 &&
		    fabs(diff) > cfg_grouplimit_tolerance) {
			fprintf(stderr, "Group bandwidth out of bounds\n");
			ok = 0;
		}
	}

	printf(" average: %f per second\n",
	    (((double)total_received)/cfg_duration)/cfg_n_connections);
	if (expected_avg_persec > 0) {
		double diff = expected_avg_persec - (((double)total_received)/cfg_duration)/cfg_n_connections;
		printf("  [Off by %lf]\n", diff);
		if (cfg_connlimit_tolerance > 0 &&
		    fabs(diff) > cfg_connlimit_tolerance) {
			fprintf(stderr, "Connection bandwidth out of bounds\n");
			ok = 0;
		}
	}

	variance = total_sq_persec/cfg_n_connections - total_persec*total_persec/(cfg_n_connections*cfg_n_connections);

	printf("  stddev: %f per second\n", sqrt(variance));
	if (cfg_stddev_tolerance > 0 &&
	    sqrt(variance) > cfg_stddev_tolerance) {
		fprintf(stderr, "Connection variance out of bounds\n");
		ok = 0;
	}

	event_base_free(base);
	free(bevs);
	free(states);

	return ok ? 0 : 1;
}

static struct option {
	const char *name; int *ptr; int min; int isbool;
} options[] = {
	{ "-v", &cfg_verbose, 0, 1 },
	{ "-h", &cfg_help, 0, 1 },
	{ "-n", &cfg_n_connections, 1, 0 },
	{ "-d", &cfg_duration, 1, 0 },
	{ "-c", &cfg_connlimit, 0, 0 },
	{ "-g", &cfg_grouplimit, 0, 0 },
	{ "-t", &cfg_tick_msec, 10, 0 },
	{ "--min-share", &cfg_min_share, 0, 0 },
	{ "--check-connlimit", &cfg_connlimit_tolerance, 0, 0 },
	{ "--check-grouplimit", &cfg_grouplimit_tolerance, 0, 0 },
	{ "--check-stddev", &cfg_stddev_tolerance, 0, 0 },
#ifdef _WIN32
	{ "--iocp", &cfg_enable_iocp, 0, 1 },
#endif
	{ NULL, NULL, -1, 0 },
};

static int
handle_option(int argc, char **argv, int *i, const struct option *opt)
{
	long val;
	char *endptr = NULL;
	if (opt->isbool) {
		*opt->ptr = 1;
		return 0;
	}
	if (*i + 1 == argc) {
		fprintf(stderr, "Too few arguments to '%s'\n",argv[*i]);
		return -1;
	}
	val = strtol(argv[*i+1], &endptr, 10);
	if (*argv[*i+1] == '\0' || !endptr || *endptr != '\0') {
		fprintf(stderr, "Couldn't parse numeric value '%s'\n",
		    argv[*i+1]);
		return -1;
	}
	if (val < opt->min || val > 0x7fffffff) {
		fprintf(stderr, "Value '%s' is out-of-range'\n",
		    argv[*i+1]);
		return -1;
	}
	*opt->ptr = (int)val;
	++*i;
	return 0;
}

static void
usage(void)
{
	fprintf(stderr,
"test-ratelim [-v] [-n INT] [-d INT] [-c INT] [-g INT] [-t INT]\n\n"
"Pushes bytes through a number of possibly rate-limited connections, and\n"
"displays average throughput.\n\n"
"  -n INT: Number of connections to open (default: 30)\n"
"  -d INT: Duration of the test in seconds (default: 5 sec)\n");
	fprintf(stderr,
"  -c INT: Connection-rate limit applied to each connection in bytes per second\n"
"	   (default: None.)\n"
"  -g INT: Group-rate limit applied to sum of all usage in bytes per second\n"
"	   (default: None.)\n"
"  -t INT: Granularity of timing, in milliseconds (default: 1000 msec)\n");
}

int
main(int argc, char **argv)
{
	int i,j;
	double ratio;

#ifdef WIN32
	WORD wVersionRequested = MAKEWORD(2,2);
	WSADATA wsaData;

	(void) WSAStartup(wVersionRequested, &wsaData);
#endif

#ifndef WIN32
	if (signal(SIGPIPE, SIG_IGN) == SIG_ERR)
		return 1;
#endif
	for (i = 1; i < argc; ++i) {
		for (j = 0; options[j].name; ++j) {
			if (!strcmp(argv[i],options[j].name)) {
				if (handle_option(argc,argv,&i,&options[j])<0)
					return 1;
				goto again;
			}
		}
		fprintf(stderr, "Unknown option '%s'\n", argv[i]);
		usage();
		return 1;
	again:
		;
	}
	if (cfg_help) {
		usage();
		return 0;
	}

	cfg_tick.tv_sec = cfg_tick_msec / 1000;
	cfg_tick.tv_usec = (cfg_tick_msec % 1000)*1000;

	seconds_per_tick = ratio = cfg_tick_msec / 1000.0;

	cfg_connlimit *= ratio;
	cfg_grouplimit *= ratio;

	{
		struct timeval tv;
		evutil_gettimeofday(&tv, NULL);
#ifdef WIN32
		srand(tv.tv_usec);
#else
		srandom(tv.tv_usec);
#endif
	}

#ifndef _EVENT_DISABLE_THREAD_SUPPORT
	evthread_enable_lock_debuging();
#endif

	return test_ratelimiting();
}
