/*
 * Copyright (c) 2003-2007 Niels Provos <provos@citi.umich.edu>
 * Copyright (c) 2007-2012 Niels Provos and Nick Mathewson
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

#ifdef WIN32
#include <winsock2.h>
#include <windows.h>
#endif

#include "event2/event-config.h"

#include <sys/types.h>
#include <sys/stat.h>
#ifdef _EVENT_HAVE_SYS_TIME_H
#include <sys/time.h>
#endif
#include <sys/queue.h>
#ifndef WIN32
#include <sys/socket.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>
#include <netdb.h>
#endif
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <ctype.h>

#include "event2/event.h"
#include "event2/event_struct.h"
#include "event2/event_compat.h"
#include "event2/tag.h"
#include "event2/buffer.h"
#include "event2/buffer_compat.h"
#include "event2/util.h"
#include "event-internal.h"
#include "evthread-internal.h"
#include "util-internal.h"
#include "log-internal.h"

#include "regress.h"

#ifndef WIN32
#include "regress.gen.h"
#endif

evutil_socket_t pair[2];
int test_ok;
int called;
struct event_base *global_base;

static char wbuf[4096];
static char rbuf[4096];
static int woff;
static int roff;
static int usepersist;
static struct timeval tset;
static struct timeval tcalled;


#define TEST1	"this is a test"
#define SECONDS	1

#ifndef SHUT_WR
#define SHUT_WR 1
#endif

#ifdef WIN32
#define write(fd,buf,len) send((fd),(buf),(int)(len),0)
#define read(fd,buf,len) recv((fd),(buf),(int)(len),0)
#endif

struct basic_cb_args
{
	struct event_base *eb;
	struct event *ev;
	unsigned int callcount;
};

static void
simple_read_cb(evutil_socket_t fd, short event, void *arg)
{
	char buf[256];
	int len;

	len = read(fd, buf, sizeof(buf));

	if (len) {
		if (!called) {
			if (event_add(arg, NULL) == -1)
				exit(1);
		}
	} else if (called == 1)
		test_ok = 1;

	called++;
}

static void
basic_read_cb(evutil_socket_t fd, short event, void *data)
{
	char buf[256];
	int len;
	struct basic_cb_args *arg = data;

	len = read(fd, buf, sizeof(buf));

	if (len < 0) {
		tt_fail_perror("read (callback)");
	} else {
		switch (arg->callcount++) {
		case 0:	 /* first call: expect to read data; cycle */
			if (len > 0)
				return;

			tt_fail_msg("EOF before data read");
			break;

		case 1:	 /* second call: expect EOF; stop */
			if (len > 0)
				tt_fail_msg("not all data read on first cycle");
			break;

		default:  /* third call: should not happen */
			tt_fail_msg("too many cycles");
		}
	}

	event_del(arg->ev);
	event_base_loopexit(arg->eb, NULL);
}

static void
dummy_read_cb(evutil_socket_t fd, short event, void *arg)
{
}

static void
simple_write_cb(evutil_socket_t fd, short event, void *arg)
{
	int len;

	len = write(fd, TEST1, strlen(TEST1) + 1);
	if (len == -1)
		test_ok = 0;
	else
		test_ok = 1;
}

static void
multiple_write_cb(evutil_socket_t fd, short event, void *arg)
{
	struct event *ev = arg;
	int len;

	len = 128;
	if (woff + len >= (int)sizeof(wbuf))
		len = sizeof(wbuf) - woff;

	len = write(fd, wbuf + woff, len);
	if (len == -1) {
		fprintf(stderr, "%s: write\n", __func__);
		if (usepersist)
			event_del(ev);
		return;
	}

	woff += len;

	if (woff >= (int)sizeof(wbuf)) {
		shutdown(fd, SHUT_WR);
		if (usepersist)
			event_del(ev);
		return;
	}

	if (!usepersist) {
		if (event_add(ev, NULL) == -1)
			exit(1);
	}
}

static void
multiple_read_cb(evutil_socket_t fd, short event, void *arg)
{
	struct event *ev = arg;
	int len;

	len = read(fd, rbuf + roff, sizeof(rbuf) - roff);
	if (len == -1)
		fprintf(stderr, "%s: read\n", __func__);
	if (len <= 0) {
		if (usepersist)
			event_del(ev);
		return;
	}

	roff += len;
	if (!usepersist) {
		if (event_add(ev, NULL) == -1)
			exit(1);
	}
}

static void
timeout_cb(evutil_socket_t fd, short event, void *arg)
{
	struct timeval tv;
	int diff;

	evutil_gettimeofday(&tcalled, NULL);
	if (evutil_timercmp(&tcalled, &tset, >))
		evutil_timersub(&tcalled, &tset, &tv);
	else
		evutil_timersub(&tset, &tcalled, &tv);

	diff = tv.tv_sec*1000 + tv.tv_usec/1000 - SECONDS * 1000;
	if (diff < 0)
		diff = -diff;

	if (diff < 100)
		test_ok = 1;
}

struct both {
	struct event ev;
	int nread;
};

static void
combined_read_cb(evutil_socket_t fd, short event, void *arg)
{
	struct both *both = arg;
	char buf[128];
	int len;

	len = read(fd, buf, sizeof(buf));
	if (len == -1)
		fprintf(stderr, "%s: read\n", __func__);
	if (len <= 0)
		return;

	both->nread += len;
	if (event_add(&both->ev, NULL) == -1)
		exit(1);
}

static void
combined_write_cb(evutil_socket_t fd, short event, void *arg)
{
	struct both *both = arg;
	char buf[128];
	int len;

	len = sizeof(buf);
	if (len > both->nread)
		len = both->nread;

	memset(buf, 'q', len);

	len = write(fd, buf, len);
	if (len == -1)
		fprintf(stderr, "%s: write\n", __func__);
	if (len <= 0) {
		shutdown(fd, SHUT_WR);
		return;
	}

	both->nread -= len;
	if (event_add(&both->ev, NULL) == -1)
		exit(1);
}

/* These macros used to replicate the work of the legacy test wrapper code */
#define setup_test(x) do {						\
	if (!in_legacy_test_wrapper) {					\
		TT_FAIL(("Legacy test %s not wrapped properly", x));	\
		return;							\
	}								\
	} while (0)
#define cleanup_test() setup_test("cleanup")

static void
test_simpleread(void)
{
	struct event ev;

	/* Very simple read test */
	setup_test("Simple read: ");

	if (write(pair[0], TEST1, strlen(TEST1)+1) < 0) {
		tt_fail_perror("write");
	}

	shutdown(pair[0], SHUT_WR);

	event_set(&ev, pair[1], EV_READ, simple_read_cb, &ev);
	if (event_add(&ev, NULL) == -1)
		exit(1);
	event_dispatch();

	cleanup_test();
}

static void
test_simplewrite(void)
{
	struct event ev;

	/* Very simple write test */
	setup_test("Simple write: ");

	event_set(&ev, pair[0], EV_WRITE, simple_write_cb, &ev);
	if (event_add(&ev, NULL) == -1)
		exit(1);
	event_dispatch();

	cleanup_test();
}

static void
simpleread_multiple_cb(evutil_socket_t fd, short event, void *arg)
{
	if (++called == 2)
		test_ok = 1;
}

static void
test_simpleread_multiple(void)
{
	struct event one, two;

	/* Very simple read test */
	setup_test("Simple read to multiple evens: ");

	if (write(pair[0], TEST1, strlen(TEST1)+1) < 0) {
		tt_fail_perror("write");
	}

	shutdown(pair[0], SHUT_WR);

	event_set(&one, pair[1], EV_READ, simpleread_multiple_cb, NULL);
	if (event_add(&one, NULL) == -1)
		exit(1);
	event_set(&two, pair[1], EV_READ, simpleread_multiple_cb, NULL);
	if (event_add(&two, NULL) == -1)
		exit(1);
	event_dispatch();

	cleanup_test();
}

static int have_closed = 0;
static int premature_event = 0;
static void
simpleclose_close_fd_cb(evutil_socket_t s, short what, void *ptr)
{
	evutil_socket_t **fds = ptr;
	TT_BLATHER(("Closing"));
	evutil_closesocket(*fds[0]);
	evutil_closesocket(*fds[1]);
	*fds[0] = -1;
	*fds[1] = -1;
	have_closed = 1;
}

static void
record_event_cb(evutil_socket_t s, short what, void *ptr)
{
	short *whatp = ptr;
	if (!have_closed)
		premature_event = 1;
	*whatp = what;
	TT_BLATHER(("Recorded %d on socket %d", (int)what, (int)s));
}

static void
test_simpleclose(void *ptr)
{
	/* Test that a close of FD is detected as a read and as a write. */
	struct event_base *base = event_base_new();
	evutil_socket_t pair1[2]={-1,-1}, pair2[2] = {-1, -1};
	evutil_socket_t *to_close[2];
	struct event *rev=NULL, *wev=NULL, *closeev=NULL;
	struct timeval tv;
	short got_read_on_close = 0, got_write_on_close = 0;
	char buf[1024];
	memset(buf, 99, sizeof(buf));
#ifdef WIN32
#define LOCAL_SOCKETPAIR_AF AF_INET
#else
#define LOCAL_SOCKETPAIR_AF AF_UNIX
#endif
	if (evutil_socketpair(LOCAL_SOCKETPAIR_AF, SOCK_STREAM, 0, pair1)<0)
		TT_DIE(("socketpair: %s", strerror(errno)));
	if (evutil_socketpair(LOCAL_SOCKETPAIR_AF, SOCK_STREAM, 0, pair2)<0)
		TT_DIE(("socketpair: %s", strerror(errno)));
	if (evutil_make_socket_nonblocking(pair1[1]) < 0)
		TT_DIE(("make_socket_nonblocking"));
	if (evutil_make_socket_nonblocking(pair2[1]) < 0)
		TT_DIE(("make_socket_nonblocking"));

	/** Stuff pair2[1] full of data, until write fails */
	while (1) {
		int r = write(pair2[1], buf, sizeof(buf));
		if (r<0) {
			int err = evutil_socket_geterror(pair2[1]);
			if (! EVUTIL_ERR_RW_RETRIABLE(err))
				TT_DIE(("write failed strangely: %s",
					evutil_socket_error_to_string(err)));
			break;
		}
	}
	to_close[0] = &pair1[0];
	to_close[1] = &pair2[0];

	closeev = event_new(base, -1, EV_TIMEOUT, simpleclose_close_fd_cb,
	    to_close);
	rev = event_new(base, pair1[1], EV_READ, record_event_cb,
	    &got_read_on_close);
	TT_BLATHER(("Waiting for read on %d", (int)pair1[1]));
	wev = event_new(base, pair2[1], EV_WRITE, record_event_cb,
	    &got_write_on_close);
	TT_BLATHER(("Waiting for write on %d", (int)pair2[1]));
	tv.tv_sec = 0;
	tv.tv_usec = 100*1000; /* Close pair1[0] after a little while, and make
			       * sure we get a read event. */
	event_add(closeev, &tv);
	event_add(rev, NULL);
	event_add(wev, NULL);
	/* Don't let the test go on too long. */
	tv.tv_sec = 0;
	tv.tv_usec = 200*1000;
	event_base_loopexit(base, &tv);
	event_base_loop(base, 0);

	tt_int_op(got_read_on_close, ==, EV_READ);
	tt_int_op(got_write_on_close, ==, EV_WRITE);
	tt_int_op(premature_event, ==, 0);

end:
	if (pair1[0] >= 0)
		evutil_closesocket(pair1[0]);
	if (pair1[1] >= 0)
		evutil_closesocket(pair1[1]);
	if (pair2[0] >= 0)
		evutil_closesocket(pair2[0]);
	if (pair2[1] >= 0)
		evutil_closesocket(pair2[1]);
	if (rev)
		event_free(rev);
	if (wev)
		event_free(wev);
	if (closeev)
		event_free(closeev);
	if (base)
		event_base_free(base);
}


static void
test_multiple(void)
{
	struct event ev, ev2;
	int i;

	/* Multiple read and write test */
	setup_test("Multiple read/write: ");
	memset(rbuf, 0, sizeof(rbuf));
	for (i = 0; i < (int)sizeof(wbuf); i++)
		wbuf[i] = i;

	roff = woff = 0;
	usepersist = 0;

	event_set(&ev, pair[0], EV_WRITE, multiple_write_cb, &ev);
	if (event_add(&ev, NULL) == -1)
		exit(1);
	event_set(&ev2, pair[1], EV_READ, multiple_read_cb, &ev2);
	if (event_add(&ev2, NULL) == -1)
		exit(1);
	event_dispatch();

	if (roff == woff)
		test_ok = memcmp(rbuf, wbuf, sizeof(wbuf)) == 0;

	cleanup_test();
}

static void
test_persistent(void)
{
	struct event ev, ev2;
	int i;

	/* Multiple read and write test with persist */
	setup_test("Persist read/write: ");
	memset(rbuf, 0, sizeof(rbuf));
	for (i = 0; i < (int)sizeof(wbuf); i++)
		wbuf[i] = i;

	roff = woff = 0;
	usepersist = 1;

	event_set(&ev, pair[0], EV_WRITE|EV_PERSIST, multiple_write_cb, &ev);
	if (event_add(&ev, NULL) == -1)
		exit(1);
	event_set(&ev2, pair[1], EV_READ|EV_PERSIST, multiple_read_cb, &ev2);
	if (event_add(&ev2, NULL) == -1)
		exit(1);
	event_dispatch();

	if (roff == woff)
		test_ok = memcmp(rbuf, wbuf, sizeof(wbuf)) == 0;

	cleanup_test();
}

static void
test_combined(void)
{
	struct both r1, r2, w1, w2;

	setup_test("Combined read/write: ");
	memset(&r1, 0, sizeof(r1));
	memset(&r2, 0, sizeof(r2));
	memset(&w1, 0, sizeof(w1));
	memset(&w2, 0, sizeof(w2));

	w1.nread = 4096;
	w2.nread = 8192;

	event_set(&r1.ev, pair[0], EV_READ, combined_read_cb, &r1);
	event_set(&w1.ev, pair[0], EV_WRITE, combined_write_cb, &w1);
	event_set(&r2.ev, pair[1], EV_READ, combined_read_cb, &r2);
	event_set(&w2.ev, pair[1], EV_WRITE, combined_write_cb, &w2);
	tt_assert(event_add(&r1.ev, NULL) != -1);
	tt_assert(!event_add(&w1.ev, NULL));
	tt_assert(!event_add(&r2.ev, NULL));
	tt_assert(!event_add(&w2.ev, NULL));
	event_dispatch();

	if (r1.nread == 8192 && r2.nread == 4096)
		test_ok = 1;

end:
	cleanup_test();
}

static void
test_simpletimeout(void)
{
	struct timeval tv;
	struct event ev;

	setup_test("Simple timeout: ");

	tv.tv_usec = 0;
	tv.tv_sec = SECONDS;
	evtimer_set(&ev, timeout_cb, NULL);
	evtimer_add(&ev, &tv);

	evutil_gettimeofday(&tset, NULL);
	event_dispatch();

	cleanup_test();
}

static void
periodic_timeout_cb(evutil_socket_t fd, short event, void *arg)
{
	int *count = arg;

	(*count)++;
	if (*count == 6) {
		/* call loopexit only once - on slow machines(?), it is
		 * apparently possible for this to get called twice. */
		test_ok = 1;
		event_base_loopexit(global_base, NULL);
	}
}

static void
test_persistent_timeout(void)
{
	struct timeval tv;
	struct event ev;
	int count = 0;

	evutil_timerclear(&tv);
	tv.tv_usec = 10000;

	event_assign(&ev, global_base, -1, EV_TIMEOUT|EV_PERSIST,
	    periodic_timeout_cb, &count);
	event_add(&ev, &tv);

	event_dispatch();

	event_del(&ev);
}

static void
test_persistent_timeout_jump(void *ptr)
{
	struct basic_test_data *data = ptr;
	struct event ev;
	int count = 0;
	struct timeval msec100 = { 0, 100 * 1000 };
	struct timeval msec50 = { 0, 50 * 1000 };

	event_assign(&ev, data->base, -1, EV_PERSIST, periodic_timeout_cb, &count);
	event_add(&ev, &msec100);
	/* Wait for a bit */
#ifdef _WIN32
	Sleep(1000);
#else
	sleep(1);
#endif
	event_base_loopexit(data->base, &msec50);
	event_base_dispatch(data->base);
	tt_int_op(count, ==, 1);

end:
	event_del(&ev);
}

struct persist_active_timeout_called {
	int n;
	short events[16];
	struct timeval tvs[16];
};

static void
activate_cb(evutil_socket_t fd, short event, void *arg)
{
	struct event *ev = arg;
	event_active(ev, EV_READ, 1);
}

static void
persist_active_timeout_cb(evutil_socket_t fd, short event, void *arg)
{
	struct persist_active_timeout_called *c = arg;
	if (c->n < 15) {
		c->events[c->n] = event;
		evutil_gettimeofday(&c->tvs[c->n], NULL);
		++c->n;
	}
}

static void
test_persistent_active_timeout(void *ptr)
{
	struct timeval tv, tv2, tv_exit, start;
	struct event ev;
	struct persist_active_timeout_called res;

	struct basic_test_data *data = ptr;
	struct event_base *base = data->base;

	memset(&res, 0, sizeof(res));

	tv.tv_sec = 0;
	tv.tv_usec = 200 * 1000;
	event_assign(&ev, base, -1, EV_TIMEOUT|EV_PERSIST,
	    persist_active_timeout_cb, &res);
	event_add(&ev, &tv);

	tv2.tv_sec = 0;
	tv2.tv_usec = 100 * 1000;
	event_base_once(base, -1, EV_TIMEOUT, activate_cb, &ev, &tv2);

	tv_exit.tv_sec = 0;
	tv_exit.tv_usec = 600 * 1000;
	event_base_loopexit(base, &tv_exit);

	event_base_assert_ok(base);
	evutil_gettimeofday(&start, NULL);

	event_base_dispatch(base);
	event_base_assert_ok(base);

	tt_int_op(res.n, ==, 3);
	tt_int_op(res.events[0], ==, EV_READ);
	tt_int_op(res.events[1], ==, EV_TIMEOUT);
	tt_int_op(res.events[2], ==, EV_TIMEOUT);
	test_timeval_diff_eq(&start, &res.tvs[0], 100);
	test_timeval_diff_eq(&start, &res.tvs[1], 300);
	test_timeval_diff_eq(&start, &res.tvs[2], 500);
end:
	event_del(&ev);
}

struct common_timeout_info {
	struct event ev;
	struct timeval called_at;
	int which;
	int count;
};

static void
common_timeout_cb(evutil_socket_t fd, short event, void *arg)
{
	struct common_timeout_info *ti = arg;
	++ti->count;
	evutil_gettimeofday(&ti->called_at, NULL);
	if (ti->count >= 6)
		event_del(&ti->ev);
}

static void
test_common_timeout(void *ptr)
{
	struct basic_test_data *data = ptr;

	struct event_base *base = data->base;
	int i;
	struct common_timeout_info info[100];

	struct timeval now;
	struct timeval tmp_100_ms = { 0, 100*1000 };
	struct timeval tmp_200_ms = { 0, 200*1000 };

	const struct timeval *ms_100, *ms_200;

	ms_100 = event_base_init_common_timeout(base, &tmp_100_ms);
	ms_200 = event_base_init_common_timeout(base, &tmp_200_ms);
	tt_assert(ms_100);
	tt_assert(ms_200);
	tt_ptr_op(event_base_init_common_timeout(base, &tmp_200_ms),
	    ==, ms_200);
	tt_int_op(ms_100->tv_sec, ==, 0);
	tt_int_op(ms_200->tv_sec, ==, 0);
	tt_int_op(ms_100->tv_usec, ==, 100000|0x50000000);
	tt_int_op(ms_200->tv_usec, ==, 200000|0x50100000);

	memset(info, 0, sizeof(info));

	for (i=0; i<100; ++i) {
		info[i].which = i;
		event_assign(&info[i].ev, base, -1, EV_TIMEOUT|EV_PERSIST,
		    common_timeout_cb, &info[i]);
		if (i % 2) {
			event_add(&info[i].ev, ms_100);
		} else {
			event_add(&info[i].ev, ms_200);
		}
	}

	event_base_assert_ok(base);
	event_base_dispatch(base);

	evutil_gettimeofday(&now, NULL);
	event_base_assert_ok(base);

	for (i=0; i<10; ++i) {
		struct timeval tmp;
		int ms_diff;
		tt_int_op(info[i].count, ==, 6);
		evutil_timersub(&now, &info[i].called_at, &tmp);
		ms_diff = tmp.tv_usec/1000 + tmp.tv_sec*1000;
		if (i % 2) {
			tt_int_op(ms_diff, >, 500);
			tt_int_op(ms_diff, <, 700);
		} else {
			tt_int_op(ms_diff, >, -100);
			tt_int_op(ms_diff, <, 100);
		}
	}

	/* Make sure we can free the base with some events in. */
	for (i=0; i<100; ++i) {
		if (i % 2) {
			event_add(&info[i].ev, ms_100);
		} else {
			event_add(&info[i].ev, ms_200);
		}
	}

end:
	event_base_free(data->base); /* need to do this here before info is
				      * out-of-scope */
	data->base = NULL;
}

#ifndef WIN32
static void signal_cb(evutil_socket_t fd, short event, void *arg);

#define lwrrent_base ompi_event_global_lwrrent_base_
extern struct event_base *lwrrent_base;

static void
child_signal_cb(evutil_socket_t fd, short event, void *arg)
{
	struct timeval tv;
	int *pint = arg;

	*pint = 1;

	tv.tv_usec = 500000;
	tv.tv_sec = 0;
	event_loopexit(&tv);
}

static void
test_fork(void)
{
	int status, got_sigchld = 0;
	struct event ev, sig_ev;
	pid_t pid;

	setup_test("After fork: ");

	tt_assert(lwrrent_base);
	evthread_make_base_notifiable(lwrrent_base);

	if (write(pair[0], TEST1, strlen(TEST1)+1) < 0) {
		tt_fail_perror("write");
	}

	event_set(&ev, pair[1], EV_READ, simple_read_cb, &ev);
	if (event_add(&ev, NULL) == -1)
		exit(1);

	evsignal_set(&sig_ev, SIGCHLD, child_signal_cb, &got_sigchld);
	evsignal_add(&sig_ev, NULL);

	event_base_assert_ok(lwrrent_base);
	TT_BLATHER(("Before fork"));
	if ((pid = regress_fork()) == 0) {
		/* in the child */
		TT_BLATHER(("In child, before reinit"));
		event_base_assert_ok(lwrrent_base);
		if (event_reinit(lwrrent_base) == -1) {
			fprintf(stdout, "FAILED (reinit)\n");
			exit(1);
		}
		TT_BLATHER(("After reinit"));
		event_base_assert_ok(lwrrent_base);
		TT_BLATHER(("After assert-ok"));

		evsignal_del(&sig_ev);

		called = 0;

		event_dispatch();

		event_base_free(lwrrent_base);

		/* we do not send an EOF; simple_read_cb requires an EOF
		 * to set test_ok.  we just verify that the callback was
		 * called. */
		exit(test_ok != 0 || called != 2 ? -2 : 76);
	}

	/* wait for the child to read the data */
	sleep(1);

	if (write(pair[0], TEST1, strlen(TEST1)+1) < 0) {
		tt_fail_perror("write");
	}

	TT_BLATHER(("Before waitpid"));
	if (waitpid(pid, &status, 0) == -1) {
		fprintf(stdout, "FAILED (fork)\n");
		exit(1);
	}
	TT_BLATHER(("After waitpid"));

	if (WEXITSTATUS(status) != 76) {
		fprintf(stdout, "FAILED (exit): %d\n", WEXITSTATUS(status));
		exit(1);
	}

	/* test that the current event loop still works */
	if (write(pair[0], TEST1, strlen(TEST1)+1) < 0) {
		fprintf(stderr, "%s: write\n", __func__);
	}

	shutdown(pair[0], SHUT_WR);

	event_dispatch();

	if (!got_sigchld) {
		fprintf(stdout, "FAILED (sigchld)\n");
		exit(1);
	}

	evsignal_del(&sig_ev);

	end:
	cleanup_test();
}

static void
signal_cb_sa(int sig)
{
	test_ok = 2;
}

static void
signal_cb(evutil_socket_t fd, short event, void *arg)
{
	struct event *ev = arg;

	evsignal_del(ev);
	test_ok = 1;
}

static void
test_simplesignal(void)
{
	struct event ev;
	struct itimerval itv;

	setup_test("Simple signal: ");
	evsignal_set(&ev, SIGALRM, signal_cb, &ev);
	evsignal_add(&ev, NULL);
	/* find bugs in which operations are re-ordered */
	evsignal_del(&ev);
	evsignal_add(&ev, NULL);

	memset(&itv, 0, sizeof(itv));
	itv.it_value.tv_sec = 1;
	if (setitimer(ITIMER_REAL, &itv, NULL) == -1)
		goto skip_simplesignal;

	event_dispatch();
 skip_simplesignal:
	if (evsignal_del(&ev) == -1)
		test_ok = 0;

	cleanup_test();
}

static void
test_multiplesignal(void)
{
	struct event ev_one, ev_two;
	struct itimerval itv;

	setup_test("Multiple signal: ");

	evsignal_set(&ev_one, SIGALRM, signal_cb, &ev_one);
	evsignal_add(&ev_one, NULL);

	evsignal_set(&ev_two, SIGALRM, signal_cb, &ev_two);
	evsignal_add(&ev_two, NULL);

	memset(&itv, 0, sizeof(itv));
	itv.it_value.tv_sec = 1;
	if (setitimer(ITIMER_REAL, &itv, NULL) == -1)
		goto skip_simplesignal;

	event_dispatch();

 skip_simplesignal:
	if (evsignal_del(&ev_one) == -1)
		test_ok = 0;
	if (evsignal_del(&ev_two) == -1)
		test_ok = 0;

	cleanup_test();
}

static void
test_immediatesignal(void)
{
	struct event ev;

	test_ok = 0;
	evsignal_set(&ev, SIGUSR1, signal_cb, &ev);
	evsignal_add(&ev, NULL);
	raise(SIGUSR1);
	event_loop(EVLOOP_NONBLOCK);
	evsignal_del(&ev);
	cleanup_test();
}

static void
test_signal_dealloc(void)
{
	/* make sure that evsignal_event is event_del'ed and pipe closed */
	struct event ev;
	struct event_base *base = event_init();
	evsignal_set(&ev, SIGUSR1, signal_cb, &ev);
	evsignal_add(&ev, NULL);
	evsignal_del(&ev);
	event_base_free(base);
	/* If we got here without asserting, we're fine. */
	test_ok = 1;
	cleanup_test();
}

static void
test_signal_pipeloss(void)
{
	/* make sure that the base1 pipe is closed correctly. */
	struct event_base *base1, *base2;
	int pipe1;
	test_ok = 0;
	base1 = event_init();
	pipe1 = base1->sig.ev_signal_pair[0];
	base2 = event_init();
	event_base_free(base2);
	event_base_free(base1);
	if (close(pipe1) != -1 || errno!=EBADF) {
		/* fd must be closed, so second close gives -1, EBADF */
		printf("signal pipe not closed. ");
		test_ok = 0;
	} else {
		test_ok = 1;
	}
	cleanup_test();
}

/*
 * make two bases to catch signals, use both of them.  this only works
 * for event mechanisms that use our signal pipe trick.	 kqueue handles
 * signals internally, and all interested kqueues get all the signals.
 */
static void
test_signal_switchbase(void)
{
	struct event ev1, ev2;
	struct event_base *base1, *base2;
	int is_kqueue;
	test_ok = 0;
	base1 = event_init();
	base2 = event_init();
	is_kqueue = !strcmp(event_get_method(),"kqueue");
	evsignal_set(&ev1, SIGUSR1, signal_cb, &ev1);
	evsignal_set(&ev2, SIGUSR1, signal_cb, &ev2);
	if (event_base_set(base1, &ev1) ||
	    event_base_set(base2, &ev2) ||
	    event_add(&ev1, NULL) ||
	    event_add(&ev2, NULL)) {
		fprintf(stderr, "%s: cannot set base, add\n", __func__);
		exit(1);
	}

	tt_ptr_op(event_get_base(&ev1), ==, base1);
	tt_ptr_op(event_get_base(&ev2), ==, base2);

	test_ok = 0;
	/* can handle signal before loop is called */
	raise(SIGUSR1);
	event_base_loop(base2, EVLOOP_NONBLOCK);
	if (is_kqueue) {
		if (!test_ok)
			goto end;
		test_ok = 0;
	}
	event_base_loop(base1, EVLOOP_NONBLOCK);
	if (test_ok && !is_kqueue) {
		test_ok = 0;

		/* set base1 to handle signals */
		event_base_loop(base1, EVLOOP_NONBLOCK);
		raise(SIGUSR1);
		event_base_loop(base1, EVLOOP_NONBLOCK);
		event_base_loop(base2, EVLOOP_NONBLOCK);
	}
end:
	event_base_free(base1);
	event_base_free(base2);
	cleanup_test();
}

/*
 * assert that a signal event removed from the event queue really is
 * removed - with no possibility of it's parent handler being fired.
 */
static void
test_signal_assert(void)
{
	struct event ev;
	struct event_base *base = event_init();
	test_ok = 0;
	/* use SIGCONT so we don't kill ourselves when we signal to nowhere */
	evsignal_set(&ev, SIGCONT, signal_cb, &ev);
	evsignal_add(&ev, NULL);
	/*
	 * if evsignal_del() fails to reset the handler, it's current handler
	 * will still point to evsig_handler().
	 */
	evsignal_del(&ev);

	raise(SIGCONT);
#if 0
	/* only way to verify we were in evsig_handler() */
	/* XXXX Now there's no longer a good way. */
	if (base->sig.evsig_caught)
		test_ok = 0;
	else
		test_ok = 1;
#else
	test_ok = 1;
#endif

	event_base_free(base);
	cleanup_test();
	return;
}

/*
 * assert that we restore our previous signal handler properly.
 */
static void
test_signal_restore(void)
{
	struct event ev;
	struct event_base *base = event_init();
#ifdef _EVENT_HAVE_SIGACTION
	struct sigaction sa;
#endif

	test_ok = 0;
#ifdef _EVENT_HAVE_SIGACTION
	sa.sa_handler = signal_cb_sa;
	sa.sa_flags = 0x0;
	sigemptyset(&sa.sa_mask);
	if (sigaction(SIGUSR1, &sa, NULL) == -1)
		goto out;
#else
	if (signal(SIGUSR1, signal_cb_sa) == SIG_ERR)
		goto out;
#endif
	evsignal_set(&ev, SIGUSR1, signal_cb, &ev);
	evsignal_add(&ev, NULL);
	evsignal_del(&ev);

	raise(SIGUSR1);
	/* 1 == signal_cb, 2 == signal_cb_sa, we want our previous handler */
	if (test_ok != 2)
		test_ok = 0;
out:
	event_base_free(base);
	cleanup_test();
	return;
}

static void
signal_cb_swp(int sig, short event, void *arg)
{
	called++;
	if (called < 5)
		raise(sig);
	else
		event_loopexit(NULL);
}
static void
timeout_cb_swp(evutil_socket_t fd, short event, void *arg)
{
	if (called == -1) {
		struct timeval tv = {5, 0};

		called = 0;
		evtimer_add((struct event *)arg, &tv);
		raise(SIGUSR1);
		return;
	}
	test_ok = 0;
	event_loopexit(NULL);
}

static void
test_signal_while_processing(void)
{
	struct event_base *base = event_init();
	struct event ev, ev_timer;
	struct timeval tv = {0, 0};

	setup_test("Receiving a signal while processing other signal: ");

	called = -1;
	test_ok = 1;
	signal_set(&ev, SIGUSR1, signal_cb_swp, NULL);
	signal_add(&ev, NULL);
	evtimer_set(&ev_timer, timeout_cb_swp, &ev_timer);
	evtimer_add(&ev_timer, &tv);
	event_dispatch();

	event_base_free(base);
	cleanup_test();
	return;
}
#endif

static void
test_free_active_base(void *ptr)
{
	struct basic_test_data *data = ptr;
	struct event_base *base1;
	struct event ev1;

	base1 = event_init();
	if (base1) {
		event_assign(&ev1, base1, data->pair[1], EV_READ,
			     dummy_read_cb, NULL);
		event_add(&ev1, NULL);
		event_base_free(base1);	 /* should not crash */
	} else {
		tt_fail_msg("failed to create event_base for test");
	}

	base1 = event_init();
	tt_assert(base1);
	event_assign(&ev1, base1, 0, 0, dummy_read_cb, NULL);
	event_active(&ev1, EV_READ, 1);
	event_base_free(base1);
end:
	;
}

static void
test_manipulate_active_events(void *ptr)
{
	struct basic_test_data *data = ptr;
	struct event_base *base = data->base;
	struct event ev1;

	event_assign(&ev1, base, -1, EV_TIMEOUT, dummy_read_cb, NULL);

	/* Make sure an active event is pending. */
	event_active(&ev1, EV_READ, 1);
	tt_int_op(event_pending(&ev1, EV_READ|EV_TIMEOUT|EV_WRITE, NULL),
	    ==, EV_READ);

	/* Make sure that activating an event twice works. */
	event_active(&ev1, EV_WRITE, 1);
	tt_int_op(event_pending(&ev1, EV_READ|EV_TIMEOUT|EV_WRITE, NULL),
	    ==, EV_READ|EV_WRITE);

end:
	event_del(&ev1);
}

static void
test_bad_assign(void *ptr)
{
	struct event ev;
	int r;
	/* READ|SIGNAL is not allowed */
	r = event_assign(&ev, NULL, -1, EV_SIGNAL|EV_READ, dummy_read_cb, NULL);
	tt_int_op(r,==,-1);

end:
	;
}

static int reentrant_cb_run = 0;

static void
bad_reentrant_run_loop_cb(evutil_socket_t fd, short what, void *ptr)
{
	struct event_base *base = ptr;
	int r;
	reentrant_cb_run = 1;
	/* This reentrant call to event_base_loop should be detected and
	 * should fail */
	r = event_base_loop(base, 0);
	tt_int_op(r, ==, -1);
end:
	;
}

static void
test_bad_reentrant(void *ptr)
{
	struct basic_test_data *data = ptr;
	struct event_base *base = data->base;
	struct event ev;
	int r;
	event_assign(&ev, base, -1,
	    0, bad_reentrant_run_loop_cb, base);

	event_active(&ev, EV_WRITE, 1);
	r = event_base_loop(base, 0);
	tt_int_op(r, ==, 1);
	tt_int_op(reentrant_cb_run, ==, 1);
end:
	;
}

static void
test_event_base_new(void *ptr)
{
	struct basic_test_data *data = ptr;
	struct event_base *base = 0;
	struct event ev1;
	struct basic_cb_args args;

	int towrite = (int)strlen(TEST1)+1;
	int len = write(data->pair[0], TEST1, towrite);

	if (len < 0)
		tt_abort_perror("initial write");
	else if (len != towrite)
		tt_abort_printf(("initial write fell short (%d of %d bytes)",
				 len, towrite));

	if (shutdown(data->pair[0], SHUT_WR))
		tt_abort_perror("initial write shutdown");

	base = event_base_new();
	if (!base)
		tt_abort_msg("failed to create event base");

	args.eb = base;
	args.ev = &ev1;
	args.callcount = 0;
	event_assign(&ev1, base, data->pair[1],
		     EV_READ|EV_PERSIST, basic_read_cb, &args);

	if (event_add(&ev1, NULL))
		tt_abort_perror("initial event_add");

	if (event_base_loop(base, 0))
		tt_abort_msg("unsuccessful exit from event loop");

end:
	if (base)
		event_base_free(base);
}

static void
test_loopexit(void)
{
	struct timeval tv, tv_start, tv_end;
	struct event ev;

	setup_test("Loop exit: ");

	tv.tv_usec = 0;
	tv.tv_sec = 60*60*24;
	evtimer_set(&ev, timeout_cb, NULL);
	evtimer_add(&ev, &tv);

	tv.tv_usec = 0;
	tv.tv_sec = 1;
	event_loopexit(&tv);

	evutil_gettimeofday(&tv_start, NULL);
	event_dispatch();
	evutil_gettimeofday(&tv_end, NULL);
	evutil_timersub(&tv_end, &tv_start, &tv_end);

	evtimer_del(&ev);

	tt_assert(event_base_got_exit(global_base));
	tt_assert(!event_base_got_break(global_base));

	if (tv.tv_sec < 2)
		test_ok = 1;

end:
	cleanup_test();
}

static void
test_loopexit_multiple(void)
{
	struct timeval tv;
	struct event_base *base;

	setup_test("Loop Multiple exit: ");

	base = event_base_new();

	tv.tv_usec = 0;
	tv.tv_sec = 1;
	event_base_loopexit(base, &tv);

	tv.tv_usec = 0;
	tv.tv_sec = 2;
	event_base_loopexit(base, &tv);

	event_base_dispatch(base);

	tt_assert(event_base_got_exit(base));
	tt_assert(!event_base_got_break(base));

	event_base_free(base);

	test_ok = 1;

end:
	cleanup_test();
}

static void
break_cb(evutil_socket_t fd, short events, void *arg)
{
	test_ok = 1;
	event_loopbreak();
}

static void
fail_cb(evutil_socket_t fd, short events, void *arg)
{
	test_ok = 0;
}

static void
test_loopbreak(void)
{
	struct event ev1, ev2;
	struct timeval tv;

	setup_test("Loop break: ");

	tv.tv_sec = 0;
	tv.tv_usec = 0;
	evtimer_set(&ev1, break_cb, NULL);
	evtimer_add(&ev1, &tv);
	evtimer_set(&ev2, fail_cb, NULL);
	evtimer_add(&ev2, &tv);

	event_dispatch();

	tt_assert(!event_base_got_exit(global_base));
	tt_assert(event_base_got_break(global_base));

	evtimer_del(&ev1);
	evtimer_del(&ev2);

end:
	cleanup_test();
}

static struct event *readd_test_event_last_added = NULL;
static void
re_add_read_cb(evutil_socket_t fd, short event, void *arg)
{
	char buf[256];
	struct event *ev_other = arg;
	readd_test_event_last_added = ev_other;

	if (read(fd, buf, sizeof(buf)) < 0) {
		tt_fail_perror("read");
	}

	event_add(ev_other, NULL);
	++test_ok;
}

static void
test_nonpersist_readd(void)
{
	struct event ev1, ev2;

	setup_test("Re-add nonpersistent events: ");
	event_set(&ev1, pair[0], EV_READ, re_add_read_cb, &ev2);
	event_set(&ev2, pair[1], EV_READ, re_add_read_cb, &ev1);

	if (write(pair[0], "Hello", 5) < 0) {
		tt_fail_perror("write(pair[0])");
	}

	if (write(pair[1], "Hello", 5) < 0) {
		tt_fail_perror("write(pair[1])\n");
	}

	if (event_add(&ev1, NULL) == -1 ||
	    event_add(&ev2, NULL) == -1) {
		test_ok = 0;
	}
	if (test_ok != 0)
		exit(1);
	event_loop(EVLOOP_ONCE);
	if (test_ok != 2)
		exit(1);
	/* At this point, we exelwted both callbacks.  Whichever one got
	 * called first added the second, but the second then immediately got
	 * deleted before its callback was called.  At this point, though, it
	 * re-added the first.
	 */
	if (!readd_test_event_last_added) {
		test_ok = 0;
	} else if (readd_test_event_last_added == &ev1) {
		if (!event_pending(&ev1, EV_READ, NULL) ||
		    event_pending(&ev2, EV_READ, NULL))
			test_ok = 0;
	} else {
		if (event_pending(&ev1, EV_READ, NULL) ||
		    !event_pending(&ev2, EV_READ, NULL))
			test_ok = 0;
	}

	event_del(&ev1);
	event_del(&ev2);

	cleanup_test();
}

struct test_pri_event {
	struct event ev;
	int count;
};

static void
test_priorities_cb(evutil_socket_t fd, short what, void *arg)
{
	struct test_pri_event *pri = arg;
	struct timeval tv;

	if (pri->count == 3) {
		event_loopexit(NULL);
		return;
	}

	pri->count++;

	evutil_timerclear(&tv);
	event_add(&pri->ev, &tv);
}

static void
test_priorities_impl(int npriorities)
{
	struct test_pri_event one, two;
	struct timeval tv;

	TT_BLATHER(("Testing Priorities %d: ", npriorities));

	event_base_priority_init(global_base, npriorities);

	memset(&one, 0, sizeof(one));
	memset(&two, 0, sizeof(two));

	timeout_set(&one.ev, test_priorities_cb, &one);
	if (event_priority_set(&one.ev, 0) == -1) {
		fprintf(stderr, "%s: failed to set priority", __func__);
		exit(1);
	}

	timeout_set(&two.ev, test_priorities_cb, &two);
	if (event_priority_set(&two.ev, npriorities - 1) == -1) {
		fprintf(stderr, "%s: failed to set priority", __func__);
		exit(1);
	}

	evutil_timerclear(&tv);

	if (event_add(&one.ev, &tv) == -1)
		exit(1);
	if (event_add(&two.ev, &tv) == -1)
		exit(1);

	event_dispatch();

	event_del(&one.ev);
	event_del(&two.ev);

	if (npriorities == 1) {
		if (one.count == 3 && two.count == 3)
			test_ok = 1;
	} else if (npriorities == 2) {
		/* Two is called once because event_loopexit is priority 1 */
		if (one.count == 3 && two.count == 1)
			test_ok = 1;
	} else {
		if (one.count == 3 && two.count == 0)
			test_ok = 1;
	}
}

static void
test_priorities(void)
{
	test_priorities_impl(1);
	if (test_ok)
		test_priorities_impl(2);
	if (test_ok)
		test_priorities_impl(3);
}

/* priority-active-ilwersion: activate a higher-priority event, and make sure
 * it keeps us from running a lower-priority event first. */
static int n_pai_calls = 0;
static struct event pai_events[3];

static void
prio_active_ilwersion_cb(evutil_socket_t fd, short what, void *arg)
{
	int *call_order = arg;
	*call_order = n_pai_calls++;
	if (n_pai_calls == 1) {
		/* This should activate later, even though it shares a
		   priority with us. */
		event_active(&pai_events[1], EV_READ, 1);
		/* This should activate next, since its priority is higher,
		   even though we activated it second. */
		event_active(&pai_events[2], EV_TIMEOUT, 1);
	}
}

static void
test_priority_active_ilwersion(void *data_)
{
	struct basic_test_data *data = data_;
	struct event_base *base = data->base;
	int call_order[3];
	int i;
	tt_int_op(event_base_priority_init(base, 8), ==, 0);

	n_pai_calls = 0;
	memset(call_order, 0, sizeof(call_order));

	for (i=0;i<3;++i) {
		event_assign(&pai_events[i], data->base, -1, 0,
		    prio_active_ilwersion_cb, &call_order[i]);
	}

	event_priority_set(&pai_events[0], 4);
	event_priority_set(&pai_events[1], 4);
	event_priority_set(&pai_events[2], 0);

	event_active(&pai_events[0], EV_WRITE, 1);

	event_base_dispatch(base);
	tt_int_op(n_pai_calls, ==, 3);
	tt_int_op(call_order[0], ==, 0);
	tt_int_op(call_order[1], ==, 2);
	tt_int_op(call_order[2], ==, 1);
end:
	;
}


static void
test_multiple_cb(evutil_socket_t fd, short event, void *arg)
{
	if (event & EV_READ)
		test_ok |= 1;
	else if (event & EV_WRITE)
		test_ok |= 2;
}

static void
test_multiple_events_for_same_fd(void)
{
   struct event e1, e2;

   setup_test("Multiple events for same fd: ");

   event_set(&e1, pair[0], EV_READ, test_multiple_cb, NULL);
   event_add(&e1, NULL);
   event_set(&e2, pair[0], EV_WRITE, test_multiple_cb, NULL);
   event_add(&e2, NULL);
   event_loop(EVLOOP_ONCE);
   event_del(&e2);

   if (write(pair[1], TEST1, strlen(TEST1)+1) < 0) {
	   tt_fail_perror("write");
   }

   event_loop(EVLOOP_ONCE);
   event_del(&e1);

   if (test_ok != 3)
	   test_ok = 0;

   cleanup_test();
}

int evtag_decode_int(ev_uint32_t *pnumber, struct evbuffer *evbuf);
int evtag_decode_int64(ev_uint64_t *pnumber, struct evbuffer *evbuf);
int evtag_encode_tag(struct evbuffer *evbuf, ev_uint32_t number);
int evtag_decode_tag(ev_uint32_t *pnumber, struct evbuffer *evbuf);

static void
read_once_cb(evutil_socket_t fd, short event, void *arg)
{
	char buf[256];
	int len;

	len = read(fd, buf, sizeof(buf));

	if (called) {
		test_ok = 0;
	} else if (len) {
		/* Assumes global pair[0] can be used for writing */
		if (write(pair[0], TEST1, strlen(TEST1)+1) < 0) {
			tt_fail_perror("write");
			test_ok = 0;
		} else {
			test_ok = 1;
		}
	}

	called++;
}

static void
test_want_only_once(void)
{
	struct event ev;
	struct timeval tv;

	/* Very simple read test */
	setup_test("Want read only once: ");

	if (write(pair[0], TEST1, strlen(TEST1)+1) < 0) {
		tt_fail_perror("write");
	}

	/* Setup the loop termination */
	evutil_timerclear(&tv);
	tv.tv_sec = 1;
	event_loopexit(&tv);

	event_set(&ev, pair[1], EV_READ, read_once_cb, &ev);
	if (event_add(&ev, NULL) == -1)
		exit(1);
	event_dispatch();

	cleanup_test();
}

#define TEST_MAX_INT	6

static void
evtag_int_test(void *ptr)
{
	struct evbuffer *tmp = evbuffer_new();
	ev_uint32_t integers[TEST_MAX_INT] = {
		0xaf0, 0x1000, 0x1, 0xdeadbeef, 0x00, 0xbef000
	};
	ev_uint32_t integer;
	ev_uint64_t big_int;
	int i;

	evtag_init();

	for (i = 0; i < TEST_MAX_INT; i++) {
		int oldlen, newlen;
		oldlen = (int)EVBUFFER_LENGTH(tmp);
		evtag_encode_int(tmp, integers[i]);
		newlen = (int)EVBUFFER_LENGTH(tmp);
		TT_BLATHER(("encoded 0x%08x with %d bytes",
			(unsigned)integers[i], newlen - oldlen));
		big_int = integers[i];
		big_int *= 1000000000; /* 1 billion */
		evtag_encode_int64(tmp, big_int);
	}

	for (i = 0; i < TEST_MAX_INT; i++) {
		tt_int_op(evtag_decode_int(&integer, tmp), !=, -1);
		tt_uint_op(integer, ==, integers[i]);
		tt_int_op(evtag_decode_int64(&big_int, tmp), !=, -1);
		tt_assert((big_int / 1000000000) == integers[i]);
	}

	tt_uint_op(EVBUFFER_LENGTH(tmp), ==, 0);
end:
	evbuffer_free(tmp);
}

static void
evtag_fuzz(void *ptr)
{
	u_char buffer[4096];
	struct evbuffer *tmp = evbuffer_new();
	struct timeval tv;
	int i, j;

	int not_failed = 0;

	evtag_init();

	for (j = 0; j < 100; j++) {
		for (i = 0; i < (int)sizeof(buffer); i++)
			buffer[i] = rand();
		evbuffer_drain(tmp, -1);
		evbuffer_add(tmp, buffer, sizeof(buffer));

		if (evtag_unmarshal_timeval(tmp, 0, &tv) != -1)
			not_failed++;
	}

	/* The majority of decodes should fail */
	tt_int_op(not_failed, <, 10);

	/* Now insert some corruption into the tag length field */
	evbuffer_drain(tmp, -1);
	evutil_timerclear(&tv);
	tv.tv_sec = 1;
	evtag_marshal_timeval(tmp, 0, &tv);
	evbuffer_add(tmp, buffer, sizeof(buffer));

	((char *)EVBUFFER_DATA(tmp))[1] = '\xff';
	if (evtag_unmarshal_timeval(tmp, 0, &tv) != -1) {
		tt_abort_msg("evtag_unmarshal_timeval should have failed");
	}

end:
	evbuffer_free(tmp);
}

static void
evtag_tag_encoding(void *ptr)
{
	struct evbuffer *tmp = evbuffer_new();
	ev_uint32_t integers[TEST_MAX_INT] = {
		0xaf0, 0x1000, 0x1, 0xdeadbeef, 0x00, 0xbef000
	};
	ev_uint32_t integer;
	int i;

	evtag_init();

	for (i = 0; i < TEST_MAX_INT; i++) {
		int oldlen, newlen;
		oldlen = (int)EVBUFFER_LENGTH(tmp);
		evtag_encode_tag(tmp, integers[i]);
		newlen = (int)EVBUFFER_LENGTH(tmp);
		TT_BLATHER(("encoded 0x%08x with %d bytes",
			(unsigned)integers[i], newlen - oldlen));
	}

	for (i = 0; i < TEST_MAX_INT; i++) {
		tt_int_op(evtag_decode_tag(&integer, tmp), !=, -1);
		tt_uint_op(integer, ==, integers[i]);
	}

	tt_uint_op(EVBUFFER_LENGTH(tmp), ==, 0);

end:
	evbuffer_free(tmp);
}

static void
evtag_test_peek(void *ptr)
{
	struct evbuffer *tmp = evbuffer_new();
	ev_uint32_t u32;

	evtag_marshal_int(tmp, 30, 0);
	evtag_marshal_string(tmp, 40, "Hello world");

	tt_int_op(evtag_peek(tmp, &u32), ==, 1);
	tt_int_op(u32, ==, 30);
	tt_int_op(evtag_peek_length(tmp, &u32), ==, 0);
	tt_int_op(u32, ==, 1+1+1);
	tt_int_op(evtag_consume(tmp), ==, 0);

	tt_int_op(evtag_peek(tmp, &u32), ==, 1);
	tt_int_op(u32, ==, 40);
	tt_int_op(evtag_peek_length(tmp, &u32), ==, 0);
	tt_int_op(u32, ==, 1+1+11);
	tt_int_op(evtag_payload_length(tmp, &u32), ==, 0);
	tt_int_op(u32, ==, 11);

end:
	evbuffer_free(tmp);
}


static void
test_methods(void *ptr)
{
	const char **methods = event_get_supported_methods();
	struct event_config *cfg = NULL;
	struct event_base *base = NULL;
	const char *backend;
	int n_methods = 0;

	tt_assert(methods);

	backend = methods[0];
	while (*methods != NULL) {
		TT_BLATHER(("Support method: %s", *methods));
		++methods;
		++n_methods;
	}

	cfg = event_config_new();
	assert(cfg != NULL);

	tt_int_op(event_config_avoid_method(cfg, backend), ==, 0);
	event_config_set_flag(cfg, EVENT_BASE_FLAG_IGNORE_ELW);

	base = event_base_new_with_config(cfg);
	if (n_methods > 1) {
		tt_assert(base);
		tt_str_op(backend, !=, event_base_get_method(base));
	} else {
		tt_assert(base == NULL);
	}

end:
	if (base)
		event_base_free(base);
	if (cfg)
		event_config_free(cfg);
}

static void
test_version(void *arg)
{
	const char *vstr;
	ev_uint32_t vint;
	int major, minor, patch, n;

	vstr = event_get_version();
	vint = event_get_version_number();

	tt_assert(vstr);
	tt_assert(vint);

	tt_str_op(vstr, ==, LIBEVENT_VERSION);
	tt_int_op(vint, ==, LIBEVENT_VERSION_NUMBER);

	n = sscanf(vstr, "%d.%d.%d", &major, &minor, &patch);
	tt_assert(3 == n);
	tt_int_op((vint&0xffffff00), ==, ((major<<24)|(minor<<16)|(patch<<8)));
end:
	;
}

static void
test_base_features(void *arg)
{
	struct event_base *base = NULL;
	struct event_config *cfg = NULL;

	cfg = event_config_new();

	tt_assert(0 == event_config_require_features(cfg, EV_FEATURE_ET));

	base = event_base_new_with_config(cfg);
	if (base) {
		tt_int_op(EV_FEATURE_ET, ==,
		    event_base_get_features(base) & EV_FEATURE_ET);
	} else {
		base = event_base_new();
		tt_int_op(0, ==, event_base_get_features(base) & EV_FEATURE_ET);
	}

end:
	if (base)
		event_base_free(base);
	if (cfg)
		event_config_free(cfg);
}

#ifdef _EVENT_HAVE_SETELW
#define SETELW_OK
#elif !defined(_EVENT_HAVE_SETELW) && defined(_EVENT_HAVE_PUTELW)
static void setelw(const char *k, const char *v, int _o)
{
	char b[256];
	evutil_snprintf(b, sizeof(b), "%s=%s",k,v);
	putelw(b);
}
#define SETELW_OK
#endif

#ifdef _EVENT_HAVE_UNSETELW
#define UNSETELW_OK
#elif !defined(_EVENT_HAVE_UNSETELW) && defined(_EVENT_HAVE_PUTELW)
static void unsetelw(const char *k)
{
	char b[256];
	evutil_snprintf(b, sizeof(b), "%s=",k);
	putelw(b);
}
#define UNSETELW_OK
#endif

#if defined(SETELW_OK) && defined(UNSETELW_OK)
static void
methodname_to_elwvar(const char *mname, char *buf, size_t buflen)
{
	char *cp;
	evutil_snprintf(buf, buflen, "EVENT_NO%s", mname);
	for (cp = buf; *cp; ++cp) {
		*cp = EVUTIL_TOUPPER(*cp);
	}
}
#endif

static void
test_base_elwiron(void *arg)
{
	struct event_base *base = NULL;
	struct event_config *cfg = NULL;

#if defined(SETELW_OK) && defined(UNSETELW_OK)
	const char **basenames;
	int i, n_methods=0;
	char varbuf[128];
	const char *defaultname, *ignoreelwname;

	/* See if unsetelw works before we rely on it. */
	setelw("EVENT_NOWAFFLES", "1", 1);
	unsetelw("EVENT_NOWAFFLES");
	if (getelw("EVENT_NOWAFFLES") != NULL) {
#ifndef _EVENT_HAVE_UNSETELW
		TT_DECLARE("NOTE", ("Can't fake unsetelw; skipping test"));
#else
		TT_DECLARE("NOTE", ("unsetelw doesn't work; skipping test"));
#endif
		tt_skip();
	}

	basenames = event_get_supported_methods();
	for (i = 0; basenames[i]; ++i) {
		methodname_to_elwvar(basenames[i], varbuf, sizeof(varbuf));
		unsetelw(varbuf);
		++n_methods;
	}

	base = event_base_new();
	tt_assert(base);

	defaultname = event_base_get_method(base);
	TT_BLATHER(("default is <%s>", defaultname));
	event_base_free(base);
	base = NULL;

	/* Can we disable the method with EVENT_NOfoo ? */
	if (!strcmp(defaultname, "epoll (with changelist)")) {
 		setelw("EVENT_NOEPOLL", "1", 1);
		ignoreelwname = "epoll";
	} else {
		methodname_to_elwvar(defaultname, varbuf, sizeof(varbuf));
		setelw(varbuf, "1", 1);
		ignoreelwname = defaultname;
	}

	/* Use an empty cfg rather than NULL so a failure doesn't exit() */
	cfg = event_config_new();
	base = event_base_new_with_config(cfg);
	event_config_free(cfg);
	cfg = NULL;
	if (n_methods == 1) {
		tt_assert(!base);
	} else {
		tt_assert(base);
		tt_str_op(defaultname, !=, event_base_get_method(base));
		event_base_free(base);
		base = NULL;
	}

	/* Can we disable looking at the environment with IGNORE_ELW ? */
	cfg = event_config_new();
	event_config_set_flag(cfg, EVENT_BASE_FLAG_IGNORE_ELW);
	base = event_base_new_with_config(cfg);
	tt_assert(base);
	tt_str_op(ignoreelwname, ==, event_base_get_method(base));
#else
	tt_skip();
#endif

end:
	if (base)
		event_base_free(base);
	if (cfg)
		event_config_free(cfg);
}

static void
read_called_once_cb(evutil_socket_t fd, short event, void *arg)
{
	tt_int_op(event, ==, EV_READ);
	called += 1;
end:
	;
}

static void
timeout_called_once_cb(evutil_socket_t fd, short event, void *arg)
{
	tt_int_op(event, ==, EV_TIMEOUT);
	called += 100;
end:
	;
}

static void
test_event_once(void *ptr)
{
	struct basic_test_data *data = ptr;
	struct timeval tv;
	int r;

	tv.tv_sec = 0;
	tv.tv_usec = 50*1000;
	called = 0;
	r = event_base_once(data->base, data->pair[0], EV_READ,
	    read_called_once_cb, NULL, NULL);
	tt_int_op(r, ==, 0);
	r = event_base_once(data->base, -1, EV_TIMEOUT,
	    timeout_called_once_cb, NULL, &tv);
	tt_int_op(r, ==, 0);
	r = event_base_once(data->base, -1, 0, NULL, NULL, NULL);
	tt_int_op(r, <, 0);

	if (write(data->pair[1], TEST1, strlen(TEST1)+1) < 0) {
		tt_fail_perror("write");
	}

	shutdown(data->pair[1], SHUT_WR);

	event_base_dispatch(data->base);

	tt_int_op(called, ==, 101);
end:
	;
}

static void
test_event_pending(void *ptr)
{
	struct basic_test_data *data = ptr;
	struct event *r=NULL, *w=NULL, *t=NULL;
	struct timeval tv, now, tv2, diff;

	tv.tv_sec = 0;
	tv.tv_usec = 500 * 1000;
	r = event_new(data->base, data->pair[0], EV_READ, simple_read_cb,
	    NULL);
	w = event_new(data->base, data->pair[1], EV_WRITE, simple_write_cb,
	    NULL);
	t = evtimer_new(data->base, timeout_cb, NULL);

	tt_assert(r);
	tt_assert(w);
	tt_assert(t);

	evutil_gettimeofday(&now, NULL);
	event_add(r, NULL);
	event_add(t, &tv);

	tt_assert( event_pending(r, EV_READ, NULL));
	tt_assert(!event_pending(w, EV_WRITE, NULL));
	tt_assert(!event_pending(r, EV_WRITE, NULL));
	tt_assert( event_pending(r, EV_READ|EV_WRITE, NULL));
	tt_assert(!event_pending(r, EV_TIMEOUT, NULL));
	tt_assert( event_pending(t, EV_TIMEOUT, NULL));
	tt_assert( event_pending(t, EV_TIMEOUT, &tv2));

	tt_assert(evutil_timercmp(&tv2, &now, >));
	evutil_timeradd(&now, &tv, &tv);
	evutil_timersub(&tv2, &tv, &diff);
	tt_int_op(diff.tv_sec, ==, 0);
	tt_int_op(labs(diff.tv_usec), <, 1000);

end:
	if (r) {
		event_del(r);
		event_free(r);
	}
	if (w) {
		event_del(w);
		event_free(w);
	}
	if (t) {
		event_del(t);
		event_free(t);
	}
}

#ifndef WIN32
/* You can't do this test on windows, since dup2 doesn't work on sockets */

static void
dfd_cb(evutil_socket_t fd, short e, void *data)
{
	*(int*)data = (int)e;
}

/* Regression test for our workaround for a fun epoll/linux related bug
 * where fd2 = dup(fd1); add(fd2); close(fd2); dup2(fd1,fd2); add(fd2)
 * will get you an EEXIST */
static void
test_dup_fd(void *arg)
{
	struct basic_test_data *data = arg;
	struct event_base *base = data->base;
	struct event *ev1=NULL, *ev2=NULL;
	int fd, dfd=-1;
	int ev1_got, ev2_got;

	tt_int_op(write(data->pair[0], "Hello world",
		strlen("Hello world")), >, 0);
	fd = data->pair[1];

	dfd = dup(fd);
	tt_int_op(dfd, >=, 0);

	ev1 = event_new(base, fd, EV_READ|EV_PERSIST, dfd_cb, &ev1_got);
	ev2 = event_new(base, dfd, EV_READ|EV_PERSIST, dfd_cb, &ev2_got);
	ev1_got = ev2_got = 0;
	event_add(ev1, NULL);
	event_add(ev2, NULL);
	event_base_loop(base, EVLOOP_ONCE);
	tt_int_op(ev1_got, ==, EV_READ);
	tt_int_op(ev2_got, ==, EV_READ);

	/* Now close and delete dfd then dispatch.  We need to do the
	 * dispatch here so that when we add it later, we think there
	 * was an intermediate delete. */
	close(dfd);
	event_del(ev2);
	ev1_got = ev2_got = 0;
	event_base_loop(base, EVLOOP_ONCE);
	tt_want_int_op(ev1_got, ==, EV_READ);
	tt_int_op(ev2_got, ==, 0);

	/* Re-duplicate the fd.  We need to get the same duplicated
	 * value that we closed to provoke the epoll quirk.  Also, we
	 * need to change the events to write, or else the old lingering
	 * read event will make the test pass whether the change was
	 * successful or not. */
	tt_int_op(dup2(fd, dfd), ==, dfd);
	event_free(ev2);
	ev2 = event_new(base, dfd, EV_WRITE|EV_PERSIST, dfd_cb, &ev2_got);
	event_add(ev2, NULL);
	ev1_got = ev2_got = 0;
	event_base_loop(base, EVLOOP_ONCE);
	tt_want_int_op(ev1_got, ==, EV_READ);
	tt_int_op(ev2_got, ==, EV_WRITE);

end:
	if (ev1)
		event_free(ev1);
	if (ev2)
		event_free(ev2);
	if (dfd >= 0)
		close(dfd);
}
#endif

#ifdef _EVENT_DISABLE_MM_REPLACEMENT
static void
test_mm_functions(void *arg)
{
	_tinytest_set_test_skipped();
}
#else
static int
check_dummy_mem_ok(void *_mem)
{
	char *mem = _mem;
	mem -= 16;
	return !memcmp(mem, "{[<guardedram>]}", 16);
}

static void *
dummy_malloc(size_t len)
{
	char *mem = malloc(len+16);
	memcpy(mem, "{[<guardedram>]}", 16);
	return mem+16;
}

static void *
dummy_realloc(void *_mem, size_t len)
{
	char *mem = _mem;
	if (!mem)
		return dummy_malloc(len);
	tt_want(check_dummy_mem_ok(_mem));
	mem -= 16;
	mem = realloc(mem, len+16);
	return mem+16;
}

static void
dummy_free(void *_mem)
{
	char *mem = _mem;
	tt_want(check_dummy_mem_ok(_mem));
	mem -= 16;
	free(mem);
}

static void
test_mm_functions(void *arg)
{
	struct event_base *b = NULL;
	struct event_config *cfg = NULL;
	event_set_mem_functions(dummy_malloc, dummy_realloc, dummy_free);
	cfg = event_config_new();
	event_config_avoid_method(cfg, "Nonesuch");
	b = event_base_new_with_config(cfg);
	tt_assert(b);
	tt_assert(check_dummy_mem_ok(b));
end:
	if (cfg)
		event_config_free(cfg);
	if (b)
		event_base_free(b);
}
#endif

static void
many_event_cb(evutil_socket_t fd, short event, void *arg)
{
	int *calledp = arg;
	*calledp += 1;
}

static void
test_many_events(void *arg)
{
	/* Try 70 events that should all be ready at once.  This will
	 * exercise the "resize" code on most of the backends, and will make
	 * sure that we can get past the 64-handle limit of some windows
	 * functions. */
#define MANY 70

	struct basic_test_data *data = arg;
	struct event_base *base = data->base;
	int one_at_a_time = data->setup_data != NULL;
	evutil_socket_t sock[MANY];
	struct event *ev[MANY];
	int called[MANY];
	int i;
	int loopflags = EVLOOP_NONBLOCK, evflags=0;
	const int is_evport = !strcmp(event_base_get_method(base),"evport");
	if (one_at_a_time) {
		loopflags |= EVLOOP_ONCE;
		evflags = EV_PERSIST;
	}

	memset(sock, 0xff, sizeof(sock));
	memset(ev, 0, sizeof(ev));
	memset(called, 0, sizeof(called));
	if (is_evport && one_at_a_time) {
		TT_DECLARE("NOTE", ("evport can't pass this in 2.0; skipping\n"));
		tt_skip();
	}

	for (i = 0; i < MANY; ++i) {
		/* We need an event that will hit the backend, and that will
		 * be ready immediately.  "Send a datagram" is an easy
		 * instance of that. */
		sock[i] = socket(AF_INET, SOCK_DGRAM, 0);
		tt_assert(sock[i] >= 0);
		called[i] = 0;
		ev[i] = event_new(base, sock[i], EV_WRITE|evflags,
		    many_event_cb, &called[i]);
		event_add(ev[i], NULL);
		if (one_at_a_time)
			event_base_loop(base, EVLOOP_NONBLOCK|EVLOOP_ONCE);
	}

	event_base_loop(base, loopflags);

	for (i = 0; i < MANY; ++i) {
		if (one_at_a_time)
			tt_int_op(called[i], ==, MANY - i + 1);
		else
			tt_int_op(called[i], ==, 1);
	}

end:
	for (i = 0; i < MANY; ++i) {
		if (ev[i])
			event_free(ev[i]);
		if (sock[i] >= 0)
			evutil_closesocket(sock[i]);
	}
#undef MANY
}

static void
test_struct_event_size(void *arg)
{
	tt_int_op(event_get_struct_event_size(), <=, sizeof(struct event));
end:
	;
}

struct testcase_t main_testcases[] = {
	/* Some colwerted-over tests */
	{ "methods", test_methods, TT_FORK, NULL, NULL },
	{ "version", test_version, 0, NULL, NULL },
	BASIC(base_features, TT_FORK|TT_NO_LOGS),
	{ "base_elwiron", test_base_elwiron, TT_FORK, NULL, NULL },

	BASIC(event_base_new, TT_FORK|TT_NEED_SOCKETPAIR),
	BASIC(free_active_base, TT_FORK|TT_NEED_SOCKETPAIR),

	BASIC(manipulate_active_events, TT_FORK|TT_NEED_BASE),

	BASIC(bad_assign, TT_FORK|TT_NEED_BASE|TT_NO_LOGS),
	BASIC(bad_reentrant, TT_FORK|TT_NEED_BASE|TT_NO_LOGS),

	LEGACY(persistent_timeout, TT_FORK|TT_NEED_BASE),
	{ "persistent_timeout_jump", test_persistent_timeout_jump, TT_FORK|TT_NEED_BASE, &basic_setup, NULL },
	{ "persistent_active_timeout", test_persistent_active_timeout,
	  TT_FORK|TT_NEED_BASE, &basic_setup, NULL },
	LEGACY(priorities, TT_FORK|TT_NEED_BASE),
	BASIC(priority_active_ilwersion, TT_FORK|TT_NEED_BASE),
	{ "common_timeout", test_common_timeout, TT_FORK|TT_NEED_BASE,
	  &basic_setup, NULL },

	/* These legacy tests may not all need all of these flags. */
	LEGACY(simpleread, TT_ISOLATED),
	LEGACY(simpleread_multiple, TT_ISOLATED),
	LEGACY(simplewrite, TT_ISOLATED),
	{ "simpleclose", test_simpleclose, TT_FORK, &basic_setup,
	  NULL },
	LEGACY(multiple, TT_ISOLATED),
	LEGACY(persistent, TT_ISOLATED),
	LEGACY(combined, TT_ISOLATED),
	LEGACY(simpletimeout, TT_ISOLATED),
	LEGACY(loopbreak, TT_ISOLATED),
	LEGACY(loopexit, TT_ISOLATED),
	LEGACY(loopexit_multiple, TT_ISOLATED),
	LEGACY(nonpersist_readd, TT_ISOLATED),
	LEGACY(multiple_events_for_same_fd, TT_ISOLATED),
	LEGACY(want_only_once, TT_ISOLATED),
	{ "event_once", test_event_once, TT_ISOLATED, &basic_setup, NULL },
	{ "event_pending", test_event_pending, TT_ISOLATED, &basic_setup,
	  NULL },
#ifndef WIN32
	{ "dup_fd", test_dup_fd, TT_ISOLATED, &basic_setup, NULL },
#endif
	{ "mm_functions", test_mm_functions, TT_FORK, NULL, NULL },
	{ "many_events", test_many_events, TT_ISOLATED, &basic_setup, NULL },
	{ "many_events_slow_add", test_many_events, TT_ISOLATED, &basic_setup, (void*)1 },

	{ "struct_event_size", test_struct_event_size, 0, NULL, NULL },

#ifndef WIN32
	LEGACY(fork, TT_ISOLATED),
#endif
	END_OF_TESTCASES
};

struct testcase_t evtag_testcases[] = {
	{ "int", evtag_int_test, TT_FORK, NULL, NULL },
	{ "fuzz", evtag_fuzz, TT_FORK, NULL, NULL },
	{ "encoding", evtag_tag_encoding, TT_FORK, NULL, NULL },
	{ "peek", evtag_test_peek, 0, NULL, NULL },

	END_OF_TESTCASES
};

struct testcase_t signal_testcases[] = {
#ifndef WIN32
	LEGACY(simplesignal, TT_ISOLATED),
	LEGACY(multiplesignal, TT_ISOLATED),
	LEGACY(immediatesignal, TT_ISOLATED),
	LEGACY(signal_dealloc, TT_ISOLATED),
	LEGACY(signal_pipeloss, TT_ISOLATED),
	LEGACY(signal_switchbase, TT_ISOLATED|TT_NO_LOGS),
	LEGACY(signal_restore, TT_ISOLATED),
	LEGACY(signal_assert, TT_ISOLATED),
	LEGACY(signal_while_processing, TT_ISOLATED),
#endif
	END_OF_TESTCASES
};

