/*
 * Copyright (c) 2009-2012 Nick Mathewson and Niels Provos
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
#include <ws2tcpip.h>
#endif

#include "event2/event-config.h"

#include <sys/types.h>

#ifndef WIN32
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif
#ifdef _EVENT_HAVE_NETINET_IN6_H
#include <netinet/in6.h>
#endif
#ifdef _EVENT_HAVE_SYS_WAIT_H
#include <sys/wait.h>
#endif
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "event2/event.h"
#include "event2/util.h"
#include "../ipv6-internal.h"
#include "../util-internal.h"
#include "../log-internal.h"
#include "../strlcpy-internal.h"

#include "regress.h"

enum entry_status { NORMAL, CANONICAL, BAD };

/* This is a big table of results we expect from generating and parsing */
static struct ipv4_entry {
	const char *addr;
	ev_uint32_t res;
	enum entry_status status;
} ipv4_entries[] = {
	{ "1.2.3.4", 0x01020304u, CANONICAL },
	{ "255.255.255.255", 0xffffffffu, CANONICAL },
	{ "256.0.0.0", 0, BAD },
	{ "ABC", 0, BAD },
	{ "1.2.3.4.5", 0, BAD },
	{ "176.192.208.244", 0xb0c0d0f4, CANONICAL },
	{ NULL, 0, BAD },
};

static struct ipv6_entry {
	const char *addr;
	ev_uint32_t res[4];
	enum entry_status status;
} ipv6_entries[] = {
	{ "::", { 0, 0, 0, 0, }, CANONICAL },
	{ "0:0:0:0:0:0:0:0", { 0, 0, 0, 0, }, NORMAL },
	{ "::1", { 0, 0, 0, 1, }, CANONICAL },
	{ "::1.2.3.4", { 0, 0, 0, 0x01020304, }, CANONICAL },
	{ "ffff:1::", { 0xffff0001u, 0, 0, 0, }, CANONICAL },
	{ "ffff:0000::", { 0xffff0000u, 0, 0, 0, }, NORMAL },
	{ "ffff::1234", { 0xffff0000u, 0, 0, 0x1234, }, CANONICAL },
	{ "0102::1.2.3.4", {0x01020000u, 0, 0, 0x01020304u }, NORMAL },
	{ "::9:c0a8:1:1", { 0, 0, 0x0009c0a8u, 0x00010001u }, CANONICAL },
	{ "::ffff:1.2.3.4", { 0, 0, 0x000ffffu, 0x01020304u }, CANONICAL },
	{ "FFFF::", { 0xffff0000u, 0, 0, 0 }, NORMAL },
	{ "foobar.", { 0, 0, 0, 0 }, BAD },
	{ "foobar", { 0, 0, 0, 0 }, BAD },
	{ "fo:obar", { 0, 0, 0, 0 }, BAD },
	{ "ffff", { 0, 0, 0, 0 }, BAD },
	{ "fffff::", { 0, 0, 0, 0 }, BAD },
	{ "fffff::", { 0, 0, 0, 0 }, BAD },
	{ "::1.0.1.1000", { 0, 0, 0, 0 }, BAD },
	{ "1:2:33333:4::", { 0, 0, 0, 0 }, BAD },
	{ "1:2:3:4:5:6:7:8:9", { 0, 0, 0, 0 }, BAD },
	{ "1::2::3", { 0, 0, 0, 0 }, BAD },
	{ ":::1", { 0, 0, 0, 0 }, BAD },
	{ NULL, { 0, 0, 0, 0,  }, BAD },
};

static void
regress_ipv4_parse(void *ptr)
{
	int i;
	for (i = 0; ipv4_entries[i].addr; ++i) {
		char written[128];
		struct ipv4_entry *ent = &ipv4_entries[i];
		struct in_addr in;
		int r;
		r = evutil_inet_pton(AF_INET, ent->addr, &in);
		if (r == 0) {
			if (ent->status != BAD) {
				TT_FAIL(("%s did not parse, but it's a good address!",
					ent->addr));
			}
			continue;
		}
		if (ent->status == BAD) {
			TT_FAIL(("%s parsed, but we expected an error", ent->addr));
			continue;
		}
		if (ntohl(in.s_addr) != ent->res) {
			TT_FAIL(("%s parsed to %lx, but we expected %lx", ent->addr,
				(unsigned long)ntohl(in.s_addr),
				(unsigned long)ent->res));
			continue;
		}
		if (ent->status == CANONICAL) {
			const char *w = evutil_inet_ntop(AF_INET, &in, written,
											 sizeof(written));
			if (!w) {
				TT_FAIL(("Tried to write out %s; got NULL.", ent->addr));
				continue;
			}
			if (strcmp(written, ent->addr)) {
				TT_FAIL(("Tried to write out %s; got %s",
					ent->addr, written));
				continue;
			}
		}

	}

}

static void
regress_ipv6_parse(void *ptr)
{
#ifdef AF_INET6
	int i, j;

	for (i = 0; ipv6_entries[i].addr; ++i) {
		char written[128];
		struct ipv6_entry *ent = &ipv6_entries[i];
		struct in6_addr in6;
		int r;
		r = evutil_inet_pton(AF_INET6, ent->addr, &in6);
		if (r == 0) {
			if (ent->status != BAD)
				TT_FAIL(("%s did not parse, but it's a good address!",
					ent->addr));
			continue;
		}
		if (ent->status == BAD) {
			TT_FAIL(("%s parsed, but we expected an error", ent->addr));
			continue;
		}
		for (j = 0; j < 4; ++j) {
			/* Can't use s6_addr32 here; some don't have it. */
			ev_uint32_t u =
				(in6.s6_addr[j*4  ] << 24) |
				(in6.s6_addr[j*4+1] << 16) |
				(in6.s6_addr[j*4+2] << 8) |
				(in6.s6_addr[j*4+3]);
			if (u != ent->res[j]) {
				TT_FAIL(("%s did not parse as expected.", ent->addr));
				continue;
			}
		}
		if (ent->status == CANONICAL) {
			const char *w = evutil_inet_ntop(AF_INET6, &in6, written,
											 sizeof(written));
			if (!w) {
				TT_FAIL(("Tried to write out %s; got NULL.", ent->addr));
				continue;
			}
			if (strcmp(written, ent->addr)) {
				TT_FAIL(("Tried to write out %s; got %s", ent->addr, written));
				continue;
			}
		}

	}
#else
	TT_BLATHER(("Skipping IPv6 address parsing."));
#endif
}

static struct sa_port_ent {
	const char *parse;
	int safamily;
	const char *addr;
	int port;
} sa_port_ents[] = {
	{ "[ffff::1]:1000", AF_INET6, "ffff::1", 1000 },
	{ "[ffff::1]", AF_INET6, "ffff::1", 0 },
	{ "[ffff::1", 0, NULL, 0 },
	{ "[ffff::1]:65599", 0, NULL, 0 },
	{ "[ffff::1]:0", 0, NULL, 0 },
	{ "[ffff::1]:-1", 0, NULL, 0 },
	{ "::1", AF_INET6, "::1", 0 },
	{ "1:2::1", AF_INET6, "1:2::1", 0 },
	{ "192.168.0.1:50", AF_INET, "192.168.0.1", 50 },
	{ "1.2.3.4", AF_INET, "1.2.3.4", 0 },
	{ NULL, 0, NULL, 0 },
};

static void
regress_sockaddr_port_parse(void *ptr)
{
	struct sockaddr_storage ss;
	int i, r;

	for (i = 0; sa_port_ents[i].parse; ++i) {
		struct sa_port_ent *ent = &sa_port_ents[i];
		int len = sizeof(ss);
		memset(&ss, 0, sizeof(ss));
		r = evutil_parse_sockaddr_port(ent->parse, (struct sockaddr*)&ss, &len);
		if (r < 0) {
			if (ent->safamily)
				TT_FAIL(("Couldn't parse %s!", ent->parse));
			continue;
		} else if (! ent->safamily) {
			TT_FAIL(("Shouldn't have been able to parse %s!", ent->parse));
			continue;
		}
		if (ent->safamily == AF_INET) {
			struct sockaddr_in sin;
			memset(&sin, 0, sizeof(sin));
#ifdef _EVENT_HAVE_STRUCT_SOCKADDR_IN_SIN_LEN
			sin.sin_len = sizeof(sin);
#endif
			sin.sin_family = AF_INET;
			sin.sin_port = htons(ent->port);
			r = evutil_inet_pton(AF_INET, ent->addr, &sin.sin_addr);
			if (1 != r) {
				TT_FAIL(("Couldn't parse ipv4 target %s.", ent->addr));
			} else if (memcmp(&sin, &ss, sizeof(sin))) {
				TT_FAIL(("Parse for %s was not as expected.", ent->parse));
			} else if (len != sizeof(sin)) {
				TT_FAIL(("Length for %s not as expected.",ent->parse));
			}
		} else {
			struct sockaddr_in6 sin6;
			memset(&sin6, 0, sizeof(sin6));
#ifdef _EVENT_HAVE_STRUCT_SOCKADDR_IN6_SIN6_LEN
			sin6.sin6_len = sizeof(sin6);
#endif
			sin6.sin6_family = AF_INET6;
			sin6.sin6_port = htons(ent->port);
			r = evutil_inet_pton(AF_INET6, ent->addr, &sin6.sin6_addr);
			if (1 != r) {
				TT_FAIL(("Couldn't parse ipv6 target %s.", ent->addr));
			} else if (memcmp(&sin6, &ss, sizeof(sin6))) {
				TT_FAIL(("Parse for %s was not as expected.", ent->parse));
			} else if (len != sizeof(sin6)) {
				TT_FAIL(("Length for %s not as expected.",ent->parse));
			}
		}
	}
}


static void
regress_sockaddr_port_format(void *ptr)
{
	struct sockaddr_storage ss;
	int len;
	const char *cp;
	char cbuf[128];
	int r;

	len = sizeof(ss);
	r = evutil_parse_sockaddr_port("192.168.1.1:80",
	    (struct sockaddr*)&ss, &len);
	tt_int_op(r,==,0);
	cp = evutil_format_sockaddr_port(
		(struct sockaddr*)&ss, cbuf, sizeof(cbuf));
	tt_ptr_op(cp,==,cbuf);
	tt_str_op(cp,==,"192.168.1.1:80");

	len = sizeof(ss);
	r = evutil_parse_sockaddr_port("[ff00::8010]:999",
	    (struct sockaddr*)&ss, &len);
	tt_int_op(r,==,0);
	cp = evutil_format_sockaddr_port(
		(struct sockaddr*)&ss, cbuf, sizeof(cbuf));
	tt_ptr_op(cp,==,cbuf);
	tt_str_op(cp,==,"[ff00::8010]:999");

	ss.ss_family=99;
	cp = evutil_format_sockaddr_port(
		(struct sockaddr*)&ss, cbuf, sizeof(cbuf));
	tt_ptr_op(cp,==,cbuf);
	tt_str_op(cp,==,"<addr with socktype 99>");
end:
	;
}

static struct sa_pred_ent {
	const char *parse;

	int is_loopback;
} sa_pred_entries[] = {
	{ "127.0.0.1",	 1 },
	{ "127.0.3.2",	 1 },
	{ "128.1.2.3",	 0 },
	{ "18.0.0.1",	 0 },
	{ "129.168.1.1", 0 },

	{ "::1",	 1 },
	{ "::0",	 0 },
	{ "f::1",	 0 },
	{ "::501",	 0 },
	{ NULL,		 0 },

};

static void
test_evutil_sockaddr_predicates(void *ptr)
{
	struct sockaddr_storage ss;
	int r, i;

	for (i=0; sa_pred_entries[i].parse; ++i) {
		struct sa_pred_ent *ent = &sa_pred_entries[i];
		int len = sizeof(ss);

		r = evutil_parse_sockaddr_port(ent->parse, (struct sockaddr*)&ss, &len);

		if (r<0) {
			TT_FAIL(("Couldn't parse %s!", ent->parse));
			continue;
		}

		/* sockaddr_is_loopback */
		if (ent->is_loopback != evutil_sockaddr_is_loopback((struct sockaddr*)&ss)) {
			TT_FAIL(("evutil_sockaddr_loopback(%s) not as expected",
				ent->parse));
		}
	}
}

static void
test_evutil_strtoll(void *ptr)
{
	const char *s;
	char *endptr;

	tt_want(evutil_strtoll("5000000000", NULL, 10) ==
		((ev_int64_t)5000000)*1000);
	tt_want(evutil_strtoll("-5000000000", NULL, 10) ==
		((ev_int64_t)5000000)*-1000);
	s = " 99999stuff";
	tt_want(evutil_strtoll(s, &endptr, 10) == (ev_int64_t)99999);
	tt_want(endptr == s+6);
	tt_want(evutil_strtoll("foo", NULL, 10) == 0);
 }

static void
test_evutil_snprintf(void *ptr)
{
	char buf[16];
	int r;
	ev_uint64_t u64 = ((ev_uint64_t)1000000000)*200;
	ev_int64_t i64 = -1 * (ev_int64_t) u64;
	size_t size = 8000;
	ev_ssize_t ssize = -9000;

	r = evutil_snprintf(buf, sizeof(buf), "%d %d", 50, 100);
	tt_str_op(buf, ==, "50 100");
	tt_int_op(r, ==, 6);

	r = evutil_snprintf(buf, sizeof(buf), "longish %d", 1234567890);
	tt_str_op(buf, ==, "longish 1234567");
	tt_int_op(r, ==, 18);

	r = evutil_snprintf(buf, sizeof(buf), EV_U64_FMT, EV_U64_ARG(u64));
	tt_str_op(buf, ==, "200000000000");
	tt_int_op(r, ==, 12);

	r = evutil_snprintf(buf, sizeof(buf), EV_I64_FMT, EV_I64_ARG(i64));
	tt_str_op(buf, ==, "-200000000000");
	tt_int_op(r, ==, 13);

	r = evutil_snprintf(buf, sizeof(buf), EV_SIZE_FMT" "EV_SSIZE_FMT,
	    EV_SIZE_ARG(size), EV_SSIZE_ARG(ssize));
	tt_str_op(buf, ==, "8000 -9000");
	tt_int_op(r, ==, 10);

      end:
	;
}

static void
test_evutil_casecmp(void *ptr)
{
	tt_int_op(evutil_ascii_strcasecmp("ABC", "ABC"), ==, 0);
	tt_int_op(evutil_ascii_strcasecmp("ABC", "abc"), ==, 0);
	tt_int_op(evutil_ascii_strcasecmp("ABC", "abcd"), <, 0);
	tt_int_op(evutil_ascii_strcasecmp("ABC", "abb"), >, 0);
	tt_int_op(evutil_ascii_strcasecmp("ABCd", "abc"), >, 0);

	tt_int_op(evutil_ascii_strncasecmp("Libevent", "LibEvEnT", 100), ==, 0);
	tt_int_op(evutil_ascii_strncasecmp("Libevent", "LibEvEnT", 4), ==, 0);
	tt_int_op(evutil_ascii_strncasecmp("Libevent", "LibEXXXX", 4), ==, 0);
	tt_int_op(evutil_ascii_strncasecmp("Libevent", "LibE", 4), ==, 0);
	tt_int_op(evutil_ascii_strncasecmp("Libe", "LibEvEnT", 4), ==, 0);
	tt_int_op(evutil_ascii_strncasecmp("Lib", "LibEvEnT", 4), <, 0);
	tt_int_op(evutil_ascii_strncasecmp("abc", "def", 99), <, 0);
	tt_int_op(evutil_ascii_strncasecmp("Z", "qrst", 1), >, 0);
end:
	;
}

static int logsev = 0;
static char *logmsg = NULL;

static void
logfn(int severity, const char *msg)
{
	logsev = severity;
	tt_want(msg);
	if (msg) {
		if (logmsg)
			free(logmsg);
		logmsg = strdup(msg);
	}
}

static int fatal_want_severity = 0;
static const char *fatal_want_message = NULL;
static void
fatalfn(int exitcode)
{
	if (logsev != fatal_want_severity ||
	    !logmsg ||
	    strcmp(logmsg, fatal_want_message))
		exit(0);
	else
		exit(exitcode);
}

#ifndef WIN32
#define CAN_CHECK_ERR
static void
check_error_logging(void (*fn)(void), int wantexitcode,
    int wantseverity, const char *wantmsg)
{
	pid_t pid;
	int status = 0, exitcode;
	fatal_want_severity = wantseverity;
	fatal_want_message = wantmsg;
	if ((pid = regress_fork()) == 0) {
		/* child process */
		fn();
		exit(0); /* should be unreachable. */
	} else {
		wait(&status);
		exitcode = WEXITSTATUS(status);
		tt_int_op(wantexitcode, ==, exitcode);
	}
end:
	;
}

static void
errx_fn(void)
{
	event_errx(2, "Fatal error; too many kumquats (%d)", 5);
}

static void
err_fn(void)
{
	errno = ENOENT;
	event_err(5,"Couldn't open %s", "/very/bad/file");
}

static void
sock_err_fn(void)
{
	evutil_socket_t fd = socket(AF_INET, SOCK_STREAM, 0);
#ifdef WIN32
	EVUTIL_SET_SOCKET_ERROR(WSAEWOULDBLOCK);
#else
	errno = EAGAIN;
#endif
	event_sock_err(20, fd, "Unhappy socket");
}
#endif

static void
test_evutil_log(void *ptr)
{
	evutil_socket_t fd = -1;
	char buf[128];

	event_set_log_callback(logfn);
	event_set_fatal_callback(fatalfn);
#define RESET() do {				\
		logsev = 0;	\
		if (logmsg) free(logmsg);	\
		logmsg = NULL;			\
	} while (0)
#define LOGEQ(sev,msg) do {			\
		tt_int_op(logsev,==,sev);	\
		tt_assert(logmsg != NULL);	\
		tt_str_op(logmsg,==,msg);	\
	} while (0)

#ifdef CAN_CHECK_ERR
	/* We need to disable these tests for now.  Previously, the logging
	 * module didn't enforce the requirement that a fatal callback
	 * actually exit.  Now, it exits no matter what, so if we wan to
	 * reinstate these tests, we'll need to fork for each one. */
	check_error_logging(errx_fn, 2, _EVENT_LOG_ERR,
	    "Fatal error; too many kumquats (5)");
	RESET();
#endif

	event_warnx("Far too many %s (%d)", "wombats", 99);
	LOGEQ(_EVENT_LOG_WARN, "Far too many wombats (99)");
	RESET();

	event_msgx("Connecting lime to coconut");
	LOGEQ(_EVENT_LOG_MSG, "Connecting lime to coconut");
	RESET();

	event_debug(("A millisecond passed! We should log that!"));
#ifdef USE_DEBUG
	LOGEQ(_EVENT_LOG_DEBUG, "A millisecond passed! We should log that!");
#else
	tt_int_op(logsev,==,0);
	tt_ptr_op(logmsg,==,NULL);
#endif
	RESET();

	/* Try with an errno. */
	errno = ENOENT;
	event_warn("Couldn't open %s", "/bad/file");
	evutil_snprintf(buf, sizeof(buf),
	    "Couldn't open /bad/file: %s",strerror(ENOENT));
	LOGEQ(_EVENT_LOG_WARN,buf);
	RESET();

#ifdef CAN_CHECK_ERR
	evutil_snprintf(buf, sizeof(buf),
	    "Couldn't open /very/bad/file: %s",strerror(ENOENT));
	check_error_logging(err_fn, 5, _EVENT_LOG_ERR, buf);
	RESET();
#endif

	/* Try with a socket errno. */
	fd = socket(AF_INET, SOCK_STREAM, 0);
#ifdef WIN32
	evutil_snprintf(buf, sizeof(buf),
	    "Unhappy socket: %s",
	    evutil_socket_error_to_string(WSAEWOULDBLOCK));
	EVUTIL_SET_SOCKET_ERROR(WSAEWOULDBLOCK);
#else
	evutil_snprintf(buf, sizeof(buf),
	    "Unhappy socket: %s", strerror(EAGAIN));
	errno = EAGAIN;
#endif
	event_sock_warn(fd, "Unhappy socket");
	LOGEQ(_EVENT_LOG_WARN, buf);
	RESET();

#ifdef CAN_CHECK_ERR
	check_error_logging(sock_err_fn, 20, _EVENT_LOG_ERR, buf);
	RESET();
#endif

#undef RESET
#undef LOGEQ
end:
	if (logmsg)
		free(logmsg);
	if (fd >= 0)
		evutil_closesocket(fd);
}

static void
test_evutil_strlcpy(void *arg)
{
	char buf[8];

	/* Successful case. */
	tt_int_op(5, ==, strlcpy(buf, "Hello", sizeof(buf)));
	tt_str_op(buf, ==, "Hello");

	/* Overflow by a lot. */
	tt_int_op(13, ==, strlcpy(buf, "pentasyllabic", sizeof(buf)));
	tt_str_op(buf, ==, "pentasy");

	/* Overflow by exactly one. */
	tt_int_op(8, ==, strlcpy(buf, "overlong", sizeof(buf)));
	tt_str_op(buf, ==, "overlon");
end:
	;
}

struct example_struct {
	const char *a;
	const char *b;
	long c;
};

static void
test_evutil_upcast(void *arg)
{
	struct example_struct es1;
	const char **cp;
	es1.a = "World";
	es1.b = "Hello";
	es1.c = -99;

	tt_int_op(evutil_offsetof(struct example_struct, b), ==, sizeof(char*));

	cp = &es1.b;
	tt_ptr_op(EVUTIL_UPCAST(cp, struct example_struct, b), ==, &es1);

end:
	;
}

static void
test_evutil_integers(void *arg)
{
	ev_int64_t i64;
	ev_uint64_t u64;
	ev_int32_t i32;
	ev_uint32_t u32;
	ev_int16_t i16;
	ev_uint16_t u16;
	ev_int8_t  i8;
	ev_uint8_t  u8;

	void *ptr;
	ev_intptr_t iptr;
	ev_uintptr_t uptr;

	ev_ssize_t ssize;

	tt_int_op(sizeof(u64), ==, 8);
	tt_int_op(sizeof(i64), ==, 8);
	tt_int_op(sizeof(u32), ==, 4);
	tt_int_op(sizeof(i32), ==, 4);
	tt_int_op(sizeof(u16), ==, 2);
	tt_int_op(sizeof(i16), ==, 2);
	tt_int_op(sizeof(u8), ==,  1);
	tt_int_op(sizeof(i8), ==,  1);

	tt_int_op(sizeof(ev_ssize_t), ==, sizeof(size_t));
	tt_int_op(sizeof(ev_intptr_t), >=, sizeof(void *));
	tt_int_op(sizeof(ev_uintptr_t), ==, sizeof(intptr_t));

	u64 = 1000000000;
	u64 *= 1000000000;
	tt_assert(u64 / 1000000000 == 1000000000);
	i64 = -1000000000;
	i64 *= 1000000000;
	tt_assert(i64 / 1000000000 == -1000000000);

	u64 = EV_UINT64_MAX;
	i64 = EV_INT64_MAX;
	tt_assert(u64 > 0);
	tt_assert(i64 > 0);
	u64++;
	i64++;
	tt_assert(u64 == 0);
	tt_assert(i64 == EV_INT64_MIN);
	tt_assert(i64 < 0);

	u32 = EV_UINT32_MAX;
	i32 = EV_INT32_MAX;
	tt_assert(u32 > 0);
	tt_assert(i32 > 0);
	u32++;
	i32++;
	tt_assert(u32 == 0);
	tt_assert(i32 == EV_INT32_MIN);
	tt_assert(i32 < 0);

	u16 = EV_UINT16_MAX;
	i16 = EV_INT16_MAX;
	tt_assert(u16 > 0);
	tt_assert(i16 > 0);
	u16++;
	i16++;
	tt_assert(u16 == 0);
	tt_assert(i16 == EV_INT16_MIN);
	tt_assert(i16 < 0);

	u8 = EV_UINT8_MAX;
	i8 = EV_INT8_MAX;
	tt_assert(u8 > 0);
	tt_assert(i8 > 0);
	u8++;
	i8++;
	tt_assert(u8 == 0);
	tt_assert(i8 == EV_INT8_MIN);
	tt_assert(i8 < 0);

	ssize = EV_SSIZE_MAX;
	tt_assert(ssize > 0);
	ssize++;
	tt_assert(ssize < 0);
	tt_assert(ssize == EV_SSIZE_MIN);

	ptr = &ssize;
	iptr = (ev_intptr_t)ptr;
	uptr = (ev_uintptr_t)ptr;
	ptr = (void *)iptr;
	tt_assert(ptr == &ssize);
	ptr = (void *)uptr;
	tt_assert(ptr == &ssize);

	iptr = -1;
	tt_assert(iptr < 0);
end:
	;
}

struct evutil_addrinfo *
ai_find_by_family(struct evutil_addrinfo *ai, int family)
{
	while (ai) {
		if (ai->ai_family == family)
			return ai;
		ai = ai->ai_next;
	}
	return NULL;
}

struct evutil_addrinfo *
ai_find_by_protocol(struct evutil_addrinfo *ai, int protocol)
{
	while (ai) {
		if (ai->ai_protocol == protocol)
			return ai;
		ai = ai->ai_next;
	}
	return NULL;
}


int
_test_ai_eq(const struct evutil_addrinfo *ai, const char *sockaddr_port,
    int socktype, int protocol, int line)
{
	struct sockaddr_storage ss;
	int slen = sizeof(ss);
	int gotport;
	char buf[128];
	memset(&ss, 0, sizeof(ss));
	if (socktype > 0)
		tt_int_op(ai->ai_socktype, ==, socktype);
	if (protocol > 0)
		tt_int_op(ai->ai_protocol, ==, protocol);

	if (evutil_parse_sockaddr_port(
		    sockaddr_port, (struct sockaddr*)&ss, &slen)<0) {
		TT_FAIL(("Couldn't parse expected address %s on line %d",
			sockaddr_port, line));
		return -1;
	}
	if (ai->ai_family != ss.ss_family) {
		TT_FAIL(("Address family %d did not match %d on line %d",
			ai->ai_family, ss.ss_family, line));
		return -1;
	}
	if (ai->ai_addr->sa_family == AF_INET) {
		struct sockaddr_in *sin = (struct sockaddr_in*)ai->ai_addr;
		evutil_inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf));
		gotport = ntohs(sin->sin_port);
		if (ai->ai_addrlen != sizeof(struct sockaddr_in)) {
			TT_FAIL(("Addr size mismatch on line %d", line));
			return -1;
		}
	} else {
		struct sockaddr_in6 *sin6 = (struct sockaddr_in6*)ai->ai_addr;
		evutil_inet_ntop(AF_INET6, &sin6->sin6_addr, buf, sizeof(buf));
		gotport = ntohs(sin6->sin6_port);
		if (ai->ai_addrlen != sizeof(struct sockaddr_in6)) {
			TT_FAIL(("Addr size mismatch on line %d", line));
			return -1;
		}
	}
	if (evutil_sockaddr_cmp(ai->ai_addr, (struct sockaddr*)&ss, 1)) {
		TT_FAIL(("Wanted %s, got %s:%d on line %d", sockaddr_port,
			buf, gotport, line));
		return -1;
	} else {
		TT_BLATHER(("Wanted %s, got %s:%d on line %d", sockaddr_port,
			buf, gotport, line));
	}
	return 0;
end:
	TT_FAIL(("Test failed on line %d", line));
	return -1;
}

static void
test_evutil_rand(void *arg)
{
	char buf1[32];
	char buf2[32];
	int counts[256];
	int i, j, k, n=0;

	memset(buf2, 0, sizeof(buf2));
	memset(counts, 0, sizeof(counts));

	for (k=0;k<32;++k) {
		/* Try a few different start and end points; try to catch
		 * the various misaligned cases of arc4random_buf */
		int startpoint = _evutil_weakrand() % 4;
		int endpoint = 32 - (_evutil_weakrand() % 4);

		memset(buf2, 0, sizeof(buf2));

		/* Do 6 runs over buf1, or-ing the result into buf2 each
		 * time, to make sure we're setting each byte that we mean
		 * to set. */
		for (i=0;i<8;++i) {
			memset(buf1, 0, sizeof(buf1));
			evutil_selwre_rng_get_bytes(buf1 + startpoint,
			    endpoint-startpoint);
			n += endpoint - startpoint;
			for (j=0; j<32; ++j) {
				if (j >= startpoint && j < endpoint) {
					buf2[j] |= buf1[j];
					++counts[(unsigned char)buf1[j]];
				} else {
					tt_assert(buf1[j] == 0);
					tt_int_op(buf1[j], ==, 0);

				}
			}
		}

		/* This will give a false positive with P=(256**8)==(2**64)
		 * for each character. */
		for (j=startpoint;j<endpoint;++j) {
			tt_int_op(buf2[j], !=, 0);
		}
	}

	/* for (i=0;i<256;++i) { printf("%3d %2d\n", i, counts[i]); } */
end:
	;
}

static void
test_evutil_getaddrinfo(void *arg)
{
	struct evutil_addrinfo *ai = NULL, *a;
	struct evutil_addrinfo hints;

	struct sockaddr_in6 *sin6;
	struct sockaddr_in *sin;
	char buf[128];
	const char *cp;
	int r;

	/* Try using it as a pton. */
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = PF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	r = evutil_getaddrinfo("1.2.3.4", "8080", &hints, &ai);
	tt_int_op(r, ==, 0);
	tt_assert(ai);
	tt_ptr_op(ai->ai_next, ==, NULL); /* no ambiguity */
	test_ai_eq(ai, "1.2.3.4:8080", SOCK_STREAM, IPPROTO_TCP);
	evutil_freeaddrinfo(ai);
	ai = NULL;

	memset(&hints, 0, sizeof(hints));
	hints.ai_family = PF_UNSPEC;
	hints.ai_protocol = IPPROTO_UDP;
	r = evutil_getaddrinfo("1001:b0b::f00f", "4321", &hints, &ai);
	tt_int_op(r, ==, 0);
	tt_assert(ai);
	tt_ptr_op(ai->ai_next, ==, NULL); /* no ambiguity */
	test_ai_eq(ai, "[1001:b0b::f00f]:4321", SOCK_DGRAM, IPPROTO_UDP);
	evutil_freeaddrinfo(ai);
	ai = NULL;

	/* Try out the behavior of nodename=NULL */
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = PF_INET;
	hints.ai_protocol = IPPROTO_TCP;
	hints.ai_flags = EVUTIL_AI_PASSIVE; /* as if for bind */
	r = evutil_getaddrinfo(NULL, "9999", &hints, &ai);
	tt_int_op(r,==,0);
	tt_assert(ai);
	tt_ptr_op(ai->ai_next, ==, NULL);
	test_ai_eq(ai, "0.0.0.0:9999", SOCK_STREAM, IPPROTO_TCP);
	evutil_freeaddrinfo(ai);
	ai = NULL;
	hints.ai_flags = 0; /* as if for connect */
	r = evutil_getaddrinfo(NULL, "9998", &hints, &ai);
	tt_assert(ai);
	tt_int_op(r,==,0);
	test_ai_eq(ai, "127.0.0.1:9998", SOCK_STREAM, IPPROTO_TCP);
	tt_ptr_op(ai->ai_next, ==, NULL);
	evutil_freeaddrinfo(ai);
	ai = NULL;

	hints.ai_flags = 0; /* as if for connect */
	hints.ai_family = PF_INET6;
	r = evutil_getaddrinfo(NULL, "9997", &hints, &ai);
	tt_assert(ai);
	tt_int_op(r,==,0);
	tt_ptr_op(ai->ai_next, ==, NULL);
	test_ai_eq(ai, "[::1]:9997", SOCK_STREAM, IPPROTO_TCP);
	evutil_freeaddrinfo(ai);
	ai = NULL;

	hints.ai_flags = EVUTIL_AI_PASSIVE; /* as if for bind. */
	hints.ai_family = PF_INET6;
	r = evutil_getaddrinfo(NULL, "9996", &hints, &ai);
	tt_assert(ai);
	tt_int_op(r,==,0);
	tt_ptr_op(ai->ai_next, ==, NULL);
	test_ai_eq(ai, "[::]:9996", SOCK_STREAM, IPPROTO_TCP);
	evutil_freeaddrinfo(ai);
	ai = NULL;

	/* Now try an unspec one. We should get a v6 and a v4. */
	hints.ai_family = PF_UNSPEC;
	r = evutil_getaddrinfo(NULL, "9996", &hints, &ai);
	tt_assert(ai);
	tt_int_op(r,==,0);
	a = ai_find_by_family(ai, PF_INET6);
	tt_assert(a);
	test_ai_eq(a, "[::]:9996", SOCK_STREAM, IPPROTO_TCP);
	a = ai_find_by_family(ai, PF_INET);
	tt_assert(a);
	test_ai_eq(a, "0.0.0.0:9996", SOCK_STREAM, IPPROTO_TCP);
	evutil_freeaddrinfo(ai);
	ai = NULL;

	/* Try out AI_NUMERICHOST: successful case.  Also try
	 * multiprotocol. */
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = PF_UNSPEC;
	hints.ai_flags = EVUTIL_AI_NUMERICHOST;
	r = evutil_getaddrinfo("1.2.3.4", NULL, &hints, &ai);
	tt_int_op(r, ==, 0);
	a = ai_find_by_protocol(ai, IPPROTO_TCP);
	tt_assert(a);
	test_ai_eq(a, "1.2.3.4", SOCK_STREAM, IPPROTO_TCP);
	a = ai_find_by_protocol(ai, IPPROTO_UDP);
	tt_assert(a);
	test_ai_eq(a, "1.2.3.4", SOCK_DGRAM, IPPROTO_UDP);
	evutil_freeaddrinfo(ai);
	ai = NULL;

	/* Try the failing case of AI_NUMERICHOST */
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = PF_UNSPEC;
	hints.ai_flags = EVUTIL_AI_NUMERICHOST;
	r = evutil_getaddrinfo("www.google.com", "80", &hints, &ai);
	tt_int_op(r, ==, EVUTIL_EAI_NONAME);
	tt_ptr_op(ai, ==, NULL);

	/* Try symbolic service names wit AI_NUMERICSERV */
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = PF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_flags = EVUTIL_AI_NUMERICSERV;
	r = evutil_getaddrinfo("1.2.3.4", "http", &hints, &ai);
	tt_int_op(r,==,EVUTIL_EAI_NONAME);

	/* Try symbolic service names */
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = PF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	r = evutil_getaddrinfo("1.2.3.4", "http", &hints, &ai);
	if (r!=0) {
		TT_DECLARE("SKIP", ("Symbolic service names seem broken."));
	} else {
		tt_assert(ai);
		test_ai_eq(ai, "1.2.3.4:80", SOCK_STREAM, IPPROTO_TCP);
		evutil_freeaddrinfo(ai);
		ai = NULL;
	}

	/* Now do some actual lookups. */
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = PF_INET;
	hints.ai_protocol = IPPROTO_TCP;
	hints.ai_socktype = SOCK_STREAM;
	r = evutil_getaddrinfo("www.google.com", "80", &hints, &ai);
	if (r != 0) {
		TT_DECLARE("SKIP", ("Couldn't resolve www.google.com"));
	} else {
		tt_assert(ai);
		tt_int_op(ai->ai_family, ==, PF_INET);
		tt_int_op(ai->ai_protocol, ==, IPPROTO_TCP);
		tt_int_op(ai->ai_socktype, ==, SOCK_STREAM);
		tt_int_op(ai->ai_addrlen, ==, sizeof(struct sockaddr_in));
		sin = (struct sockaddr_in*)ai->ai_addr;
		tt_int_op(sin->sin_family, ==, AF_INET);
		tt_int_op(sin->sin_port, ==, htons(80));
		tt_int_op(sin->sin_addr.s_addr, !=, 0xffffffff);

		cp = evutil_inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf));
		TT_BLATHER(("www.google.com resolved to %s",
			cp?cp:"<unwriteable>"));
		evutil_freeaddrinfo(ai);
		ai = NULL;
	}

	hints.ai_family = PF_INET6;
	r = evutil_getaddrinfo("ipv6.google.com", "80", &hints, &ai);
	if (r != 0) {
		TT_BLATHER(("Couldn't do an ipv6 lookup for ipv6.google.com"));
	} else {
		tt_assert(ai);
		tt_int_op(ai->ai_family, ==, PF_INET6);
		tt_int_op(ai->ai_addrlen, ==, sizeof(struct sockaddr_in6));
		sin6 = (struct sockaddr_in6*)ai->ai_addr;
		tt_int_op(sin6->sin6_port, ==, htons(80));

		cp = evutil_inet_ntop(AF_INET6, &sin6->sin6_addr, buf,
		    sizeof(buf));
		TT_BLATHER(("ipv6.google.com resolved to %s",
			cp?cp:"<unwriteable>"));
	}

end:
	if (ai)
		evutil_freeaddrinfo(ai);
}

#ifdef WIN32
static void
test_evutil_loadsyslib(void *arg)
{
	HANDLE h=NULL;

	h = evutil_load_windows_system_library(TEXT("kernel32.dll"));
	tt_assert(h);

end:
	if (h)
		CloseHandle(h);

}
#endif

struct testcase_t util_testcases[] = {
	{ "ipv4_parse", regress_ipv4_parse, 0, NULL, NULL },
	{ "ipv6_parse", regress_ipv6_parse, 0, NULL, NULL },
	{ "sockaddr_port_parse", regress_sockaddr_port_parse, 0, NULL, NULL },
	{ "sockaddr_port_format", regress_sockaddr_port_format, 0, NULL, NULL },
	{ "sockaddr_predicates", test_evutil_sockaddr_predicates, 0,NULL,NULL },
	{ "evutil_snprintf", test_evutil_snprintf, 0, NULL, NULL },
	{ "evutil_strtoll", test_evutil_strtoll, 0, NULL, NULL },
	{ "evutil_casecmp", test_evutil_casecmp, 0, NULL, NULL },
	{ "strlcpy", test_evutil_strlcpy, 0, NULL, NULL },
	{ "log", test_evutil_log, TT_FORK, NULL, NULL },
	{ "upcast", test_evutil_upcast, 0, NULL, NULL },
	{ "integers", test_evutil_integers, 0, NULL, NULL },
	{ "rand", test_evutil_rand, TT_FORK, NULL, NULL },
	{ "getaddrinfo", test_evutil_getaddrinfo, TT_FORK, NULL, NULL },
#ifdef WIN32
	{ "loadsyslib", test_evutil_loadsyslib, TT_FORK, NULL, NULL },
#endif
	END_OF_TESTCASES,
};

