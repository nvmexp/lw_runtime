/*
 * Copyright 2009-2012 Niels Provos and Nick Mathewson
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
#include "event2/event-config.h"

/* With glibc we need to define this to get PTHREAD_MUTEX_RELWRSIVE. */
#define _GNU_SOURCE
#include <pthread.h>

struct event_base;
#include "event2/thread.h"

#include <stdlib.h>
#include <string.h>
#include "mm-internal.h"
#include "evthread-internal.h"

static pthread_mutexattr_t attr_relwrsive;

static void *
evthread_posix_lock_alloc(unsigned locktype)
{
	pthread_mutexattr_t *attr = NULL;
	pthread_mutex_t *lock = mm_malloc(sizeof(pthread_mutex_t));
	if (!lock)
		return NULL;
	if (locktype & EVTHREAD_LOCKTYPE_RELWRSIVE)
		attr = &attr_relwrsive;
	if (pthread_mutex_init(lock, attr)) {
		mm_free(lock);
		return NULL;
	}
	return lock;
}

static void
evthread_posix_lock_free(void *_lock, unsigned locktype)
{
	pthread_mutex_t *lock = _lock;
	pthread_mutex_destroy(lock);
	mm_free(lock);
}

static int
evthread_posix_lock(unsigned mode, void *_lock)
{
	pthread_mutex_t *lock = _lock;
	if (mode & EVTHREAD_TRY)
		return pthread_mutex_trylock(lock);
	else
		return pthread_mutex_lock(lock);
}

static int
evthread_posix_unlock(unsigned mode, void *_lock)
{
	pthread_mutex_t *lock = _lock;
	return pthread_mutex_unlock(lock);
}

static unsigned long
evthread_posix_get_id(void)
{
	union {
		pthread_t thr;
#if _EVENT_SIZEOF_PTHREAD_T > _EVENT_SIZEOF_LONG
		ev_uint64_t id;
#else
		unsigned long id;
#endif
	} r;
#if _EVENT_SIZEOF_PTHREAD_T < _EVENT_SIZEOF_LONG
	memset(&r, 0, sizeof(r));
#endif
	r.thr = pthread_self();
	return (unsigned long)r.id;
}

static void *
evthread_posix_cond_alloc(unsigned condflags)
{
	pthread_cond_t *cond = mm_malloc(sizeof(pthread_cond_t));
	if (!cond)
		return NULL;
	if (pthread_cond_init(cond, NULL)) {
		mm_free(cond);
		return NULL;
	}
	return cond;
}

static void
evthread_posix_cond_free(void *_cond)
{
	pthread_cond_t *cond = _cond;
	pthread_cond_destroy(cond);
	mm_free(cond);
}

static int
evthread_posix_cond_signal(void *_cond, int broadcast)
{
	pthread_cond_t *cond = _cond;
	int r;
	if (broadcast)
		r = pthread_cond_broadcast(cond);
	else
		r = pthread_cond_signal(cond);
	return r ? -1 : 0;
}

static int
evthread_posix_cond_wait(void *_cond, void *_lock, const struct timeval *tv)
{
	int r;
	pthread_cond_t *cond = _cond;
	pthread_mutex_t *lock = _lock;

	if (tv) {
		struct timeval now, abstime;
		struct timespec ts;
		evutil_gettimeofday(&now, NULL);
		evutil_timeradd(&now, tv, &abstime);
		ts.tv_sec = abstime.tv_sec;
		ts.tv_nsec = abstime.tv_usec*1000;
		r = pthread_cond_timedwait(cond, lock, &ts);
		if (r == ETIMEDOUT)
			return 1;
		else if (r)
			return -1;
		else
			return 0;
	} else {
		r = pthread_cond_wait(cond, lock);
		return r ? -1 : 0;
	}
}

int
evthread_use_pthreads(void)
{
	struct evthread_lock_callbacks cbs = {
		EVTHREAD_LOCK_API_VERSION,
		EVTHREAD_LOCKTYPE_RELWRSIVE,
		evthread_posix_lock_alloc,
		evthread_posix_lock_free,
		evthread_posix_lock,
		evthread_posix_unlock
	};
	struct evthread_condition_callbacks cond_cbs = {
		EVTHREAD_CONDITION_API_VERSION,
		evthread_posix_cond_alloc,
		evthread_posix_cond_free,
		evthread_posix_cond_signal,
		evthread_posix_cond_wait
	};
	/* Set ourselves up to get relwrsive locks. */
	if (pthread_mutexattr_init(&attr_relwrsive))
		return -1;
	if (pthread_mutexattr_settype(&attr_relwrsive, PTHREAD_MUTEX_RELWRSIVE))
		return -1;

	evthread_set_lock_callbacks(&cbs);
	evthread_set_condition_callbacks(&cond_cbs);
	evthread_set_id_callback(evthread_posix_get_id);
	return 0;
}
