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

#include "modsdrv.h"

struct event_base;
#include "event2/thread.h"

#include <stdlib.h>
#include <string.h>
#include "mm-internal.h"
#include "evthread-internal.h"

static void *
evthread_mods_lock_alloc(unsigned locktype)
{
	void *lock = ModsDrvAllocMutex();
	if (!lock)
		return NULL;
	return lock;
}

static void
evthread_mods_lock_free(void *lock, unsigned locktype)
{
    ModsDrvFreeMutex(lock);
}

static int
evthread_mods_lock(unsigned mode, void *lock)
{
    int r;
	if (mode & EVTHREAD_TRY)
		r = ModsDrvTryAcquireMutex(lock);
	else
    {
		ModsDrvAcquireMutex(lock);
        r = 1;
    }
    return r ? 0 : -1;
}

static int
evthread_mods_unlock(unsigned mode, void *lock)
{
	ModsDrvReleaseMutex(lock);
    return 0;
}

static unsigned long
evthread_mods_get_id(void)
{
	return (unsigned long)ModsDrvGetLwrrentThreadId();
}

static void *
evthread_mods_cond_alloc(unsigned condflags)
{
	void *cond = ModsDrvAllocCondition();
	if (!cond)
		return NULL;
	return cond;
}

static void
evthread_mods_cond_free(void *cond)
{
    ModsDrvFreeCondition(cond);
}

static int
evthread_mods_cond_signal(void *cond, int broadcast)
{
	if (broadcast)
		ModsDrvBroadcastCondition(cond);
	else
		ModsDrvSignalCondition(cond);
	return 0;
}

static int
evthread_mods_cond_wait(void *cond, void *lock, const struct timeval *tv)
{
	int r;

	if (tv) {
		struct timeval now, abstime;
		struct timespec ts;
		evutil_gettimeofday(&now, NULL);
		evutil_timeradd(&now, tv, &abstime);
		double ms = (double)abstime.tv_sec*1000 + (double)abstime.tv_usec/1000;
		r = ModsDrvWaitConditionTimeout(cond, lock, ms);
        return r ? 0 : 1;
	} else {
		r = ModsDrvWaitCondition(cond, lock);
		return r ? 0 : -1;
	}
}

int
evthread_use_pthreads(void)
{
	struct evthread_lock_callbacks cbs = {
		EVTHREAD_LOCK_API_VERSION,
		EVTHREAD_LOCKTYPE_RELWRSIVE,
		evthread_mods_lock_alloc,
		evthread_mods_lock_free,
		evthread_mods_lock,
		evthread_mods_unlock
	};
	struct evthread_condition_callbacks cond_cbs = {
		EVTHREAD_CONDITION_API_VERSION,
		evthread_mods_cond_alloc,
		evthread_mods_cond_free,
		evthread_mods_cond_signal,
		evthread_mods_cond_wait
	};

	evthread_set_lock_callbacks(&cbs);
	evthread_set_condition_callbacks(&cond_cbs);
	evthread_set_id_callback(evthread_mods_get_id);
	return 0;
}
