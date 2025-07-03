/*
 * Copyright (c) 2008      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OPAL_SYS_ARCH_TIMER_H
#define OPAL_SYS_ARCH_TIMER_H 1

#include <sys/times.h>

typedef uint64_t opal_timer_t;

static inline opal_timer_t
opal_sys_timer_get_cycles(void)
{
    opal_timer_t ret;
    struct tms aclwrate_clock;

    times(&aclwrate_clock);
    ret = aclwrate_clock.tms_utime + aclwrate_clock.tms_stime;

    return ret;
}

#define OPAL_HAVE_SYS_TIMER_GET_CYCLES 1

#endif /* ! OPAL_SYS_ARCH_TIMER_H */
