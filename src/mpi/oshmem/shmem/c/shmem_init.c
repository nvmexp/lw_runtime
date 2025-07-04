/*
 * Copyright (c) 2013-2015 Mellanox Technologies, Inc.
 *                         All rights reserved.
 * Copyright (c) 2016      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "oshmem_config.h"

#include <stdlib.h>

#include "orte/util/show_help.h"

#include "opal/runtime/opal_cr.h"
#include "opal/util/output.h"

#include "oshmem/constants.h"
#include "oshmem/include/shmem.h"
#include "oshmem/runtime/params.h"
#include "oshmem/runtime/runtime.h"
#include "oshmem/shmem/shmem_api_logger.h"

#if OSHMEM_PROFILING
#include "oshmem/include/pshmem.h"
#pragma weak shmem_init = pshmem_init
#pragma weak shmem_init_thread = pshmem_init_thread
#pragma weak start_pes = pstart_pes
#include "oshmem/shmem/c/profile/defines.h"
#endif

extern int oshmem_shmem_globalexit_status;

static inline void _shmem_init(int required, int *provided);

void shmem_init(void)
{
    int provided;
    /* spec says that npes are ignored for now */
    _shmem_init(SHMEM_THREAD_SINGLE, &provided);
}

int shmem_init_thread(int requested, int *provided)
{
    _shmem_init(requested, provided);
    return 0;
}

void start_pes(int npes)
{
    int provided;
    /* spec says that npes are ignored for now */
    _shmem_init(SHMEM_THREAD_SINGLE, &provided);
}

static void shmem_onexit(int exitcode, void *arg)
{
    oshmem_shmem_globalexit_status = exitcode;
    shmem_finalize();
}

static inline void _shmem_init(int required, int *provided)
{
    int err = OSHMEM_SUCCESS;

    if (oshmem_shmem_initialized) {
        /*
         * SPEC: If start_pes() is called multiple times, subsequent calls have no effect.
         */
        return;
    }

    err = oshmem_shmem_init(0, NULL, required, provided);
    if (OSHMEM_SUCCESS != err) {
        /* since spec does not propagete error to user we can only abort */
        SHMEM_API_ERROR("SHMEM failed to initialize - aborting");
        oshmem_shmem_abort(-1);
    }

    OPAL_CR_INIT_LIBRARY();
#if HAVE_ON_EXIT
    on_exit(shmem_onexit, NULL);
#endif
}

