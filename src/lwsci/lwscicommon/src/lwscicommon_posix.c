/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/**
 * \file
 * \brief <b>LwSciCommon posix layer Implementation</b>
 *
 * @b Description: The APIs in this file use posix functions
 */
#include "lwscicommon_os.h"

#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <pthread.h>
#include <fcntl.h>
#include <time.h>
#include <limits.h>
#include "lwscicommon_covanalysis.h"
#include "lwscicommon_libc.h"
#include "lwscilog.h"

void LwSciCommonPanic(void)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 21_8), "Approved TID-1010")
    abort();
}

LwSciError LwSciCommonMutexCreate(LwSciCommonMutex* mutex)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_6), "LwSciCommon-ADV-MISRAC2012-003")
    int retVal = 0;
    LwSciError err = LwSciError_Success;

    if (NULL == mutex) {
        LWSCI_ERR_STR("Input is NULL\n");
        LwSciCommonPanic();
    }

    retVal = pthread_mutex_init(mutex, NULL);
    if (EAGAIN == retVal) {
        err = LwSciError_ResourceError;
        LWSCI_ERR_STR("Pthread mutex init failed due to resource error.\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciCommon-ADV-MISRAC2012-006")
        goto ret;
    }

    if (ENOMEM == retVal) {
        err = LwSciError_InsufficientMemory;
        LWSCI_ERR_STR("Pthread mutex init failed due to insufficient memory\n");
        LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 15_1), "LwSciCommon-ADV-MISRAC2012-006")
        goto ret;
    }

    if (0 != retVal) {
        LWSCI_ERR_STR("Pthread mutex init failed\n");
        LwSciCommonPanic();
    }

ret:
    return err;
}

void LwSciCommonMutexLock(LwSciCommonMutex* mutex)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_6), "LwSciCommon-ADV-MISRAC2012-003")
    int retVal = 0;

    if (NULL == mutex) {
        LWSCI_ERR_STR("Input is NULL\n");
        LwSciCommonPanic();
    }

    retVal = pthread_mutex_lock(mutex);

    if (0 != retVal) {
        LWSCI_ERR_STR("Pthread mutex init failed\n");
        LwSciCommonPanic();
    }
}

void LwSciCommonMutexUnlock(LwSciCommonMutex* mutex)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_6), "LwSciCommon-ADV-MISRAC2012-003")
    int retVal = 0;

    if (NULL == mutex) {
        LWSCI_ERR_STR("Input is NULL\n");
        LwSciCommonPanic();
    }

    retVal = pthread_mutex_unlock(mutex);

    if (0 != retVal) {
        LWSCI_ERR_STR("Pthread mutex init failed\n");
        LwSciCommonPanic();
    }
}

void LwSciCommonMutexDestroy(LwSciCommonMutex* mutex)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_6), "LwSciCommon-ADV-MISRAC2012-003")
    int retVal = 0;

    if (NULL == mutex) {
        LWSCI_ERR_STR("Input is NULL\n");
        LwSciCommonPanic();
    }

    retVal = pthread_mutex_destroy(mutex);

    if (0 != retVal) {
        LWSCI_ERR_STR("Pthread mutex init failed\n");
        LwSciCommonPanic();
    }
}


LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Rule, 8_7), "LwSciCommon-ADV-MISRAC2012-009")
void LwSciCommonSleepNs(uint64_t timeNs)
{
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_6), "LwSciCommon-ADV-MISRAC2012-003")
    int err;

    struct timespec rqtp;
    struct timespec rmtp;
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_6), "LwSciCommon-ADV-MISRAC2012-003")
    long signedTimeNs = 0;

    if ((uint64_t)LONG_MAX < timeNs) {
        LwSciCommonPanic();
    }
    LWCOV_ALLOWLIST_LINE(LWCOV_MISRA(Directive, 4_6), "LwSciCommon-ADV-MISRAC2012-003")
    signedTimeNs = (long)timeNs;

    rqtp.tv_sec = signedTimeNs / 1000000000;
    rqtp.tv_nsec = signedTimeNs % 1000000000;

    do {
        err = nanosleep(&rqtp, &rmtp);
        /* If signals are being used, it is possible for the thread to be woken
         * and return with a non-zero return code due to signal delivery. This
         * is not considered an error.
         *
         * In this case, we should set the requested time to the remaining time
         * and try again. */
        rqtp = rmtp;
        (void)memset(&rmtp, 0x0, sizeof(rmtp));
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(ERR30_C), "LwSciCommon-REQ-CERTC-001")
    } while ((0 != err) && (EINTR == errno));

    if (0 != err) {
        LwSciCommonPanic();
    }
}

