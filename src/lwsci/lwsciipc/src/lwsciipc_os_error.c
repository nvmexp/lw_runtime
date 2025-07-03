/*
 * Copyright (c) 2020, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <sys/types.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <errno.h>
#include <lwscierror.h>

#include <lwos_static_analysis.h>
#include "lwsciipc_os_error.h"


typedef struct LwSciErrMap {
    LwSciError scierr;
    int32_t oserr;
    const char *oserrstr;
} LwSciErrMap;

static const LwSciErrMap ZeroMap = {
    LwSciError_Success,
    0,
    ""
};

static const LwSciErrMap LwSciCoreErrs[] =
{
    {LwSciError_Success,
     0,               "EOK"},
    {LwSciError_NotImplemented,
     ENOSYS,          "ENOSYS"},
    {LwSciError_NotSupported,
     ENOTSUP,         "ENOTSUP"},
    {LwSciError_BadParameter,
     EILWAL,          "EILWAL"},
    {LwSciError_Timeout,
     ETIMEDOUT,       "ETIMEDOUT"},
    {LwSciError_InsufficientMemory,
     ENOMEM,          "ENOMEM"},
    {LwSciError_AccessDenied,
     EACCES,          "EACCES"},
    {LwSciError_TooBig,
     E2BIG,           "E2BIG"},
    {LwSciError_TryItAgain,
     EAGAIN,          "EAGAIN"},
    {LwSciError_BadFileDesc,
     EBADF,           "EBADF"},
    {LwSciError_Busy,
     EBUSY,           "EBUSY"},
    {LwSciError_ConnectionReset,
     ECONNRESET,      "ECONNRESET"},
    {LwSciError_ResourceDeadlock,
     EDEADLK,         "EDEADLK"},
    {LwSciError_FileExists,
     EEXIST,          "EEXIST"},
    {LwSciError_BadAddress,
     EFAULT,          "EFAULT"},
    {LwSciError_FileTooBig,
     EFBIG,           "EFBIG"},
    {LwSciError_InterruptedCall,
     EINTR,           "EINTR"},
    {LwSciError_IO,
     EIO,             "EIO"},
    {LwSciError_IsDirectory,
     EISDIR,          "EISDIR"},
    {LwSciError_TooManySymbolLinks,
     ELOOP,           "ELOOP"},
    {LwSciError_TooManyOpenFiles,
     EMFILE,          "EMFILE"},
    {LwSciError_FileNameTooLong,
     ENAMETOOLONG,    "ENAMETOOLONG"},
    {LwSciError_FileTableOverflow,
     ENFILE,          "ENFILE"},
    {LwSciError_NoSuchDevice,
     ENODEV,          "ENODEV"},
    {LwSciError_NoSuchEntry,
     ENOENT,          "ENOENT"},
    {LwSciError_NoSpace,
     ENOSPC,          "ENOSPC"},
#ifdef ENOTRECOVERABLE
    {LwSciError_MutexNotRecoverable,
     ENOTRECOVERABLE, "ENOTRECOVERABLE"},
#endif
    {LwSciError_NoSuchDevAddr,
     ENXIO,           "ENXIO"},
    {LwSciError_Overflow,
     EOVERFLOW,       "EOVERFLOW"},
#ifdef EOWNERDEAD
    {LwSciError_LockOwnerDead,
     EOWNERDEAD,      "EOWNERDEAD"},
#endif
    {LwSciError_NotPermitted,
     EPERM,           "EPERM"},
    {LwSciError_ReadOnlyFileSys,
     EROFS,           "EROFS"},
    {LwSciError_NoSuchProcess,
     ESRCH,           "ESRCH"},
    {LwSciError_TextFileBusy,
     ETXTBSY,         "ETXTBSY"},
    {LwSciError_IlwalidIoctlNum,
     ENOTTY,          "ENOTTY"},
    {LwSciError_NoData,
     ENODATA,         "ENODATA"},
    {LwSciError_AlreadyInProgress,
     EALREADY,        "EALREADY"},
    {LwSciError_NoDesiredMessage,
     ENOMSG,          "ENOMSG"},
    {LwSciError_MessageSize,
     EMSGSIZE,        "EMSGSIZE"},
};

LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 8_9), "<QNXBSP>:<lwpu>:<1>:<Bug 200736827>")
static const size_t LwSciCoreErrCount = sizeof(LwSciCoreErrs)
                                       / sizeof(LwSciCoreErrs[0]);

/* TODO: This ifdef probably not allowed by MISRA. Should move os-specific
 *       table definitions into their own files and just access them here.
 */
#ifdef __QNX__
static const LwSciErrMap LwSciOsErrs[] = {
    {LwSciError_NoRemote,
     ENOREMOTE, "ENOREMOTE"},
    {LwSciError_CorruptedFileSys,
     EBADFSYS,  "EBADFSYS"},
};

LWOS_COV_WHITELIST(deviate, LWOS_MISRA(Rule, 8_9), "<QNXBSP>:<lwpu>:<1>:<Bug 200736827>")
static const size_t LwSciOsErrCount = sizeof(LwSciOsErrs)
                                     / sizeof(LwSciOsErrs[0]);
#endif /* __QNX__ */

typedef bool (*MatchFunc)(const LwSciErrMap* mapA,
                         const LwSciErrMap* mapB);

static bool
MatchLwSciError(
    const LwSciErrMap* mapA,
    const LwSciErrMap* mapB)
{
    return (mapA->scierr == mapB->scierr);
}

static bool
MatchErrno(
    const LwSciErrMap* mapA,
    const LwSciErrMap* mapB)
{
    return (mapA->oserr == mapB->oserr);
}


static const LwSciErrMap*
LookupErrMap(
    MatchFunc matchfn,
    const LwSciErrMap* matchval)
{
    /*
     * The loops in this functions could be replaced with searching
     * a pre-built hash table if efficiency is a concern. But since this is
     * only ilwoked when a failure oclwrs, which shouldn't happen at all on
     * a safety certified system, optimization probably isn't important.
     */

    const LwSciErrMap* found = NULL;
    uint64_t i = 0U;

    /* search common map list */
    for (i = 0U; (i < LwSciCoreErrCount); i++) {
        if (matchfn(matchval, &LwSciCoreErrs[i])) {
            found = &LwSciCoreErrs[i];
            break;
        }
    }

#ifdef __QNX__
    /* search os-specific map list */
    for (i = 0U; (i < LwSciOsErrCount); i++) {
        if (matchfn(matchval, &LwSciOsErrs[i])) {
            found = &LwSciOsErrs[i];
            break;
        }
    }
#endif /* __QNX__ */

    return found;
}

/*
 * colwert errno to LwSciError enum value
 */
LwSciError ErrnoToLwSciErr(int32_t err)
{
    LwSciErrMap matchval = ZeroMap;
    const LwSciErrMap* errMap;
    LwSciError lwSciErr = LwSciError_Unknown;

    if (err == INT32_MIN) {
        lwSciErr = LwSciError_Unknown;
        goto fail;
    }

    matchval.oserr = abs(err);
    errMap = LookupErrMap(MatchErrno, &matchval);
    if (errMap != NULL) {
        lwSciErr = errMap->scierr;
    }

fail:
    return lwSciErr;
}

/*
 * colwert LwSciError to OS errno
 */
int32_t LwSciErrToErrno(LwSciError lwSciErr)
{
    LwSciErrMap matchval = ZeroMap;
    const LwSciErrMap* errMap;
    int32_t err = -1;

    matchval.scierr = lwSciErr;
    errMap = LookupErrMap(MatchLwSciError, &matchval);
    if (errMap != NULL) {
        err = errMap->oserr;
    }

    return err;
}

