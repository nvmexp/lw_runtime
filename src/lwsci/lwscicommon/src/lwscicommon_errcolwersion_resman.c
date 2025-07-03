/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#include "lwscicommon_errcolwersion.h"

#include <sys/types.h>
#include <stdlib.h>
#include <stdbool.h>
#include <errno.h>
#include "lwscierror.h"

static const struct LwStatusCodeString
{
    LW_STATUS statusCode;
    const char *statusString;
} statusCodeList[] = {
   #include "lwstatuscodes.h"
   { 0xffffffff, "Unknown error code!" } // Some compilers don't like the trailing ','
};

typedef struct LwSciErrMap {
    LwSciError scierr;
    LW_STATUS status;
    int32_t oserr;
    const char *oserrstr;
} LwSciErrMap;

static const LwSciErrMap ZeroMap = {
    LwSciError_Success,
    LW_OK,
    0,
    ""
};

static const LwSciErrMap LwSciCoreErrs[] =
{
    {LwSciError_Success,
     LW_OK,
     0,               "EOK"},
    {LwSciError_NotImplemented,
     LW_ERR_GENERIC,
     ENOSYS,          "ENOSYS"},
    {LwSciError_NotSupported,
     LW_ERR_NOT_SUPPORTED,
     ENOTSUP,         "ENOTSUP"},
    {LwSciError_BadParameter,
     LW_ERR_ILWALID_PARAMETER,
     EILWAL,          "EILWAL"},
    {LwSciError_Timeout,
     LW_ERR_TIMEOUT,
     ETIMEDOUT,       "ETIMEDOUT"},
    {LwSciError_InsufficientMemory,
     LW_ERR_NO_MEMORY,
     ENOMEM,          "ENOMEM"},
    {LwSciError_AccessDenied,
     LW_ERR_INSUFFICIENT_PERMISSIONS,
     EACCES,          "EACCES"},
    {LwSciError_TooBig,
     LW_ERR_GENERIC,
     E2BIG,           "E2BIG"},
    {LwSciError_TryItAgain,
     LW_ERR_TIMEOUT_RETRY,
     EAGAIN,          "EAGAIN"},
    {LwSciError_TryItAgain,
     LW_ERR_BUSY_RETRY,
     EAGAIN,          "EAGAIN"},
    {LwSciError_BadFileDesc,
     LW_ERR_GENERIC,
     EBADF,           "EBADF"},
    {LwSciError_Busy,
     LW_ERR_IN_USE,
     EBUSY,           "EBUSY"},
    {LwSciError_ConnectionReset,
     LW_ERR_RESET_REQUIRED,
     ECONNRESET,      "ECONNRESET"},
    {LwSciError_ResourceDeadlock,
     LW_ERR_GENERIC,
     EDEADLK,         "EDEADLK"},
    {LwSciError_FileExists,
     LW_ERR_GENERIC,
     EEXIST,          "EEXIST"},
    {LwSciError_BadAddress,
     LW_ERR_GENERIC,
     EFAULT,          "EFAULT"},
    {LwSciError_FileTooBig,
     LW_ERR_GENERIC,
     EFBIG,           "EFBIG"},
    {LwSciError_InterruptedCall,
     LW_ERR_GENERIC,
     EINTR,           "EINTR"},
    {LwSciError_IO,
     LW_ERR_GENERIC,
     EIO,             "EIO"},
    {LwSciError_IsDirectory,
     LW_ERR_GENERIC,
     EISDIR,          "EISDIR"},
    {LwSciError_TooManySymbolLinks,
     LW_ERR_GENERIC,
     ELOOP,           "ELOOP"},
    {LwSciError_TooManyOpenFiles,
     LW_ERR_GENERIC,
     EMFILE,          "EMFILE"},
    {LwSciError_FileNameTooLong,
     LW_ERR_GENERIC,
     ENAMETOOLONG,    "ENAMETOOLONG"},
    {LwSciError_FileTableOverflow,
     LW_ERR_GENERIC,
     ENFILE,          "ENFILE"},
    {LwSciError_NoSuchDevice,
     LW_ERR_ILWALID_DEVICE,
     ENODEV,          "ENODEV"},
    {LwSciError_NoSuchEntry,
     LW_ERR_MISSING_TABLE_ENTRY,
     ENOENT,          "ENOENT"},
    {LwSciError_NoSpace,
     LW_ERR_NO_MEMORY,
     ENOSPC,          "ENOSPC"},
#ifdef ENOTRECOVERABLE
    {LwSciError_MutexNotRecoverable,
     LW_ERR_GENERIC,
     ENOTRECOVERABLE, "ENOTRECOVERABLE"},
#endif
    {LwSciError_NoSuchDevAddr,
     LW_ERR_GENERIC,
     ENXIO,           "ENXIO"},
    {LwSciError_Overflow,
     LW_ERR_GENERIC,
     EOVERFLOW,       "EOVERFLOW"},
#ifdef EOWNERDEAD
    {LwSciError_LockOwnerDead,
     LW_ERR_GENERIC,
     EOWNERDEAD,      "EOWNERDEAD"},
#endif
    {LwSciError_NotPermitted,
     LW_ERR_GENERIC,
     EPERM,           "EPERM"},
    {LwSciError_ReadOnlyFileSys,
     LW_ERR_GENERIC,
     EROFS,           "EROFS"},
    {LwSciError_NoSuchProcess,
     LW_ERR_GENERIC,
     ESRCH,           "ESRCH"},
    {LwSciError_TextFileBusy,
     LW_ERR_GENERIC,
     ETXTBSY,         "ETXTBSY"},
    {LwSciError_IlwalidIoctlNum,
     LW_ERR_GENERIC,
     ENOTTY,          "ENOTTY"},
    {LwSciError_NoData,
     LW_ERR_GENERIC,
     ENODATA,         "ENODATA"},
    {LwSciError_AlreadyInProgress,
     LW_ERR_GENERIC,
     EALREADY,        "EALREADY"},
    {LwSciError_NoDesiredMessage,
     LW_ERR_GENERIC,
     ENOMSG,          "ENOMSG"},
    {LwSciError_MessageSize,
     LW_ERR_GENERIC,
     EMSGSIZE,        "EMSGSIZE"},
    {LwSciError_NotInitialized,
     LW_ERR_GPU_DMA_NOT_INITIALIZED,
     -1,              "UNKNOWN"},
    {LwSciError_IlwalidState,
     LW_ERR_ILWALID_STATE,
     -1,              "UNKNOWN"},
    {LwSciError_ResourceError,
     LW_ERR_GENERIC,
     -1,              "UNKNOWN"},
    {LwSciError_EndOfFile,
     LW_ERR_GENERIC,
     -1,              "UNKNOWN"},
    {LwSciError_IlwalidOperation,
     LW_ERR_ILWALID_OPERATION,
     -1,              "UNKNOWN"},
    {LwSciError_ReconciliationFailed,
     LW_ERR_GENERIC,
     -1,              "UNKNOWN"},
    {LwSciError_UnsupportedConfig,
     LW_ERR_GENERIC,
     -1,              "UNKNOWN"},
    {LwSciError_AttrListValidationFailed,
     LW_ERR_GENERIC,
     -1,              "UNKNOWN"},
};
static const ssize_t LwSciCoreErrCount = sizeof(LwSciCoreErrs)
                                       / sizeof(LwSciCoreErrs[0]);

static const LwSciErrMap LwSciOsErrs[] = {
    // Dummy entry to avoid compilation issues
    {LwSciError_Unknown,
     LW_ERR_GENERIC,
     -1, "UNKNOWN"},
};
static const ssize_t LwSciOsErrCount = 0;

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
MatchLWStatus(
    const LwSciErrMap* mapA,
    const LwSciErrMap* mapB)
{
    return (mapA->status == mapB->status);
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
    MatchFunc matchfunc,
    const LwSciErrMap* matchval)
{
    /*
     * The loops in this functions could be replaced with searching
     * a pre-built hash table if efficiency is a concern. But since this is
     * only ilwoked when a failure oclwrs, which shouldn't happen at all on
     * a safety certified system, optimization probably isn't important.
     */

    const LwSciErrMap* found = NULL;
    int32_t i;

    /* search common map list */
    for (i=0; (i<LwSciCoreErrCount) && (found==NULL); i++) {
        if (matchfunc(matchval, &LwSciCoreErrs[i])) {
            found = &LwSciCoreErrs[i];
        }
    }

    /* search os-specific map list */
    for (i=0; (i<LwSciOsErrCount) && (found==NULL); i++) {
        if (matchfunc(matchval, &LwSciOsErrs[i])) {
            found = &LwSciOsErrs[i];
        }
    }

    return found;
}

LwSciError
LwStatusToLwSciErr(
    LW_STATUS status)
{
    LwSciError lwSciErr = LwSciError_Unknown;
    LwSciErrMap matchval = ZeroMap;
    matchval.status = status;
    const LwSciErrMap* errMap = LookupErrMap(MatchLWStatus, &matchval);
    if (NULL != errMap) {
        lwSciErr = errMap->scierr;
    }
    return lwSciErr;
}

LW_STATUS
LwSciErrToLwStatus(
    LwSciError lwSciErr)
{
    LW_STATUS status = LW_ERR_GENERIC;
    LwSciErrMap matchval = ZeroMap;
    matchval.scierr = lwSciErr;
    const LwSciErrMap* errMap = LookupErrMap(MatchLwSciError, &matchval);
    if (NULL != errMap) {
        status = errMap->status;
    }
    return status;
}

/*
 * colwert errno to LwSciError enum value
 */
LwSciError
ErrnoToLwSciErr(
    int32_t err)
{
    LwSciError lwSciErr = LwSciError_Unknown;
    LwSciErrMap matchval = ZeroMap;
    matchval.oserr = abs(err);
    const LwSciErrMap* errMap = LookupErrMap(MatchErrno, &matchval);
    if (NULL != errMap) {
        lwSciErr = errMap->scierr;
    }
    return lwSciErr;
}

/*
 * colwert LwSciError to OS errno
 */
int32_t
LwSciErrToErrno(
    LwSciError lwSciErr)
{
    int32_t err = -1;
    LwSciErrMap matchval = ZeroMap;
    matchval.scierr = lwSciErr;
    const LwSciErrMap* errMap = LookupErrMap(MatchLwSciError, &matchval);
    if (NULL != errMap) {
        err = errMap->oserr;
    }
    return err;
}

/*
 * colwert LwSciError to OS errno string
 */
const char*
LwSciErrToErrnoStr(
    LwSciError lwSciErr)
{
    const char* str = "UNKNOWN";
    LwSciErrMap matchval = ZeroMap;
    matchval.scierr = lwSciErr;
    const LwSciErrMap* errMap = LookupErrMap(MatchLwSciError, &matchval);
    if (NULL != errMap) {
        str = errMap->oserrstr;
    }
    return str;
}

/*
 * resman error code to string
 */
const char* LwStatusToString(LW_STATUS lwStatusIn)
{
    LwU32 i;
    LwU32 n = ((LwU32)(sizeof(statusCodeList))/(LwU32)(sizeof(statusCodeList[0])));
    const char* statusString = "Unknown error code!";
    for (i = 0U; i < n; i++)
    {
        if (lwStatusIn == statusCodeList[i].statusCode)
        {
            statusString = statusCodeList[i].statusString;
            break;
        }
    }

    return statusString;
}
