/*
 * Copyright (c) 2019, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <lwscicommon_errcolwersion.h>
//This is to avoid rtti compilation error in x86 build
#define GTEST_HAS_RTTI 0
#include "gtest/gtest.h"

#define MAX_STRING_LENGTH 100

#define TEST_LWSCI_TO_ERRNO(lwSciErr, err)                                     \
({                                                                             \
    EXPECT_EQ(LwSciErrToErrno(lwSciErr), err);                                 \
})

#define TEST_ERRNO_TO_LWSCI(err, lwSciErr)                                     \
({                                                                             \
    EXPECT_EQ(LwSciErrToErrno(lwSciErr), err);                                 \
    if ((err) != -1) {                                                         \
        EXPECT_EQ(ErrnoToLwSciErr((err)), (lwSciErr));                         \
    }                                                                          \
})

#define TEST_ERR_LWSCI_TO_ERRNO_STR(lwSciErr, str)                             \
({                                                                             \
    EXPECT_TRUE(strncmp(LwSciErrToErrnoStr((lwSciErr)), (str), MAX_STRING_LENGTH) == 0); \
})

#define TEST_ERR(lwSciErr, lwErr, err, str)                                    \
({                                                                             \
    TEST_LWSCI_TO_ERRNO(lwSciErr, err);                                        \
    TEST_ERRNO_TO_LWSCI(err, lwSciErr);                                        \
    TEST_ERR_LWSCI_TO_ERRNO_STR(lwSciErr, str);                                \
})

TEST(LwSciCommon, ErrorColwersion) {
    TEST_ERR(LwSciError_Success, LwError_Success, 0, "EOK");
    TEST_ERR(LwSciError_NotImplemented, LwError_NotImplemented, ENOSYS, "ENOSYS");
    TEST_ERR(LwSciError_NotSupported, LwError_NotSupported, ENOTSUP, "ENOTSUP");
    TEST_ERR(LwSciError_BadParameter, LwError_BadParameter, EILWAL, "EILWAL");
    TEST_ERR(LwSciError_Timeout, LwError_Timeout, ETIMEDOUT, "ETIMEDOUT");
    TEST_ERR(LwSciError_InsufficientMemory, LwError_InsufficientMemory, ENOMEM, "ENOMEM");
    TEST_ERR(LwSciError_AccessDenied, LwError_AccessDenied, EACCES, "EACCES");
    TEST_ERR(LwSciError_TooBig, LwError_Force32, E2BIG, "E2BIG");
    TEST_ERR(LwSciError_TryItAgain, LwError_Force32, EAGAIN, "EAGAIN");
    TEST_ERR(LwSciError_BadFileDesc, LwError_Force32, EBADF, "EBADF");
    TEST_ERR(LwSciError_Busy, LwError_Force32, EBUSY, "EBUSY");
    TEST_ERR(LwSciError_ConnectionReset, LwError_Force32, ECONNRESET, "ECONNRESET");
    TEST_ERR(LwSciError_ResourceDeadlock, LwError_Force32, EDEADLK, "EDEADLK");
    TEST_ERR(LwSciError_FileExists, LwError_Force32, EEXIST, "EEXIST");
    TEST_ERR(LwSciError_BadAddress, LwError_Force32, EFAULT, "EFAULT");
    TEST_ERR(LwSciError_FileTooBig, LwError_Force32, EFBIG, "EFBIG");
    TEST_ERR(LwSciError_InterruptedCall, LwError_Force32, EINTR, "EINTR");
    TEST_ERR(LwSciError_IO, LwError_Force32, EIO, "EIO");
    TEST_ERR(LwSciError_IsDirectory, LwError_Force32, EISDIR, "EISDIR");
    TEST_ERR(LwSciError_TooManySymbolLinks, LwError_Force32, ELOOP, "ELOOP");
    TEST_ERR(LwSciError_TooManyOpenFiles, LwError_Force32, EMFILE, "EMFILE");
    TEST_ERR(LwSciError_FileNameTooLong, LwError_Force32, ENAMETOOLONG, "ENAMETOOLONG");
    TEST_ERR(LwSciError_FileTableOverflow, LwError_Force32, ENFILE, "ENFILE");
    TEST_ERR(LwSciError_NoSuchDevice, LwError_Force32, ENODEV, "ENODEV");
    TEST_ERR(LwSciError_NoSuchEntry, LwError_Force32, ENOENT, "ENOENT");
    TEST_ERR(LwSciError_NoSpace, LwError_Force32, ENOSPC, "ENOSPC");
#ifdef ENOTRECOVERABLE
    TEST_ERR(LwSciError_MutexNotRecoverable, LwError_Force32, ENOTRECOVERABLE, "ENOTRECOVERABLE");
#endif
    TEST_ERR(LwSciError_NoSuchDevAddr, LwError_Force32, ENXIO, "ENXIO");
    TEST_ERR(LwSciError_Overflow, LwError_Force32, EOVERFLOW, "EOVERFLOW");
#ifdef EOWNERDEAD
    TEST_ERR(LwSciError_LockOwnerDead, LwError_Force32, EOWNERDEAD, "EOWNERDEAD");
#endif
    TEST_ERR(LwSciError_NotPermitted, LwError_Force32, EPERM, "EPERM");
    TEST_ERR(LwSciError_ReadOnlyFileSys, LwError_Force32, EROFS, "EROFS");
    TEST_ERR(LwSciError_NoSuchProcess, LwError_Force32, ESRCH, "ESRCH");
    TEST_ERR(LwSciError_TextFileBusy, LwError_Force32, ETXTBSY, "ETXTBSY");
    TEST_ERR(LwSciError_IlwalidIoctlNum, LwError_Force32, ENOTTY, "ENOTTY");
    TEST_ERR(LwSciError_NoData, LwError_Force32, ENODATA, "ENODATA");
    TEST_ERR(LwSciError_AlreadyInProgress, LwError_Force32, EALREADY, "EALREADY");
    TEST_ERR(LwSciError_NoDesiredMessage, LwError_Force32, ENOMSG, "ENOMSG");
    TEST_ERR(LwSciError_MessageSize, LwError_Force32, EMSGSIZE, "EMSGSIZE");

    // This error codes do not map back from errno:
    TEST_ERR(LwSciError_NotInitialized, LwError_NotInitialized, -1, "UNKNOWN");
    TEST_ERR(LwSciError_IlwalidState, LwError_IlwalidState, -1, "UNKNOWN");
    TEST_ERR(LwSciError_ResourceError, LwError_ResourceError, -1, "UNKNOWN");
    TEST_ERR(LwSciError_EndOfFile, LwError_EndOfFile, -1, "UNKNOWN");
    TEST_ERR(LwSciError_IlwalidOperation, LwError_IlwalidOperation, -1, "UNKNOWN");

    // This error codes do not map back from LwSciError to LwError and errno:
    // LwSciError_ReconciliationFailed
    TEST_LWSCI_TO_ERRNO(LwSciError_ReconciliationFailed, -1);
    TEST_ERR_LWSCI_TO_ERRNO_STR(LwSciError_ReconciliationFailed, "UNKNOWN");
    // LwSciError_UnsupportedConfig
    TEST_LWSCI_TO_ERRNO(LwSciError_UnsupportedConfig, -1);
    TEST_ERR_LWSCI_TO_ERRNO_STR(LwSciError_UnsupportedConfig, "UNKNOWN");
    // LwSciError_AttrListValidationFailed
    TEST_LWSCI_TO_ERRNO(LwSciError_AttrListValidationFailed, -1);
    TEST_ERR_LWSCI_TO_ERRNO_STR(LwSciError_AttrListValidationFailed, "UNKNOWN");
}
