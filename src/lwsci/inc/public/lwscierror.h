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
  * @file
  *
  * <b> LWPU Software Communications Interface (SCI): Error Handling </b>
  *
  * @b Description: This file declares error codes for LwSci APIs.
  */

#ifndef INCLUDED_LWSCI_ERROR_H
#define INCLUDED_LWSCI_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup LwSciError SCI Error Handling
 *
 * Contains error code enumeration and helper macros.
 *
 * @ingroup lwsci_top
 * @{
 */

/**
 * @brief Return/error codes for all LwSci functions.
 *
 * This enumeration contains unique return/error codes to identify the
 * source of a failure. Some errors have direct correspondence to standard
 * errno.h codes, indicated [IN BRACKETS], and may result from failures in
 * lower level system calls. Others indicate failures specific to misuse
 * of LwSci library function.
 *
 */
typedef enum {
    /* Range 0x00000000 - 0x00FFFFFF : Common errors
     * This range is used for errors common to all LwSci libraries. */

    /** [EOK] No error */
    LwSciError_Success                  = 0x00000000,

    /** Unidentified error with no additional info */
    LwSciError_Unknown                  = 0x00000001,

    /* Generic errors */
    /** [ENOSYS] Feature is not implemented */
    LwSciError_NotImplemented           = 0x00000010,
    /** [ENOTSUP] Feature is not supported */
    LwSciError_NotSupported             = 0x00000011,
    /** [EACCES] Access to resource denied */
    LwSciError_AccessDenied             = 0x00000020,
    /** [EPERM] No permission to perform operation */
    LwSciError_NotPermitted             = 0x00000021,
    /** Resource is in wrong state to perform operation */
    LwSciError_IlwalidState             = 0x00000022,
    /** Requested operation is not legal */
    LwSciError_IlwalidOperation         = 0x00000023,
    /** Required resource is not initialized */
    LwSciError_NotInitialized           = 0x00000024,
    /** Requested resource is already in use */
    LwSciError_AlreadyInUse             = 0x00000025,
    /** Operation has already been performed */
    LwSciError_AlreadyDone              = 0x00000026,
    /** Resource/information not yet available */
    LwSciError_NotYetAvailable          = 0x00000027,
    /** Resource/information no longer available */
    LwSciError_NoLongerAvailable        = 0x00000028,
    /** [ENOMEM] Not enough memory */
    LwSciError_InsufficientMemory       = 0x00000030,
    /** Not enough (non-memory) resources */
    LwSciError_InsufficientResource     = 0x00000031,
    /** Resource failed */
    LwSciError_ResourceError            = 0x00000032,

    /* Function parameter errors */
    /** [EILWAL] Invalid parameter value */
    LwSciError_BadParameter             = 0x00000100,
    /** [EFAULT] Invalid address */
    LwSciError_BadAddress               = 0x00000101,
    /** [E2BIG] Parameter list too long */
    LwSciError_TooBig                   = 0x00000102,
    /** [EOVERFLOW] Value too large for data type */
    LwSciError_Overflow                 = 0x00000103,
    /** Parameters are inconsistent with each other or prior settings  */
    LwSciError_InconsistentData         = 0x00000104,
    /** Parameters or prior settings are insufficient */
    LwSciError_InsufficientData         = 0x00000105,
    /** An index is not in the allowed range */
    LwSciError_IndexOutOfRange          = 0x00000106,
    /** A value is not in the allowed range */
    LwSciError_ValueOutOfRange          = 0x00000107,

    /* Timing/temporary errors */
    /** [ETIMEDOUT] Operation timed out*/
    LwSciError_Timeout                  = 0x00000200,
    /** [EAGAIN] Resource unavailable. Try again. */
    LwSciError_TryItAgain               = 0x00000201,
    /** [EBUSY] Resource is busy */
    LwSciError_Busy                     = 0x00000202,
    /** [EINTR] An interrupt olwrred */
    LwSciError_InterruptedCall          = 0x00000203,

    /* Device errors */
    /** [ENODEV] No such device */
    LwSciError_NoSuchDevice             = 0x00001000,
    /** [ENOSPC] No space left on device */
    LwSciError_NoSpace                  = 0x00001001,
    /** [ENXIO] No such device or address */
    LwSciError_NoSuchDevAddr            = 0x00001002,
    /** [EIO] Input/output error */
    LwSciError_IO                       = 0x00001003,
    /** [ENOTTY] Inappropriate I/O control operation */
    LwSciError_IlwalidIoctlNum          = 0x00001004,

    /* File system errors */
    /** [ENOENT] No such file or directory*/
    LwSciError_NoSuchEntry              = 0x00001100,
    /** [EBADF] Bad file descriptor */
    LwSciError_BadFileDesc              = 0x00001101,
    /** [EBADFSYS] Corrupted file system detected */
    LwSciError_CorruptedFileSys         = 0x00001102,
    /** [EEXIST] File already exists */
    LwSciError_FileExists               = 0x00001103,
    /** [EISDIR] File is a directory */
    LwSciError_IsDirectory              = 0x00001104,
    /** [EROFS] Read-only file system */
    LwSciError_ReadOnlyFileSys          = 0x00001105,
    /** [ETXTBSY] Text file is busy */
    LwSciError_TextFileBusy             = 0x00001106,
    /** [ENAMETOOLONG] File name is too long */
    LwSciError_FileNameTooLong          = 0x00001107,
    /** [EFBIG] File is too large */
    LwSciError_FileTooBig               = 0x00001108,
    /** [ELOOP] Too many levels of symbolic links */
    LwSciError_TooManySymbolLinks       = 0x00001109,
    /** [EMFILE] Too many open files in process*/
    LwSciError_TooManyOpenFiles         = 0x0000110A,
    /** [ENFILE] Too many open files in system */
    LwSciError_FileTableOverflow        = 0x0000110B,
    /** End of file reached */
    LwSciError_EndOfFile                = 0x0000110C,


    /* Communication errors */
    /** [ECONNRESET] Connection was closed or lost */
    LwSciError_ConnectionReset          = 0x00001200,
    /** [EALREADY] Pending connection is already in progress */
    LwSciError_AlreadyInProgress        = 0x00001201,
    /** [ENODATA] No message data available */
    LwSciError_NoData                   = 0x00001202,
    /** [ENOMSG] No message of the desired type available*/
    LwSciError_NoDesiredMessage         = 0x00001203,
    /** [EMSGSIZE] Message is too large */
    LwSciError_MessageSize              = 0x00001204,
    /** [ENOREMOTE] Remote node doesn't exist */
    LwSciError_NoRemote                 = 0x00001205,

    /* Process/thread errors */
    /** [ESRCH] No such process */
    LwSciError_NoSuchProcess            = 0x00002000,

    /* Mutex errors */
    /** [ENOTRECOVERABLE] Mutex damaged by previous owner's death */
    LwSciError_MutexNotRecoverable      = 0x00002100,
    /** [EOWNERDEAD] Previous owner died while holding mutex */
    LwSciError_LockOwnerDead            = 0x00002101,
    /** [EDEADLK] Taking ownership would cause deadlock */
    LwSciError_ResourceDeadlock         = 0x00002102,

    /* LwSci attribute list errors */
    /** Could not reconcile attributes */
    LwSciError_ReconciliationFailed     = 0x00010100,
    /** Could not validate attributes */
    LwSciError_AttrListValidationFailed = 0x00010101,

    /** End of range for common error codes */
    LwSciError_CommonEnd                = 0x00FFFFFF,


    /* Range 0x01000000 - 0x01FFFFFF : LwSciBuf errors */
    /** Unidentified LwSciBuf error with no additional info */
    LwSciError_LwSciBufUnknown          = 0x01000000,
    /** End of range for LwSciBuf errors */
    LwSciError_LwSciBufEnd              = 0x01FFFFFF,


    /* Range 0x02000000 - 0x02FFFFFF : LwSciSync errors */
    /** Unidentified LwSciSync error with no additional info */
    LwSciError_LwSciSynlwnknown         = 0x02000000,
    /** Unsupported configuration */
    LwSciError_UnsupportedConfig        = 0x02000001,
    /** Provided fence is cleared */
    LwSciError_ClearedFence             = 0x02000002,
    /* End of range for LwScSync errors */
    LwSciError_LwSciSyncEnd             = 0x02FFFFFF,


    /* Range 0x03000000 - 0x03FFFFFF : LwSciStream errors */

    /** Unidentified LwSciStream error with no additional info */
    LwSciError_LwSciStreamUnknown       = 0x03000000,
    /** Internal stream resource failure oclwrred */
    LwSciError_StreamInternalError      = 0x03000001,
    /** Unrecognized block handle */
    LwSciError_StreamBadBlock           = 0x03000100,
    /** Unrecognized packet handle */
    LwSciError_StreamBadPacket          = 0x03000101,
    /** Invalid packet cookie value */
    LwSciError_StreamBadCookie          = 0x03000102,
    /** Operation requires stream be fully connected */
    LwSciError_StreamNotConnected       = 0x03000200,
    /** Operation can only be performed in setup phase */
    LwSciError_StreamNotSetupPhase      = 0x03000201,
    /** Operation can only be performed in safety phase */
    LwSciError_StreamNotSafetyPhase     = 0x03000202,
    /** No stream packet available */
    LwSciError_NoStreamPacket           = 0x03001000,
    /** Referenced packet's current location does not allow this operation */
    LwSciError_StreamPacketInaccessible = 0x03001001,
    /** Internal error due to operation on deleted packet */
    LwSciError_StreamPacketDeleted      = 0x03001002,
    /** Queried info not exist */
    LwSciError_StreamInfoNotProvided    = 0x03003000,

    /**
     * These stream errors represent failures detected from lower level
     *    system components. They generally are not due to any user error,
     *    but might be caused by the system running out of resources.
     */
    /** Failed to acquire lock on mutex used to ensure thread safety */
    LwSciError_StreamLockFailed         = 0x03400000,

    /**
     * These stream errors represent internal failures which should never
     *    be possible in a production system. They exist only for internal
     *    unit testing.
     */
    /** Invalid input index was passed to a block. */
    LwSciError_StreamBadSrcIndex        = 0x03800000,
    /** Invalid output index was passed to a block. */
    LwSciError_StreamBadDstIndex        = 0x03800001,

    /** End of range for LwSciStream errors */
    LwSciError_LwSciStreamEnd           = 0x03FFFFFF,


    /* Range 0x04000000 - 0x04FFFFFF : LwSciIpc errors */
    /** Unidentified LwSciIpc error with no additional info */
    LwSciError_LwSciIplwnknown          = 0x04000000,
    /** End of range for LwSciIpc errors */
    LwSciError_LwSciIpcEnd              = 0x04FFFFFF,


    /* Range 0x05000000 - 0x05FFFFFF : LwSciEvent errors */
    /** Unidentified LwSciEvent error with no additional info */
    LwSciError_LwSciEventUnknown        = 0x05000000,
    /** End of range for LwSciEvent errors */
    LwSciError_LwSciEventEnd            = 0x05FFFFFF,

} LwSciError;

/**
 * @}
*/

#ifdef __cplusplus
}
#endif

#endif /* INCLUDED_LWSCI_ERROR_H */
