/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/**
 * \file
 * \brief <b>LwSciSync common utilities</b>
 *
 * @b Description: This file contains small
 * utilities common to multiple LwSciSync units
 */

#ifndef INCLUDED_LWSCISYNC_CORE_H
#define INCLUDED_LWSCISYNC_CORE_H

#include "lwsciipc.h"
#include "lwsciipc_internal.h"
#include "lwscilog.h"
#include "lwscisync.h"
#include "lwscisync_internal.h"
#include "lwscicommon_covanalysis.h"

/**
 * \brief Maximal primitive type
 *
 *  \implements{18845802}
 */
#define MAX_PRIMITIVE_TYPE \
        ((size_t)LwSciSyncInternalAttrValPrimitiveType_UpperBound - \
         (size_t)LwSciSyncInternalAttrValPrimitiveType_LowerBound - 1U)

/**
 * \brief Maximal number of hardware engines per peer in an LwSciSyncAttrList
 *
 * \implements{22831599}
 */
#define MAX_HW_ENGINE_TYPE \
    (128U)

/**
 * @defgroup lwsci_sync Synchronization APIs
 * @{
 */

/**
 * \brief Enum representing LwSciSyncAttrList and LwSciSyncObj keys in export
 * descriptor
 *
 * \implements{18845793}
 */
LWCOV_ALLOWLIST_BEGIN(LWCOV_MISRA(Rule, 2_3), "LwSciSync-ADV-MISRAC2012-002")
typedef enum {
    /** Represents the LwSciSyncAttrList section in the export descriptor */
    LwSciSyncCoreDescKey_AttrList,
    /** Represents the LwSciSyncObj section in the export descriptor */
    LwSciSyncCoreDescKey_SyncObj,
} LwSciSyncCoreDescKey;
LWCOV_ALLOWLIST_END(LWCOV_MISRA(Rule, 2_3))

/**
 * \brief Contains additional information associated with an LwSciIpcEndpoint
 * that LwSciIpc provides.
 *
 * \implements{TBD}
 */
typedef struct {
    LwSciIpcTopoId topoId;
    LwSciIpcEndpointVuid vuId;
} __attribute__((packed)) LwSciSyncIpcTopoId;

/**
 * \brief Returns a 64 bit integer with lower 32 bits containing library minor
 * version and upper 32 bits containing library major version.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - This is a pure function with no side effects
 *
 * \implements{18844413}
*/
static inline uint64_t LwSciSyncCoreGetLibVersion(void)
{
    return ((uint64_t)LwSciSyncMajorVersion << 32U) |
            (uint64_t)LwSciSyncMinorVersion;
}

/**
 * \brief Validates the input LwSciIpcEndpoint
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization:
 *      - Conlwrrent access to the LwSciIpcEndpoint is handled via
 *        LwSciIpcGetEndpointInfo()
 *
 * \param[in] handle LwSciIpcEndpoint to be validated
 *
 * \return LwSciError
 * - LwSciError_Success if handle is a valid LwSciIpcEndpoint
 * - LwSciError_BadParameter if handle not a valid LwSciIpcEndpoint
 *
 * \implements{18844416}
 */

static inline LwSciError LwSciSyncCoreValidateIpcEndpoint(
    const LwSciIpcEndpoint handle)
{
    struct LwSciIpcEndpointInfo info;
    LwSciError err = LwSciIpcGetEndpointInfo(handle, &info);

    if (err != LwSciError_Success) {
        LWSCI_ERR_STR("Failed to validate LwSciIpcEndpoint.\n");
        err = LwSciError_BadParameter;
    }

    return err;
}

/**
 * \brief Checks whether the input LwSciIpcTopoId is from a C2C channel
 *
 * The check is done thanks to the fact that SocId equals LWSCIIPC_SELF_SOCID
 * in non C2C cases.
 *
 * \param[in] topoId topoId of LwSciIpcEndpoint
 *
 * \return bool
 * - true if the topoId is from a C2C channel
 * - false otherwise
 *
 * \implements{TBD}
 */
bool LwSciSyncCoreIsTopoIdC2c(
    LwSciIpcTopoId topoId);

/**
 * \brief Gets topo id information associated with the local input LwSciIpcEndpoint
 */
#if (LW_IS_SAFETY == 0)
/**
 * Retrieves the data with LwSciIpcEndpointGetTopoId()
 */
#if !defined(__x86_64__)
/**
 * and LwSciIpcEndpointGetVuid().
 */
#else
/**
 * .
 */
#endif
#endif
/**
 *
 * \param[in] ipcEndpoint LwSciIpcEndpoint to get topo id information about
 * \param[out] syncTopoId sync topo id information provided by LwSciIpc
 *
 * \return LwSciError
 * - LwSciError_Success if successfully retrieved the topo id information
 * - LwSciError_BadParameter if ipcEndpoint is not a valid LwSciIpcEndpoint
 * - Panics if syncTopoId is NULL
 *
 * \implements{TBD}
 */
LwSciError LwSciSyncCoreGetSyncTopoId(
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncIpcTopoId* syncTopoId);

/**
 * \brief Check if LwSciIpcEndpoint is Inter-Chip
 *
 * \param[in] ipcEndpoint LwSciIpcEndpoint to check for
 * \param[out] hasC2C to store output
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if ipcEndpoint is not a valid LwSciIpcEndpoint
 * - Panics if hasC2C is NULL
 *
 * \implements{TBD}
 */
LwSciError LwSciSyncCoreIsIpcEndpointC2c(
    LwSciIpcEndpoint ipcEndpoint,
    bool* hasC2C);

/**
 * \brief Compares (<) the given LwSciSyncAccessPerm values.
 *
 * Doesn't check if parameters are valid enums from LwSciSyncAccessPerm.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - This is a pure function with no side effects
 *
 * \param[in] permA left side of the comparison.
 * Valid value: any bitmask fitting in LwSciSyncAccessPerm
 * \param[in] permB right side of the comparison.
 * Valid value: any bitmask fitting in LwSciSyncAccessPerm
 *
 * \return bool
 * - true if permA is smaller than permB
 * - false otherwise
 *
 * \implements{18844290}
 */
bool LwSciSyncCorePermLessThan(
    LwSciSyncAccessPerm permA,
    LwSciSyncAccessPerm permB);

/**
 * \brief Compares (<=) the given LwSciSyncAccessPerm values.
 *
 * Doesn't check if parameters are valid enums from LwSciSyncAccessPerm.
 * Can be used to check this validity by comparing against
 * LwSciSyncAccessPerm_WaitSignal.
 *
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - This is a pure function with no side effects
 *
 * \param[in] permA left side of the comparison.
 * Valid value: any bitmask fitting in LwSciSyncAccessPerm
 * \param[in] permB right side of the comparison.
 * Valid value: any bitmask fitting in LwSciSyncAccessPerm
 *
 * \return bool
 * - true if permA is smaller or equal permB
 * - false otherwise
 *
 * \implements{21423605}
 */
bool LwSciSyncCorePermLEq(
    LwSciSyncAccessPerm permA,
    LwSciSyncAccessPerm permB);

/**
 * \brief Check whether the input is a correct LwSciSyncAccessPerm.
 *
 * \param[in] perm input to validate
 *
 * \return bool
 * - true if perm is at most LwSciSyncAccessPerm_WaitSignal bitmaskwise
 *   and not empty
 * - false otherwise
 *
 * \implements{TBD}
 */
bool LwSciSyncCorePermValid(
    LwSciSyncAccessPerm perm);

static LwSciError LwSciSyncCoreValidateHwEngNamespace(
    LwSciSyncHwEngNamespace engNamespace)
{
    LwSciError err = LwSciError_Success;

    if ((LwSciSyncHwEngine_TegraNamespaceId != engNamespace) &&
        (LwSciSyncHwEngine_ResmanNamespaceId != engNamespace)) {
        err = LwSciError_BadParameter;
    }
    return err;
}

/**
 * \brief Sanity check for input LwSciSyncHwEngine values
 *
 * \param[in] engineArray array of LwSciSyncHwEngine
 * \param[in] size size of @a engineArray array
 * Valid value: [0, SIZE_MAX]
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if invalid LwSciSyncHwEngine
 *
 * \implements{22831593}
 */
static inline LwSciError LwSciSyncCoreCheckHwEngineValues(
    const LwSciSyncHwEngine* engineArray,
    size_t size)
{
    LwSciError error = LwSciError_Success;
    size_t i;
    for (i = 0U; i < size; i++) {
        LwSciSyncHwEngName engName = LwSciSyncHwEngName_LowerBound;

        error = LwSciSyncCoreValidateHwEngNamespace(engineArray[i].engNamespace);
        if (LwSciError_Success != error) {
            break;
        }

        /* Call an API to fetch the engine name which will also validate the ID */
        error = LwSciSyncHwEngGetNameFromId(engineArray[i].rmModuleID, &engName);
        if (LwSciError_Success != error) {
            break;
        }
    }
    return error;
}

/**
 * \brief Compares whether two LwSciSyncHwEngines are equal. Equality is
 * determined if each of the members are identical.
 *
 * \param[in] engineA The first LwSciSyncHwEngine to compare
 * \param[in] engineB The second LwSciSyncHwEngine to compare
 *
 * \return bool
 * - true if engineA and engineB are identical
 * - false otherwise
 * - Panics if any of the following oclwrs:
 *      - engineA is NULL
 *      - engineB is NULL
 *      - engineA is invalid
 *      - engineB is invalid
 *
 * \implements{22837636}
 */
bool LwSciSyncHwEngineEqual(
    const LwSciSyncHwEngine* engineA,
    const LwSciSyncHwEngine* engineB);

/**
 * \brief Appends the LwSciSyncHwEngine entries in srcEngineArray to
 * dstEngineArray if they do not already exist.
 *
 * \param[out] dstEngineArray Buffer to write the LwSciSyncHwEngine array into
 * \param[in] dstEngineArrayMaxLen Maximum number of LwSciSyncHwEngine
 * dstEngineArray is capable of holding
 * \param[in] srcEngineArray Source array of LwSciSyncHwEngine to copy from
 * \param[in] srcEngineArrayLen Number of entries in srcEngineArray
 * \param[in,out] dstEngineArrayLen Number of entries dstEngineArray contains
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - dstEngineArray is NULL
 *      - srcEngineArray is NULL
 *      - dstEngineArrayLen is NULL
 *      - dstEngineArray does not have sufficient capacity to contain all the
 *        LwSciSyncHwEngine entries
 *
 * \implements{22837635}
 */
void LwSciSyncAppendHwEngineToArrayUnion(
    LwSciSyncHwEngine* dstEngineArray,
    size_t dstEngineArrayMaxLen,
    const LwSciSyncHwEngine* srcEngineArray,
    size_t srcEngineArrayLen,
    size_t* dstEngineArrayLen);

/**
 * Conlwrrency:
 *  - Thread-safe: Yes
 *  - Synchronization
 *      - The function only operates using immutable data provided via input
 *        parameters (so there is no data-dependency)
 *
 * \implements{18844407}
 *
 * \fn LwSciError LwSciSyncCheckVersionCompatibility(
 *   uint32_t majorVer,
 *   uint32_t minorVer,
 *   bool* isCompatible);
 */

/**
 * The generated hardware engine ID represents the combination of the
 * LwSciSyncHwEngName and instance of LwSciSyncHwEngName. For this interface,
 * assume that the instance is 0.
 *
 * \implements{22823502}
 *
 * \fn LwSciError LwSciSyncHwEngCreateIdWithoutInstance(
 *   LwSciSyncHwEngName engName,
 *   int64_t* engId);
 */

/**
 * \implements{22823503}
 *
 * \fn LwSciError LwSciSyncHwEngGetNameFromId(
 *   int64_t engId,
 *   LwSciSyncHwEngName* engName);
 */

/**
 * \implements{22823504}
 *
 * \fn LwSciError LwSciSyncHwEngGetInstanceFromId(
 *   int64_t engId,
 *   uint32_t* instance);
 */

/** @} */
#endif
