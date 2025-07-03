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
 * \brief <b>LwSciSync timestamps buffer management</b>
 *
 * @b Description: This file contains LwSciSync timestamps
 * structures and interfaces.
 */

#ifndef INCLUDED_LWSCISYNC_TIMESTAMPS_H
#define INCLUDED_LWSCISYNC_TIMESTAMPS_H

#include "lwscisync_internal.h"
#include "lwscisync_primitive_type.h"

/**
 * \brief Defines pointer to core timestamp structure
 *
 *  \implements{18864546}
 */
typedef struct LwSciSyncCoreTimestampsRec* LwSciSyncCoreTimestamps;

/**
 * \brief Defines invalid value for timestamp slot
 *
 *  \implements{18864549}
 */
#define TIMESTAMPS_ILWALID_SLOT UINT32_MAX

/**
 * Core interfaces declaration.
 */

/**
 * \brief Allocate and initialize timestamps unit structure
 *
 * \param[in] reconciledList reconciled LwSciSyncAttrList
 * \param[out] timestamps structure to be allocated and filled
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_InsufficientMemory if memory allocation failed.
 *
 *  \implements{18864564}
 */
LwSciError LwSciSyncCoreTimestampsInit(
    LwSciSyncAttrList reconciledList,
    LwSciSyncCorePrimitive* primitive,
    LwSciSyncCoreTimestamps* timestamps);

/**
 * \brief Destroys the allocated timestamps structure
 *
 * \param[in] timestamps struct to deinit
 *
 *  \implements{18864567}
 *
 */
void LwSciSyncCoreTimestampsDeinit(
    LwSciSyncCoreTimestamps timestamps);

/**
 * \brief Import the timestamps info from blob to sync object
 *
 * \param[in] ipcEndpoint LwSciIpcEndpoint through which import happens
 * \param[in] reconciledList reconciled LwSciSyncAttrList
 * \param[in] data blob containing timestamps information
 * \param[in] len length of the data blob
 * \param[out] timestamps struct to be initialized
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_InsufficientMemory if memory allocation failed.
 *
 *  \implements{18864570}
 */
LwSciError LwSciSyncCoreTimestampsImport(
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAttrList reconciledList,
    const void* data,
    size_t len,
    LwSciSyncCoreTimestamps* timestamps);

/**
 * \brief Export the timestamps info to the blob
 *
 * \param[in] timestamps struct to be exported
 * \param[in] ipcEndpoint LwSciIpcEndpoint through which import happens
 * \param[out] data blob
 * \param[out] len length of the constructed data blob
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_InsufficientMemory if memory allocation failed.
 *
 *  \implements{18864573}
 */
LwSciError LwSciSyncCoreTimestampsExport(
    LwSciSyncCoreTimestamps timestamps,
    LwSciIpcEndpoint ipcEndpoint,
    void** data,
    size_t* length);

/**
 * \brief Returns the next timestamp buffer slot index.
 *
 * LwSciSync maintains the buffer index in round-robin manner.
 *
 * \param[in] timestamps LwSciSyncCoreTimestamps structure
 * \param[out] slotIndex index in timestamp buffer
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_Overflow if there is no slot for the next fence.
 *
 *  \implements{18864576}
 */
LwSciError LwSciSyncCoreTimestampsGetNextSlot(
    LwSciSyncCoreTimestamps timestamps,
    uint32_t* slotIndex);

/**
 * \brief Returns the timestamp buffer info
 *
 * \param[in] timestamps LwSciSyncCoreTimestamps structure
 * \param[out] bufferInfo the buffer information to be filled in
 *
 *  \implements{18864579}
 */
void LwSciSyncCoreTimestampsGetBufferInfo(
    LwSciSyncCoreTimestamps timestamp,
    LwSciSyncTimestampBufferInfo* bufferInfo);

/**
 * \brief Return the decoded value of a timestamp buffer slot
 *
 * \param[in] timestamps LwSciSyncCoreTimestamps structure
 * \param[in] slotIndex slot in the buffer to read from
 * \param[in] primitive the primitive to obtain the buffer to
 * \param[in] fenceId the fence id corresponding to the primitive id
 * \param[out] stamp decoded timestamp value pointed to by slotIndex
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError_BadParameter if slotIndex is not a valid slotIndex
 *
 *  \implements{18864582}
 */
LwSciError LwSciSyncCoreTimestampsGetTimestamp(
    LwSciSyncCoreTimestamps timestamps,
    uint32_t slotIndex,
    LwSciSyncCorePrimitive primitive,
    uint64_t fenceId,
    uint64_t* stamp);

/**
 * \brief Checks if the slot index is a valid index for the timestamp buffer
 *
 * \param[in] timestamps LwSciSyncCoreTimestamps structure
 * \param[in] slotIndex slot in the buffer to read from
 *
 * \return bool
 * - true if slotIndex is valid
 * - false if slotIndex is not valid
 *
 *  \implements{18864585}
 */
bool LwSciSyncCoreTimestampsIsSlotValid(
    LwSciSyncCoreTimestamps timestamps,
    uint64_t slotIndex);

/**
 * \brief Writes current time to the timestamp slot next
 *  to the last one used by this operation
 *
 * \param[in,out] timestamps LwSciSyncCoreTimestamps structure
 * \param[in] primitive LwSciSyncCorePrimitive structure
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - LwSciError errcode if writing failed
 *
 *  \implements{18864588}
 */
LwSciError LwSciSyncCoreTimestampsWriteTime(
    LwSciSyncCoreTimestamps timestamps,
    LwSciSyncCorePrimitive primitive);
#endif
