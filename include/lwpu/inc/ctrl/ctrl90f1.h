/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl90f1.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#include "mmu_fmt_types.h"

#define GMMU_FMT_MAX_LEVELS  6

/* Fermi+ GPU VASpace control commands and parameters */
#define LW90F1_CTRL_CMD(cat,idx)          LWXXXX_CTRL_CMD(0x90F1, LW90F1_CTRL_##cat, idx)

/* Command categories (6bits) */
#define LW90F1_CTRL_RESERVED (0x00)
#define LW90F1_CTRL_VASPACE  (0x01)

/*!
 * Does nothing.
 */
#define LW90F1_CTRL_CMD_NULL (0x90f10000) /* finn: Evaluated from "(FINN_FERMI_VASPACE_A_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*!
 * Get VAS GPU MMU format.
 */
#define LW90F1_CTRL_CMD_VASPACE_GET_GMMU_FORMAT (0x90f10101) /* finn: Evaluated from "(FINN_FERMI_VASPACE_A_VASPACE_INTERFACE_ID << 8) | LW90F1_CTRL_VASPACE_GET_GMMU_FORMAT_PARAMS_MESSAGE_ID" */

#define LW90F1_CTRL_VASPACE_GET_GMMU_FORMAT_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW90F1_CTRL_VASPACE_GET_GMMU_FORMAT_PARAMS {
    /*!
     * [in] GPU sub-device handle - this API only supports unicast.
     *      Pass 0 to use subDeviceId instead.
     */
    LwHandle hSubDevice;

    /*!
     * [in] GPU sub-device ID. Ignored if hSubDevice is non-zero.
     */
    LwU32    subDeviceId;

    /*!
     * [out] GMMU format struct. This is of RM-internal type "struct GMMU_FMT*"
     *       which can only be accessed by kernel builds since this is a kernel
     *       only API.
     */
    LW_DECLARE_ALIGNED(LwP64 pFmt, 8);
} LW90F1_CTRL_VASPACE_GET_GMMU_FORMAT_PARAMS;

/*!
 * Get VAS page level information.
 */
#define LW90F1_CTRL_CMD_VASPACE_GET_PAGE_LEVEL_INFO (0x90f10102) /* finn: Evaluated from "(FINN_FERMI_VASPACE_A_VASPACE_INTERFACE_ID << 8) | 0x2" */

typedef struct LW90F1_CTRL_VASPACE_GET_PAGE_LEVEL_INFO_PARAMS {
    /*!
     * [in] GPU sub-device handle - this API only supports unicast.
     *      Pass 0 to use subDeviceId instead.
     */
    LwHandle hSubDevice;

    /*!
     * [in] GPU sub-device ID. Ignored if hSubDevice is non-zero.
     */
    LwU32    subDeviceId;

    /*!
     * [in] GPU virtual address to query.
     */
    LW_DECLARE_ALIGNED(LwU64 virtAddress, 8);

    /*!
     * [in] Page size to query.
     */
    LW_DECLARE_ALIGNED(LwU64 pageSize, 8);

    /*!
     * [out] Number of levels populated.
     */
    LwU32    numLevels;

    /*!
     * [out] Per-level information.
     */
    struct {
        /*!
         * Format of this level.
         */
        LW_DECLARE_ALIGNED(struct MMU_FMT_LEVEL *pFmt, 8);

       /*!
        * Level/Sublevel Formats flattened
        */
        LW_DECLARE_ALIGNED(MMU_FMT_LEVEL levelFmt, 8);
        LW_DECLARE_ALIGNED(MMU_FMT_LEVEL sublevelFmt[MMU_FMT_MAX_SUB_LEVELS], 8);

        /*!
         * Physical address of this page level instance.
         */
        LW_DECLARE_ALIGNED(LwU64 physAddress, 8);

        /*!
         * Aperture in which this page level instance resides.
         */
        LwU32 aperture;

        /*!
         * Size in bytes allocated for this level instance.
         */
        LW_DECLARE_ALIGNED(LwU64 size, 8);
    } levels[GMMU_FMT_MAX_LEVELS];
} LW90F1_CTRL_VASPACE_GET_PAGE_LEVEL_INFO_PARAMS;

/*!
 * Reserve (allocate and bind) page directory/table entries up to
 * a given level of the MMU format. Also referred to as "lock-down".
 *
 * Each range that has been reserved must be released
 * eventually with @ref LW90F1_CTRL_CMD_VASPACE_RELEASE_ENTRIES.
 * A particular VA range and level (page size) combination may only be
 * locked down once at a given time, but each level is independent.
 */
#define LW90F1_CTRL_CMD_VASPACE_RESERVE_ENTRIES (0x90f10103) /* finn: Evaluated from "(FINN_FERMI_VASPACE_A_VASPACE_INTERFACE_ID << 8) | LW90F1_CTRL_VASPACE_RESERVE_ENTRIES_PARAMS_MESSAGE_ID" */

#define LW90F1_CTRL_VASPACE_RESERVE_ENTRIES_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW90F1_CTRL_VASPACE_RESERVE_ENTRIES_PARAMS {
    /*!
     * [in] GPU sub-device handle - this API only supports unicast.
     *      Pass 0 to use subDeviceId instead.
     */
    LwHandle hSubDevice;

    /*!
     * [in] GPU sub-device ID. Ignored if hSubDevice is non-zero.
     */
    LwU32    subDeviceId;

    /*!
     * [in] Page size (VA coverage) of the level to reserve.
     *      This need not be a leaf (page table) page size - it can be
     *      the coverage of an arbitrary level (including root page directory).
     */
    LW_DECLARE_ALIGNED(LwU64 pageSize, 8);

    /*!
     * [in] First GPU virtual address of the range to reserve.
     *      This must be aligned to pageSize.
     */
    LW_DECLARE_ALIGNED(LwU64 virtAddrLo, 8);

    /*!
     * [in] Last GPU virtual address of the range to reserve.
     *      This (+1) must be aligned to pageSize.
     */
    LW_DECLARE_ALIGNED(LwU64 virtAddrHi, 8);
} LW90F1_CTRL_VASPACE_RESERVE_ENTRIES_PARAMS;

/*!
 * Release (unbind and free) page directory/table entries up to
 * a given level of the MMU format that has been reserved through a call to
 * @ref LW90F1_CTRL_CMD_VASPACE_RESERVE_ENTRIES. Also referred to as "unlock".
 */
#define LW90F1_CTRL_CMD_VASPACE_RELEASE_ENTRIES (0x90f10104) /* finn: Evaluated from "(FINN_FERMI_VASPACE_A_VASPACE_INTERFACE_ID << 8) | LW90F1_CTRL_VASPACE_RELEASE_ENTRIES_PARAMS_MESSAGE_ID" */

#define LW90F1_CTRL_VASPACE_RELEASE_ENTRIES_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW90F1_CTRL_VASPACE_RELEASE_ENTRIES_PARAMS {
    /*!
     * [in] GPU sub-device handle - this API only supports unicast.
     *      Pass 0 to use subDeviceId instead.
     */
    LwHandle hSubDevice;

    /*!
     * [in] GPU sub-device ID. Ignored if hSubDevice is non-zero.
     */
    LwU32    subDeviceId;

    /*!
     * [in] Page size (VA coverage) of the level to release.
     *      This need not be a leaf (page table) page size - it can be
     *      the coverage of an arbitrary level (including root page directory).
     */
    LW_DECLARE_ALIGNED(LwU64 pageSize, 8);

    /*!
     * [in] First GPU virtual address of the range to release.
     *      This must be aligned to pageSize.
     */
    LW_DECLARE_ALIGNED(LwU64 virtAddrLo, 8);

    /*!
     * [in] Last GPU virtual address of the range to release.
     *      This (+1) must be aligned to pageSize.
     */
    LW_DECLARE_ALIGNED(LwU64 virtAddrHi, 8);
} LW90F1_CTRL_VASPACE_RELEASE_ENTRIES_PARAMS;

/*!
 * Get VAS page level information without kernel priviledge. This will internally call
 * LW90F1_CTRL_CMD_VASPACE_GET_PAGE_LEVEL_INFO.
 */
#define LW90F1_CTRL_CMD_VASPACE_GET_PAGE_LEVEL_INFO_VERIF (0x90f10105) /* finn: Evaluated from "(FINN_FERMI_VASPACE_A_VASPACE_INTERFACE_ID << 8) | 0x5" */

/*!
 * Pin PDEs for a given VA range on the server RM and then mirror the client's page 
 * directory/tables in the server. 
 *  
 * @ref
 */
#define LW90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES (0x90f10106) /* finn: Evaluated from "(FINN_FERMI_VASPACE_A_VASPACE_INTERFACE_ID << 8) | LW90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_MESSAGE_ID" */

#define LW90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS {
    /*!
     * [in] GPU sub-device handle - this API only supports unicast.
     *      Pass 0 to use subDeviceId instead.
     */
    LwHandle hSubDevice;

    /*!
     * [in] GPU sub-device ID. Ignored if hSubDevice is non-zero.
     */
    LwU32    subDeviceId;

    /*!
     * [in] Page size (VA coverage) of the level to reserve.
     *      This need not be a leaf (page table) page size - it can be
     *      the coverage of an arbitrary level (including root page directory).
     */
    LW_DECLARE_ALIGNED(LwU64 pageSize, 8);

    /*!
     * [in] First GPU virtual address of the range to reserve.
     *      This must be aligned to pageSize.
     */
    LW_DECLARE_ALIGNED(LwU64 virtAddrLo, 8);

    /*!
     * [in] Last GPU virtual address of the range to reserve.
     *      This (+1) must be aligned to pageSize.
     */
    LW_DECLARE_ALIGNED(LwU64 virtAddrHi, 8);

    /*! 
     * [in] Number of PDE levels to copy.
     */
    LwU32    numLevelsToCopy;

   /*!
     * [in] Per-level information.
     */
    struct {
        /*!
         * Physical address of this page level instance.
         */
        LW_DECLARE_ALIGNED(LwU64 physAddress, 8);

        /*!
         * Size in bytes allocated for this level instance.
         */
        LW_DECLARE_ALIGNED(LwU64 size, 8);

        /*!
         * Aperture in which this page level instance resides.
         */
        LwU32 aperture;

        /*!
         * Page shift corresponding to the level
         */
        LwU8  pageShift;
    } levels[GMMU_FMT_MAX_LEVELS];
} LW90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS;

/* _ctrl90f1_h_ */
