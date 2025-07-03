/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2009-2015 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl208f/ctrl208fgpu.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl2080/ctrl2080gr.h"        /* 208F is partially derivative of 2080 */
#include "ctrl/ctrl208f/ctrl208fbase.h"

/*
 * LW208F_CTRL_CMD_GPU_GET_RAM_SVOP_VALUES
 *
 * This command can be used to get the RAM SVOP values.
 *
 *    sp
 *     This field outputs RAM_SVOP_SP
 *    rg
 *     This field outputs RAM_SVOP_REG
 *    pdp
 *     This field outputs RAM_SVOP_PDP
 *    dp
 *     This field outputs RAM_SVOP_DP
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW208F_CTRL_CMD_GPU_GET_RAM_SVOP_VALUES (0x208f1101) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_GPU_INTERFACE_ID << 8) | 0x1" */

typedef struct LW208F_CTRL_GPU_RAM_SVOP_VALUES_PARAMS {
    LwU32 sp;
    LwU32 rg;
    LwU32 pdp;
    LwU32 dp;
} LW208F_CTRL_GPU_RAM_SVOP_VALUES_PARAMS;

/*
 * LW208F_CTRL_CMD_GPU_SET_RAM_SVOP_VALUES
 *
 * This command can be used to set the RAM SVOP values.
 *
 *    sp
 *     Input for RAM_SVOP_SP
 *    rg
 *     Input for RAM_SVOP_REG
 *    pdp
 *     Input for RAM_SVOP_PDP
 *    dp
 *     Input for RAM_SVOP_DP
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW208F_CTRL_CMD_GPU_SET_RAM_SVOP_VALUES (0x208f1102) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_GPU_INTERFACE_ID << 8) | 0x2" */

/*
 * LW208F_CTRL_CMD_GPU_GET_JTAG_CHAIN
 *
 * This command can be used to read values from the JTAG.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW208F_CTRL_CMD_GPU_GET_JTAG_CHAIN      (0x208f1103) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_GPU_INTERFACE_ID << 8) | 0x3" */


/*
 * LW208F_CTRL_CMD_GPU_SET_JTAG_CHAIN
 *
 * This command can be used to write values to the JTAG.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW208F_CTRL_CMD_GPU_SET_JTAG_CHAIN      (0x208f1104) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_GPU_INTERFACE_ID << 8) | 0x4" */

/* XAPI-TODO: Colwert this interface to iterable form since it can reasonably
              return much more than 4k (megabyte or more).
 */

typedef struct LW208F_CTRL_GPU_JTAG_CHAIN_PARAMS {
    LwU32 chainLen;
    LwU32 chipletSel;
    LwU32 instrId;
    LwU32 dataArrayLen;
    LW_DECLARE_ALIGNED(LwP64 data, 8);
} LW208F_CTRL_GPU_JTAG_CHAIN_PARAMS;

/*
 * LW208F_CTRL_CMD_GPU_VERIFY_INFOROM
 *
 * This command can be used by clients to determine if an InfoROM
 * with a valid image is present. If the SKU in question does
 * not feature an InfoROM, the LW_ERR_NOT_SUPPORTED
 * error is returned. Else the RM attempts to read the ROM object
 * and any objects listed in the ROM object. The checksum of
 * each object read is verified. If all checksums are valid, the
 * RM will report the InfoROM as being valid. If image is valid then
 * RM will return a checksum for all of the dynamically configurable
 * data in InfoROM. This checksum is expected to be same for all the
 * boards with an identical InfoROM version and similarly configured.
 *
 * result
 *    The result of the InfoROM verification attempt. Possible
 *    values are:
 *      LW208F_CTRL_GPU_INFOROM_VERIFICATION_RESULTS_NONE
 *         This value indicates that a validation couldn't be done
 *         due to some software/OS related error.
 *      LW208F_CTRL_GPU_INFOROM_VERIFICATION_RESULTS_IO_ERROR
 *         This value indicates that a validation couldn't be done
 *         due to some IO error.
 *      LW208F_CTRL_GPU_INFOROM_VERIFICATION_RESULTS_VALID
 *         This value indicates that all InfoROM objects have valid
 *         checksum.
 *      LW208F_CTRL_GPU_INFOROM_VERIFICATION_RESULTS_ILWALID
 *         This value indicates that some InfoROM objects have invalid
 *         checksum.
 * checksum
 *    Checksum for all of the dynamically configurable data
 *    in InfoROM for e.g. PWR and CFG objects.
 *
 * NOTE: For the result values to be valid, return status should be:
 *   LW_OK
 *
 * Possible return status values:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *
 */
#define LW208F_CTRL_CMD_GPU_VERIFY_INFOROM (0x208f1105) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_GPU_INTERFACE_ID << 8) | LW208F_CTRL_GPU_VERIFY_INFOROM_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_GPU_VERIFY_INFOROM_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW208F_CTRL_GPU_VERIFY_INFOROM_PARAMS {
    LwU32 result;
    LwU32 checksum;
} LW208F_CTRL_GPU_VERIFY_INFOROM_PARAMS;

/* valid result values */
#define LW208F_CTRL_GPU_INFOROM_VERIFICATION_RESULT_NONE     (0x00000000)
#define LW208F_CTRL_GPU_INFOROM_VERIFICATION_RESULT_IO_ERROR (0x00000001)
#define LW208F_CTRL_GPU_INFOROM_VERIFICATION_RESULT_VALID    (0x00000002)
#define LW208F_CTRL_GPU_INFOROM_VERIFICATION_RESULT_ILWALID  (0x00000003)

/*
 * LW208F_CTRL_CMD_GPU_DISABLE_ECC_INFOROM_REPORTING
 *
 * This command disables ECC errors from being reported on the
 * inforom.  This should be used by programs that intentually inject
 * ECC errors (such as MODS), to prevent those errors from appearing
 * as genuine errors.
 *
 * Due to other structures with ECC information in the RM, re-enabling
 * inforom reporting can not occur without re-initializing the RM.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW208F_CTRL_CMD_GPU_DISABLE_ECC_INFOROM_REPORTING    (0x208f1107) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_GPU_INTERFACE_ID << 8) | 0x7" */

/*
 * LW208F_CTRL_CMD_GPU_GET_EMULATION_INFO
 *
 * This control command has been added to fetch information under
 * emulation conditions. It returns info like current netlist number,
 * hardware changelist for current netlist, available DRAM and info from 
 * chunk bits.
 *
 * [OUT]  lwrrentNetlistNumber
 *            This field returns the current netlist number.
 * [OUT]  hwChangelistNumber
 *            This field returns the hardware changelist number corresponding to the
 *            current netlist.
 * [OUT]  totalDram
 *            This field returns the total DRAM in the database.
 * [OUT]  chunkBits
 *            This field returns the value of chunk bits.
 * [IN]   grRouteInfo
 *            This parameter specifies the routing information used to
 *            disambiguate the target GR engine.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */

#define LW208F_CTRL_CMD_GPU_GET_EMULATION_INFO               (0x208f1108) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_GPU_INTERFACE_ID << 8) | LW208F_CTRL_GPU_GET_EMULATION_INFO_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_GPU_GET_EMULATION_INFO_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW208F_CTRL_GPU_GET_EMULATION_INFO_PARAMS {
    LwU32 lwrrentNetlistNumber;
    LwU32 hwChangelistNumber;
    LwU32 totalDram;
    LwU32 chunkBits;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW208F_CTRL_GPU_GET_EMULATION_INFO_PARAMS;

/*
 * LW208F_CTRL_CMD_GPU_SET_USER_REGISTER_ACCESS_PERMISSIONS
 *
 * This control tests the mechanism that restricts user-space access to
 * privileged registers.  It exposes sensitive information and functionality
 * so should only ever be enabled on verification builds (LW_VERIF_FEATURES).
 *
 * This control provides two functions.  First, it takes a range and whether or
 * not that range should be accessible from userspace.  Second, it has the
 * ability to swap RM's buffer for a provided one.  This is to ease saving and
 * restoring RM's internal buffer before and after running a test, to make sure
 * the buffer is in a known format when testing.
 *
 * The new buffer is inserted before applying the provided update.
 *
 * Note size parameters are provided for completeness.  Using a size that is
 * not large enough to contain all of BAR0 will yield undefined results.
 *
 *   offset
 *       This field is the offset within the GPU register space to change.  It
 *       must be 4-byte aligned.
 *   size
 *       This field is the size of register space to change.  It must be a
 *       4-byte multiple.
 *   bAllow
 *      This field indicates whether the given range should be accessible from
 *      user space or not.
 *   pOldAccessMap
 *      This field returns the pointer to RM's internal buffer.
 *   oldAccessMapSize
 *      This field returns the size of the buffer held in pOldAccessMap
 *   pNewAccessMap
 *      This field contains a pointer to a buffer used to override the existing
 *      internal buffer.
 *   newAccessMapSize
 *      This field contains the size of the buffer held in pNewAccessMap.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW208F_CTRL_CMD_GPU_SET_USER_REGISTER_ACCESS_PERMISSIONS (0x208f1109) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_GPU_INTERFACE_ID << 8) | LW208F_CTRL_GPU_SET_USER_REGISTER_ACCESS_PERMISSIONS_MESSAGE_ID" */

#define LW208F_CTRL_GPU_SET_USER_REGISTER_ACCESS_PERMISSIONS_MESSAGE_ID (0x9U)

typedef struct LW208F_CTRL_GPU_SET_USER_REGISTER_ACCESS_PERMISSIONS {
    LwU32  offset;
    LwU32  size;
    LwBool bAllow;

    LW_DECLARE_ALIGNED(LwP64 pOldAccessMap, 8);
    LwU32  oldAccessMapSize;

    LW_DECLARE_ALIGNED(LwP64 pNewAccessMap, 8);
    LwU32  newAccessMapSize;
} LW208F_CTRL_GPU_SET_USER_REGISTER_ACCESS_PERMISSIONS;

/* _ctrl208fgpu_h_ */

