/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2001-2020 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl9096.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#define LW9096_CTRL_CMD(cat,idx)  \
    LWXXXX_CTRL_CMD(0x9096, LW9096_CTRL_##cat, idx)

/* LW9096 command categories (6bits) */
#define LW9096_CTRL_RESERVED                                            (0x00U)
#define LW9096_CTRL_ZBC                                                 (0x01U)


/*
 * LW9096_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW9096_CTRL_CMD_NULL                                            (0x90960000U) /* finn: Evaluated from "(FINN_GF100_ZBC_CLEAR_RESERVED_INTERFACE_ID << 8) | 0x0" */

/*
 * LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL_
 * These are various possible CtFormats which 
 * the Client passes down to RM to set in the DS
 * Color Table.These ZBC ENUMS are based on the CT Format ENUM class 
 * equivalence described in :-
 * //hw/fermi1_gf100/class/mfs/class/3d/fermi.mfs
 */

#define LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL_ILWALID             0x00000000U
#define LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL_ZERO                0x00000001U
#define LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL_UNORM_ONE           0x00000002U
#define LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL_RF32_GF32_BF32_AF32 0x00000004U
#define LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL_R16_G16_B16_A16     0x00000008U
#define LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL_RN16_GN16_BN16_AN16 0x0000000lw
#define LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL_RS16_GS16_BS16_AS16 0x00000010U
#define LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL_RU16_GU16_BU16_AU16 0x00000014U
#define LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL_RF16_GF16_BF16_AF16 0x00000016U
#define LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL_A8R8G8B8            0x00000018U
#define LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL_A8RL8GL8BL8         0x0000001lw
#define LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL_A2B10G10R10         0x00000020U
#define LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL_AU2BU10GU10RU10     0x00000024U
#define LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL_A8B8G8R8            0x00000028U
#define LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL_A8BL8GL8RL8         0x0000002lw
#define LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL_AN8BN8GN8RN8        0x00000030U
#define LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL_AS8BS8GS8RS8        0x00000034U
#define LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL_AU8BU8GU8RU8        0x00000038U
#define LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL_A2R10G10B10         0x0000003lw
#define LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL_BF10GF11RF11        0x00000040U

/*
 * LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR
 *
 * This command attempts to add a new entry to Color ZBC Tables.
 *
 *  colorFB
 *    This field sets the raw framebuffer values for the ZBC table entries. Prior to 
 *    GA10x, these values are written into the "L2" table.
 *    From GA10x and later, these values are written in "CROP" table.
 *  colorDS
 *    This field sets format-independent values for ZBC table entries. Prior to GA10X, 
 *    these values are written in the "DS" table and  matched with the format-independent 
 *    clear color sent in the 3D class. These values are ignored on GA10X and later 
 *  format
 *    This field specifies color format for ZBC table entries and should be one of the 
 *    LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT* enums. Prior to GA10X, these values 
 *    are written in the format field of the hardware ZBC table entry and are matched 
 *    against the color format sent in the 3D class. These values are ignored on GA10X and later.
 *  bSkipL2Table
 *    This parameter specifies if the L2 ZBC table should be updated or not. If
 *    this parameter is set to a non-zero value, the L2 ZBC table will not be
 *    updated. If this parameter is set to 0, the L2 ZBC table will be updated. This 
 *    parameter will skip programming DS table values in Pre-GA10x. From GA10x and later
 *    this parameter will skip programming CROP table entries. 
 *    Note: This parameter will only be supported in verification platforms. 
 *
 * Possible status values returned are:
 *   LW_OK
 *   LWOS_STATUS_INSUFFICIENT_RESOURCES
 */


#define LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR        (0x90960101U) /* finn: Evaluated from "(FINN_GF100_ZBC_CLEAR_ZBC_INTERFACE_ID << 8) | LW9096_CTRL_SET_ZBC_COLOR_CLEAR_PARAMS_MESSAGE_ID" */

#define LW9096_CTRL_SET_ZBC_COLOR_CLEAR_VALUE_SIZE 4U
#define LW9096_CTRL_SET_ZBC_COLOR_CLEAR_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW9096_CTRL_SET_ZBC_COLOR_CLEAR_PARAMS {
    LwU32  colorFB[LW9096_CTRL_SET_ZBC_COLOR_CLEAR_VALUE_SIZE];
    LwU32  colorDS[LW9096_CTRL_SET_ZBC_COLOR_CLEAR_VALUE_SIZE];
    LwU32  format;
    LwBool bSkipL2Table;
} LW9096_CTRL_SET_ZBC_COLOR_CLEAR_PARAMS;


/*
 * LW9096_CTRL_CMD_SET_ZBC_DEPTH_CLEAR_FMT_VAL
 * These are various possible formats which the Client passes down to RM to set in the ZBC clear 
 * Table.
 * 
 * Though the "Depth" data written in both the DS as well as L2 Version of the 
 * depth table are of (only available!) FP32 Format. Still the format 
 * support is lwrrently given with _CTRL_CMD_SET_ZBC_DEPTH_CLEAR as this format will
 * be used later on to disable/remove an entry from the table.
 * In future this field is going to be significant for
 * "Depth" entries too.
 */

#define LW9096_CTRL_CMD_SET_ZBC_DEPTH_CLEAR_FMT_VAL_ILWALID 0x00000000U
// Fix me: Fix the name to FMT_VAL_FP32
#define LW9096_CTRL_CMD_SET_ZBC_DEPTH_CLEAR_FMT_FP32        0x00000001U

/*
 * LW9096_CTRL_CMD_SET_ZBC_DEPTH_CLEAR
 *
 * This command attempts to add a new entry to Depth ZBC Tables.
 *
 *  depth
 *    This field specifies the ZBC depth clear value to be set.
 *  format
 *    This field specifies the Depth format for the data send in  by the client. 
 *  bSkipL2Table
 *    This parameter specifies if the L2 ZBC table should be updated or not. If
 *    this parameter is set to a non-zero value, the L2 ZBC table will not be
 *    updated. If this parameter is set to 0, the L2 ZBC table will be updated.
 *    Note: This parameter will only be supported in verification platforms. 
 * 

 * Possible status values returned are:
 *   LW_OK
 *   LWOS_STATUS_INSUFFICIENT_RESOURCES
 */
#define LW9096_CTRL_CMD_SET_ZBC_DEPTH_CLEAR                 (0x90960102U) /* finn: Evaluated from "(FINN_GF100_ZBC_CLEAR_ZBC_INTERFACE_ID << 8) | LW9096_CTRL_SET_ZBC_DEPTH_CLEAR_PARAMS_MESSAGE_ID" */

#define LW9096_CTRL_SET_ZBC_DEPTH_CLEAR_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW9096_CTRL_SET_ZBC_DEPTH_CLEAR_PARAMS {
    LwU32  depth;
    LwU32  format;
    LwBool bSkipL2Table;
} LW9096_CTRL_SET_ZBC_DEPTH_CLEAR_PARAMS;

/*
 * LW9096_CTRL_CMD_GET_ZBC_CLEAR_TABLE
 *
 *
 * This command is used to get the ZBC Clear Color/Depth/Stencil table data i.e. color 
 * values and the status "Use-satatus" of each value in the table from RM. 
 *
 *   colorFB
 *      This value returns raw framebuffer values for color ZBC table entries. Pre-GA10x, these values
 *      are taken from the "L2" table. From GA10x+, these values are taken from CROP table. This value is
 *      set only when valType is chosen as "LW9096_CTRL_ZBC_CLEAR_OBJECT_TYPE_COLOR"
 *   colorDS
 *      Pre-GA10x, returns the DS color value set for ZBC. From GA10x+, returns zeroes since this format is
 *      deprecated in HW. This value is set only when valType is chosen as "LW9096_CTRL_ZBC_CLEAR_OBJECT_TYPE_COLOR"
 *   depth
 *     This field returns the ZBC depth clear value set, when valType is chosen as "LW9096_CTRL_ZBC_CLEAR_OBJECT_TYPE_DEPTH".
 *   stencil
 *     This field returns the ZBC stencil clear value set, when valType is chosen as "LW9096_CTRL_ZBC_CLEAR_OBJECT_TYPE_STENCIL"
 *   format
 *      This field returns the format of color, depth, or stencil ZBC table entries, using the
 *      LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT,
 *      LW9096_CTRL_CMD_SET_ZBC_DEPTH_CLEAR_FMT,
 *      LW9096_CTRL_CMD_SET_ZBC_STENCIL_CLEAR_FMT* enums depending on the table identified by valType.
 *      On GA10X and later, color ZBC table entries do not have a format in hardware and this query 
 *      returns a format of "INVALID".
 *   valType
 *     This filed specifies the type of the Table  data to be fetched.
 *     Possible Types are :
 *       LW9096_CTRL_ZBC_CLEAR_OBJECT_TYPE_COLOR
 *       LW9096_CTRL_ZBC_CLEAR_OBJECT_TYPE_DEPTH
 *       LW9096_CTRL_ZBC_CLEAR_OBJECT_TYPE_STENCIL
 *   indexUsed
 *     This boolean parameter indicates if a particular index of the table is valid or not.
 *
 *   Note: The following parameters are deprecated after CheetAh interface is also changed. Clients need 
 *   to use LW9096_CTRL_CMD_GET_ZBC_CLEAR_TABLE_SIZE to query the (start, end) indexes of respective ZBC tables
 *
 *   indexSize
 *     This parameter is used to fetch the table size when "valType" parameter
 *     is specified as " 0 ".(INVALID TYPE). It is also used to pass in the
 *     index of the ZBC table for which we want the COLOR/DEPTH info.
 *   indexStart
 *      This parameter is used to return the valid starting index of ZBC table, when
 *      "valType" parameter is specified as "ILWALID_TYPE". It will also be used
 *       as input index to query the ZBC table for COLOR/QUERY/STENCIL Info. 
 *   indexEnd
 *       This parameter is used to return the valid ending index of ZBC table, when
 *       "valType" parameter is specified as "ILWALID_TYPE".
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW9096_CTRL_CMD_GET_ZBC_CLEAR_TABLE       (0x90960103U) /* finn: Evaluated from "(FINN_GF100_ZBC_CLEAR_ZBC_INTERFACE_ID << 8) | LW9096_CTRL_GET_ZBC_CLEAR_TABLE_PARAMS_MESSAGE_ID" */

#define LW9096_CTRL_ZBC_CLEAR_OBJECT_TYPE_ILWALID 0U
#define LW9096_CTRL_ZBC_CLEAR_OBJECT_TYPE_COLOR   1U
#define LW9096_CTRL_ZBC_CLEAR_OBJECT_TYPE_DEPTH   2U
#define LW9096_CTRL_ZBC_CLEAR_OBJECT_TYPE_STENCIL 3U
#define LW9096_CTRL_GET_ZBC_CLEAR_TABLE_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW9096_CTRL_GET_ZBC_CLEAR_TABLE_PARAMS {
    struct {
        LwU32 colorFB[LW9096_CTRL_SET_ZBC_COLOR_CLEAR_VALUE_SIZE];
        LwU32 colorDS[LW9096_CTRL_SET_ZBC_COLOR_CLEAR_VALUE_SIZE];
        LwU32 depth;
        LwU32 stencil;
    } value;
    LwU32 indexSize;
    LwU32 indexStart;
    LwU32 indexEnd;
    LwU32 indexUsed; /* TODO: Change to LwBool, need to fix the cheetah interface too */
    LwU32 format;
    LwU32 valType;
} LW9096_CTRL_GET_ZBC_CLEAR_TABLE_PARAMS;

/*
 * Note: This ctrl call is deprecated. To program ZBC table entries, please use 
 *  LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR
 *  LW9096_CTRL_CMD_SET_ZBC_DEPTH_CLEAR
 *  LW9096_CTRL_CMD_SET_ZBC_STENCIL_CLEAR
 *  which will update a single entry in single table at a time. 
 *  
 *  
 * LW9096_CTRL_CMD_SET_ZBC_CLEAR_TABLE
 *
 * This command is used to set the ZBC Clear Color/Depth/Stencil table data at a specified
 * index. The parameters to this command are described below.
 *
 *  colorFB
 *    This array field specifies the L2 color value to be written to the ZBC table.
 *  colorDS
 *    This array field specifies the DS color value to be written to the ZBC table.
 *  colorFormat
 *    This field specifies the ZBC color format to be set. This field must be set
 *    to one of the valid LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL* defines.
 *  depth
 *    This field specifies the ZBC depth clear value to be set.
 *  depthFormat
 *    This field specifies the ZBC depth format to be set. This field must be set
 *    to one of the valid LW9096_CTRL_CMD_SET_ZBC_DEPTH_CLEAR_FMT_VAL* defines.
 *  stencil
 *    This field specifies the ZBC stencil clear value to be set.
 *  stencilFormat
 *    This field specifies the ZBC stencil format to be set. This field must be set
 *    to one of the valid LW9096_CTRL_CMD_SET_ZBC_STENCIL_CLEAR_FMT_VAL* defines.
 *  index
 *    This field specifies the index at which the color/depth data is to be
 *    written into the ZBC table. Legal values for this field must lie between
 *    1 and the value returned in the indexSize parameter of the
 *    LW9096_CTRL_CMD_GET_ZBC_CLEAR_TABLE control call when called with the
 *    parameter valType set to LW9096_CTRL_ZBC_CLEAR_OBJECT_TYPE_ILWALID.
 *  bSkipL2Table
 *    This parameter specifies if the L2 ZBC table should be updated or not. If
 *    this parameter is set to a non-zero value, the L2 ZBC table will not be
 *    updated. If this parameter is set to 0, the L2 ZBC table will be updated.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 *    LW_ERR_INSUFFICIENT_RESOURCES
 */
#define LW9096_CTRL_CMD_SET_ZBC_CLEAR_TABLE (0x90960104U) /* finn: Evaluated from "(FINN_GF100_ZBC_CLEAR_ZBC_INTERFACE_ID << 8) | LW9096_CTRL_SET_ZBC_CLEAR_TABLE_PARAMS_MESSAGE_ID" */

#define LW9096_CTRL_SET_ZBC_CLEAR_TABLE_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW9096_CTRL_SET_ZBC_CLEAR_TABLE_PARAMS {
    LwU32  colorFB[LW9096_CTRL_SET_ZBC_COLOR_CLEAR_VALUE_SIZE];
    LwU32  colorDS[LW9096_CTRL_SET_ZBC_COLOR_CLEAR_VALUE_SIZE];
    LwU32  colorFormat;
    LwU32  depth;
    LwU32  depthFormat;
    LwU32  stencil;
    LwU32  stencilFormat;
    LwU32  index;
    LwBool bSkipL2Table;
} LW9096_CTRL_SET_ZBC_CLEAR_TABLE_PARAMS;

/*
 * LW9096_CTRL_CMD_SET_ZBC_STENCIL_CLEAR_FMT_VAL_
 * These are various possible Formats which the Client passes down to RM to set in the 
 * ZBC clear(DSS) Table.
 */

#define LW9096_CTRL_CMD_SET_ZBC_STENCIL_CLEAR_FMT_VAL_ILWALID 0x00000000U
// Fix me: Change it to CLEAR_FMT_VAL_U8
#define LW9096_CTRL_CMD_SET_ZBC_STENCIL_CLEAR_FMT_U8          0x00000001U

/*
 * LW9096_CTRL_CMD_SET_ZBC_STENCIL_CLEAR
 *
 * This command attempts to add a new entry to Stencil ZBC Tables.
 *
 *  stencil
 *    This field specifies the ZBC stencil clear value to be set.
 *  format
 *    This field specifies the stencil format for the data send in  by the client.
 *  bSkipL2Table
 *    This parameter specifies if the L2 ZBC table should be updated or not. If
 *    this parameter is set to a non-zero value, the L2 ZBC table will not be
 *    updated. If this parameter is set to 0, the L2 ZBC table will be updated.
 *    Note: This parameter will only be supported in verification platforms.
 * Possible status values returned are:
 *   LW_OK
 *   LWOS_STATUS_INSUFFICIENT_RESOURCES
 */
#define LW9096_CTRL_CMD_SET_ZBC_STENCIL_CLEAR                 (0x90960105U) /* finn: Evaluated from "(FINN_GF100_ZBC_CLEAR_ZBC_INTERFACE_ID << 8) | LW9096_CTRL_SET_ZBC_STENCIL_CLEAR_PARAMS_MESSAGE_ID" */

#define LW9096_CTRL_SET_ZBC_STENCIL_CLEAR_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW9096_CTRL_SET_ZBC_STENCIL_CLEAR_PARAMS {
    LwU32  stencil;
    LwU32  format;
    LwBool bSkipL2Table;
} LW9096_CTRL_SET_ZBC_STENCIL_CLEAR_PARAMS;

/*
 * LW9096_CTRL_CMD_GET_ZBC_CLEAR_TABLE_SIZE
 *   This command returns the range of valid indices in the color, depth, or stencil ZBC tables.
 * 
 *   indexStart
 *      This parameter is used to return the first valid index in the color, depth, or stencil ZBC table,
 *      depending on the value passed in the tableType 
 *   indexEnd
 *      This parameter is used to return the last valid index in the color, depth, or stencil ZBC table,
 *      depending on the value passed in the tableType
 *   tableType
 *     This field specifies the type of the Table  data to be fetched.
 *     Possible Types are :
 *       LW9096_CTRL_ZBC_CLEAR_TABLE_TYPE_COLOR
 *       LW9096_CTRL_ZBC_CLEAR_TABLE_TYPE_DEPTH
 *       LW9096_CTRL_ZBC_CLEAR_TABLE_TYPE_STENCIL
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

typedef enum LW9096_CTRL_ZBC_CLEAR_TABLE_TYPE {
    LW9096_CTRL_ZBC_CLEAR_TABLE_TYPE_ILWALID = 0,
    LW9096_CTRL_ZBC_CLEAR_TABLE_TYPE_COLOR = 1,
    LW9096_CTRL_ZBC_CLEAR_TABLE_TYPE_DEPTH = 2,
    LW9096_CTRL_ZBC_CLEAR_TABLE_TYPE_STENCIL = 3,
    LW9096_CTRL_ZBC_CLEAR_TABLE_TYPE_COUNT = 4,
} LW9096_CTRL_ZBC_CLEAR_TABLE_TYPE;

#define LW9096_CTRL_CMD_GET_ZBC_CLEAR_TABLE_SIZE (0x90960106U) /* finn: Evaluated from "(FINN_GF100_ZBC_CLEAR_ZBC_INTERFACE_ID << 8) | LW9096_CTRL_GET_ZBC_CLEAR_TABLE_SIZE_PARAMS_MESSAGE_ID" */

#define LW9096_CTRL_GET_ZBC_CLEAR_TABLE_SIZE_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW9096_CTRL_GET_ZBC_CLEAR_TABLE_SIZE_PARAMS {
    LwU32                            indexStart;
    LwU32                            indexEnd;
    LW9096_CTRL_ZBC_CLEAR_TABLE_TYPE tableType;
} LW9096_CTRL_GET_ZBC_CLEAR_TABLE_SIZE_PARAMS;

/*
 * LW9096_CTRL_CMD_GET_ZBC_CLEAR_TABLE_ENTRY
 *   This command returns the ZBC entry stored in the color, depth or stencil ZBC tables
 *
 *   colorFB[out]
 *      This value returns raw framebuffer values for color ZBC table entries. Pre-GA10x, these values
 *      are taken from the "L2" table. From GA10x+, these values are taken from CROP table. This value is
 *      set only when valType is chosen as "LW9096_CTRL_ZBC_CLEAR_TABLE_TYPE_COLOR"
 *   colorDS[out]
 *      Pre-GA10x, returns the DS color value set for ZBC. From GA10x+, returns zeroes since this format is
 *      deprecated in HW. This value is set only when valType is chosen as "LW9096_CTRL_ZBC_CLEAR_TABLE_TYPE_COLOR"
 *   depth[out]
 *      This field specifies the ZBC depth clear value set, when valType is chosen as "LW9096_CTRL_ZBC_CLEAR_TABLE_TYPE_DEPTH".
 *   stencil[out]
 *      This field specifies the ZBC stencil clear value set, when valType is chosen as "LW9096_CTRL_ZBC_CLEAR_TABLE_TYPE_STENCIL"
 *   format
 *      This field returns the format of color, depth, or stencil ZBC table entries, using the
 *      LW9096_CTRL_CMD_SET_ZBC_COLOR_CLEAR_FMT_VAL*,
 *      LW9096_CTRL_CMD_SET_ZBC_DEPTH_CLEAR_FMT*,
 *      LW9096_CTRL_CMD_SET_ZBC_STENCIL_CLEAR_FMT* enums depending on the table identified by valType.
 *      On GA10X and later, color ZBC table entries do not have a format in hardware and this query 
 *      returns a format of "INVALID".
 *   index[in]
 *      This field specifies table index for which the ZBC entry information needs to be fetched. 
 *   bIndexValid[out]
 *      This field specifies whether the entry is valid or not. 
 *   tableType[in]
 *      This field specifies the type of the Table  data to be fetched.
 *      Possible Types are :
 *       LW9096_CTRL_ZBC_CLEAR_TABLE_TYPE_COLOR
 *       LW9096_CTRL_ZBC_CLEAR_TABLE_TYPE_DEPTH
 *       LW9096_CTRL_ZBC_CLEAR_TABLE_TYPE_STENCIL
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW9096_CTRL_CMD_GET_ZBC_CLEAR_TABLE_ENTRY (0x90960107U) /* finn: Evaluated from "(FINN_GF100_ZBC_CLEAR_ZBC_INTERFACE_ID << 8) | LW9096_CTRL_GET_ZBC_CLEAR_TABLE_ENTRY_PARAMS_MESSAGE_ID" */

#define LW9096_CTRL_GET_ZBC_CLEAR_TABLE_ENTRY_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW9096_CTRL_GET_ZBC_CLEAR_TABLE_ENTRY_PARAMS {
    struct {
        LwU32 colorFB[LW9096_CTRL_SET_ZBC_COLOR_CLEAR_VALUE_SIZE];
        LwU32 colorDS[LW9096_CTRL_SET_ZBC_COLOR_CLEAR_VALUE_SIZE];
        LwU32 depth;
        LwU32 stencil;
    } value;
    LwU32                            format;
    LwU32                            index;
    LwBool                           bIndexValid;
    LW9096_CTRL_ZBC_CLEAR_TABLE_TYPE tableType;
} LW9096_CTRL_GET_ZBC_CLEAR_TABLE_ENTRY_PARAMS;

/* _ctrl9096_h_ */
