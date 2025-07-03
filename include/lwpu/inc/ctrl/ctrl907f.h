/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2007-2017 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl907f.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




/* GF100_REMAPPER control commands and parameters */

#include "ctrl/ctrlxxxx.h"
#define LW907F_CTRL_CMD(cat,idx)  \
    LWXXXX_CTRL_CMD(0x907F, LW907F_CTRL_##cat, idx)

/* GF100_REMAPPER command categories (6bits) */
#define LW907F_CTRL_RESERVED (0x00)
#define LW907F_CTRL_REMAPPER (0x01)


/*
 * LW907F_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW907F_CTRL_CMD_NULL (0x907f0000) /* finn: Evaluated from "(FINN_GF100_REMAPPER_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*
 * LW907F_CTRL_CMD_SET_SURFACE
 *
 * This command sets the surface's parameters for the remapper
 *
 *   pMemory
 *       This is a CPU address space pointer (used in context of current process
 *       if not a kernel pointer)  used to specify which mapping is to be affected.
 *   hMemory 
 *       This field specifies the surface pointed at by the remapper.
 *   hSubDevice
 *      This parameter determines to which subdevice the remapper operation
 *      applies.
 *   size
 *       This field specifies the surface size in gob units.
 *   format
 *       This field contains pitch and bytes-per-pixel of the surface.
 *       pitch
 *           This field specifies the number of bytes that
 *           separates 2 pixels of same X coordinate in adjacent scanlines. The
 *           maximum pitch size is 64K (texels) * 16 (Bytes/texel) = 1 M bytes.
 *       bytes_pixel
 *           This field specifies bytes per pixel.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_STATE_IN_USE
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW907F_CTRL_CMD_SET_SURFACE (0x907f0101) /* finn: Evaluated from "(FINN_GF100_REMAPPER_REMAPPER_INTERFACE_ID << 8) | LW907F_CTRL_SET_SURFACE_PARAMS_MESSAGE_ID" */

#define LW907F_CTRL_SET_SURFACE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW907F_CTRL_SET_SURFACE_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 pMemory, 8);
    LwHandle hMemory;
    LwHandle hSubDevice;
    LwU32    size;
    LwU32    format;
} LW907F_CTRL_SET_SURFACE_PARAMS;

#define LW907F_CTRL_CMD_SET_SURFACE_FORMAT_MEM_SPACE               0:0
#define LW907F_CTRL_CMD_SET_SURFACE_FORMAT_MEM_SPACE_CLIENT (0x00000000)
#define LW907F_CTRL_CMD_SET_SURFACE_FORMAT_MEM_SPACE_USER   (0x00000001)

#define LW907F_CTRL_CMD_SET_SURFACE_FORMAT_PITCH                   20:1
#define LW907F_CTRL_CMD_SET_SURFACE_FORMAT_BYTES_PIXEL             27:25
#define LW907F_CTRL_CMD_SET_SURFACE_FORMAT_BYTES_PIXEL_1    (0x00000001)
#define LW907F_CTRL_CMD_SET_SURFACE_FORMAT_BYTES_PIXEL_2    (0x00000002)
#define LW907F_CTRL_CMD_SET_SURFACE_FORMAT_BYTES_PIXEL_4    (0x00000003)
#define LW907F_CTRL_CMD_SET_SURFACE_FORMAT_BYTES_PIXEL_8    (0x00000004)
#define LW907F_CTRL_CMD_SET_SURFACE_FORMAT_BYTES_PIXEL_16   (0x00000005)

/*
 * LW907F_CTRL_CMD_SET_BLOCKLINEAR
 * This command sets the block linear specific parameters for the remapper.
 *
 *   pMemory
 *       This is a CPU address space pointer (used in context of current process
 *       if not a kernel pointer)  used to specify which mapping is to be affected.
 *   hMemory
 *       Memory handle to heap allocation that will be remapped.
 *   hSubDevice
 *      This parameter determines to which subdevice the remapper operation
 *      applies.
 *   gob
 *       This field contains gob width and height.
 *       width
 *           This field specifies the log2 of gob width in texels.
 *       height
 *           This field specifies the log2 of gob height in texels.
 *   block
 *       This field contains block width, height, and depth.
 *       width
 *           This field specifies the log2 of block width in gobs.
 *       height
 *           This field specifies the log2 of block height in gobs.
 *       depth
 *           This field specifies the log2 of block depth in gobs.
 *   imageWH
 *       This field contains image width and height.
 *       width
 *           This field specifies the image width in blocks.
 *       height
 *           This field specifies the image height in blocks.
 *   log2ImageSlice
 *       This field specifies the Log base 2 of the number of bytes that
 *       separates 2 pixels with the same (x, y) coordinates in adjacent
 *       image slice in pitch-linear domain.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_STATE_IN_USE
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW907F_CTRL_CMD_SET_BLOCKLINEAR                     (0x907f0102) /* finn: Evaluated from "(FINN_GF100_REMAPPER_REMAPPER_INTERFACE_ID << 8) | LW907F_CTRL_SET_BLOCKLINEAR_PARAMS_MESSAGE_ID" */

#define LW907F_CTRL_SET_BLOCKLINEAR_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW907F_CTRL_SET_BLOCKLINEAR_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 pMemory, 8);
    LwHandle hMemory;
    LwHandle hSubDevice;
    LwU32    gob;
    LwU32    block;
    LwU32    imageWH;
    LwU32    log2ImageSlice;
} LW907F_CTRL_SET_BLOCKLINEAR_PARAMS;

#define LW907F_CTRL_CMD_SET_BLOCKLINEAR_GOB_MEM_SPACE              0:0
#define LW907F_CTRL_CMD_SET_BLOCKLINEAR_GOB_MEM_SPACE_CLIENT (0x00000000)
#define LW907F_CTRL_CMD_SET_BLOCKLINEAR_GOB_MEM_SPACE_USER   (0x00000001)

#define LW907F_CTRL_CMD_SET_BLOCKLINEAR_BLOCK_HEIGHT               28:26
#define LW907F_CTRL_CMD_SET_BLOCKLINEAR_BLOCK_HEIGHT_1       (0x00000000)
#define LW907F_CTRL_CMD_SET_BLOCKLINEAR_BLOCK_HEIGHT_2       (0x00000001)
#define LW907F_CTRL_CMD_SET_BLOCKLINEAR_BLOCK_HEIGHT_4       (0x00000002)
#define LW907F_CTRL_CMD_SET_BLOCKLINEAR_BLOCK_HEIGHT_8       (0x00000003)
#define LW907F_CTRL_CMD_SET_BLOCKLINEAR_BLOCK_HEIGHT_16      (0x00000004)
#define LW907F_CTRL_CMD_SET_BLOCKLINEAR_BLOCK_HEIGHT_32      (0x00000005)
#define LW907F_CTRL_CMD_SET_BLOCKLINEAR_BLOCK_DEPTH                31:29
#define LW907F_CTRL_CMD_SET_BLOCKLINEAR_BLOCK_DEPTH_1        (0x00000000)
#define LW907F_CTRL_CMD_SET_BLOCKLINEAR_BLOCK_DEPTH_2        (0x00000001)
#define LW907F_CTRL_CMD_SET_BLOCKLINEAR_BLOCK_DEPTH_4        (0x00000002)
#define LW907F_CTRL_CMD_SET_BLOCKLINEAR_BLOCK_DEPTH_8        (0x00000003)
#define LW907F_CTRL_CMD_SET_BLOCKLINEAR_BLOCK_DEPTH_16       (0x00000004)

#define LW907F_CTRL_CMD_SET_BLOCKLINEAR_IMAGEWH_HEIGHT             14:0
#define LW907F_CTRL_CMD_SET_BLOCKLINEAR_IMAGEWH_WIDTH              30:16

// This will set the log2ImageSlice to a large value that will set Z
// to 0 for 2D surfaces.
#define LW907F_CTRL_CMD_SET_BLOCKLINEAR_LOG2IMAGESLICE_0_Z   (0)


/*
 * LW907F_CTRL_CMD_START_BLOCKLINEAR_REMAP
 *
 * This command enables block linear remapping in the hardware.
 *
 *   hSubDevice
 *      This parameter determines to which subdevice the remapper operation
 *      applies.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 */
#define LW907F_CTRL_CMD_START_BLOCKLINEAR_REMAP              (0x907f0104) /* finn: Evaluated from "(FINN_GF100_REMAPPER_REMAPPER_INTERFACE_ID << 8) | LW907F_CTRL_START_BLOCKLINEAR_REMAP_PARAMS_MESSAGE_ID" */

#define LW907F_CTRL_START_BLOCKLINEAR_REMAP_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW907F_CTRL_START_BLOCKLINEAR_REMAP_PARAMS {
    LwHandle hSubDevice;
} LW907F_CTRL_START_BLOCKLINEAR_REMAP_PARAMS;

/*
 * LW907F_CTRL_CMD_STOP_BLOCKLINEAR_REMAP
 *
 * This command disables block linear remapping in the hardware.
 *
 *   hSubDevice
 *      This parameter determines to which subdevice the remapper operation
 *      applies.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_STATE_IN_USE
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 */
#define LW907F_CTRL_CMD_STOP_BLOCKLINEAR_REMAP (0x907f0105) /* finn: Evaluated from "(FINN_GF100_REMAPPER_REMAPPER_INTERFACE_ID << 8) | LW907F_CTRL_STOP_BLOCKLINEAR_REMAP_PARAMS_MESSAGE_ID" */

#define LW907F_CTRL_STOP_BLOCKLINEAR_REMAP_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW907F_CTRL_STOP_BLOCKLINEAR_REMAP_PARAMS {
    LwHandle hSubDevice;
} LW907F_CTRL_STOP_BLOCKLINEAR_REMAP_PARAMS;

/* _ctrl907f_h_ */
