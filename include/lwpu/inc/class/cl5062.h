/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2003-2004 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef _cl_lw50_context_surfaces_2d_h_
#define _cl_lw50_context_surfaces_2d_h_

/* This file is generated - do not edit. */

#include "lwtypes.h"

#define LW50_CONTEXT_SURFACES_2D    0x5062

typedef volatile struct _cl5062_tag0 {
    LwU32 SetObject;
    LwU32 Reserved_0x04[0x3F];
    LwU32 NoOperation;
    LwU32 Notify;
    LwU32 Reserved_0x108[0x2];
    LwU32 WaitForIdle;
    LwU32 Reserved_0x114[0xB];
    LwU32 PmTrigger;
    LwU32 Reserved_0x144[0xF];
    LwU32 SetContextDmaNotify;
    LwU32 SetContextDmaImageSource;
    LwU32 SetContextDmaImageDestin;
    LwU32 Reserved_0x18C[0x1D];
    LwU32 SetSrcMemoryLayout;
    LwU32 SetSrcBlockSize;
    LwU32 SetSrcWidth;
    LwU32 SetSrcHeight;
    LwU32 SetSrcDepth;
    LwU32 SetSrcLayer;
    LwU32 SetDstMemoryLayout;
    LwU32 SetDstBlockSize;
    LwU32 SetDstWidth;
    LwU32 SetDstHeight;
    LwU32 SetDstDepth;
    LwU32 SetDstLayer;
    LwU32 SetOffsetSourceUpper;
    LwU32 SetOffsetDestinUpper;
    LwU32 Reserved_0x238[0x32];
    LwU32 SetColorFormat;
    LwU32 SetPitch;
    LwU32 SetOffsetSource;
    LwU32 SetOffsetDestin;
} lw50_context_surfaces_2d_t;


#define LW5062_SET_OBJECT                                                                                  0x0000
#define LW5062_SET_OBJECT_POINTER                                                                            15:0

#define LW5062_NO_OPERATION                                                                                0x0100
#define LW5062_NO_OPERATION_V                                                                                31:0

#define LW5062_NOTIFY                                                                                      0x0104
#define LW5062_NOTIFY_TYPE                                                                                   31:0
#define LW5062_NOTIFY_TYPE_WRITE_ONLY                                                                  0x00000000
#define LW5062_NOTIFY_TYPE_WRITE_THEN_AWAKEN                                                           0x00000001

#define LW5062_WAIT_FOR_IDLE                                                                               0x0110
#define LW5062_WAIT_FOR_IDLE_V                                                                               31:0

#define LW5062_PM_TRIGGER                                                                                  0x0140
#define LW5062_PM_TRIGGER_V                                                                                  31:0

#define LW5062_SET_CONTEXT_DMA_NOTIFY                                                                      0x0180
#define LW5062_SET_CONTEXT_DMA_NOTIFY_HANDLE                                                                 31:0

#define LW5062_SET_CONTEXT_DMA_IMAGE_SOURCE                                                                0x0184
#define LW5062_SET_CONTEXT_DMA_IMAGE_SOURCE_HANDLE                                                           31:0

#define LW5062_SET_CONTEXT_DMA_IMAGE_DESTIN                                                                0x0188
#define LW5062_SET_CONTEXT_DMA_IMAGE_DESTIN_HANDLE                                                           31:0

#define LW5062_SET_SRC_MEMORY_LAYOUT                                                                       0x0200
#define LW5062_SET_SRC_MEMORY_LAYOUT_V                                                                        0:0
#define LW5062_SET_SRC_MEMORY_LAYOUT_V_BLOCKLINEAR                                                     0x00000000
#define LW5062_SET_SRC_MEMORY_LAYOUT_V_PITCH                                                           0x00000001

#define LW5062_SET_SRC_BLOCK_SIZE                                                                          0x0204
#define LW5062_SET_SRC_BLOCK_SIZE_WIDTH                                                                       3:0
#define LW5062_SET_SRC_BLOCK_SIZE_WIDTH_ONE_GOB                                                        0x00000000
#define LW5062_SET_SRC_BLOCK_SIZE_HEIGHT                                                                      7:4
#define LW5062_SET_SRC_BLOCK_SIZE_HEIGHT_ONE_GOB                                                       0x00000000
#define LW5062_SET_SRC_BLOCK_SIZE_HEIGHT_TWO_GOBS                                                      0x00000001
#define LW5062_SET_SRC_BLOCK_SIZE_HEIGHT_FOUR_GOBS                                                     0x00000002
#define LW5062_SET_SRC_BLOCK_SIZE_HEIGHT_EIGHT_GOBS                                                    0x00000003
#define LW5062_SET_SRC_BLOCK_SIZE_HEIGHT_SIXTEEN_GOBS                                                  0x00000004
#define LW5062_SET_SRC_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS                                                0x00000005
#define LW5062_SET_SRC_BLOCK_SIZE_DEPTH                                                                      11:8
#define LW5062_SET_SRC_BLOCK_SIZE_DEPTH_ONE_GOB                                                        0x00000000
#define LW5062_SET_SRC_BLOCK_SIZE_DEPTH_TWO_GOBS                                                       0x00000001
#define LW5062_SET_SRC_BLOCK_SIZE_DEPTH_FOUR_GOBS                                                      0x00000002
#define LW5062_SET_SRC_BLOCK_SIZE_DEPTH_EIGHT_GOBS                                                     0x00000003
#define LW5062_SET_SRC_BLOCK_SIZE_DEPTH_SIXTEEN_GOBS                                                   0x00000004
#define LW5062_SET_SRC_BLOCK_SIZE_DEPTH_THIRTYTWO_GOBS                                                 0x00000005

#define LW5062_SET_SRC_WIDTH                                                                               0x0208
#define LW5062_SET_SRC_WIDTH_V                                                                               31:0

#define LW5062_SET_SRC_HEIGHT                                                                              0x020c
#define LW5062_SET_SRC_HEIGHT_V                                                                              31:0

#define LW5062_SET_SRC_DEPTH                                                                               0x0210
#define LW5062_SET_SRC_DEPTH_V                                                                               31:0

#define LW5062_SET_SRC_LAYER                                                                               0x0214
#define LW5062_SET_SRC_LAYER_V                                                                               31:0

#define LW5062_SET_DST_MEMORY_LAYOUT                                                                       0x0218
#define LW5062_SET_DST_MEMORY_LAYOUT_V                                                                        0:0
#define LW5062_SET_DST_MEMORY_LAYOUT_V_BLOCKLINEAR                                                     0x00000000
#define LW5062_SET_DST_MEMORY_LAYOUT_V_PITCH                                                           0x00000001

#define LW5062_SET_DST_BLOCK_SIZE                                                                          0x021c
#define LW5062_SET_DST_BLOCK_SIZE_WIDTH                                                                       3:0
#define LW5062_SET_DST_BLOCK_SIZE_WIDTH_ONE_GOB                                                        0x00000000
#define LW5062_SET_DST_BLOCK_SIZE_HEIGHT                                                                      7:4
#define LW5062_SET_DST_BLOCK_SIZE_HEIGHT_ONE_GOB                                                       0x00000000
#define LW5062_SET_DST_BLOCK_SIZE_HEIGHT_TWO_GOBS                                                      0x00000001
#define LW5062_SET_DST_BLOCK_SIZE_HEIGHT_FOUR_GOBS                                                     0x00000002
#define LW5062_SET_DST_BLOCK_SIZE_HEIGHT_EIGHT_GOBS                                                    0x00000003
#define LW5062_SET_DST_BLOCK_SIZE_HEIGHT_SIXTEEN_GOBS                                                  0x00000004
#define LW5062_SET_DST_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS                                                0x00000005
#define LW5062_SET_DST_BLOCK_SIZE_DEPTH                                                                      11:8
#define LW5062_SET_DST_BLOCK_SIZE_DEPTH_ONE_GOB                                                        0x00000000
#define LW5062_SET_DST_BLOCK_SIZE_DEPTH_TWO_GOBS                                                       0x00000001
#define LW5062_SET_DST_BLOCK_SIZE_DEPTH_FOUR_GOBS                                                      0x00000002
#define LW5062_SET_DST_BLOCK_SIZE_DEPTH_EIGHT_GOBS                                                     0x00000003
#define LW5062_SET_DST_BLOCK_SIZE_DEPTH_SIXTEEN_GOBS                                                   0x00000004
#define LW5062_SET_DST_BLOCK_SIZE_DEPTH_THIRTYTWO_GOBS                                                 0x00000005

#define LW5062_SET_DST_WIDTH                                                                               0x0220
#define LW5062_SET_DST_WIDTH_V                                                                               31:0

#define LW5062_SET_DST_HEIGHT                                                                              0x0224
#define LW5062_SET_DST_HEIGHT_V                                                                              31:0

#define LW5062_SET_DST_DEPTH                                                                               0x0228
#define LW5062_SET_DST_DEPTH_V                                                                               31:0

#define LW5062_SET_DST_LAYER                                                                               0x022c
#define LW5062_SET_DST_LAYER_V                                                                               31:0

#define LW5062_SET_OFFSET_SOURCE_UPPER                                                                     0x0230
#define LW5062_SET_OFFSET_SOURCE_UPPER_V                                                                      7:0

#define LW5062_SET_OFFSET_DESTIN_UPPER                                                                     0x0234
#define LW5062_SET_OFFSET_DESTIN_UPPER_V                                                                      7:0

#define LW5062_SET_COLOR_FORMAT                                                                            0x0300
#define LW5062_SET_COLOR_FORMAT_V                                                                            31:0
#define LW5062_SET_COLOR_FORMAT_V_LE_Y8                                                                0x00000001
#define LW5062_SET_COLOR_FORMAT_V_LE_X1R5G5B5_Z1R5G5B5                                                 0x00000002
#define LW5062_SET_COLOR_FORMAT_V_LE_X1R5G5B5_O1R5G5B5                                                 0x00000003
#define LW5062_SET_COLOR_FORMAT_V_LE_R5G6B5                                                            0x00000004
#define LW5062_SET_COLOR_FORMAT_V_LE_Y16                                                               0x00000005
#define LW5062_SET_COLOR_FORMAT_V_LE_X8R8G8B8_Z8R8G8B8                                                 0x00000006
#define LW5062_SET_COLOR_FORMAT_V_LE_X8R8G8B8_O8R8G8B8                                                 0x00000007
#define LW5062_SET_COLOR_FORMAT_V_LE_X1A7R8G8B8_Z1A7R8G8B8                                             0x00000008
#define LW5062_SET_COLOR_FORMAT_V_LE_X1A7R8G8B8_O1A7R8G8B8                                             0x00000009
#define LW5062_SET_COLOR_FORMAT_V_LE_A8R8G8B8                                                          0x0000000A
#define LW5062_SET_COLOR_FORMAT_V_LE_Y32                                                               0x0000000B
#define LW5062_SET_COLOR_FORMAT_V_LE_X8B8G8R8_Z8B8G8R8                                                 0x0000000C
#define LW5062_SET_COLOR_FORMAT_V_LE_X8B8G8R8_O8B8G8R8                                                 0x0000000D
#define LW5062_SET_COLOR_FORMAT_V_LE_A8B8G8R8                                                          0x0000000E

#define LW5062_SET_PITCH                                                                                   0x0304
#define LW5062_SET_PITCH_SOURCE                                                                              15:0
#define LW5062_SET_PITCH_DESTIN                                                                             31:16

#define LW5062_SET_OFFSET_SOURCE                                                                           0x0308
#define LW5062_SET_OFFSET_SOURCE_V                                                                           31:0

#define LW5062_SET_OFFSET_DESTIN                                                                           0x030c
#define LW5062_SET_OFFSET_DESTIN_V                                                                           31:0

#endif /* _cl_lw50_context_surfaces_2d_h_ */
