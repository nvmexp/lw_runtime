/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2001-2001 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl8274_h_
#define _cl8274_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  LW82_VIDEOACCEL                                            (0x00008274)
#define  LW8274                                            0x00001fff:0x00000000
/* LwNotification[] elements */
#define LW8274_NOTIFIERS_NOTIFY                                     (0)
#define LW8274_NOTIFIERS_EOB                                        (1)
#define LW8274_NOTIFIERS_SYNC                                       (2)
#define LW8274_NOTIFIERS_GET_SESSION_KEY                            (3)
#define LW8274_NOTIFIERS_MAXCOUNT                                   (4)
/* LwNotification[] fields and values */
#define LW8274_NOTIFICATION_STATUS_IN_PROGRESS                      (0x8000)
#define LW8274_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)
#define LW8274_NOTIFICATION_STATUS_ERROR_BAD_ARGUMENT               (0x2000)
#define LW8274_NOTIFICATION_STATUS_ERROR_ILWALID_STATE              (0x1000)
#define LW8274_NOTIFICATION_STATUS_ERROR_STATE_IN_USE               (0x0800)
#define LW8274_NOTIFICATION_STATUS_DONE_SUCCESS                     (0x0000)
/* mpeg method data structure */
typedef volatile struct _cl8274_tag0 {
 LwV32 Instance;                /* instance / set object            0000-0003*/
 LwV32 Reserved00[0x003];
 LwV32 PioFree;                 /* pio free count                   0010-0013*/
 LwV32 PioInfo;                 /* pio info/idle                    0014-0017*/
 LwV32 Reserved01[0x03c];
 LwV32 Synchronize;             /* synchronize                      0108-010b*/
 LwV32 Reserved02[0x00d];
 LwV32 QuerySessionKey;         /* generate session key             0140-0143*/
 LwV32 GetSessionKey;           /* get session key                  0144-0147*/
 LwV32 SetSessionKey;           /* set session key                  0148-014b*/
 LwV32 Reserved03[0x005];
 LwV32 SetImageFormat;          /* set the image stride and chroma  0160-0163*/
 LwV32 SetPicSize;              /* set the picture size             0164-0167*/
 LwV32 SetPicMBSize;            /* set the picture mb size          0168-016b*/
 LwV32 Reserved04[0x005];
 LwV32 SetContextDmaNotify;     /* set context dma notify           0180-0183*/
 LwV32 Reserved05[0x003];
 LwV32 SetContextDmaCommand;    /* set context dma of cmd buffer    0190-0193*/
 LwV32 Reserved06[0x003];
 LwV32 SetContextDmaData;       /* set context dma of data buffer   01a0-01a3*/
 LwV32 Reserved07[0x003];
 LwV32 SetContextDmaImage;      /* set context dma of image buffer  01b0-01b3*/
 LwV32 Reserved08[0x003];
 LwV32 SetContextDmaSemaphore;  /* Set context dma of the semaphore 01c0-01c3*/
 LwV32 Reserved09[0x00f];
 struct {
     LwV32 Y;
     LwV32 C;
 } SetImageOffset[8];           /* set offset of image buffers      0200-023f*/
 LwV32 Reserved0a[0x030];
 LwV32 SetPicIndex;             /* select image buffer to use       0300-0303*/
 LwV32 SetPicParam0;            /* set picture parameter 0          0304-0307*/
 LwV32 SetPicParam1;            /* set picture parameter 1          0308-030b*/
 LwV32 Reserved0b[0x001];
 LwV32 SetDataParam0;           /* set data parameter 0             0310-0313*/
 LwV32 SetDataParam1;           /* set data parameter 1             0314-0317*/
 LwV32 Reserved0c[0x01];        /*                                  0318-031b*/
 LwV32 SetSemaphoreOffset;      /* set semaphore offset             031c-031f*/
 LwV32 SetSemaphoreRelease;     /* set semaphore release            0320-0323*/
 LwV32 Reserved0d[0x017];
 LwV32 SetCommandStart;         /* set command start                0380-0383*/
 LwV32 SetCommandSize;          /* set command size                 0384-0387*/
 LwV32 SetCommandMBCnt;         /* set command mb count             0388-038b*/
 LwV32 SetDataStart;            /* set data start offset            038c-038f*/
 LwV32 SetDataSize;             /* set size of data                 0390-0393*/
 LwV32 Execute;                 /* execute the current buffer       0394-0397*/
 LwV32 Reserved0e[0x71A];
} Lw8274Typedef, Lw82Mpeg;
#define LW8274_TYPEDEF                                              Lw8274Mpeg
/* mpeg channel method offsets, fields, and values */
#define LW8274_INSTANCE                                           (0x00000000)
#define LW8274_PIO_FREE                                           (0x00000010)
#define LW8274_PIO_FREE_COUNT                                     11:2
#define LW8274_PIO_FREE_COUNT_FIFO_FULL                           0x0000
#define LW8274_PIO_INFO                                           (0x00000014)
#define LW8274_PIO_INFO_PIO                                       0:0
#define LW8274_PIO_INFO_PIO_EMPTY_AND_IDLE                        0x0000
#define LW8274_PIO_INFO_PIO_BUSY                                  0x0001
#define LW8274_SYNCHRONIZE                                        (0x00000108)
#define LW8274_SYNCHRONIZE_STYLE                                  31:0
#define LW8274_SYNCHRONIZE_STYLE_NO_OPERATION                     0x0000
#define LW8274_SYNCHRONIZE_STYLE_WAIT4IDLE                        0x0001
#define LW8274_SYNCHRONIZE_STYLE_WAIT4IDLE_NOTIFY                 0x0002
#define LW8274_SYNCHRONIZE_STYLE_WAIT4IDLE_NOTIFY_AWAKE           0x0003
#define LW8274_QUERY_SESSION_KEY                                  (0x00000140)
#define LW8274_QUERY_SESSION_KEY_HOST_KEY                         31:0
#define LW8274_GET_SESSION_KEY                                    (0x00000144)
#define LW8274_GET_SESSION_KEY_PARAMETER                          31:0
#define LW8274_GET_SESSION_KEY_PARAMETER_VALUE                    0x0000
#define LW8274_SET_ENCRYPTION_KEY                                 (0x00000148)
#define LW8274_SET_ENCRYPTION_KEY_FRAME_KEY                       15:0
#define LW8274_SET_IMAGE_FORMAT                                   (0x00000160)
#define LW8274_SET_IMAGE_FORMAT_STRIDE                            15:0
#define LW8274_SET_IMAGE_FORMAT_CHROMA                            31:16
#define LW8274_SET_IMAGE_FORMAT_CHROMA_420                        0x0001
#define LW8274_SET_IMAGE_FORMAT_CHROMA_422                        0x0002
#define LW8274_SET_IMAGE_FORMAT_CHROMA_444                        0x0003
#define LW8274_SET_PIC_SIZE                                       (0x00000164)
#define LW8274_SET_PIC_SIZE_WIDTH                                 15:0
#define LW8274_SET_PIC_SIZE_HEIGHT                                31:16
#define LW8274_SET_PIC_MBSIZE                                     (0x00000168)
#define LW8274_SET_PIC_MBSIZE_WIDTH                               15:0
#define LW8274_SET_PIC_MBSIZE_HEIGHT                              31:16
#define LW8274_SET_CONTEXT_DMA_NOTIFY                             (0x00000180)
#define LW8274_SET_CONTEXT_DMA_NOTIFY_PARAMETER                   31:0
#define LW8274_SET_CONTEXT_DMA_COMMAND                            (0x00000190)
#define LW8274_SET_CONTEXT_DMA_COMMAND_PARAMETER                  31:0
#define LW8274_SET_CONTEXT_DMA_DATA                               (0x000001A0)
#define LW8274_SET_CONTEXT_DMA_DATA_PARAMETER                     31:0
#define LW8274_SET_CONTEXT_DMA_IMAGE                              (0x000001B0)
#define LW8274_SET_CONTEXT_DMA_IMAGE_PARAMETER                    31:0
#define LW8274_SET_IMAGE_OFFSET_Y(i)                              (0x00000200\
                                                                  +(i)*0x0008)
#define LW8274_SET_IMAGE_OFFSET_Y__SIZE_1                         8
#define LW8274_SET_IMAGE_OFFSET_Y_PARAMETER                       31:0
#define LW8274_SET_IMAGE_OFFSET_C(i)                              (0x00000204\
                                                                  +(i)*0x0008)
#define LW8274_SET_IMAGE_OFFSET_C__SIZE_1                         8
#define LW8274_SET_IMAGE_OFFSET_C_PARAMETER                       31:0
#define LW8274_SET_PIC_INDEX                                      (0x00000300)
#define LW8274_SET_PIC_INDEX_DECODED                              2:0
#define LW8274_SET_PIC_INDEX_FORWARD                              18:16
#define LW8274_SET_PIC_INDEX_BACKWARD                             26:24
#define LW8274_SET_PICPARAM0                                      (0x00000304)
#define LW8274_SET_PICPARAM0_STRUCTURE                            1:0
#define LW8274_SET_PICPARAM0_STRUCTURE_TOP_FIELD                  0x1
#define LW8274_SET_PICPARAM0_STRUCTURE_BOTTOM_FIELD               0x2
#define LW8274_SET_PICPARAM0_STRUCTURE_FRAME                      0x3
#define LW8274_SET_PICPARAM0_SECONDFIELD                          8:8
#define LW8274_SET_PICPARAM0_SECONDFIELD_FALSE                    0x0
#define LW8274_SET_PICPARAM0_SECONDFIELD_TRUE                     0x1
#define LW8274_SET_PICPARAM0_INTRA                                16:16
#define LW8274_SET_PICPARAM0_INTRA_FALSE                          0x0
#define LW8274_SET_PICPARAM0_INTRA_TRUE                           0x1
#define LW8274_SET_PICPARAM0_BACKWARDPREDICTION                   24:24
#define LW8274_SET_PICPARAM0_BACKWARDPREDICTION_FALSE             0x0
#define LW8274_SET_PICPARAM0_BACKWARDPREDICTION_TRUE              0x1
#define LW8274_SET_PICPARAM1                                      (0x00000308)
#define LW8274_SET_PICPARAM1_SPATIALRESID                         0:0
#define LW8274_SET_PICPARAM1_SPATIALRESID_16BIT                   0x0
#define LW8274_SET_PICPARAM1_SPATIALRESID_8BIT                    0x1
#define LW8274_SET_PICPARAM1_OVERFLOWBLOCKS                       8:8
#define LW8274_SET_PICPARAM1_OVERFLOWBLOCKS_NOTALLOWED            0x0
#define LW8274_SET_PICPARAM1_OVERFLOWBLOCKS_ALLOWED               0x1
#define LW8274_SET_PICPARAM1_EXTRAPOLATION                        16:16
#define LW8274_SET_PICPARAM1_EXTRAPOLATION_DISABLED               0x0
#define LW8274_SET_PICPARAM1_EXTRAPOLATION_ENABLED                0x1
#define LW8274_SET_PICPARAM1_SCANMETHOD                           24:24
#define LW8274_SET_PICPARAM1_SCANMETHOD_PERMACROBLOCK             0x0
#define LW8274_SET_PICPARAM1_SCANMETHOD_FIXED_ZIGZAG              0x4
#define LW8274_SET_PICPARAM1_SCANMETHOD_FIXED_ALTVERT             0x5
#define LW8274_SET_PICPARAM1_SCANMETHOD_FIXED_ALTHORIZ            0x6
#define LW8274_SET_PICPARAM1_SCANMETHOD_FIXED_ARBITRARY           0x7
#define LW8274_SET_DATAPARAM0                                     (0x00000310)
#define LW8274_SET_DATAPARAM0_MCFORMAT                            0:0
#define LW8274_SET_DATAPARAM0_MCFORMAT_S16BIT                     0x0
#define LW8274_SET_DATAPARAM0_MCFORMAT_S8BIT                      0x1
#define LW8274_SET_DATAPARAM0_IDCTENCRYPTED                       8:8
#define LW8274_SET_DATAPARAM0_IDCTENCRYPTED_FALSE                 0x0
#define LW8274_SET_DATAPARAM0_IDCTENCRYPTED_TRUE                  0x1
#define LW8274_SET_DATAPARAM1                                     (0x00000314)
#define LW8274_SET_DATAPARAM1_HWIDCT                              0:0
#define LW8274_SET_DATAPARAM1_HWIDCT_DISABLE                      0x0
#define LW8274_SET_DATAPARAM1_HWIDCT_ENABLE                       0x1
#define LW8274_SET_CONTEXT_DMA_SEMAPHORE                          (0x000001C0)
#define LW8274_SET_CONTEXT_DMA_SEMAPHORE_HANDLE                   31:0
#define LW8274_SET_SEMAPHORE_OFFSET                               (0x0000031C)
#define LW8274_SET_SEMAPHORE_OFFSET_VALUE                         31:0
#define LW8274_SET_SEMAPHORE_RELEASE                              (0x00000320)
#define LW8274_SET_SEMAPHORE_RELEASE_VALUE                        31:0
#define LW8274_SET_COMMAND_START                                  (0x00000380)
#define LW8274_SET_COMMAND_START_OFFSET                           31:0
#define LW8274_SET_COMMAND_SIZE                                   (0x00000384)
#define LW8274_SET_COMMAND_SIZE_PARAMETER                         31:0
#define LW8274_SET_COMMAND_MBCNT                                  (0x00000388)
#define LW8274_SET_COMMAND_MBCNT_VALUE                            31:0
#define LW8274_SET_DATA_START                                     (0x0000038C)
#define LW8274_SET_DATA_START_OFFSET                              31:0
#define LW8274_SET_DATA_SIZE                                      (0x00000390)
#define LW8274_SET_DATA_SIZE_PARAMETER                            31:0
#define LW8274_EXELWTE                                            (0x00000394)
#define LW8274_EXELWTE_TYPE                                       31:0
#define LW8274_EXELWTE_TYPE_LEGACY                                0x1

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl8274_h_ */
