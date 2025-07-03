/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2005-2006 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl824d_h_
#define _cl824d_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  G82_EXTERNAL_VIDEO_DECODER                                 (0x0000824D)
/* LwNotification[] elements */
#define LW824D_NOTIFIERS_NOTIFY                                     (0)
#define LW824D_NOTIFIERS_SET_VBI_FORMAT_NOTIFY(b)                   (1+(b))
#define LW824D_NOTIFIERS_GET_VBI_OFFSET_NOTIFY(b)                   (3+(b))
#define LW824D_NOTIFIERS_SET_IMAGE_FORMAT_NOTIFY(b)                 (5+(b))
#define LW824D_NOTIFIERS_GET_IMAGE_OFFSET_NOTIFY(b)                 (7+(b))
#define LW824D_NOTIFIERS_MAXCOUNT                                   (9)
/* LwNotification[] fields and values */
#define LW824D_NOTIFICATION_INFO16_FIELD_NOT_STARTED                (0x0000)
#define LW824D_NOTIFICATION_INFO16_FIELD_VALID_OFFSET               (0x0001)
#define LW824D_NOTIFICATION_INFO16_FIELD_DONE                       (0x0002)
#define LW824D_NOTIFICATION_STATUS_IN_PROGRESS                      (0x8000)
#define LW824D_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)
#define LW824D_NOTIFICATION_STATUS_ERROR_BAD_ARGUMENT               (0x2000)
#define LW824D_NOTIFICATION_STATUS_ERROR_ILWALID_STATE              (0x1000)
#define LW824D_NOTIFICATION_STATUS_ERROR_STATE_IN_USE               (0x0800)
#define LW824D_NOTIFICATION_STATUS_WARNING_ILWALID_DATA             (0x0001)
#define LW824D_NOTIFICATION_STATUS_DONE_SUCCESS                     (0x0000)
/* pio method data structure */
typedef volatile struct _cl824d_tag0 {
 LwV32 NoOperation;             /* ignored                           0100-0103*/
 LwV32 Notify;                  /* LW824D_NOTIFY_*                   0104-0107*/
 LwV32 StopTransferVbi;         /* LW824D_STOP_TRANSFER_VBI_VALUE    0108-010b*/
 LwV32 StopTransferImage;       /* LW824D_STOP_TRANSFER_IMAGE_VALUE  010c-010f*/
 LwV32 Reserved00[0x01c];
 LwV32 SetContextDmaNotifies;   /* LW01_CONTEXT_DMA                  0180-0183*/
 LwV32 SetContextDmaVbi[2];     /* LW01_CONTEXT_DMA                  0184-018b*/
 LwV32 SetContextDmaImage[2];   /* LW01_CONTEXT_DMA                  018c-0193*/
 LwV32 Reserved01[0x059];
 LwU32 SetImageConfig;          /* data width, task bit, null data   02f8-02fb*/
 LwU32 SetImageStartLine;       /* lines                             02fc-02ff*/
 struct {                       /* start of methods in array         0300-    */
  LwV32 size;                   /* height_firstLine U16_U16            0-   3*/
  LwU32 offset;                 /* byte offset of top-left pixel       4-   7*/
  LwV32 format;                 /* notify_field_pitch V8_V8_U16        8-   b*/
 } SetVbi[2];                   /* end of methods in array              -0317*/
 LwV32 GetVbiOffsetNotify[2];   /* LW824D_GET_VBI_OFFSET_NOTIFY_*    0318-031f*/
 struct {                       /* start of methods in array         0320-    */
  LwV32 sizeIn;                 /* height_width U16_U16 in pixels      0-   3*/
  LwV32 sizeOut;                /* height_width U16_U16 in pixels      4-   7*/
  LwU32 offset;                 /* byte offset of top-left pixel       8-   b*/
  LwV32 format;                 /* notify_field_pitch V8_V8_U16        c-   f*/
 } SetImage[2];                 /* end of methods in array              -033f*/
 LwV32 GetImageOffsetNotify[2]; /* LW824D_GET_IMAGE_OFFSET_NOTIFY_*  0340-0347*/
 LwV32 SetMemConfig;            /* LW824D_SET_MEM_CONFIG             0348-34b */
 LwV32 Reserved02[0x72a];
} Lw824dTypedef, Lw05ExternalVideoDecoder;
#define LW824D_TYPEDEF                                  Lw05ExternalVideoDecoder
/* dma method offsets, fields, and values */
#define LW824D_SET_OBJECT                                           (0x00000000)
#define LW824D_NO_OPERATION                                         (0x00000100)
#define LW824D_NOTIFY                                               (0x00000104)
#define LW824D_NOTIFY_WRITE_ONLY                                    (0x00000000)
#define LW824D_NOTIFY_WRITE_THEN_AWAKEN                             (0x00000001)
#define LW824D_STOP_TRANSFER_VBI                                    (0x00000108)
#define LW824D_STOP_TRANSFER_VBI_VALUE                              (0x00000000)
#define LW824D_STOP_TRANSFER_IMAGE                                  (0x0000010C)
#define LW824D_STOP_TRANSFER_IMAGE_VALUE                            (0x00000000)
#define LW824D_SET_CONTEXT_DMA_NOTIFIES                             (0x00000180)
#define LW824D_SET_CONTEXT_DMA_VBI(b)                               (0x00000184\
                                                                    +(b)*0x0004)
#define LW824D_SET_CONTEXT_DMA_IMAGE(b)                             (0x0000018C\
                                                                    +(b)*0x0004)
#define LW824D_SET_IMAGE_CONFIG                                     (0x000002f8)
#define LW824D_SET_IMAGE_CONFIG_DATA_WIDTH                          7:0
#define LW824D_SET_IMAGE_CONFIG_TASK                                8:8
#define LW824D_SET_IMAGE_CONFIG_TASK_A                              (0x00000000)
#define LW824D_SET_IMAGE_CONFIG_TASK_B                              (0x00000001)
#define LW824D_SET_IMAGE_CONFIG_ANC_MODE                            9:9
#define LW824D_SET_IMAGE_CONFIG_ANC_MODE_VBI                        (0x00000000)
#define LW824D_SET_IMAGE_CONFIG_ANC_MODE_ANC                        (0x00000001)
#define LW824D_SET_IMAGE_CONFIG_NULL_DATA                           13:12
#define LW824D_SET_IMAGE_CONFIG_NULL_DATA_DISABLED                  (0x00000000)
#define LW824D_SET_IMAGE_CONFIG_NULL_DATA_BYTE_ENABLED              (0x00000001)
#define LW824D_SET_IMAGE_CONFIG_NULL_DATA_LINE_ENABLED              (0x00000002)
#define LW824D_SET_IMAGE_CONFIG_NULL_VALUE                          31:16
#define LW824D_SET_IMAGE_START_LINE                                 (0x000002FC)
#define LW824D_SET_VBI(b)                                           (0x00000300\
                                                                    +(b)*0x000C)
#define LW824D_SET_VBI_SIZE(b)                                      (0x00000300\
                                                                    +(b)*0x000C)
#define LW824D_SET_VBI_SIZE_FIRST_LINE                              15:0
#define LW824D_SET_VBI_SIZE_HEIGHT                                  31:16
#define LW824D_SET_VBI_OFFSET(b)                                    (0x00000304\
                                                                    +(b)*0x000C)
#define LW824D_SET_VBI_FORMAT(b)                                    (0x00000308\
                                                                    +(b)*0x000C)
#define LW824D_SET_VBI_FORMAT_PITCH                                 15:0
#define LW824D_SET_VBI_FORMAT_FIELD                                 23:16
#define LW824D_SET_VBI_FORMAT_FIELD_PROGRESSIVE                     (0x00000000)
#define LW824D_SET_VBI_FORMAT_FIELD_EVEN_FIELD                      (0x00000001)
#define LW824D_SET_VBI_FORMAT_FIELD_ODD_FIELD                       (0x00000002)
#define LW824D_SET_VBI_FORMAT_NOTIFY                                31:24
#define LW824D_SET_VBI_FORMAT_NOTIFY_WRITE_ONLY                     (0x00000000)
#define LW824D_SET_VBI_FORMAT_NOTIFY_WRITE_THEN_AWAKEN              (0x00000001)
#define LW824D_GET_VBI_OFFSET_NOTIFY(b)                             (0x00000318\
                                                                    +(b)*0x0004)
#define LW824D_GET_VBI_OFFSET_NOTIFY_WRITE_ONLY                     (0x00000000)
#define LW824D_GET_VBI_OFFSET_NOTIFY_WRITE_THEN_AWAKEN              (0x00000001)
#define LW824D_SET_IMAGE(b)                                         (0x00000320\
                                                                    +(b)*0x0010)
#define LW824D_SET_IMAGE_SIZE_IN(b)                                 (0x00000320\
                                                                    +(b)*0x0010)
#define LW824D_SET_IMAGE_SIZE_IN_WIDTH                              15:0
#define LW824D_SET_IMAGE_SIZE_IN_HEIGHT                             31:16
#define LW824D_SET_IMAGE_SIZE_OUT(b)                                (0x00000324\
                                                                    +(b)*0x0010)
#define LW824D_SET_IMAGE_SIZE_OUT_WIDTH                             15:0
#define LW824D_SET_IMAGE_SIZE_OUT_HEIGHT                            31:16
#define LW824D_SET_IMAGE_OFFSET(b)                                  (0x00000328\
                                                                    +(b)*0x0010)
#define LW824D_SET_IMAGE_FORMAT(b)                                  (0x0000032C\
                                                                    +(b)*0x0010)
#define LW824D_SET_IMAGE_FORMAT_PITCH                               15:0
#define LW824D_SET_IMAGE_FORMAT_FIELD                               23:16
#define LW824D_SET_IMAGE_FORMAT_FIELD_PROGRESSIVE                   (0x00000000)
#define LW824D_SET_IMAGE_FORMAT_FIELD_EVEN_FIELD                    (0x00000001)
#define LW824D_SET_IMAGE_FORMAT_FIELD_ODD_FIELD                     (0x00000002)
#define LW824D_SET_IMAGE_FORMAT_NOTIFY                              31:24
#define LW824D_SET_IMAGE_FORMAT_NOTIFY_WRITE_ONLY                   (0x00000000)
#define LW824D_SET_IMAGE_FORMAT_NOTIFY_WRITE_THEN_AWAKEN            (0x00000001)
#define LW824D_GET_IMAGE_OFFSET_NOTIFY(b)                           (0x00000340\
                                                                    +(b)*0x0004)
#define LW824D_GET_IMAGE_OFFSET_NOTIFY_WRITE_ONLY                   (0x00000000)
#define LW824D_GET_IMAGE_OFFSET_NOTIFY_WRITE_THEN_AWAKEN            (0x00000001)
#define LW824D_SET_MEM_CONFIG                                       (0x00000348)

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl824d_h_ */
