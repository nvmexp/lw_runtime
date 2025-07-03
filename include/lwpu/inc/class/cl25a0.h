/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2001-2002 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl25a0_h_
#define _cl25a0_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

/* class LW25_MULTICHIP_VIDEO_SPLIT */
#define LW25_MULTICHIP_VIDEO_SPLIT                                 (0x000025A0)

/* LwNotifidation[] elements */
#define LW25AO_NOTIFIERS_NOTIFY                                     (0)
#define LW25AO_NOTIFIERS_SET_VIDEO_FORMAT                           (1)
#define LW25A0_NOTIFIERS_MAXCOUNT                                   2

/* LwNotification[] fields and values */
#define LW25A0_NOTIFICATION_STATUS_IN_PROGRESS                      (0x8000)
#define LW25A0_NOTIFICATION_STATUS_ERROR_BAD_ARGUMENT               (0x2000)
#define LW25A0_NOTIFICATION_STATUS_ERROR_ILWALID_STATE              (0x4000)
#define LW25A0_NOTIFICATION_STATUS_ERROR_STATE_IN_USE               (0x8000)
#define LW25A0_NOTIFICATION_STATUS_DONE_SUCCESS                     (0x0000)

#define LW25A0_NUMBER_OF_DEVICES                                    (7)

/* pio method data structure */
typedef volatile struct _cl25a0_tag0 {        
 LwV32 NoOperation;             /* ignored                          0100-0103*/
 LwV32 Notify;                  /* LW25A0_NOTIFY_*                  0104-0107*/
 LwV32 StopFormat;              /* LW25A0_STOP                      0108-010B*/
 LwV32 Reserved00[0x01D];
 LwV32 SetContextDmaNotifies;   /* LW25A0_CONTEXT_DMA               0180-0183*/
 LwV32 Reserved01[0x5F];
 struct {                       /* LW25A0_SET_FORMAT                0300-337 */
     LwV32 mode;                
     LwV32 field;              
 } SetSplit[LW25A0_NUMBER_OF_DEVICES];
 LwV32 SetFormat;               /* LW25A0_SET_FORMAT                0338-033B*/
 LwV32 SetDacObject;            /* LW25A0_SET_DAC_OBJECT            033C-033F*/
 LwV32 Reserved02[0x730];                                         /*0340-1FFF*/
} Lw25A0Typedef, Lw25MultichipVideoSplit;

/* dma method offsets, fields, and values */
#define LW25A0_SET_OBJECT                                         (0x00000000)
#define LW25A0_NO_OPERATION                                       (0x00000100)
#define LW25A0_NOTIFY                                             (0x00000104)
#define LW25A0_STOP_FORMAT                                        (0x00000108)
#define LW25A0_SET_CONTEXT_DMA_NOTIFIES                           (0x00000180)

#define LW25A0_SET_SPLIT_MODE(b)                                  (0x00000300\
                                                                  +(b)*0x0008)
#define LW25A0_SET_SPLIT_MODE_SCAN                                3:0
#define LW25A0_SET_SPLIT_MODE_SCAN_NEVER                          (0x00000000)
#define LW25A0_SET_SPLIT_MODE_SCAN_ODD                            (0x00000001)
#define LW25A0_SET_SPLIT_MODE_SCAN_EVEN                           (0x00000002)
#define LW25A0_SET_SPLIT_MODE_SCAN_ALWAYS                         (0x00000003)
#define LW25A0_SET_SPLIT_MODE_SCAN_PROG                           (0x00000004)
#define LW25A0_SET_SPLIT_MODE_SCAN_AFR                            (0x00000005)
#define LW25A0_SET_SPLIT_MODE_SCAN_SINGLE                         (0x00000006)
#define LW25A0_SET_SPLIT_MODE_SCAN_AA                             (0x00000007)
#define LW25A0_SET_SPLIT_MODE_SCAN_AFROFSFR                       (0x00000008)
#define LW25A0_SET_SPLIT_MODE_SCAN_AFROFAA                        (0x00000009)
#define LW25A0_SET_SPLIT_MODE_SCAN_SFROFAA                        (0x0000000A)

#define LW25A0_SET_SPLIT_MODE_AAROP                               8:8
#define LW25A0_SET_SPLIT_MODE_AAROP_LINEAR                        (0x00000000)
#define LW25A0_SET_SPLIT_MODE_AAROP_GAMMA_COMP                    (0x00000001)

#define LW25A0_SET_SPLIT_FIELD(b)                                 (0x00000304\
                                                                  +(b)*0x0008)
#define LW25A0_SET_SPLIT_FIELD_START                              15:0
#define LW25A0_SET_SPLIT_FIELD_END                                31:16
#define LW25A0_SET_FORMAT                                         (0x00000338)
#define LW25A0_SET_FORMAT_WHEN                                    31:31
#define LW25A0_SET_FORMAT_WHEN_IMMEDIATELY                        (0x00000000)
#define LW25A0_SET_FORMAT_WHEN_BLANK                              (0X00000001)
#define LW25A0_SET_DAC_OBJECT                                     (0x0000033C)
#define LW25A0_SET_HW_FLIP                                        (0x00000340)
#define LW25A0_SET_HW_FLIP_EN                                     1:0
#define LW25A0_SET_HW_FLIP_EN_DISABLE                             (0x00000000)
#define LW25A0_SET_HW_FLIP_EN_ENABLE                              (0x00000001)
#define LW25A0_SET_AFR_GPU_ACTIVE                                 (0x00000344)

//
// b defines each subdevice index but not a group of subdevices.  By default, 
// peer masks will be set in groups of 2 devices. i.e. PeerMask[0] = 0x3, 
// PeerMask[1] = 0x3, PeerMask[2] = 0xC, PeerMask[3] = 0xC, PeerMask[4] = 0x30, etc.
//
#define LW25A0_SET_PEER_MASK(b)                                   (0x00000348\
                                                                  +(b)*0x0004)



#ifdef __cplusplus
};     /* extern "C" */
#endif

typedef struct
{
    LwU32   logicalHeadId;
    LwU32   deviceMask; // Everyone other than display driver should pass 0 as the device mask
    LwU32   caps;       // Capabilities
                        // Note: Any new fields added must pad to 64 bit boundary
} LW25A0_ALLOCATION_PARAMETERS;

// Class-specific allocation capabilities
#define LW25A0_ALLOCATION_PARAMETERS_CAPS_AA_CAPABLE                                0:0  /* RW--F */
#define LW25A0_ALLOCATION_PARAMETERS_CAPS_AA_CAPABLE_FALSE                  (0x00000000) /* RW--V */
#define LW25A0_ALLOCATION_PARAMETERS_CAPS_AA_CAPABLE_TRUE                   (0x00000001) /* RW--V */

#endif /* _cl25A0_h_ */

