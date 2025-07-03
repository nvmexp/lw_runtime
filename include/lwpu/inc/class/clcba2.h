// WARNING!!! THIS HEADER INCLUDES SOFTWARE METHODS!!!
// ********** DO NOT USE IN HW TREE.  ********** 
/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2021 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


#include "lwtypes.h"

#ifndef _clcba2_h_
#define _clcba2_h_

#ifdef __cplusplus
extern "C" {
#endif

#define HOPPER_SEC2_WORK_LAUNCH_A                                               (0x0000CBA2)

typedef volatile struct _clcba2_tag0 {
    LwV32 Reserved00[0x100];
    LwV32 DecryptCopySrcAddrHi;                                                 // 0x00000400 - 0x00000403
    LwV32 DecryptCopySrcAddrLo;                                                 // 0x00000404 - 0x00000407
    LwV32 DecryptCopyDstAddrHi;                                                 // 0x00000408 - 0x0000040B
    LwV32 DecryptCopyDstAddrLo;                                                 // 0x0000040c - 0x0000040F
    LwU32 DecryptCopySize;                                                      // 0x00000410 - 0x00000413
    LwU32 DecryptCopyAuthTagAddrHi;                                             // 0x00000414 - 0x00000417
    LwU32 DecryptCopyAuthTagAddrLo;                                             // 0x00000418 - 0x0000041B
    LwV32 DigestAddrHi;                                                         // 0x0000041C - 0x0000041F
    LwV32 DigestAddrLo;                                                         // 0x00000420 - 0x00000423
    LwV32 Reserved01[0x7];
    LwV32 SemaphoreA;                                                           // 0x00000440 - 0x00000443
    LwV32 SemaphoreB;                                                           // 0x00000444 - 0x00000447
    LwV32 SemaphoreSetPayloadLower;                                             // 0x00000448 - 0x0000044B
    LwV32 SemaphoreSetPayloadUppper;                                            // 0x0000044C - 0x0000044F
    LwV32 SemaphoreD;                                                           // 0x00000450 - 0x00000453
    LwU32 Reserved02[0x7];
    LwV32 Execute;                                                              // 0x00000470 - 0x00000473
    LwV32 Reserved03[0x23];
} LWCBA2_HOPPER_SEC2_WORK_LAUNCH_AControlPio;

#define LWCBA2_DECRYPT_COPY_SRC_ADDR_HI                                         (0x00000400)
#define LWCBA2_DECRYPT_COPY_SRC_ADDR_HI_DATA                                    24:0
#define LWCBA2_DECRYPT_COPY_SRC_ADDR_LO                                         (0x00000404)
#define LWCBA2_DECRYPT_COPY_SRC_ADDR_LO_DATA                                    31:4
#define LWCBA2_DECRYPT_COPY_DST_ADDR_HI                                         (0x00000408)
#define LWCBA2_DECRYPT_COPY_DST_ADDR_HI_DATA                                    24:0
#define LWCBA2_DECRYPT_COPY_DST_ADDR_LO                                         (0x0000040c)
#define LWCBA2_DECRYPT_COPY_DST_ADDR_LO_DATA                                    31:4
#define LWCBA2_DECRYPT_COPY_SIZE                                                (0x00000410)
#define LWCBA2_DECRYPT_COPY_SIZE_DATA                                           31:2
#define LWCBA2_DECRYPT_COPY_AUTH_TAG_ADDR_HI                                    (0x00000414)
#define LWCBA2_DECRYPT_COPY_AUTH_TAG_ADDR_HI_DATA                               24:0
#define LWCBA2_DECRYPT_COPY_AUTH_TAG_ADDR_LO                                    (0x00000418)
#define LWCBA2_DECRYPT_COPY_AUTH_TAG_ADDR_LO_DATA                               31:4
#define LWCBA2_DIGEST_ADDR_HI                                                   (0x0000041C)
#define LWCBA2_DIGEST_ADDR_HI_DATA                                              24:0
#define LWCBA2_DIGEST_ADDR_LO                                                   (0x00000420)
#define LWCBA2_DIGEST_ADDR_LO_DATA                                              31:4
#define LWCBA2_SEMAPHORE_A                                                      (0x00000440)
#define LWCBA2_SEMAPHORE_A_UPPER                                                24:0
#define LWCBA2_SEMAPHORE_B                                                      (0x00000444)
#define LWCBA2_SEMAPHORE_B_LOWER                                                31:2
#define LWCBA2_SET_SEMAPHORE_PAYLOAD_LOWER                                      (0x00000448)
#define LWCBA2_SET_SEMAPHORE_PAYLOAD_LOWER_DATA                                 31:0
#define LWCBA2_SET_SEMAPHORE_PAYLOAD_UPPER                                      (0x0000044C)
#define LWCBA2_SET_SEMAPHORE_PAYLOAD_UPPER_DATA                                 31:0
#define LWCBA2_SEMAPHORE_D                                                      (0x00000450)
#define LWCBA2_SEMAPHORE_D_NOTIFY_INTR                                          0:0
#define LWCBA2_SEMAPHORE_D_NOTIFY_INTR_DISABLE                                  (0x00000000)
#define LWCBA2_SEMAPHORE_D_NOTIFY_INTR_ENABLE                                   (0x00000001)
#define LWCBA2_SEMAPHORE_D_PAYLOAD_SIZE                                         1:1
#define LWCBA2_SEMAPHORE_D_PAYLOAD_SIZE_32_BIT                                  (0x00000000)
#define LWCBA2_SEMAPHORE_D_PAYLOAD_SIZE_64_BIT                                  (0x00000001)
#define LWCBA2_SEMAPHORE_D_TIMESTAMP                                            2:2
#define LWCBA2_SEMAPHORE_D_TIMESTAMP_DISABLE                                    (0x00000000)
#define LWCBA2_SEMAPHORE_D_TIMESTAMP_ENABLE                                     (0x00000001)
#define LWCBA2_EXELWTE                                                          (0x00000470)
#define LWCBA2_EXELWTE_NOTIFY                                                   0:0
#define LWCBA2_EXELWTE_NOTIFY_DISABLE                                           (0x00000000)
#define LWCBA2_EXELWTE_NOTIFY_ENABLE                                            (0x00000001)
#define LWCBA2_EXELWTE_NOTIFY_ON                                                1:1
#define LWCBA2_EXELWTE_NOTIFY_ON_END                                            (0x00000000)
#define LWCBA2_EXELWTE_NOTIFY_ON_BEGIN                                          (0x00000001)
#define LWCBA2_EXELWTE_FLUSH_DISABLE                                            2:2
#define LWCBA2_EXELWTE_FLUSH_DISABLE_FALSE                                      (0x00000000)
#define LWCBA2_EXELWTE_FLUSH_DISABLE_TRUE                                       (0x00000001)
#define LWCBA2_EXELWTE_NOTIFY_INTR                                              3:3
#define LWCBA2_EXELWTE_NOTIFY_INTR_DISABLE                                      (0x00000000)
#define LWCBA2_EXELWTE_NOTIFY_INTR_ENABLE                                       (0x00000001)
#define LWCBA2_EXELWTE_PAYLOAD_SIZE                                             4:4
#define LWCBA2_EXELWTE_PAYLOAD_SIZE_32_BIT                                      (0x00000000)
#define LWCBA2_EXELWTE_PAYLOAD_SIZE_64_BIT                                      (0x00000001)
#define LWCBA2_EXELWTE_TIMESTAMP                                                5:5
#define LWCBA2_EXELWTE_TIMESTAMP_DISABLE                                        (0x00000000)
#define LWCBA2_EXELWTE_TIMESTAMP_ENABLE                                         (0x00000001)

// Errors
#define LWCBA2_ERROR_NONE                                                       (0x00000000)
#define LWCBA2_ERROR_DECRYPT_COPY_SRC_ADDR_MISALIGNED_POINTER                   (0x00000001)
#define LWCBA2_ERROR_DECRYPT_COPY_DEST_ADDR_MISALIGNED_POINTER                  (0x00000002)
#define LWCBA2_ERROR_DECRYPT_COPY_AUTH_TAG_ADDR_MISALIGNED_POINTER              (0x00000003)
#define LWCBA2_ERROR_DECRYPT_COPY_DMA_NACK                                      (0x00000004)
#define LWCBA2_ERROR_DECRYPT_COPY_SIGNATURE_CHECK_FAILURE                       (0x00000005) // TODO: remove once this merges to chips_a and we update ucode to use the below
#define LWCBA2_ERROR_DECRYPT_COPY_AUTH_TAG_MISMATCH                             (0x00000005)
#define LWCBA2_ERROR_DIGEST_ADDR_MISALIGNED_POINTER                             (0x00000006)
#define LWCBA2_ERROR_DIGEST_ADDR_DMA_NACK                                       (0x00000007)
#define LWCBA2_ERROR_DIGEST_CHECK_FAILURE                                       (0x00000008)
#define LWCBA2_ERROR_MISALIGNED_SIZE                                            (0x00000009)
#define LWCBA2_ERROR_MISSING_METHODS                                            (0x0000000A)
#define LWCBA2_ERROR_SEMAPHORE_RELEASE_DMA_NACK                                 (0x0000000B)

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _clcba2_h
