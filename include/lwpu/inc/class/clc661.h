/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2022 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

// AUTO GENERATED -- DO NOT EDIT - this file automatically generated by refhdr2class.pl
// Command: ../../../../../../../../bin/manuals/refhdr2class.pl clc661.h c661 HOPPER_USERMODE_A --search_str=LW_VIRTUAL_FUNCTION --exclude_regex=LW_VIRTUAL_FUNCTION_PRIV --exclude_regex=VFALIAS --exclude_regex=FULL_PHYS_OFFSET --exclude_regex=MSCG_BLOCK --input_file=lw_ref_dev_vm.h --zero_based_offsets --add_size_defines


#ifndef _clc661_h_
#define _clc661_h_

#ifdef __cplusplus
extern "C" {
#endif

#define HOPPER_USERMODE_A (0xc661)

#define LWC661                                                      0x0000ffff:0x00000000
#define LWC661_HOPPER_USERMODE_A__SIZE                                              65536
#define LWC661_CFG0                                                            0x00000000
#define LWC661_CFG0_USERMODE_CLASS_ID                                                15:0
#define LWC661_CFG0_USERMODE_CLASS_ID_VALUE                                        0xc661
#define LWC661_CFG0_RSVD                                                            31:16
#define LWC661_CFG0_RSVD_VALUE_ZERO                                                0x0000
#define LWC661_TIME_0                                                          0x00000080
#define LWC661_TIME_0_NSEC                                                           31:5
#define LWC661_TIME_1                                                          0x00000084
#define LWC661_TIME_1_NSEC                                                           28:0
#define LWC661_DOORBELL                                                        0x00000090
#define LWC661_DOORBELL_HANDLE                                                       31:0
#define LWC661_DOORBELL_VECTOR                                                       11:0
#define LWC661_DOORBELL_RSVD                                                        15:12
#define LWC661_DOORBELL_RUNLIST_ID                                                  22:16
#define LWC661_DOORBELL_RUNLIST_ID_ILWALID_RUNLIST                                   0x7f
#define LWC661_DOORBELL_RUNLIST_DOORBELL                                            22:22
#define LWC661_DOORBELL_RUNLIST_DOORBELL_DISABLE                                      0x1
#define LWC661_DOORBELL_RUNLIST_DOORBELL_ENABLE                                       0x0
#define LWC661_DOORBELL_RSVD2                                                       31:23
#define LWC661_DOORBELL_GSP_DOORBELL                                                31:31
#define LWC661_DOORBELL_GSP_DOORBELL_DISABLE                                          0x1
#define LWC661_DOORBELL_GSP_DOORBELL_ENABLE                                           0x0
#define LWC661_ERR_CONT                                                        0x00000094
#define LWC661_ERR_CONT_INTERRUPT(i)                                              (i):(i)
#define LWC661_ERR_CONT_INTERRUPT__SIZE_1                                              32
#define LWC661_ERR_CONT_INTERRUPT_NOT_PENDING                                         0x0
#define LWC661_ERR_CONT_INTERRUPT_PENDING                                             0x1
#define LWC661_ERR_CONT_HW_INTERRUPT                                                 31:0

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _clc661_h_ */
