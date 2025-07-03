/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2001-2010 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

/* AUTO GENERATED FILE -- DO NOT EDIT */

#ifndef __CLCBC0PCAS_H__
#define __CLCBC0PCAS_H__

/*
** Post Compare And Swap
 */

#define LWCBC0_PCAS_QMD_ADDRESS_SHIFTED8                           MW(31:0)
#define LWCBC0_PCAS_FROM                                           MW(55:32)
#define LWCBC0_PCAS_DELTA                                          MW(63:56)


/*
** Signaling Post Compare And Swap
 */

#define LWCBC0_SPCAS_QMD_ADDRESS_SHIFTED8                          MW(31:0)
#define LWCBC0_SPCAS_ILWALIDATE                                    MW(32:32)
#define LWCBC0_SPCAS_ILWALIDATE_FALSE                              0x00000000
#define LWCBC0_SPCAS_ILWALIDATE_TRUE                               0x00000001
#define LWCBC0_SPCAS_SCHEDULE                                      MW(33:33)
#define LWCBC0_SPCAS_SCHEDULE_FALSE                                0x00000000
#define LWCBC0_SPCAS_SCHEDULE_TRUE                                 0x00000001
#define LWCBC0_SPCAS_HW_ONLY_INCREMENT_PUT                         MW(34:34)
#define LWCBC0_SPCAS_HW_ONLY_INCREMENT_PUT_FALSE                   0x00000000
#define LWCBC0_SPCAS_HW_ONLY_INCREMENT_PUT_TRUE                    0x00000001
#define LWCBC0_SPCAS_RESERVED                                      MW(55:35)
#define LWCBC0_SPCAS_MUST_BE_ZERO                                  MW(63:56)


/*
** 
 */

#define StructSignalingPcas2_QMD_ADDRESS_SHIFTED8                  MW(31:0)
#define StructSignalingPcas2_PCAS_ACTION                           MW(35:32)
#define StructSignalingPcas2_PCAS_ACTION_NOP                       0x00000000
#define StructSignalingPcas2_PCAS_ACTION_ILWALIDATE                0x00000001
#define StructSignalingPcas2_PCAS_ACTION_SCHEDULE                  0x00000002
#define StructSignalingPcas2_PCAS_ACTION_ILWALIDATE_COPY_SCHEDULE  0x00000003
#define StructSignalingPcas2_PCAS_ACTION_INCREMENT_PUT             0x00000006
#define StructSignalingPcas2_PCAS_ACTION_DECREMENT_DEPENDENCE      0x00000007
#define StructSignalingPcas2_PCAS_ACTION_PREFETCH                  0x00000008
#define StructSignalingPcas2_PCAS_ACTION_PREFETCH_SCHEDULE         0x00000009
#define StructSignalingPcas2_PCAS_ACTION_ILWALIDATE_PREFETCH_COPY_SCHEDULE 0x0000000a
#define StructSignalingPcas2_PCAS_ACTION_ILWALIDATE_PREFETCH_COPY_FORCE_REQUIRE_SCHEDULING 0x0000000b
#define StructSignalingPcas2_PCAS_ACTION_INCREMENT_DEPENDENCE      0x0000000c
#define StructSignalingPcas2_PCAS_ACTION_INCREMENT_CWD_REF_COUNTER 0x0000000d
#define StructSignalingPcas2_RESERVED_A                            MW(39:36)
#define StructSignalingPcas2_SELECT                                MW(45:40)
#define StructSignalingPcas2_OFFSET_MINUS_ONE                      MW(55:46)
#define StructSignalingPcas2_MUST_BE_ZERO                          MW(63:56)



#endif // #ifndef __CLCBC0PCAS_H__
