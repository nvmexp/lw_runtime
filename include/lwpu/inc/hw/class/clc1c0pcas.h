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

#ifndef __CLC1C0PCAS_H__
#define __CLC1C0PCAS_H__

/*
** Post Compare And Swap
 */

#define LWC1C0_PCAS_QMD_ADDRESS_SHIFTED8                           MW(31:0)
#define LWC1C0_PCAS_FROM                                           MW(55:32)
#define LWC1C0_PCAS_DELTA                                          MW(63:56)


/*
** Signaling Post Compare And Swap
 */

#define LWC1C0_SPCAS_QMD_ADDRESS_SHIFTED8                          MW(31:0)
#define LWC1C0_SPCAS_ILWALIDATE                                    MW(32:32)
#define LWC1C0_SPCAS_ILWALIDATE_FALSE                              0x00000000
#define LWC1C0_SPCAS_ILWALIDATE_TRUE                               0x00000001
#define LWC1C0_SPCAS_SCHEDULE                                      MW(33:33)
#define LWC1C0_SPCAS_SCHEDULE_FALSE                                0x00000000
#define LWC1C0_SPCAS_SCHEDULE_TRUE                                 0x00000001
#define LWC1C0_SPCAS_HW_ONLY_INCREMENT_PUT                         MW(34:34)
#define LWC1C0_SPCAS_HW_ONLY_INCREMENT_PUT_FALSE                   0x00000000
#define LWC1C0_SPCAS_HW_ONLY_INCREMENT_PUT_TRUE                    0x00000001
#define LWC1C0_SPCAS_RESERVED                                      MW(55:35)
#define LWC1C0_SPCAS_MUST_BE_ZERO                                  MW(63:56)



#endif // #ifndef __CLC1C0PCAS_H__
