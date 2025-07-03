/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2008 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl9068_h_
#define _cl9068_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  GF100_INDIRECT_FRAMEBUFFER                                 (0x00009068)

/* IFB data structure */
typedef struct {
    LwU32 Ignored00[0x03];                                          /* 0x0000 - 0x000b */
    LwU32 RdWrAddrHi;                                               /* 0x000c - 0x000f */
    LwU32 RdWrAddr;                                                 /* 0x0010 - 0x0013 */
    LwU32 RdWrData;                                                 /* 0x0014 - 0x0017 */
} Lw9068Typedef, GF100IndirectFramebuffer;
#define  LW9068_TYPEDEF                                            GF100IndirectFramebuffer

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl9068_h_ */
