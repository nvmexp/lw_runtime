/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2001-2004 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl844c_h_
#define _cl844c_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  G84_PERFBUFFER                                            (0x0000844C)

/* pio method data structure */
typedef volatile struct _cl844c_tag0 {
 LwV32 Reserved00[0x7c0];
} G844cTypedef, G84PerfBuffer;
#define  G844C_TYPEDEF                                             G84PerfBuffer

#define G844C_PERFBUFFER_MEMORY_HANDLE                             (0x844C0001)

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl844c_h_ */
