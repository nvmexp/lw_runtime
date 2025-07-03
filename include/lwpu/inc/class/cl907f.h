/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2007-2007 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl907f_h_
#define _cl907f_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  GF100_REMAPPER                                             (0x0000907F)

typedef volatile struct _cl907f_tag0 {
 LwV32 Reserved00[0x7c0];
} Lw907fTypedef, GF100Remapper;
#define  Lw907F_TYPEDEF                                            GF100Remapper

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl907f_h_ */
