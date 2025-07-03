/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 1993-2004 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl8870_h_
#define _cl8870_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  G94_DISPLAY                                               (0x00008870)

typedef struct
{
    LwU32   numHeads; // Number of HEADs in this chip/display
    LwU32   numDacs;  // Number of DACs in this chip/display
    LwU32   numSors;  // Number of SORs in this chip/display
    LwU32   numPiors; // Number of PIORs in this chip/display
} LW8870_ALLOCATION_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl8870_h_ */
