/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2007-2007 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef _cl906e_h_
#define _cl906e_h_

#include "lwtypes.h"

#define GF100_CHANNEL_RUNLIST                                  (0x0000906E)

#define LW906E_TYPEDEF                                         GF100ChannelRunlist

/* Per Engine Runlist USERD */
typedef volatile struct _cl906e_tag0 {
 LwU32 Put;
 LwU32 Get;
 LwU32 Ignored00[0x3E]; 
} Lw906eRunlistUserD, GF100RunlistUserD;

/* LW_RLST */
typedef struct {
 Lw906eRunlistUserD RunlistUserD[16];
} Lw906eControl;


/* Runlist Entry Format */
typedef volatile struct _cl906e_tag1 {
 LwU32 ChID;
 LwU32 GpPutLimit;
} Lw906eRunlistEntry, GF100RunlistEntry;


#endif /* _cl906e_h_ */
