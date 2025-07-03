/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004-2014 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// vgupta@lwpu.com - Aug 19, 2004
// lw50 rsvdmem routines
// 
//*****************************************************

//
// includes
//
#include "os.h"

//-----------------------------------------------------
// getRsvdMemStartAddr_LW50
// + Returns the offset of rsvd mem from start of FB 
//
//-----------------------------------------------------
LwU64 getRsvdMemStartAddr_LW50
(
    void
)
{
    //LwU64 baseAddr = FB_RD32_64_DRF(_PFIFO, _RESERVED_MEM, _BASE);
    //return baseAddr << 16;

    return 0;
}
