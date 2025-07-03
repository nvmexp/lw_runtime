/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2011-2014 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _KEPLER_CE_H_
#define _KEPLER_CE_H_

// CE index validity check
#define CE_IS_VALID(x) (x<=(LW_PTOP_FS_STATUS_CE_IDX__SIZE_1-1))

// Indexed base addresses
#define LW_PCE_CE_BASE(x)  ((x == 2) ? LW_PCE_CE2_BASE : ((x == 1) ? LW_PCE_CE1_BASE : LW_PCE_CE0_BASE))

typedef struct _dbg_ce
{
    LwU32 m_id;
    char *m_tag;
} dbg_ce;

#define privInfo_ce(x) {x,#x}

#endif  // _KEPLER_CE_H_
