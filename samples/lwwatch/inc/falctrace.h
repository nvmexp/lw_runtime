/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2009-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#define FALCTRACE_BUFFER_SIZE 64

void  falctraceInit       (void);
LW_STATUS falctraceInitEngine (LwU32 engineId, LwBool bIsPhyAddr, LwBool bIsSysMem, LwU64 addr, LwU32 size);
LW_STATUS falctraceDump       (LwU32 engineId, LwU32 numEntries);
