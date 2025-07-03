/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _SEQ_H_
#define _SEQ_H_

#include "os.h"
#include "hal.h"

void seqExec       (char *pCmd);
void seqDumpScript (LwU32 *pScript, LwU32 sizeBytes);

#endif // _SEQ_H_

