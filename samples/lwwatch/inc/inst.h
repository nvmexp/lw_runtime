/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2004-2014 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 2.1.2002
// inst.h
//
//*****************************************************

#ifndef _INST_H_
#define _INST_H_

#include "os.h"
#include "hal.h"
#include "fifo.h"
//
// defines
//
#define LW_INST_START_ADDR              (hal.instStartAddr)

typedef struct
{
    const char* name;
    LwU32       bit_offset;
} formattedMemoryEntry;

#include "g_instmem_hal.h"                    // (rmconfig) public interface

#endif // _INST_H_
