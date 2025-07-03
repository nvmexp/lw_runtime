/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2007-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _PRODVAL_H_
#define _PRODVAL_H_

#include "os.h"

#include "g_hwprod_private.h"     // (rmconfig)  implementation prototypes

//
// Defines
//
#define MAX_REGNAME_LEN         256
#define MAX_FIELDNAME_LEN       256
#define MAX_DATNAME_LEN         64

//
// Data Structures
//
typedef struct
{
    LwU32 addr;
    LwU32 flags;
    LwU32 fieldCnt;
    LwU32 nameLen;
} PRODREGDEF;

typedef struct
{
    LwU32 addr;
    LwU32 val;
} PRODREGIDXDEF;

typedef struct
{
    LwU32 startBit;
    LwU32 endBit;
    LwU32 value;
    LwU32 nameLen;
} PRODFIELDDEF;

#endif // _PRODVAL_H_
