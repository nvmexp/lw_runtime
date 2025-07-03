/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2002 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _DRVCD_H_
#define _DRVCD_H_

//******************************************************************************
//
// Module Name: DRVCD.H
//
// This file contains structures and constants that define the driver specific
// data for the crash dump file. The record definitions defined here are always
// stored after the crash dump file header. Each record defined here is preceded
// by the LWCD_RECORD structure.
//
//******************************************************************************
#include "lwtypes.h"
#if !defined XAPIGEN          /* avoid duplicate xapi fns for <lwcd.h> */
#include "lwcd.h"
#endif

typedef struct
{
    LwU32   usec;                       // usec elapsed since last log entry
    /* XAPIGEN: hack around colwenience union with no discriminant */
#if defined(XAPIGEN)
    struct {
        LwU8    bytes[32];
    } u;
#else
    union {
        LwU8    bytes[32];
        char    str[32];
    } u;
#endif
} DrvSwLogEntry;
typedef DrvSwLogEntry *PDrvSwLogEntry;

#endif // _DRVCD_H_
