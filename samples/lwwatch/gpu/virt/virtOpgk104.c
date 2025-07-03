/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2005-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "vmem.h"
#include "virtOp.h"
#include "g_virt_private.h"     // (rmconfig)  implementation prototypes

#ifdef WIN32
#include <windows.h>
#endif
#include <math.h>

// Constants for fixed hardware gob sizes.
#define LW_FERMI_BLOCK_LINEAR_LOG_GOB_WIDTH       6    /*    64 bytes (2^6) */
#define LW_FERMI_BLOCK_LINEAR_LOG_GOB_HEIGHT      3    /* x   8 rows  (2^3) */
#define LW_FERMI_BLOCK_LINEAR_LOG_GOB_DEPTH       0    /* x   1 layer (2^0) */
#define LW_FERMI_BLOCK_LINEAR_GOB_SIZE            512  /* = 512 bytes (2^9) */

// Derived constants for fixed hardware gob sizes.
#define LW_FERMI_BLOCK_LINEAR_GOB_WIDTH               \
        (1 << LW_FERMI_BLOCK_LINEAR_LOG_GOB_WIDTH)
#define LW_FERMI_BLOCK_LINEAR_GOB_HEIGHT              \
        (1 << LW_FERMI_BLOCK_LINEAR_LOG_GOB_HEIGHT)
#define LW_FERMI_BLOCK_LINEAR_GOB_DEPTH               \
        (1 << LW_FERMI_BLOCK_LINEAR_LOG_GOB_DEPTH) 

//-----------------------------------------------------
// virtDisplayVirtual_GK104
//
//-----------------------------------------------------
LW_STATUS virtDisplayVirtual_GK104(LwU32 memType, LwU32 chId, LwU64 va,
                         LwU32 width, LwU32 height, //No depth since it displays a 2D image
                         LwU32 logBlockWidth, LwU32 logBlockHeight, LwU32 logBlockDepth, //Still need block depth though!
                         LwU32 format)
{
    return virtDisplayVirtual((VCBMEMTYPE) memType, chId, va, width, height,
                              logBlockWidth, logBlockHeight, logBlockDepth,
                              format, LW_FERMI_BLOCK_LINEAR_GOB_WIDTH, LW_FERMI_BLOCK_LINEAR_GOB_HEIGHT);
}
