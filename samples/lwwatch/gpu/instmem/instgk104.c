/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2010-2014 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//
// includes
//

#include "kepler/gk104/dev_ram.h"
#include "os.h"
#include "chip.h"
#include "hal.h"
#include "mmu.h"
#include "inst.h"
#include "print.h"
#include "lwwatchErrors.h"

#include "g_instmem_private.h"      // (rmconfig) implementation prototypes


void instmemGetUserdFormattedMemory_GK104(
        formattedMemoryEntry** ppOutFormattedMemoryEntry, 
        LwU32* pOutNumEntries
        )
{
    static formattedMemoryEntry userd[] =
    {
        {"  LW_RAMUSERD_PUT                        ",DEVICE_BASE(LW_RAMUSERD_PUT                )}, 
        {"  LW_RAMUSERD_GET                        ",DEVICE_BASE(LW_RAMUSERD_GET                )}, 
        {"  LW_RAMUSERD_REF                        ",DEVICE_BASE(LW_RAMUSERD_REF                )}, 
        {"  LW_RAMUSERD_PUT_HI                     ",DEVICE_BASE(LW_RAMUSERD_PUT_HI             )}, 
        {"  LW_RAMUSERD_REF_THRESHOLD              ",DEVICE_BASE(LW_RAMUSERD_REF_THRESHOLD      )}, 
        {"  LW_RAMUSERD_GP_TOP_LEVEL_GET           ",DEVICE_BASE(LW_RAMUSERD_GP_TOP_LEVEL_GET   )}, 
        {"  LW_RAMUSERD_GP_TOP_LEVEL_GET_HI        ",DEVICE_BASE(LW_RAMUSERD_GP_TOP_LEVEL_GET_HI)}, 
        {"  LW_RAMUSERD_GET_HI                     ",DEVICE_BASE(LW_RAMUSERD_GET_HI             )}, 
        {"  LW_RAMUSERD_GP_GET                     ",DEVICE_BASE(LW_RAMUSERD_GP_GET             )}, 
        {"  LW_RAMUSERD_GP_PUT                     ",DEVICE_BASE(LW_RAMUSERD_GP_PUT             )}, 
    };

    if ( ppOutFormattedMemoryEntry )
    {
        *ppOutFormattedMemoryEntry = userd;
    }

    if ( pOutNumEntries )
    {
        *pOutNumEntries = sizeof(userd)/sizeof(formattedMemoryEntry);
    }
};

/*!  
 * Determines how data will be read from BAR1 to fill in the USERD
 * struct.
 *
 * @param pOutSize Output parameter corresponding to USERD size.
 * @param pOutChStride Output parameter corresponding to channel stride.
 */
void instmemGetUserdParams_GK104(LwU32* pOutSize, LwU32* pOutChStride)
{
    if ( pOutSize )
    {
        // Some of the data at the end of USERDInfo is ignored, so don't bother reading it
        *pOutSize = LW_RAMUSERD_CHAN_SIZE;
    }

    if ( pOutChStride )
    {
        *pOutChStride = 0x200;
    }
}

