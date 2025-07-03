/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021-2022 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//
// includes
//

#include "blackwell/gb100/dev_ram.h"

#include "inst.h"

#include "lwmisc.h"

#include "g_instmem_private.h"


/*
 * Returns a formattedMemoryEntry list of RAMFC memory. This list can be
 * printed using PrintFormattedMemory.
 *
 * @param ppOutFormattedMemoryEntry Output variable corresponding to the formatted list.
 * @param pOutNumEntries Output variable corresponding to the number of entries in the list.
 */
void instmemGetRamfcFormattedMemory_GB100(
        formattedMemoryEntry** ppOutFormattedMemoryEntry,
        LwU32* pOutNumEntries
        )
{
    static formattedMemoryEntry ramfc[] =
    {
        {"  LW_RAMFC_MEM_OP_A                   ",DEVICE_BASE(LW_RAMFC_MEM_OP_A                 )},
        {"  LW_RAMFC_SIGNATURE                  ",DEVICE_BASE(LW_RAMFC_SIGNATURE                )},
        {"  LW_RAMFC_GP_GET                     ",DEVICE_BASE(LW_RAMFC_GP_GET                   )},
        {"  LW_RAMFC_PB_GET                     ",DEVICE_BASE(LW_RAMFC_PB_GET                   )},
        {"  LW_RAMFC_PB_GET_HI                  ",DEVICE_BASE(LW_RAMFC_PB_GET_HI                )},
        {"  LW_RAMFC_SEM_ADDR_HI                ",DEVICE_BASE(LW_RAMFC_SEM_ADDR_HI              )},
        {"  LW_RAMFC_SEM_ADDR_LO                ",DEVICE_BASE(LW_RAMFC_SEM_ADDR_LO              )},
        {"  LW_RAMFC_SEM_PAYLOAD_HI             ",DEVICE_BASE(LW_RAMFC_SEM_PAYLOAD_HI           )},
        {"  LW_RAMFC_SEM_PAYLOAD_LO             ",DEVICE_BASE(LW_RAMFC_SEM_PAYLOAD_LO           )},
        {"  LW_RAMFC_SEM_EXELWTE                ",DEVICE_BASE(LW_RAMFC_SEM_EXELWTE              )},
        {"  LW_RAMFC_GP_BASE                    ",DEVICE_BASE(LW_RAMFC_GP_BASE                  )},
        {"  LW_RAMFC_GP_BASE_HI                 ",DEVICE_BASE(LW_RAMFC_GP_BASE_HI               )},
        {"  LW_RAMFC_PB_PUT                     ",DEVICE_BASE(LW_RAMFC_PB_PUT                   )},
        {"  LW_RAMFC_PB_PUT_HI                  ",DEVICE_BASE(LW_RAMFC_PB_PUT_HI                )},
        {"  LW_RAMFC_MEM_OP_B                   ",DEVICE_BASE(LW_RAMFC_MEM_OP_B                 )},
        {"  LW_RAMFC_SPLITTER_REM_LINES         ",DEVICE_BASE(LW_RAMFC_SPLITTER_REM_LINES       )},
        {"  LW_RAMFC_SPLITTER_OFFSET_IN_LOWER   ",DEVICE_BASE(LW_RAMFC_SPLITTER_OFFSET_IN_LOWER )},
        {"  LW_RAMFC_SPLITTER_OFFSET_IN_UPPER   ",DEVICE_BASE(LW_RAMFC_SPLITTER_OFFSET_IN_UPPER )},
        {"  LW_RAMFC_PB_HEADER                  ",DEVICE_BASE(LW_RAMFC_PB_HEADER                )},
        {"  LW_RAMFC_PB_COUNT                   ",DEVICE_BASE(LW_RAMFC_PB_COUNT                 )},
        {"  LW_RAMFC_PB_DATA0                   ",DEVICE_BASE(LW_RAMFC_PB_DATA0                 )},
        {"  LW_RAMFC_PB_DATA1                   ",DEVICE_BASE(LW_RAMFC_PB_DATA1                 )},
        {"  LW_RAMFC_SUBDEVICE                  ",DEVICE_BASE(LW_RAMFC_SUBDEVICE                )},
        {"  LW_RAMFC_MEM_OP_C                   ",DEVICE_BASE(LW_RAMFC_MEM_OP_C                 )},
        {"  LW_RAMFC_SPLITTER_REM_PIXELS        ",DEVICE_BASE(LW_RAMFC_SPLITTER_REM_PIXELS      )},
        {"  LW_RAMFC_SPLITTER_OFFSET_OUT_LOWER  ",DEVICE_BASE(LW_RAMFC_SPLITTER_OFFSET_OUT_LOWER )},
        {"  LW_RAMFC_SPLITTER_OFFSET_OUT_UPPER  ",DEVICE_BASE(LW_RAMFC_SPLITTER_OFFSET_OUT_UPPER )},
        {"  LW_RAMFC_METHOD0                    ",DEVICE_BASE(LW_RAMFC_METHOD0                  )},
        {"  LW_RAMFC_DATA0                      ",DEVICE_BASE(LW_RAMFC_DATA0                    )},
        {"  LW_RAMFC_METHOD1                    ",DEVICE_BASE(LW_RAMFC_METHOD1                  )},
        {"  LW_RAMFC_DATA1                      ",DEVICE_BASE(LW_RAMFC_DATA1                    )},
        {"  LW_RAMFC_METHOD2                    ",DEVICE_BASE(LW_RAMFC_METHOD2                  )},
        {"  LW_RAMFC_DATA2                      ",DEVICE_BASE(LW_RAMFC_DATA2                    )},
        {"  LW_RAMFC_METHOD3                    ",DEVICE_BASE(LW_RAMFC_METHOD3                  )},
        {"  LW_RAMFC_DATA3                      ",DEVICE_BASE(LW_RAMFC_DATA3                    )},
        {"  LW_RAMFC_HCE_CTRL                   ",DEVICE_BASE(LW_RAMFC_HCE_CTRL                 )},
        {"  LW_RAMFC_PB_DATA2                   ",DEVICE_BASE(LW_RAMFC_PB_DATA2                 )},
        {"  LW_RAMFC_SET_CHANNEL_INFO           ",DEVICE_BASE(LW_RAMFC_SET_CHANNEL_INFO         )}
    };

    if ( ppOutFormattedMemoryEntry )
    {
        *ppOutFormattedMemoryEntry = ramfc;
    }

    if ( pOutNumEntries )
    {
        *pOutNumEntries = sizeof(ramfc)/sizeof(formattedMemoryEntry);
    }
}

void instmemGetUserdFormattedMemory_GB100(
        formattedMemoryEntry** ppOutFormattedMemoryEntry, 
        LwU32* pOutNumEntries
        )
{
    static formattedMemoryEntry userd[] =
    {
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