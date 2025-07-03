/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012-2014 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//
// includes
//

#include "maxwell/gm107/dev_ram.h"
#include "inst.h"

/* 
 * Returns a formattedMemoryEntry list of RAMFC memory. This list can be
 * printed using PrintFormattedMemory.  
 *
 * @param ppOutFormattedMemoryEntry Output variable corresponding to the formatted list.
 * @param pOutNumEntries Output variable corresponding to the number of entries in the list.
 */
void instmemGetRamfcFormattedMemory_GM107(
        formattedMemoryEntry** ppOutFormattedMemoryEntry,
        LwU32* pOutNumEntries
        )
{
    static formattedMemoryEntry ramfc[] =
    {
        {"  LW_RAMFC_GP_PUT                     ",DEVICE_BASE(LW_RAMFC_GP_PUT                   )},
        {"  LW_RAMFC_RESERVED1                  ",DEVICE_BASE(LW_RAMFC_RESERVED1                )},
        {"  LW_RAMFC_USERD                      ",DEVICE_BASE(LW_RAMFC_USERD                    )},
        {"  LW_RAMFC_USERD_HI                   ",DEVICE_BASE(LW_RAMFC_USERD_HI                 )},
        {"  LW_RAMFC_SIGNATURE                  ",DEVICE_BASE(LW_RAMFC_SIGNATURE                )},
        {"  LW_RAMFC_GP_GET                     ",DEVICE_BASE(LW_RAMFC_GP_GET                   )},
        {"  LW_RAMFC_PB_GET                     ",DEVICE_BASE(LW_RAMFC_PB_GET                   )},
        {"  LW_RAMFC_PB_GET_HI                  ",DEVICE_BASE(LW_RAMFC_PB_GET_HI                )},
        {"  LW_RAMFC_PB_TOP_LEVEL_GET           ",DEVICE_BASE(LW_RAMFC_PB_TOP_LEVEL_GET         )},
        {"  LW_RAMFC_PB_TOP_LEVEL_GET_HI        ",DEVICE_BASE(LW_RAMFC_PB_TOP_LEVEL_GET_HI      )},
        {"  LW_RAMFC_REF                        ",DEVICE_BASE(LW_RAMFC_REF                      )},
        {"  LW_RAMFC_RUNTIME                    ",DEVICE_BASE(LW_RAMFC_RUNTIME                  )},
        {"  LW_RAMFC_ACQUIRE                    ",DEVICE_BASE(LW_RAMFC_ACQUIRE                  )},
        {"  LW_RAMFC_ACQUIRE_DEADLINE           ",DEVICE_BASE(LW_RAMFC_ACQUIRE_DEADLINE         )},
        {"  LW_RAMFC_SEMAPHOREA                 ",DEVICE_BASE(LW_RAMFC_SEMAPHOREA               )},
        {"  LW_RAMFC_SEMAPHOREB                 ",DEVICE_BASE(LW_RAMFC_SEMAPHOREB               )},
        {"  LW_RAMFC_SEMAPHOREC                 ",DEVICE_BASE(LW_RAMFC_SEMAPHOREC               )},
        {"  LW_RAMFC_SEMAPHORED                 ",DEVICE_BASE(LW_RAMFC_SEMAPHORED               )},
        {"  LW_RAMFC_GP_BASE                    ",DEVICE_BASE(LW_RAMFC_GP_BASE                  )},
        {"  LW_RAMFC_GP_BASE_HI                 ",DEVICE_BASE(LW_RAMFC_GP_BASE_HI               )},
        {"  LW_RAMFC_GP_FETCH                   ",DEVICE_BASE(LW_RAMFC_GP_FETCH                 )},
        {"  LW_RAMFC_PB_FETCH                   ",DEVICE_BASE(LW_RAMFC_PB_FETCH                 )},
        {"  LW_RAMFC_PB_FETCH_HI                ",DEVICE_BASE(LW_RAMFC_PB_FETCH_HI              )},
        {"  LW_RAMFC_PB_PUT                     ",DEVICE_BASE(LW_RAMFC_PB_PUT                   )},
        {"  LW_RAMFC_PB_PUT_HI                  ",DEVICE_BASE(LW_RAMFC_PB_PUT_HI                )},
        {"  LW_RAMFC_PB_GET_STAGER              ",DEVICE_BASE(LW_RAMFC_PB_GET_STAGER            )},
        {"  LW_RAMFC_PB_PUT_STAGER              ",DEVICE_BASE(LW_RAMFC_PB_PUT_STAGER            )},
        {"  LW_RAMFC_PB_TOP_LEVEL_GET_STAGER    ",DEVICE_BASE(LW_RAMFC_PB_TOP_LEVEL_GET_STAGER  )},
        {"  LW_RAMFC_GP_CRC                     ",DEVICE_BASE(LW_RAMFC_GP_CRC                   )},
        {"  LW_RAMFC_RESERVED30                 ",DEVICE_BASE(LW_RAMFC_RESERVED30               )},
        {"  LW_RAMFC_RESERVED31                 ",DEVICE_BASE(LW_RAMFC_RESERVED31               )},
        {"  LW_RAMFC_RESERVED32                 ",DEVICE_BASE(LW_RAMFC_RESERVED32               )},
        {"  LW_RAMFC_PB_HEADER                  ",DEVICE_BASE(LW_RAMFC_PB_HEADER                )},
        {"  LW_RAMFC_PB_COUNT                   ",DEVICE_BASE(LW_RAMFC_PB_COUNT                 )},
        {"  LW_RAMFC_PB_DATA0                   ",DEVICE_BASE(LW_RAMFC_PB_DATA0                 )},
        {"  LW_RAMFC_PB_DATA1                   ",DEVICE_BASE(LW_RAMFC_PB_DATA1                 )},
        {"  LW_RAMFC_SUBDEVICE                  ",DEVICE_BASE(LW_RAMFC_SUBDEVICE                )},
        {"  LW_RAMFC_PB_CRC                     ",DEVICE_BASE(LW_RAMFC_PB_CRC                   )},
        {"  LW_RAMFC_FORMATS                    ",DEVICE_BASE(LW_RAMFC_FORMATS                  )},
        {"  LW_RAMFC_MEM_OP_A                   ",DEVICE_BASE(LW_RAMFC_MEM_OP_A                 )},
        {"  LW_RAMFC_RESERVED20                 ",DEVICE_BASE(LW_RAMFC_RESERVED20               )},
        {"  LW_RAMFC_RESERVED21                 ",DEVICE_BASE(LW_RAMFC_RESERVED21               )},
        {"  LW_RAMFC_TARGET                     ",DEVICE_BASE(LW_RAMFC_TARGET                   )},
        {"  LW_RAMFC_METHOD_CRC                 ",DEVICE_BASE(LW_RAMFC_METHOD_CRC               )},
        {"  LW_RAMFC_RESERVED45                 ",DEVICE_BASE(LW_RAMFC_RESERVED45               )},
        {"  LW_RAMFC_RESERVED46                 ",DEVICE_BASE(LW_RAMFC_RESERVED46               )},
        {"  LW_RAMFC_RESERVED47                 ",DEVICE_BASE(LW_RAMFC_RESERVED47               )},
        {"  LW_RAMFC_METHOD0                    ",DEVICE_BASE(LW_RAMFC_METHOD0                  )},
        {"  LW_RAMFC_DATA0                      ",DEVICE_BASE(LW_RAMFC_DATA0                    )},
        {"  LW_RAMFC_METHOD1                    ",DEVICE_BASE(LW_RAMFC_METHOD1                  )},
        {"  LW_RAMFC_DATA1                      ",DEVICE_BASE(LW_RAMFC_DATA1                    )},
        {"  LW_RAMFC_METHOD2                    ",DEVICE_BASE(LW_RAMFC_METHOD2                  )},
        {"  LW_RAMFC_DATA2                      ",DEVICE_BASE(LW_RAMFC_DATA2                    )},
        {"  LW_RAMFC_METHOD3                    ",DEVICE_BASE(LW_RAMFC_METHOD3                  )},
        {"  LW_RAMFC_DATA3                      ",DEVICE_BASE(LW_RAMFC_DATA3                    )},
        {"  LW_RAMFC_SPARE56                    ",DEVICE_BASE(LW_RAMFC_SPARE56                  )},
        {"  LW_RAMFC_HCE_CTRL                   ",DEVICE_BASE(LW_RAMFC_HCE_CTRL                 )},
        {"  LW_RAMFC_CHID                       ",DEVICE_BASE(LW_RAMFC_CHID                     )},
        {"  LW_RAMFC_GP_PEEK                    ",DEVICE_BASE(LW_RAMFC_GP_PEEK                  )},
        {"  LW_RAMFC_PB_DATA2                   ",DEVICE_BASE(LW_RAMFC_PB_DATA2                 )},
        {"  LW_RAMFC_L2                         ",DEVICE_BASE(LW_RAMFC_L2                       )}
    };

    if ( ppOutFormattedMemoryEntry )
    {
        *ppOutFormattedMemoryEntry = ramfc;
    }

    if ( pOutNumEntries )
    {
        *pOutNumEntries = sizeof(ramfc)/sizeof(formattedMemoryEntry);
    }
};
