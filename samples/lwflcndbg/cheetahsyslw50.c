/***************************** CHEETAH SYS State Rotuines ********************\
*                                                                           *
* Module: TEGRASYSLW50.C                                                    *
*         CHEETAH System LW50 Definitions.                                    *
\***************************************************************************/

#include "tegrasys.h"
#include "lw50/dev_bus.h"

/** 
 * @brief parses relocation table and fills out pTegraRelocTable
 * 
 * @return LW_OK if successful, error otherwise
 */
U032 tegrasysInit_LW50(PTEGRASYS pTegraSys)
{
    U032               status     = LW_ERR_GENERIC;
    PDEVICE_RELOCATION pTegraRelocTable;

    if (lwBar0 != 0)
    {
        pTegraRelocTable = (PDEVICE_RELOCATION)malloc(sizeof(DEVICE_RELOCATION));
        if (pTegraRelocTable != NULL)
        {
            U032 len = (U032)strlen("GPU");
            pTegraRelocTable->devName = (char *)malloc(sizeof(char) * (len + 1));
            if (pTegraRelocTable->devName != NULL)
            {
                strncpy(pTegraRelocTable->devName, "GPU", len + 1);

                pTegraRelocTable->devId   = 0;
                pTegraRelocTable->start   = lwBar0;
                pTegraRelocTable->end     = lwBar0 + DRF_EXTENT(LW_RSPACE);
                pTegraRelocTable->verMaj  = 0;
                pTegraRelocTable->verMin  = 0;

                pTegraSys->numDevices = 1;
                pTegraSys->pRelocationTable = pTegraRelocTable;
                status = LW_OK;
            }
        }
    }

    return status;
}

