/***************************** CHEETAH SYS State Rotuines ********************\
*                                                                           *
* Module: TEGRASYST30.C                                                     *
*         CHEETAH System T30 Definitions.                                     *
\***************************************************************************/

#include "tegrasys.h"
#include "t3x/t30/project_relocation_table.h"

/** 
 * @brief parses relocation table and fills out pTegraRelocTable
 * 
 * @return LW_OK if successful, error otherwise
 */
U032 tegrasysInit_T30(PTEGRASYS pTegraSys)
{
    DEVICE_LIST devicesToInit[] = 
    {
        { LW_DEVID_HOST1X, "HOST1X" },
        { LW_DEVID_DISPLAY, "DISPLAY" },
        { LW_DEVID_VI, "VI" },
        { LW_DEVID_HDMI, "HDMI" },
        { LW_DEVID_SHR_SEM, "SEM" },
        { LW_DEVID_CAR, "CAR" },
        { LW_DEVID_FLOW, "FLOW" },
        { LW_DEVID_GPIO, "GPIO" },
        { LW_DEVID_VECTOR, "VECTOR" },
        { LW_DEVID_MISC, "MISC" },
        { LW_DEVID_MC, "MC" },
        { LW_DEVID_FUSE, "FUSE" },
        { LW_DEVID_EMEM, "EMEM" },
        { LW_DEVID_PMIF, "PMIF" },
        { LW_DEVID_DSI, "DSI" },
        { LW_DEVID_ICTLR, "ICTLR" },
        { LW_DEVID_MPE, "MPE" }
    };

    LwU64 rawTable[] = 
    {
        LW_RELOCATION_TABLE_INIT,
    };

    LwU32 numDevices = sizeof(devicesToInit)/sizeof(devicesToInit[0]);
    U032  status     = LW_ERR_GENERIC;

    status = tegrasysParseRelocationTable(pTegraSys, rawTable, devicesToInit, numDevices);

    return status;
}

