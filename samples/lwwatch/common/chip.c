/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2003-2021 by LWPU Corporation.  All rights reserved.  All
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
// chip.c
//
//*****************************************************

//
// includes
//
#include "lwwatch.h"
#include "rmpmucmdif.h"
#include "chip.h"
#include "hal.h"
#include "g00x/g000/dev_master.h" // Include only the latest chip's dev_master
#include "tegrasys.h"

//-----------------------------------------------------
// GetChipAndRevision
// - Returns chip and revision if either parameter
//   pointer is not NULL.
//-----------------------------------------------------
void GetChipAndRevision(LwU32 *pChip, LwU32 *pRevision)
{
    LwU32 Chip, Revision;

    Chip     = ((hal.chipInfo.Architecture << 4)| hal.chipInfo.Implementation);
    Revision = hal.chipInfo.MaskRevision;

    if(pChip)
      *pChip = Chip;

    if(pRevision)
    {
        *pRevision = Revision;
    }
}

//-----------------------------------------------------
// IsTegra()
//
//-----------------------------------------------------
LwU32 isTegraHack = 0;
BOOL IsTegra()
{
    return (IsT124() || IsT210() || IsT186() || IsT194() || IsT234());
}

//-----------------------------------------------------
// IsT124()
//
//-----------------------------------------------------
BOOL IsT124()
{
    LwU32 data = ((isTegraHack >> 8) & 0xff);
    return (data == 0x40);
}

//-----------------------------------------------------
// IsT210()
//
//-----------------------------------------------------
BOOL IsT210()
{
    LwU32 data = ((isTegraHack >> 8) & 0xff);
    return (data == 0x21);
}

//-----------------------------------------------------
// IsT186()
//
//-----------------------------------------------------
BOOL IsT186()
{
    LwU32 data = ((isTegraHack >> 8) & 0xff);
    return (data == 0x18);
}

//-----------------------------------------------------
// IsT194()
//
//-----------------------------------------------------
BOOL IsT194()
{
    LwU32 data = ((isTegraHack >> 8) & 0xff);
    return (data == 0x19);
}

//-----------------------------------------------------
// IsT234()
//
//-----------------------------------------------------
BOOL IsT234()
{
    LwU32 data = ((isTegraHack >> 8) & 0xff);
    return (data == 0x23);
}

//-----------------------------------------------------
// IsGM107()
//
//-----------------------------------------------------
BOOL IsGM107()
{
    return (hal.halImpl == LWHAL_IMPL_GM107);
}

//-----------------------------------------------------
// IsGM200()
//
//-----------------------------------------------------
BOOL IsGM200()
{
    return (hal.halImpl == LWHAL_IMPL_GM200);
}

//-----------------------------------------------------
// IsGM204()
//
//-----------------------------------------------------
BOOL IsGM204()
{
    return (hal.halImpl == LWHAL_IMPL_GM204);
}

//-----------------------------------------------------
// IsGM206()
//
//-----------------------------------------------------
BOOL IsGM206()
{
    return (hal.halImpl == LWHAL_IMPL_GM206);
}

//-----------------------------------------------------
// IsGP100()
//
//-----------------------------------------------------
BOOL IsGP100()
{
    return (hal.halImpl == LWHAL_IMPL_GP100);
}

//-----------------------------------------------------
// IsGP102()
//
//-----------------------------------------------------
BOOL IsGP102()
{
    return (hal.halImpl == LWHAL_IMPL_GP102);
}

//-----------------------------------------------------
// IsGP104()
//
//-----------------------------------------------------
BOOL IsGP104()
{
    return (hal.halImpl == LWHAL_IMPL_GP104);
}

//-----------------------------------------------------
// IsGP106()
//
//-----------------------------------------------------
BOOL IsGP106()
{
    return (hal.halImpl == LWHAL_IMPL_GP106);
}

//-----------------------------------------------------
// IsGP107()
//
//-----------------------------------------------------
BOOL IsGP107()
{
    return (hal.halImpl == LWHAL_IMPL_GP107);
}

//-----------------------------------------------------
// IsGP108()
//
//-----------------------------------------------------
BOOL IsGP108()
{
    return (hal.halImpl == LWHAL_IMPL_GP108);
}

//-----------------------------------------------------
// IsGV100()
//
//-----------------------------------------------------
BOOL IsGV100()
{
    return (hal.halImpl == LWHAL_IMPL_GV100);
}

//-----------------------------------------------------
// IsTU102()
//
//-----------------------------------------------------
BOOL IsTU102()
{
    return (hal.halImpl == LWHAL_IMPL_TU102);
}

//-----------------------------------------------------
// IsTU104()
//
//-----------------------------------------------------
BOOL IsTU104()
{
    return (hal.halImpl == LWHAL_IMPL_TU104);
}

//-----------------------------------------------------
// IsTU106()
//
//-----------------------------------------------------
BOOL IsTU106()
{
    return (hal.halImpl == LWHAL_IMPL_TU106);
}

//-----------------------------------------------------
// IsTU116()
//
//-----------------------------------------------------
BOOL IsTU116()
{
    return (hal.halImpl == LWHAL_IMPL_TU116);
}

//-----------------------------------------------------
// IsTU117()
//
//-----------------------------------------------------
BOOL IsTU117()
{
    return (hal.halImpl == LWHAL_IMPL_TU117);
}

//-----------------------------------------------------
// IsGA100()
//
//-----------------------------------------------------
BOOL IsGA100()
{
    return (hal.halImpl == LWHAL_IMPL_GA100);
}

//-----------------------------------------------------
// IsGA102()
//
//-----------------------------------------------------
BOOL IsGA102()
{
    return (hal.halImpl == LWHAL_IMPL_GA102);
}

//-----------------------------------------------------
// IsGA103()
//
//-----------------------------------------------------
BOOL IsGA103()
{
    return (hal.halImpl == LWHAL_IMPL_GA103);
}

//-----------------------------------------------------
// IsGA104()
//
//-----------------------------------------------------
BOOL IsGA104()
{
    return (hal.halImpl == LWHAL_IMPL_GA104);
}

//-----------------------------------------------------
// IsGA106()
//
//-----------------------------------------------------
BOOL IsGA106()
{
    return (hal.halImpl == LWHAL_IMPL_GA106);
}

//-----------------------------------------------------
// IsGA107()
//
//-----------------------------------------------------
BOOL IsGA107()
{
    return (hal.halImpl == LWHAL_IMPL_GA107);
}

//-----------------------------------------------------
// IsAD102()
//
//-----------------------------------------------------
BOOL IsAD102()
{
    return (hal.halImpl == LWHAL_IMPL_AD102);
}

//-----------------------------------------------------
// IsAD103()
//
//-----------------------------------------------------
BOOL IsAD103()
{
    return (hal.halImpl == LWHAL_IMPL_AD103);
}

//-----------------------------------------------------
// IsAD104()
//
//-----------------------------------------------------
BOOL IsAD104()
{
    return (hal.halImpl == LWHAL_IMPL_AD104);
}

//-----------------------------------------------------
// IsAD106()
//
//-----------------------------------------------------
BOOL IsAD106()
{
    return (hal.halImpl == LWHAL_IMPL_AD106);
}

//-----------------------------------------------------
// IsAD107()
//
//-----------------------------------------------------
BOOL IsAD107()
{
    return (hal.halImpl == LWHAL_IMPL_AD107);
}

//-----------------------------------------------------
// IsGH100()
//
//-----------------------------------------------------
BOOL IsGH100()
{
    return (hal.halImpl == LWHAL_IMPL_GH100);
}

//-----------------------------------------------------
// IsGH202()
//
//-----------------------------------------------------
BOOL IsGH202()
{
    return (hal.halImpl == LWHAL_IMPL_GH202);
}

//-----------------------------------------------------
// IsGB100()
//
//-----------------------------------------------------
BOOL IsGB100()
{
    return (hal.halImpl == LWHAL_IMPL_GB100);
}

//-----------------------------------------------------
// IsG000()
//
//-----------------------------------------------------
BOOL IsG000()
{
    return (hal.halImpl == LWHAL_IMPL_G000);
}

//-----------------------------------------------------
// IsGM107orLater()
//
//-----------------------------------------------------
BOOL IsGM107orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_GM107);
}

//-----------------------------------------------------
// IsGM200orLater()
//
//-----------------------------------------------------
BOOL IsGM200orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_GM200);
}

//-----------------------------------------------------
// IsGP100orLater()
//
//-----------------------------------------------------
BOOL IsGP100orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_GP100);
}

//-----------------------------------------------------
// IsGP102orLater()
//
//-----------------------------------------------------
BOOL IsGP102orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_GP102);
}

//-----------------------------------------------------
// IsGP104orLater()
//
//-----------------------------------------------------
BOOL IsGP104orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_GP104);
}

//-----------------------------------------------------
// IsGP106orLater()
//
//-----------------------------------------------------
BOOL IsGP106orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_GP106);
}

//-----------------------------------------------------
// IsGP107orLater()
//
//-----------------------------------------------------
BOOL IsGP107orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_GP107);
}

//-----------------------------------------------------
// IsGP108orLater()
//
//-----------------------------------------------------
BOOL IsGP108orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_GP108);
}

//-----------------------------------------------------
// IsGV100orLater()
//
//-----------------------------------------------------
BOOL IsGV100orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_GV100);
}

//-----------------------------------------------------
// IsTU102orLater()
//
//-----------------------------------------------------
BOOL IsTU102orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_TU102);
}

//-----------------------------------------------------
// IsTU104orLater()
//
//-----------------------------------------------------
BOOL IsTU104orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_TU104);
}

//-----------------------------------------------------
// IsTU106orLater()
//
//-----------------------------------------------------
BOOL IsTU106orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_TU106);
}

//-----------------------------------------------------
// IsTU116orLater()
//
//-----------------------------------------------------
BOOL IsTU116orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_TU116);
}

//-----------------------------------------------------
// IsTU117orLater()
//
//-----------------------------------------------------
BOOL IsTU117orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_TU117);
}

//-----------------------------------------------------
// IsGA100orLater()
//
//-----------------------------------------------------
BOOL IsGA100orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_GA100);
}

//-----------------------------------------------------
// IsGA102orLater()
//
//-----------------------------------------------------
BOOL IsGA102orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_GA102);
}

//-----------------------------------------------------
// IsGA103orLater()
//
//-----------------------------------------------------
BOOL IsGA103orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_GA103);
}

//-----------------------------------------------------
// IsGA104orLater()
//
//-----------------------------------------------------
BOOL IsGA104orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_GA104);
}

//-----------------------------------------------------
// IsGA106orLater()
//
//-----------------------------------------------------
BOOL IsGA106orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_GA106);
}

//-----------------------------------------------------
// IsGA107orLater()
//
//-----------------------------------------------------
BOOL IsGA107orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_GA107);
}

//-----------------------------------------------------
// IsAD102orLater()
//
//-----------------------------------------------------
BOOL IsAD102orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_AD102);
}

//-----------------------------------------------------
// IsGH100orLater()
//
//-----------------------------------------------------
BOOL IsGH100orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_GH100);
}

//-----------------------------------------------------
// IsGB100orLater()
//
//-----------------------------------------------------
BOOL IsGB100orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_GB100);
}

//-----------------------------------------------------
// IsG000orLater()
//
//-----------------------------------------------------
BOOL IsG000orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_G000);
}

//-----------------------------------------------------
// IsGM100Arch()
//
// ----------------------------------------------------
BOOL    IsGM100Arch(void)
{
    return ( hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_GM100);
}

//-----------------------------------------------------
// IsGM200Arch()
//
// ----------------------------------------------------
BOOL    IsGM200Arch(void)
{
    return ( hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_GM200);

}

//-----------------------------------------------------
// IsGP100Arch()
//
// ----------------------------------------------------
BOOL    IsGP100Arch(void)
{
    return ( hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_GP100);
}

//-----------------------------------------------------
// IsGV100Arch()
//
// ----------------------------------------------------
BOOL    IsGV100Arch(void)
{
    return ( hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_GV100);
}

//-----------------------------------------------------
// IsTU100Arch()
//
// ----------------------------------------------------
BOOL    IsTU100Arch(void)
{
    return ( hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_TU100);
}

//-----------------------------------------------------
// IsGA100Arch()
//
// ----------------------------------------------------
BOOL    IsGA100Arch(void)
{
    return ( hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_GA100);
}

//-----------------------------------------------------
// IsGH100Arch()
//
// ----------------------------------------------------
BOOL    IsGH100Arch(void)
{
    return ( hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_GH100);
}

//-----------------------------------------------------
// IsGB100Arch()
//
// ----------------------------------------------------
BOOL    IsGB100Arch(void)
{
    return ( hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_GB100);
}

//-----------------------------------------------------
// IsG000Arch()
//
// ----------------------------------------------------
BOOL    IsG000Arch(void)
{
    return ( hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_G000);
}

//-----------------------------------------------------
// GpuArchitecture()
//
// ----------------------------------------------------
char*   GpuArchitecture(void)
{
    char* pGpuArchitecture;

    // Switch on the chip architecture (PMC_BOOT0 value, not RM architecture value)
    switch(hal.chipInfo.Architecture)
    {
        case LW_PMC_BOOT_0_ARCHITECTURE_GM100:;     pGpuArchitecture = "Maxwell";   break;
        case LW_PMC_BOOT_0_ARCHITECTURE_GM200:;     pGpuArchitecture = "Maxwell";   break;
        case LW_PMC_BOOT_0_ARCHITECTURE_GP100:;     pGpuArchitecture = "Pascal";    break;
        case LW_PMC_BOOT_0_ARCHITECTURE_GV100:;     pGpuArchitecture = "Volta";     break;
        case LW_PMC_BOOT_0_ARCHITECTURE_GV110:;     pGpuArchitecture = "Volta";     break;
        case LW_PMC_BOOT_0_ARCHITECTURE_TU100:;     pGpuArchitecture = "Turing";    break;
        case LW_PMC_BOOT_0_ARCHITECTURE_GA100:;     pGpuArchitecture = "Ampere";    break;
        case LW_PMC_BOOT_0_ARCHITECTURE_GH100:;     pGpuArchitecture = "Hopper";    break;
        case LW_PMC_BOOT_0_ARCHITECTURE_GB100:;     pGpuArchitecture = "Blackwell"; break;
        case LW_PMC_BOOT_0_ARCHITECTURE_G000:;      pGpuArchitecture = "G00x";      break;

        default:                                    pGpuArchitecture = "Unknown";   break;
    }
    return pGpuArchitecture;

}

//-----------------------------------------------------
// GpuImplementation()
//
// ----------------------------------------------------
char*   GpuImplementation(void)
{
    char* pGpuImplementation;

    // Switch on the chip architecture (PMC_BOOT0 value, not RM architecture value)
    switch(hal.chipInfo.Architecture)
    {

        case LW_PMC_BOOT_0_ARCHITECTURE_GM100:      // GM1xx architecture

            // Switch on the GM1xx implementation (PMC_BOOT0 value, not RM implementation value)
            switch(hal.chipInfo.Implementation)
            {
                case LW_PMC_BOOT_0_IMPLEMENTATION_4:;   pGpuImplementation = "GM104";  break;
                case LW_PMC_BOOT_0_IMPLEMENTATION_7:;   pGpuImplementation = "GM107";  break;
                case LW_PMC_BOOT_0_IMPLEMENTATION_8:;   pGpuImplementation = "GM108";  break;

                default:;                               pGpuImplementation = "GM2xx";  break;
            }
            break;

        case LW_PMC_BOOT_0_ARCHITECTURE_GM200:      // GM2xx architecture

            // Switch on the GM2xx implementation (PMC_BOOT0 value, not RM implementation value)
            switch(hal.chipInfo.Implementation)
            {
                case LW_PMC_BOOT_0_IMPLEMENTATION_0:;   pGpuImplementation = "GM200";  break;
                case LW_PMC_BOOT_0_IMPLEMENTATION_4:;   pGpuImplementation = "GM204";  break;
                case LW_PMC_BOOT_0_IMPLEMENTATION_6:;   pGpuImplementation = "GM206";  break;

                default:;                               pGpuImplementation = "GM2xx";  break;
            }
            break;

        case LW_PMC_BOOT_0_ARCHITECTURE_GP100:      // GP1xx architecture

            // Switch on the GP1xx implementation (PMC_BOOT0 value, not RM implementation value)
            switch(hal.chipInfo.Implementation)
            {
                case LW_PMC_BOOT_0_IMPLEMENTATION_0:;   pGpuImplementation = "GP100";  break;
                case LW_PMC_BOOT_0_IMPLEMENTATION_1:;   pGpuImplementation = "GP000";  break;
                case LW_PMC_BOOT_0_IMPLEMENTATION_2:;   pGpuImplementation = "GP102";  break;
                case LW_PMC_BOOT_0_IMPLEMENTATION_4:;   pGpuImplementation = "GP104";  break;
                case LW_PMC_BOOT_0_IMPLEMENTATION_6:;   pGpuImplementation = "GP106";  break;
                case LW_PMC_BOOT_0_IMPLEMENTATION_7:;   pGpuImplementation = "GP107";  break;
                case LW_PMC_BOOT_0_IMPLEMENTATION_8:;   pGpuImplementation = "GP108";  break;

                default:;                               pGpuImplementation = "GP1xx";  break;
            }
            break;

        case LW_PMC_BOOT_0_ARCHITECTURE_GV100:      // GV10x architecture

            // Switch on the GV10x implementation (PMC_BOOT0 value, not RM implementation value)
            switch(hal.chipInfo.Implementation)
            {
                case LW_PMC_BOOT_0_IMPLEMENTATION_0:;   pGpuImplementation = "GV100";  break;
                case LW_PMC_BOOT_0_IMPLEMENTATION_B:;   pGpuImplementation = "GV10B";  break;

                default:;                               pGpuImplementation = "GV10x";  break;
            }
            break;

        case LW_PMC_BOOT_0_ARCHITECTURE_GV110:      // GV11x architecture

            // Switch on the GV11x implementation (PMC_BOOT0 value, not RM implementation value)
            switch(hal.chipInfo.Implementation)
            {
                case LW_PMC_BOOT_0_IMPLEMENTATION_B:;   pGpuImplementation = "GV11B";  break;

                default:;                               pGpuImplementation = "GV11x";  break;
            }
            break;

        case LW_PMC_BOOT_0_ARCHITECTURE_TU100:      // TU1xx architecture

            // Switch on the TU1xx implementation (PMC_BOOT0 value, not RM implementation value)
            switch(hal.chipInfo.Implementation)
            {
                case LW_PMC_BOOT_0_IMPLEMENTATION_2:;   pGpuImplementation = "TU102";  break;
                case LW_PMC_BOOT_0_IMPLEMENTATION_4:;   pGpuImplementation = "TU104";  break;
                case LW_PMC_BOOT_0_IMPLEMENTATION_6:;   pGpuImplementation = "TU106";  break;
                case LW_PMC_BOOT_0_IMPLEMENTATION_8:;   pGpuImplementation = "TU116";  break;
                case LW_PMC_BOOT_0_IMPLEMENTATION_7:;   pGpuImplementation = "TU117";  break;

                default:;                               pGpuImplementation = "TU1xx";  break;
            }
            break;

        case LW_PMC_BOOT_0_ARCHITECTURE_GA100:      // GA10x architecture

            // Switch on the GA10x implementation (PMC_BOOT0 value, not RM implementation value)
            switch(hal.chipInfo.Implementation)
            {
                case LW_PMC_BOOT_0_IMPLEMENTATION_0:;   pGpuImplementation = "GA100";  break;

                default:;                               pGpuImplementation = "GA10x";  break;
            }
            break;

        case LW_PMC_BOOT_0_ARCHITECTURE_GH100:      // GH10x architecture

            // Switch on the GH1xx implementation (PMC_BOOT0 value, not RM implementation value)
            switch(hal.chipInfo.Implementation)
            {
                case LW_PMC_BOOT_0_IMPLEMENTATION_0:;   pGpuImplementation = "GH100";  break;
                case LW_PMC_BOOT_0_IMPLEMENTATION_2:;   pGpuImplementation = "GH202";  break;

                default:;                               pGpuImplementation = "GH10X";  break;
            }
            break;

        case LW_PMC_BOOT_0_ARCHITECTURE_GB100:      // GB10x architecture

            // Switch on the GB1xx implementation (PMC_BOOT0 value, not RM implementation value)
            switch(hal.chipInfo.Implementation)
            {
                case LW_PMC_BOOT_0_IMPLEMENTATION_0:;   pGpuImplementation = "GB100";  break;
                
                default:;                               pGpuImplementation = "GB10X";  break;
            }
            break;

        case LW_PMC_BOOT_0_ARCHITECTURE_G000:       // G00x architecture

            // Switch on the G00x implementation (PMC_BOOT0 value, not RM implementation value)
            switch(hal.chipInfo.Implementation)
            {
                case LW_PMC_BOOT_0_IMPLEMENTATION_0:;   pGpuImplementation = "G000";  break;

                default:;                               pGpuImplementation = "G00x";  break;
            }
            break;

        default:                                    // Unknown architecture

            // Set unknown LWxx implementation
            pGpuImplementation = "LWxx";

            break;
    }
    return pGpuImplementation;

}

//-----------------------------------------------------
// GetManualsDir
// - Gets the manuals directory for the current chip
// and the num of paths in the directory
//-----------------------------------------------------
BOOL GetManualsDir(char **pChipInfo, char *pChipClassNum, int *pNumOfPaths)
{
    int i = 0;
    char maxwellPrefix[32] = "maxwell";
    char pascalPrefix[32] = "pascal";
    char voltaPrefix[32] = "volta";
    char turingPrefix[32] = "turing";
    char amperePrefix[32] = "ampere";
    char adaPrefix[32] = "ada";
    char hopperPrefix[32] = "hopper";
    char blackwellPrefix[32] = "blackwell";
    char g00xPrefix[32] = "g00x";
    char dispPrefix[32] = "disp";
    char dpuPrefix[32] = "dpu";
    char t12xPrefix[32] = "t12x/";
    char t21xPrefix[32] = "t21x/";
    char t18xPrefix[32] = "t18x/";
    char t19xPrefix[32] = "t19x/";
    char t23xPrefix[32] = "t23x/";

    strcat(maxwellPrefix, DIR_SLASH);
    strcat(pascalPrefix, DIR_SLASH);
    strcat(voltaPrefix, DIR_SLASH);
    strcat(turingPrefix, DIR_SLASH);
    strcat(amperePrefix, DIR_SLASH);
    strcat(adaPrefix, DIR_SLASH);
    strcat(hopperPrefix, DIR_SLASH);
    strcat(blackwellPrefix, DIR_SLASH);
    strcat(g00xPrefix, DIR_SLASH);

    strcat(dispPrefix, DIR_SLASH);
    strcat(dpuPrefix, DIR_SLASH);

    if(pChipInfo == NULL)
    {
        dprintf("%s(): pChipInfo cannot be NULL.\n", __FUNCTION__);
        return LW_FALSE;
    }

    for(i = 0; i < MAX_PATHS; i++)
    {
        if(pChipInfo[i] == NULL)
        {
            dprintf("%s(): pChipInfo[%d] cannot be NULL.\n", __FUNCTION__, i);
            return LW_FALSE;
        }
    }

    if(pChipClassNum == NULL)
    {
        dprintf("%s(): pChipClassNum cannot be NULL.\n", __FUNCTION__);
        return LW_FALSE;
    }

    if(pNumOfPaths == NULL)
    {
        dprintf("%s(): pNumOfPaths cannot be NULL.\n", __FUNCTION__);
        return LW_FALSE;
    }
    // Default number of paths to one
    *pNumOfPaths = 1;

    if (IsGM107())
    {
        strcpy(pChipInfo[0], maxwellPrefix);
        strcat(pChipInfo[0], "gm107");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v02_04");

        strcpy(pChipInfo[2], dpuPrefix);
        strcat(pChipInfo[2], "v02_02");

        strcpy(pChipClassNum, "LW947");

        *pNumOfPaths = 3;
    }
    else if (IsGM200())
    {
        strcpy(pChipInfo[0], maxwellPrefix);
        strcat(pChipInfo[0], "gm200");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v02_05");

        strcpy(pChipInfo[2], dpuPrefix);
        strcat(pChipInfo[2], "v02_05");

        strcpy(pChipClassNum, "LW957");

        *pNumOfPaths = 3;
    }
    else if (IsGM204())
    {
        strcpy(pChipInfo[0], maxwellPrefix);
        strcat(pChipInfo[0], "gm204");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v02_05");

        strcpy(pChipInfo[2], dpuPrefix);
        strcat(pChipInfo[2], "v02_05");

        strcpy(pChipClassNum, "LW957");

        *pNumOfPaths = 3;
    }
    else if (IsGM206())
    {
        strcpy(pChipInfo[0], maxwellPrefix);
        strcat(pChipInfo[0], "gm206");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v02_06");

        strcpy(pChipInfo[2], dpuPrefix);
        strcat(pChipInfo[2], "v02_05");

        strcpy(pChipClassNum, "LW957");

        *pNumOfPaths = 3;
    }
    else if (IsGP100())
    {
        strcpy(pChipInfo[0], pascalPrefix);
        strcat(pChipInfo[0], "gp100");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v02_07");

        strcpy(pChipInfo[2], dpuPrefix);
        strcat(pChipInfo[2], "v02_07");

        strcpy(pChipClassNum, "LW977");

        *pNumOfPaths = 3;
    }
    else if (IsGP102())
    {
        strcpy(pChipInfo[0], pascalPrefix);
        strcat(pChipInfo[0], "gp102");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v02_08");

        strcpy(pChipInfo[2], dpuPrefix);
        strcat(pChipInfo[2], "v02_08");

        strcpy(pChipClassNum, "LW987");

        *pNumOfPaths = 3;
    }
    else if (IsGP104())
    {
        strcpy(pChipInfo[0], pascalPrefix);
        strcat(pChipInfo[0], "gp104");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v02_08");

        strcpy(pChipInfo[2], dpuPrefix);
        strcat(pChipInfo[2], "v02_08");

        strcpy(pChipClassNum, "LW987");

        *pNumOfPaths = 3;
    }
    else if (IsGP106())
    {
        strcpy(pChipInfo[0], pascalPrefix);
        strcat(pChipInfo[0], "gp106");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v02_08");

        strcpy(pChipInfo[2], dpuPrefix);
        strcat(pChipInfo[2], "v02_08");

        strcpy(pChipClassNum, "LW987");

        *pNumOfPaths = 3;
    }
    else if (IsGP107())
    {
        strcpy(pChipInfo[0], pascalPrefix);
        strcat(pChipInfo[0], "gp107");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v02_08");

        strcpy(pChipInfo[2], dpuPrefix);
        strcat(pChipInfo[2], "v02_08");

        strcpy(pChipClassNum, "LW987");

        *pNumOfPaths = 3;
    }
    else if (IsGP108())
    {
        strcpy(pChipInfo[0], pascalPrefix);
        strcat(pChipInfo[0], "gp108");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v02_08");

        strcpy(pChipInfo[2], dpuPrefix);
        strcat(pChipInfo[2], "v02_08");

        strcpy(pChipClassNum, "LW987");

        *pNumOfPaths = 3;
    }
    else if (IsGV100())
    {
        strcpy(pChipInfo[0], voltaPrefix);
        strcat(pChipInfo[0], "gv100");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v03_00");

        strcpy(pChipInfo[2], dpuPrefix);
        strcat(pChipInfo[2], "v03_00");

        strcpy(pChipClassNum, "LWC37");

        *pNumOfPaths = 3;
    }
    else if (IsTU102())
    {
        strcpy(pChipInfo[0], turingPrefix);
        strcat(pChipInfo[0], "tu102");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v04_00");

        strcpy(pChipClassNum, "LWC57");

        *pNumOfPaths = 2;
    }
    else if (IsTU104())
    {
        strcpy(pChipInfo[0], turingPrefix);
        strcat(pChipInfo[0], "tu104");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v04_00");

        strcpy(pChipClassNum, "LWC57");

        *pNumOfPaths = 2;
    }
    else if (IsTU106())
    {
        strcpy(pChipInfo[0], turingPrefix);
        strcat(pChipInfo[0], "tu106");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v04_00");

        strcpy(pChipClassNum, "LWC57");

        *pNumOfPaths = 2;
    }
    else if (IsTU116())
    {
        strcpy(pChipInfo[0], turingPrefix);
        strcat(pChipInfo[0], "tu116");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v04_00");

        strcpy(pChipClassNum, "LWC57");

        *pNumOfPaths = 2;
    }
    else if (IsTU117())
    {
        strcpy(pChipInfo[0], turingPrefix);
        strcat(pChipInfo[0], "tu117");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v04_00");

        strcpy(pChipClassNum, "LWC57");

        *pNumOfPaths = 2;
    }
    else if (IsGA100())
    {
        strcpy(pChipInfo[0], amperePrefix);
        strcat(pChipInfo[0], "ga100");

        // GA100 has no display at all
    }
    else if (IsGA102())
    {
        strcpy(pChipInfo[0], amperePrefix);
        strcat(pChipInfo[0], "ga102");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v04_01");

        strcpy(pChipClassNum, "LWC67");

        *pNumOfPaths = 2;
    }
    else if (IsGA103())
    {
        strcpy(pChipInfo[0], amperePrefix);
        strcat(pChipInfo[0], "ga103");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v04_01");

        strcpy(pChipClassNum, "LWC67");

        *pNumOfPaths = 2;
    }
    else if (IsGA104())
    {
        strcpy(pChipInfo[0], amperePrefix);
        strcat(pChipInfo[0], "ga104");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v04_01");

        strcpy(pChipClassNum, "LWC67");

        *pNumOfPaths = 2;
    }
    else if (IsGA106())
    {
        strcpy(pChipInfo[0], amperePrefix);
        strcat(pChipInfo[0], "ga106");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v04_01");

        strcpy(pChipClassNum, "LWC67");

        *pNumOfPaths = 2;
    }
    else if (IsGA107())
    {
        strcpy(pChipInfo[0], amperePrefix);
        strcat(pChipInfo[0], "ga107");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v04_01");

        strcpy(pChipClassNum, "LWC67");

        *pNumOfPaths = 2;
    }
    else if (IsAD102())
    {
        strcpy(pChipInfo[0], adaPrefix);
        strcat(pChipInfo[0], "ad102");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v04_04");

        strcpy(pChipClassNum, "LWC77");

        *pNumOfPaths = 2;
    }
    else if (IsAD103())
    {
        strcpy(pChipInfo[0], adaPrefix);
        strcat(pChipInfo[0], "ad103");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v04_04");

        strcpy(pChipClassNum, "LWC77");

        *pNumOfPaths = 2;
    }
    else if (IsAD104())
    {
        strcpy(pChipInfo[0], adaPrefix);
        strcat(pChipInfo[0], "ad104");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v04_04");

        strcpy(pChipClassNum, "LWC77");

        *pNumOfPaths = 2;
    }
    else if (IsAD106())
    {
        strcpy(pChipInfo[0], adaPrefix);
        strcat(pChipInfo[0], "ad106");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v04_04");

        strcpy(pChipClassNum, "LWC77");

        *pNumOfPaths = 2;
    }
    else if (IsAD107())
    {
        strcpy(pChipInfo[0], adaPrefix);
        strcat(pChipInfo[0], "ad107");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v04_04");

        strcpy(pChipClassNum, "LWC77");

        *pNumOfPaths = 2;
    }
    else if (IsGH100())
    {
        strcpy(pChipInfo[0], hopperPrefix);
        strcat(pChipInfo[0], "gh100");

        // GH100 has no display at all
    }
    else if (IsGH202())
    {
        strcpy(pChipInfo[0], hopperPrefix);
        strcat(pChipInfo[0], "gh202");

        // GH202 has no display defined yet
    }
    else if (IsGB100())
    {
        strcpy(pChipInfo[0], blackwellPrefix);
        strcat(pChipInfo[0], "gb100");

        // GB100 has no display 
    }
    else if (IsG000())
    {
        strcpy(pChipInfo[0], g00xPrefix);
        strcat(pChipInfo[0], "g000");
    }
    else if (IsT124())
    {
        strcpy(pChipInfo[0], t12xPrefix);
        strcat(pChipInfo[0], "t124");
    }
    else if (IsT210())
    {
        strcpy(pChipInfo[0], t21xPrefix);
        strcat(pChipInfo[0], "t210");
    }
    else if (IsT186())
    {
        strcpy(pChipInfo[0], t18xPrefix);
        strcat(pChipInfo[0], "t186");
    }
    else if (IsT194())
    {
        strcpy(pChipInfo[0], t19xPrefix);
        strcat(pChipInfo[0], "t194");
    }
    else if (IsT234())
    {
        strcpy(pChipInfo[0], t23xPrefix);
        strcat(pChipInfo[0], "t234");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v04_02");

        *pNumOfPaths = 2;
    }
    else
    {
        dprintf("Unknown or unsupported lWpu GPU. Ensure that %s() supports the chip you're working on.\n", __FUNCTION__);
        return FALSE;
    }

    return LW_TRUE;
}

//-----------------------------------------------------
// GetDispManualsDir
// - Gets the display manuals directory for the current chip.
// This will work for all G80+ plus chips.This is needed to
// provide data abstraction over changing folder names for disp manuals.
//-----------------------------------------------------
BOOL GetDispManualsDir(char *dispManualPath)
{
    char *pChipManualsDir[MAX_PATHS];
    char *pClassNum = NULL;
    int   numPaths = 1;
    BOOL status = LW_TRUE;
    int i = 0;

    if(dispManualPath == NULL)
    {
        dprintf("%s(): dispManualPath cannot be NULL.\n", __FUNCTION__);
        return LW_FALSE;
    }

    for(i = 0; i < MAX_PATHS; i++)
    {
        pChipManualsDir[i] = (char *)malloc(32  * sizeof(char));
    }
    pClassNum = (char *)malloc(32  * sizeof(char));

    if(!GetManualsDir(pChipManualsDir, pClassNum, &numPaths))
    {
        dprintf("\n: Unknown or unsupported lWpu GPU. Ensure that %s() supports the chip you're working on.\n",
            __FUNCTION__);
        status =  LW_FALSE;
        goto Cleanup;
    }

    if(numPaths == 1)
    {
        strcpy(dispManualPath, pChipManualsDir[0]);
    }
    else if(numPaths == 3)
    {
        strcpy(dispManualPath, pChipManualsDir[1]);
    }
    else
    {
        dprintf("\n:%s(): Unknown or unsupported lWpu GPU. Ensure that LwWatch supports the chip you're working on.\n",
            __FUNCTION__);
        status = LW_FALSE;
    }

Cleanup:
    // Lets free the char array
    for(i = 0; i < MAX_PATHS; i++)
    {
        free(pChipManualsDir[i]);
    }
    free(pClassNum);
    return status;
}

//-----------------------------------------------------
// GetClassNum
// - Gets the class number for the given chip.
//-----------------------------------------------------
BOOL GetClassNum(char *pClassNum)
{
    char *pChipManualsDir[MAX_PATHS];
    int   numPaths = 1;
    BOOL status = LW_TRUE;
    int i = 0;

    if(pClassNum == NULL)
    {
        dprintf("%s(): classNum cannot be NULL.\n", __FUNCTION__);
        return LW_FALSE;
    }

    for(i = 0; i < MAX_PATHS; i++)
    {
        pChipManualsDir[i] = (char *)malloc(32  * sizeof(char));
    }

    if(!GetManualsDir(pChipManualsDir, pClassNum, &numPaths))
    {
        dprintf("\n: Unknown or unsupported lWpu GPU. Ensure that %s() supports the chip you're working on.\n",
            __FUNCTION__);
        status =  LW_FALSE;
        goto Cleanup;
    }

Cleanup:
    // Lets free the char array
    for(i = 0; i < MAX_PATHS; i++)
    {
        free(pChipManualsDir[i]);
    }

    return status;
}
