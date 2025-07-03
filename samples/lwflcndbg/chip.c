/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2003-2011 by LWPU Corporation.  All rights reserved.  All
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
#include "lw_ref.h"
#include "chip.h"
#include "dac.h"
#include "hal.h"
#include "maxwell/gm200/dev_master.h" // Include only the latest chip's dev_master
#include "tegrasys.h"

//-----------------------------------------------------
// GetChipAndRevision
// - Returns chip and revision if either parameter
//   pointer is not NULL.
// - Corrected for LW11 B3 silicon
//-----------------------------------------------------
VOID GetChipAndRevision(U032 *pChip, U032 *pRevision)
{
    U032 Chip, Revision;
    U032 PllCompat;

    Chip     = ((hal.chipInfo.Architecture << 4)| hal.chipInfo.Implementation);
    Revision = hal.chipInfo.MaskRevision;

    if(pChip)
      *pChip = Chip;

    if(pRevision)
    {
        *pRevision = Revision;

        //
        // Special case for LW11 B03 detection
        // LW11 B3 silicon will show up as B2 silicon in PMC_BOOT_0
        // We need to read another register to test if it's B3 silcon.
        //
        if((Chip == 0x11) && (Revision == 0xB2))
        {
            PllCompat = GPU_REG_RD32(LW_PRAMDAC_PLL_COMPAT);
            PllCompat &= ~(DRF_DEF(_PRAMDAC, _PLL_COMPAT, _CHIP_REV, _B03));
            GPU_REG_WR32(LW_PRAMDAC_PLL_COMPAT, PllCompat);
            PllCompat = GPU_REG_RD32(LW_PRAMDAC_PLL_COMPAT);

            if(PllCompat & DRF_DEF(_PRAMDAC, _PLL_COMPAT, _CHIP_REV, _B03))
            {
                Revision = 0xB3;
            }
        }
    }
}

BOOL bIsSocBrdg = FALSE;

//-----------------------------------------------------
// IsTegra()
//
//-----------------------------------------------------
LwU32 isTegraHack = 0;
BOOL IsTegra()
{
    return (IsT30() || IsT114() || IsT124() || IsT148() || IsT210());
}

//-----------------------------------------------------
// IsT30()
//
//-----------------------------------------------------
BOOL IsT30()
{
    LwU32 data = ((isTegraHack >> 8) & 0xff);
    return (data == 0x30);
}

//-----------------------------------------------------
// IsT114()
//
//-----------------------------------------------------
BOOL IsT114()
{
    LwU32 data = ((isTegraHack >> 8) & 0xff);
    return (data == 0x35);
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
// IsT148()
//
//-----------------------------------------------------
BOOL IsT148()
{
    LwU32 data = ((isTegraHack >> 8) & 0xff);
    return (data == 0x14);
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
// IsSocBrdg()
//
//-----------------------------------------------------
BOOL IsSocBrdg()
{
    return bIsSocBrdg;
}

//-----------------------------------------------------
// IsRSX()
//
//-----------------------------------------------------
BOOL IsRSX()
{
    return (hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_LW40 &&
            hal.chipInfo.Implementation == LW_PMC_BOOT_0_IMPLEMENTATION_D);
}

//-----------------------------------------------------
// IsLW11()
// - Returns 1 if Chip is LW11, otherwise 0
//-----------------------------------------------------
BOOL IsLW11()
{
    return (hal.halImpl == LWHAL_IMPL_LW11);
}

//-----------------------------------------------------
// IsLW15()
// - Usefull for internal FPRegs or not
//-----------------------------------------------------
BOOL IsLW15()
{
    return (hal.halImpl == LWHAL_IMPL_LW15);
}

//-----------------------------------------------------
// IsLW17()
// - Returns 1 if Chip is LW17, otherwise 0
//-----------------------------------------------------
BOOL IsLW17()
{
    return (hal.halImpl == LWHAL_IMPL_LW17);
}

//-----------------------------------------------------
// IsLW18()
// - Returns 1 if Chip is LW18, otherwise 0
//-----------------------------------------------------
BOOL IsLW18()
{
    return (hal.halImpl == LWHAL_IMPL_LW18);
}

//-----------------------------------------------------
// IsLW31()
//
//-----------------------------------------------------
BOOL IsLW31()
{
    return (hal.halImpl == LWHAL_IMPL_LW31);
}

//-----------------------------------------------------
// IsLW36()
//
//-----------------------------------------------------
BOOL IsLW36()
{
    return (hal.halImpl == LWHAL_IMPL_LW36);
}

//-----------------------------------------------------
// IsLW40()
//
//-----------------------------------------------------
BOOL IsLW40()
{
    return (hal.halImpl == LWHAL_IMPL_LW40);
}

//-----------------------------------------------------
// IsLW41()
//
//-----------------------------------------------------
BOOL IsLW41()
{
    return (hal.halImpl == LWHAL_IMPL_LW41);
}

//-----------------------------------------------------
// IsLW43()
//
//-----------------------------------------------------
BOOL IsLW43()
{
    return (hal.halImpl == LWHAL_IMPL_LW43);
}

//-----------------------------------------------------
// IsLW44()
//
//-----------------------------------------------------
BOOL IsLW44()
{
    return (hal.halImpl == LWHAL_IMPL_LW44);
}

//-----------------------------------------------------
// IsLW46()
//
//-----------------------------------------------------
BOOL IsLW46()
{
     return (hal.halImpl == LWHAL_IMPL_LW46);
}

//-----------------------------------------------------
// IsLW47()
//
//-----------------------------------------------------
BOOL IsLW47()
{
     return (hal.halImpl == LWHAL_IMPL_LW47);
}

//-----------------------------------------------------
// IsLW49()
//
//-----------------------------------------------------
BOOL IsLW49()
{
     return (hal.halImpl == LWHAL_IMPL_LW49);
}

//-----------------------------------------------------
// IsLW4C()
//
//-----------------------------------------------------
BOOL IsLW4C()
{
     return (hal.halImpl == LWHAL_IMPL_LW4C);
}

//-----------------------------------------------------
// IsLW4E()
//
//-----------------------------------------------------
BOOL IsLW4E()
{
    return (hal.halImpl == LWHAL_IMPL_LW4E);
}

//-----------------------------------------------------
// IsLW63()
//
//-----------------------------------------------------
BOOL IsLW63()
{
    return (hal.halImpl == LWHAL_IMPL_LW63);
}

//-----------------------------------------------------
// IsLW67()
//
//-----------------------------------------------------
BOOL IsLW67()
{
    return (hal.halImpl == LWHAL_IMPL_LW67);
}

//-----------------------------------------------------
// IsLW50()
//
//-----------------------------------------------------
BOOL IsLW50()
{
    return (hal.halImpl == LWHAL_IMPL_LW50);
}

//
//-----------------------------------------------------
// IsG78()
//
//-----------------------------------------------------
BOOL IsG78()
{
    return (hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_LW60 &&
            hal.chipInfo.Implementation == LW_PMC_BOOT_0_IMPLEMENTATION_1);
}

//-----------------------------------------------------
// IsLW50orBetter()
//-----------------------------------------------------
BOOL IsLW50orBetter()
{
    return (hal.halImpl >= LWHAL_IMPL_LW50);
}

//
//-----------------------------------------------------
// IsG82()
//
//-----------------------------------------------------
BOOL IsG82()
{
    return (hal.halImpl == LWHAL_IMPL_G82);
}

//
//-----------------------------------------------------
// IsG84()
//
//-----------------------------------------------------
BOOL IsG84()
{
    return (hal.halImpl == LWHAL_IMPL_G84);
}

//
//-----------------------------------------------------
// IsG86()
//
//-----------------------------------------------------
BOOL IsG86()
{
    return (hal.halImpl == LWHAL_IMPL_G86);
}

//-----------------------------------------------------
// IsG92()
//
//-----------------------------------------------------
BOOL IsG92()
{
    return (hal.halImpl == LWHAL_IMPL_G92);
}

//-----------------------------------------------------
// IsG94()
//
//-----------------------------------------------------
BOOL IsG94()
{
    return (hal.halImpl == LWHAL_IMPL_G94);
}

//-----------------------------------------------------
// IsG96()
//
//-----------------------------------------------------
BOOL IsG96()
{
    return (hal.halImpl == LWHAL_IMPL_G96);
}

//-----------------------------------------------------
// IsG98()
//
//-----------------------------------------------------
BOOL IsG98()
{
    return (hal.halImpl == LWHAL_IMPL_G98);
}
//-----------------------------------------------------
// IsGT2XX()
//
//-----------------------------------------------------
BOOL IsGT2XX()
{
    return ( hal.chipInfo.Architecture >=  LW_PMC_BOOT_0_ARCHITECTURE_G100 );

}
//-----------------------------------------------------
// IsGT200()
//
//-----------------------------------------------------
BOOL IsGT200()
{
    return (hal.halImpl == LWHAL_IMPL_GT200);

}
//-----------------------------------------------------
// IsGT206()
//
//-----------------------------------------------------
BOOL IsGT206()
{
    return (hal.halImpl == LWHAL_IMPL_dGT206);

}
//-----------------------------------------------------
// IsiGT206()
//
//-----------------------------------------------------
BOOL IsiGT206()
{
     return (hal.halImpl == LWHAL_IMPL_iGT206);

}
//-----------------------------------------------------
// IsMCP77()
//
//-----------------------------------------------------
BOOL IsMCP77()
{
     return (hal.halImpl == LWHAL_IMPL_MCP77);

}
//-----------------------------------------------------
// IsMCP79()
//
//-----------------------------------------------------
BOOL IsMCP79()
{
     return (hal.halImpl == LWHAL_IMPL_MCP79);

}
//-----------------------------------------------------
// IsGT21X()
//
//-----------------------------------------------------
BOOL IsGT21X()
{
    LwU32 retVal = ((hal.halImpl == LWHAL_IMPL_GT214) ||
             (hal.halImpl == LWHAL_IMPL_GT215) ||
             (hal.halImpl == LWHAL_IMPL_GT216) ||
             (hal.halImpl == LWHAL_IMPL_GT218) ||
             (hal.halImpl == LWHAL_IMPL_iGT21A) );

    return retVal;
}

//-----------------------------------------------------
// IsGT215()
//
//-----------------------------------------------------
BOOL IsGT215()
{
    return (hal.halImpl == LWHAL_IMPL_GT215);
}

//-----------------------------------------------------
// IsGT216()
//
//-----------------------------------------------------
BOOL IsGT216()
{
    return (hal.halImpl == LWHAL_IMPL_GT216);
}

//-----------------------------------------------------
// IsGT218()
//
//-----------------------------------------------------
BOOL IsGT218()
{
    return (hal.halImpl == LWHAL_IMPL_GT218);
}

//-----------------------------------------------------
// IsiGT21A()
//
//-----------------------------------------------------
BOOL IsiGT21A()
{
    return (hal.halImpl == LWHAL_IMPL_iGT21A);
}
//-----------------------------------------------------
// IsMCP89()
//
//-----------------------------------------------------
BOOL IsMCP89()
{
     return (hal.halImpl == LWHAL_IMPL_MCP89);

}

//-----------------------------------------------------
// IsGF100()
//
//-----------------------------------------------------
BOOL IsGF100()
{
    return (hal.halImpl == LWHAL_IMPL_GF100);
}

//-----------------------------------------------------
// IsGF100B()
//
//-----------------------------------------------------
BOOL IsGF100B()
{
    return (hal.halImpl == LWHAL_IMPL_GF100B);
}

//-----------------------------------------------------
// IsGF104()
//
//-----------------------------------------------------
BOOL IsGF104()
{
    return (hal.halImpl == LWHAL_IMPL_GF104);
}

//-----------------------------------------------------
// IsGF106()
//
//-----------------------------------------------------
BOOL IsGF106()
{
    return (hal.halImpl == LWHAL_IMPL_GF106);
}

//-----------------------------------------------------
// IsGF108()
//
//-----------------------------------------------------
BOOL IsGF108()
{
    return (hal.halImpl == LWHAL_IMPL_GF108);
}

//-----------------------------------------------------
// IsGF110D()
//
//-----------------------------------------------------
BOOL IsGF110D()
{
    return (hal.halImpl == LWHAL_IMPL_GF110D);
}

//-----------------------------------------------------
// IsGF110F()
//
//-----------------------------------------------------
BOOL IsGF110F()
{
    return (hal.halImpl == LWHAL_IMPL_GF110F);
}

//-----------------------------------------------------
// IsGF110F2()
//
//-----------------------------------------------------
BOOL IsGF110F2()
{
    return (hal.halImpl == LWHAL_IMPL_GF110F2);
}

//-----------------------------------------------------
// IsGF110F3()
//
//-----------------------------------------------------
BOOL IsGF110F3()
{
    return (hal.halImpl == LWHAL_IMPL_GF110F3);
}

//-----------------------------------------------------
// IsGF117()
//
//-----------------------------------------------------
BOOL IsGF117()
{
    return (hal.halImpl == LWHAL_IMPL_GF117);
}

//-----------------------------------------------------
// IsGF119()
//
//-----------------------------------------------------
BOOL IsGF119()
{
    return (hal.halImpl == LWHAL_IMPL_GF119);
}

//-----------------------------------------------------
// IsGK104()
//
//-----------------------------------------------------
BOOL IsGK104()
{
    return (hal.halImpl == LWHAL_IMPL_GK104);
}
//-----------------------------------------------------
// IsGK106()
//
//-----------------------------------------------------
BOOL IsGK106()
{
    return (hal.halImpl == LWHAL_IMPL_GK106);
}

//-----------------------------------------------------
// IsGK107()
//
//-----------------------------------------------------
BOOL IsGK107()
{
    return (hal.halImpl == LWHAL_IMPL_GK107);
}

//-----------------------------------------------------
// IsGK110()
//
//-----------------------------------------------------
BOOL IsGK110()
{
    return (hal.halImpl == LWHAL_IMPL_GK110);
}

//-----------------------------------------------------
// IsGK208()
//
//-----------------------------------------------------
BOOL IsGK208()
{
    return (hal.halImpl == LWHAL_IMPL_GK208);
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
// IsGK20A()
//
//-----------------------------------------------------
BOOL IsGK20A()
{
    return (hal.halImpl == LWHAL_IMPL_GK20A);
}

//-----------------------------------------------------
// IsGK104orLater()
//
//-----------------------------------------------------
BOOL IsGK104orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_GK104);
}

//-----------------------------------------------------
// IsGK110orLater()
//
//-----------------------------------------------------
BOOL IsGK110orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_GK110);
}

//-----------------------------------------------------
// IsGK208orLater()
//
//-----------------------------------------------------
BOOL IsGK208orLater()
{
    return (hal.halImpl >= LWHAL_IMPL_GK208);
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
// IsLW10Arch()
//
//-----------------------------------------------------
BOOL IsLW10Arch()
{
    return (hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_LW10);
}

//-----------------------------------------------------
// IsLW20Arch()
//
//-----------------------------------------------------
BOOL IsLW20Arch()
{
    return (hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_LW20);
}

//-----------------------------------------------------
// IsLW30Arch()
//
//-----------------------------------------------------
BOOL IsLW30Arch()
{
    return ( hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_LW30);
}

//-----------------------------------------------------
// IsLW40Arch()
// xxx - need to clean this up
//-----------------------------------------------------
BOOL IsLW40Arch()
{
    return ( (hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_LW40) ||
             (hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_LW60) );
}

//-----------------------------------------------------
// IsLW50Arch()
// xxx - If IsLW40Arch needs to clean up, then this too.
//-----------------------------------------------------
BOOL IsLW50Arch()
{
    return ( hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_LW50);
}

//-----------------------------------------------------
// IsG80Arch()
//
// ----------------------------------------------------
BOOL IsG80Arch()
{
    return ( hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_LW80);

}

//-----------------------------------------------------
// IsG90Arch()
//
// ----------------------------------------------------
BOOL IsG90Arch()
{
    return ( hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_LW90);
}

//-----------------------------------------------------
// IsGT200Arch()
//
// ----------------------------------------------------
BOOL IsGT200Arch()
{
    return ( hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_G100);

}

//-----------------------------------------------------
// IsGF100Arch()
//
// ----------------------------------------------------
BOOL IsGF100Arch()
{
    return ( hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_GF100);

}

//-----------------------------------------------------
// IsGF110Arch()
//
// ----------------------------------------------------
BOOL    IsGF110Arch(void)
{
    return ( hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_GF110);

}

//-----------------------------------------------------
// IsGK100Arch()
//
// ----------------------------------------------------
BOOL    IsGK100Arch(void)
{
    return ( hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_GK100);

}

//-----------------------------------------------------
// IsGK110Arch()
//
// ----------------------------------------------------
BOOL    IsGK110Arch(void)
{
    return ( hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_GK110);

}

//-----------------------------------------------------
// IsGK208Arch()
//
// ----------------------------------------------------
BOOL    IsGK200Arch(void)
{
    return ( hal.chipInfo.Architecture == LW_PMC_BOOT_0_ARCHITECTURE_GK200);

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
// IsLW15orBetter()
// - Usefull for internal FPRegs or not
//-----------------------------------------------------
BOOL IsLW15orBetter(void)
{
    // Is this LW15 or greater, or LW11, but not LW1A, nor LW2A
 return( ( ((hal.halImpl >=  LWHAL_IMPL_LW15) || (hal.halImpl == LWHAL_IMPL_LW11)) &&
           !(hal.halImpl == LWHAL_IMPL_LW1A) ) ||
         ( (hal.halImpl >= LWHAL_IMPL_LW25) )||
         (hal.chipInfo.Architecture >= LW_PMC_BOOT_0_ARCHITECTURE_LW30) );
}

//-----------------------------------------------------
// IsLW17orBetter()
// - Usefull for Core4 VBIOSes
//-----------------------------------------------------
BOOL IsLW17orBetter()
{
    // Is this LW17 or greater, but not LW1A
    // Or is this LW25 or greater but not LW2A
    // Or is this LW30 or greater
    return ( ( (hal.halImpl >= LWHAL_IMPL_LW17) &&
              !(hal.halImpl == LWHAL_IMPL_LW1A) ) ||
             ( (hal.halImpl >= LWHAL_IMPL_LW25) ) ||
             (hal.chipInfo.Architecture >= LW_PMC_BOOT_0_ARCHITECTURE_LW30) );
}

//-----------------------------------------------------
// IsLW18orBetter()
// - Usefull for # of DVO Ports
//-----------------------------------------------------
BOOL IsLW18orBetter()
{
    return ( ( (hal.halImpl == LWHAL_IMPL_LW18) ||
               (hal.halImpl == LWHAL_IMPL_LW1F) ) ||
             (hal.chipInfo.Architecture >= LW_PMC_BOOT_0_ARCHITECTURE_LW30) );
}

//-----------------------------------------------------
// IsLW25orBetter()
//
//-----------------------------------------------------
BOOL IsLW25orBetter()
{
    return (hal.halImpl >= LWHAL_IMPL_LW25);
}

//-----------------------------------------------------
// IsLW30orBetter()
//-----------------------------------------------------
BOOL IsLW30orBetter()
{
    return (hal.chipInfo.Architecture >= LW_PMC_BOOT_0_ARCHITECTURE_LW30);
}

//-----------------------------------------------------
// IsLW40orBetter()
//-----------------------------------------------------
BOOL IsLW40orBetter()
{
    return (hal.chipInfo.Architecture >= LW_PMC_BOOT_0_ARCHITECTURE_LW40);
}

//-----------------------------------------------------
// IsLW41orBetter()
//-----------------------------------------------------
BOOL IsLW41orBetter()
{
    return (hal.halImpl >= LWHAL_IMPL_LW41);
}


//-----------------------------------------------------
// GetNumCrtcs()
// - Returns 2 if LW11, LW17, LW18, LW25, or 30 or greater
// - otherwise 1.
//-----------------------------------------------------
U032 GetNumCrtcs()
{
    U032 NumCrtcs = 1;

    if(IsLW11() || IsLW17orBetter())
      NumCrtcs = 2;

    return NumCrtcs;
}

//-----------------------------------------------------
// GetNumLinks()
// - Returns 1 if LW15
// - Returns 2 if LW11
// - Returns 3 if LW17, LW18, LW3x or LW40
// - Returns 4 if LW41+
// - otherwise 0.
//-----------------------------------------------------
U032 GetNumLinks()
{
    U032 NumLinks = 0;

    if(IsLW15())
      NumLinks = 1;
    else if(IsLW11())
      NumLinks = 2;
    else if(IsLW17() || IsLW18() || IsLW30Arch() || IsLW40())
      NumLinks = 3;
    else if(IsLW41orBetter())
      NumLinks = 4;

    return NumLinks;
}

//-----------------------------------------------------
// EnableHead
// - Switches the head access and returns the previous head
//-----------------------------------------------------
U032 EnableHead(U032 Head)
{
    U008 prevCr44, OldLock, Cr44;

    if(GetNumCrtcs() == 0x1)
      return 0;

    // Unlock Head A regs
    OldLock = UnlockExtendedCRTCs(0);

    if(Head == 0x1)
      Cr44 = 0x3;
    else
      Cr44 = 0x0;

    // Read Cr44 from Head A
    prevCr44 = REG_RDCR(0x44, 0);
    REG_WRCR(0x44, Cr44, 0);

    // Restore Head A Lock
    RestoreExtendedCRTCs(OldLock, 0);

    if(prevCr44 == 0x3)
      return 1;
    else
      return 0;
}

//-----------------------------------------------------
// GetMaxCrtcReg
// - Return the max CR reg number
//-----------------------------------------------------
U032 GetMaxCrtcReg()
{
    if(IsLW18orBetter())
      return 0x9f;
    else if(IsLW17orBetter())
      return 0x9e;
    else // use LW11's default
      return 0x52;
}

//-----------------------------------------------------
// GetMaxTMDSReg
// - Return the max TMDS reg number
//-----------------------------------------------------
U032 GetMaxTMDSReg()
{
    if(IsLW15())
      return 0x31;
    else if(IsLW11())
      return 0x29;
    else if(IsLW17() || IsLW18() || IsLW30orBetter())
      return 0x3f;
    else
      return 0;
}


//-----------------------------------------------------
// GetManualsDir
// - Gets the manuals directory for the current chip 
// and the num of paths in the directory
//-----------------------------------------------------
BOOL GetManualsDir(char **pChipInfo, char *pChipClassNum, int *pNumOfPaths)
{
    int i = 0;
    char fermiPrefix[32] = "fermi";
    char keplerPrefix[32] = "kepler";
    char maxwellPrefix[32] = "maxwell";
    char dispPrefix[32] = "disp";
    char dpuPrefix[32] = "dpu";
    char t124Prefix[32] = "t12x";

    strcat(fermiPrefix, DIR_SLASH);
    strcat(keplerPrefix, DIR_SLASH);
    strcat(maxwellPrefix, DIR_SLASH);
    strcat(dispPrefix, DIR_SLASH);
    strcat(dpuPrefix, DIR_SLASH);
    strcat(t124Prefix, DIR_SLASH);

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

    *pNumOfPaths = 1;

    if (IsG82())
    {
        strcpy(pChipInfo[0], "g82");
        strcpy(pChipClassNum, "LW827");
    }
    else if (IsG84())
    {
        strcpy(pChipInfo[0], "g84");
        strcpy(pChipClassNum, "LW827");
    }
    else if (IsG86())
    {
        strcpy(pChipInfo[0], "g86");
        strcpy(pChipClassNum, "LW827");
    }
    else if (IsG78())
    {
        strcpy(pChipInfo[0], "g78");
        dprintf("There is no disp class for this chip!\n");
        pChipClassNum[0] = '\0';
    }
    else if (IsG92())
    {
        strcpy(pChipInfo[0], "g92");
        strcpy(pChipClassNum, "LW827");
    }
    else if (IsG94())
    {
        strcpy(pChipInfo[0], "g94");
        strcpy(pChipClassNum, "LW887");
    }
    else if (IsG96())
    {
        strcpy(pChipInfo[0], "g96");
        strcpy(pChipClassNum, "LW887");
    }
    else if (IsG98())
    {
        strcpy(pChipInfo[0], "g98");
        strcpy(pChipClassNum, "LW887");
    }
    else if (IsLW50())
    {
        strcpy(pChipInfo[0], "lw50");
        strcpy(pChipClassNum, "LW507");
    }
    else if (IsLW49())
    {
        strcpy(pChipInfo[0], "lw49");
        dprintf("There is no disp class for this chip!\n");
        pChipClassNum[0] = '\0';
    }
    else if (IsLW47())
    {
        strcpy(pChipInfo[0], "lw47");
        dprintf("There is no disp class for this chip!\n");
        pChipClassNum[0] = '\0';
    }
    else if (IsLW46())
    {
        strcpy(pChipInfo[0], "lw46");
        dprintf("There is no disp class for this chip!\n");
        pChipClassNum[0] = '\0';
    }
    else if (IsLW44())
    {
        strcpy(pChipInfo[0], "lw44");
        dprintf("There is no disp class for this chip!\n");
        pChipClassNum[0] = '\0';
    }
    else if (IsLW43())
    {
        strcpy(pChipInfo[0], "lw43");
        dprintf("There is no disp class for this chip!\n");
        pChipClassNum[0] = '\0';
    }
    else if (IsLW41())
    {
        strcpy(pChipInfo[0], "lw41");
        dprintf("There is no disp class for this chip!\n");
        pChipClassNum[0] = '\0';
    }
    else if (IsLW40())
    {
        strcpy(pChipInfo[0], "lw40");
        dprintf("There is no disp class for this chip!\n");
        pChipClassNum[0] = '\0';
    }
    else if (IsGT200())
    {
        strcpy(pChipInfo[0], "gt200");
        strcpy(pChipClassNum, "LW837");
    }
    else if (IsGT206())
    {
        strcpy(pChipInfo[0], "gt206");
        strcpy(pChipClassNum, "LW887");
    }
    else if (IsiGT206() || IsMCP77())
    {
        strcpy(pChipInfo[0], "igt206");
        strcpy(pChipClassNum, "LW887");
    }
    else if (IsMCP79())
    {
        strcpy(pChipInfo[0], "igt209");
        strcpy(pChipClassNum, "LW887");
    }
    else if (IsGT215())
    {
        strcpy(pChipInfo[0], "gt215");
        strcpy(pChipClassNum, "LW857");
    }
    else if (IsGT216())
    {
        strcpy(pChipInfo[0], "gt216");
        strcpy(pChipClassNum, "LW857");
    }
    else if (IsGT218())
    {
        strcpy(pChipInfo[0], "gt218");
        strcpy(pChipClassNum, "LW857");
    }
    else if (IsiGT21A() || IsMCP89())
    {
        strcpy(pChipInfo[0], "igt21a");
        strcpy(pChipClassNum, "LW857");
    }
    else if (IsGF100())
    {
        strcpy(pChipInfo[0], fermiPrefix);
        strcat(pChipInfo[0], "gf100");
        strcpy(pChipClassNum, "LW857");
    }
    else if (IsGF100B())
    {
        strcpy(pChipInfo[0], fermiPrefix);
        strcat(pChipInfo[0], "gf100b");
        strcpy(pChipClassNum, "LW857");
    }
    else if (IsGF104())
    {
        strcpy(pChipInfo[0], fermiPrefix);
        strcat(pChipInfo[0], "gf104");
        strcpy(pChipClassNum, "LW857");
    }
    else if (IsGF106())
    {
        strcpy(pChipInfo[0], fermiPrefix);
        strcat(pChipInfo[0], "gf106");
        strcpy(pChipClassNum, "LW857");
    }
    else if (IsGF108())
    {
        strcpy(pChipInfo[0], fermiPrefix);
        strcat(pChipInfo[0], "gf108");
        strcpy(pChipClassNum, "LW857");
    }
    else if (IsGF110D() || IsGF110F())
    {
        strcpy(pChipInfo[0], fermiPrefix);
        strcat(pChipInfo[0], "gf110");
        strcpy(pChipClassNum, "LW907");
    }
    else if (IsGF117())
    {
        strcpy(pChipInfo[0], fermiPrefix);
        strcat(pChipInfo[0], "gf117");
        strcpy(pChipClassNum, "LW907");
    }
    else if (IsGF119()) 
    {
        strcpy(pChipInfo[0], fermiPrefix);
        strcat(pChipInfo[0], "gf119");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v02_00");

        strcpy(pChipInfo[2], dpuPrefix);
        strcat(pChipInfo[2], "v02_00");

        strcpy(pChipClassNum, "LW907");

        *pNumOfPaths = 3;
    }
    else if (IsGK104())
    {
        strcpy(pChipInfo[0], keplerPrefix);
        strcat(pChipInfo[0], "gk104");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v02_01");

        strcpy(pChipInfo[2], dpuPrefix);
        strcat(pChipInfo[2], "v02_01");

        strcpy(pChipClassNum, "LW917");

        *pNumOfPaths = 3;
    }
    else if (IsGK106())
    {
        strcpy(pChipInfo[0], keplerPrefix);
        strcat(pChipInfo[0], "gk106");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v02_01");

        strcpy(pChipInfo[2], dpuPrefix);
        strcat(pChipInfo[2], "v02_01");

        strcpy(pChipClassNum, "LW917");

        *pNumOfPaths = 3;
    }
    else if (IsGK107())
    {
        strcpy(pChipInfo[0], keplerPrefix);
        strcat(pChipInfo[0], "gk107");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v02_01");

        strcpy(pChipInfo[2], dpuPrefix);
        strcat(pChipInfo[2], "v02_01");

        strcpy(pChipClassNum, "LW917");

        *pNumOfPaths = 3;
    }
    else if (IsGK110())
    {
        strcpy(pChipInfo[0], keplerPrefix);
        strcat(pChipInfo[0], "gk110");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v02_02");

        strcpy(pChipInfo[2], dpuPrefix);
        strcat(pChipInfo[2], "v02_02");

        strcpy(pChipClassNum, "LW927");

        *pNumOfPaths = 3;
    }
    else if (IsGK208())
    {
        strcpy(pChipInfo[0], keplerPrefix);
        strcat(pChipInfo[0], "gk208");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v02_03");

        strcpy(pChipInfo[2], dpuPrefix);
        strcat(pChipInfo[2], "v02_02");

        strcpy(pChipClassNum, "LW927");

        *pNumOfPaths = 3;
    }
    else if (IsGM107())
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
        strcat(pChipInfo[2], "v02_02");

        strcpy(pChipClassNum, "LW947");

        *pNumOfPaths = 3;
    }
    else if (IsT124())
    {
        strcpy(pChipInfo[0], t124Prefix);
        strcat(pChipInfo[0], "t124");

        strcpy(pChipInfo[1], dispPrefix);
        strcat(pChipInfo[1], "v02_01");

        strcpy(pChipInfo[2], dpuPrefix);
        strcat(pChipInfo[2], "v02_01");

        strcpy(pChipClassNum, "LW927");

        *pNumOfPaths = 3;
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

/** 
 * @brief Determines if specified device ID corresponds to a SOC bridge device
 * 
 * @param[in] deviceId 
 *
 * @return LW_TRUE if bridge device, LW_FALSE if not
 */
LwBool socbrdgIsBridgeDevid(LwU32 deviceId)
{
    // Only one device supported right now.
    return (deviceId == 0xFA7);
}
