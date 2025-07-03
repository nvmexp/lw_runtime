/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2003-2016 by LWPU Corporation.  All rights reserved.  All
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
// clk.c
//
//*****************************************************

//
// includes
//
#include "clk.h"

typedef struct 
{
    CLKSRCWHICH clkSrc;
    char        clkName[128];
} clkMap;

const char* getClkSrcName(CLKSRCWHICH clkSrcWhich)
{
    static clkMap clkNameList[] = 
    { 
        //this has to be in same order as CLKSRCWHICH enumeration
        { clkSrcWhich_Default           , "????Default" },
        { clkSrcWhich_XTAL              , "XTAL" },
        { clkSrcWhich_XTALS             , "XTALS (Spread)" },
        { clkSrcWhich_XTAL4X            , "XTAL4X" },
        { clkSrcWhich_EXTREF            , "EXT RefClk" },
        { clkSrcWhich_QUALEXTREF        , "QUALEXTREF" },
        { clkSrcWhich_EXTSPREAD         , "External Spread" },
        { clkSrcWhich_SPPLL0            , "SPPLL0" },
        { clkSrcWhich_SPPLL1            , "SPPLL1" },
        { clkSrcWhich_XCLK              , "XCLK" },
        { clkSrcWhich_XCLK3XDIV2        , "XCLK3XDIV2" },
        { clkSrcWhich_MPLL              , "MPLL" },
        { clkSrcWhich_HOSTCLK           , "HostClk" },
        { clkSrcWhich_PEXREFCLK         , "PEXREFCLK" },
        { clkSrcWhich_VPLL0             , "VPLL0" },
        { clkSrcWhich_VPLL1             , "VPLL1" },
        { clkSrcWhich_VPLL2             , "VPLL2" },
        { clkSrcWhich_VPLL3             , "VPLL3" },
        { clkSrcWhich_PWRCLK            , "PWRCLK" },
        { clkSrcWhich_HDACLK            , "HDACLK" },
        { clkSrcWhich_GPCPLL            , "GPCPLL" },
        { clkSrcWhich_LTCPLL            , "LTCPLL" },
        { clkSrcWhich_XBARPLL           , "XBARPLL" },
        { clkSrcWhich_SYSPLL            , "SYSPLL" },
        { clkSrcWhich_REFMPLL           , "REFMPLL" },
        { clkSrcWhich_XCLK500           , "XCLK500" },
        { clkSrcWhich_XCLKGEN3          , "XCLKGEN3" },
        { clkSrcWhich_Supported         , "Supported" },
    };

#define CLKNAMELSTSIZE  (sizeof(clkNameList)/sizeof(clkMap))

    if (clkSrcWhich >= 0 && 
        clkSrcWhich < CLKNAMELSTSIZE)
    {
        return clkNameList[clkSrcWhich].clkName;
    }

    // Invalid CLKSRCWHICH passed in
    dprintf("lw: %s: Invalid clkSrc (%d) passed !!\n", 
            __FUNCTION__, clkSrcWhich);
    return "";
}
