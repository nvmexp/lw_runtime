/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2006-2014 by LWPU Corporation.  All rights reserved.  All
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
// hal.c
//
//*****************************************************

// Setup all enabled gpus
#define LWWATCHCFG_ENGINE_SETUP   1
#define LWWATCHCFG_HAL_SETUP_ALL  1

//
// includes
//
#include "hal.h"
#include "chip.h"
#include "pmu.h"
#include "socbrdg.h"
#include "tegrasys.h"
#include "pascal/gp107/dev_master.h" // Include only the latest chip's dev_master

#include "g_hal_private.h"          // (rmconfig) hal/obj setup


//
// global halObject
//
Hal hal;

int indexGpu;

PMU_LWHAL_IFACES pPmu[MAX_GPUS];
DPU_LWHAL_IFACES pDpu[MAX_GPUS];
SOCBRDG_LWHAL_IFACES pSocbrdg[MAX_GPUS];
TEGRASYS_LWHAL_IFACES pTegrasys[MAX_GPUS];

TEGRASYS TegraSysObj[MAX_GPUS];

// Mapping of chip implementations to strings.  From enum LWHAL_IMPLEMENTATION in hal.f
// TODO: check this into hal.h or hal.c so it's maintained centrally.
#ifndef QUOTE_ME
#define QUOTE_ME(x) #x
#endif

static char* gpuImplementationNames[LWHAL_IMPL_MAXIMUM] = 
{
    QUOTE_ME( LWHAL_IMPL_T20 ),
    QUOTE_ME( LWHAL_IMPL_T30 ),
    QUOTE_ME( LWHAL_IMPL_T114 ),
    QUOTE_ME( LWHAL_IMPL_T124 ),
    QUOTE_ME( LWHAL_IMPL_T148 ),
    QUOTE_ME( LWHAL_IMPL_T210 ),
    QUOTE_ME( LWHAL_IMPL_LW04 ),
    QUOTE_ME( LWHAL_IMPL_LW10 ),
    QUOTE_ME( LWHAL_IMPL_LW11 ),
    QUOTE_ME( LWHAL_IMPL_LW15 ),
    QUOTE_ME( LWHAL_IMPL_LW1A ),
    QUOTE_ME( LWHAL_IMPL_LW1F ),
    QUOTE_ME( LWHAL_IMPL_LW17 ),
    QUOTE_ME( LWHAL_IMPL_LW18 ),
    QUOTE_ME( LWHAL_IMPL_LW20 ),
    QUOTE_ME( LWHAL_IMPL_LW25 ),
    QUOTE_ME( LWHAL_IMPL_LW28 ),
    QUOTE_ME( LWHAL_IMPL_LW30 ),
    QUOTE_ME( LWHAL_IMPL_LW31 ),
    QUOTE_ME( LWHAL_IMPL_LW34 ),
    QUOTE_ME( LWHAL_IMPL_LW35 ),
    QUOTE_ME( LWHAL_IMPL_LW36 ),
    QUOTE_ME( LWHAL_IMPL_LW40 ),
    QUOTE_ME( LWHAL_IMPL_LW41 ),
    QUOTE_ME( LWHAL_IMPL_LW42 ),
    QUOTE_ME( LWHAL_IMPL_LW43 ),
    QUOTE_ME( LWHAL_IMPL_LW44 ),
    QUOTE_ME( LWHAL_IMPL_LW44A ),
    QUOTE_ME( LWHAL_IMPL_LW46 ),
    QUOTE_ME( LWHAL_IMPL_LW47 ),
    QUOTE_ME( LWHAL_IMPL_LW48 ),
    QUOTE_ME( LWHAL_IMPL_LW49 ),
    QUOTE_ME( LWHAL_IMPL_LW4B ),
    QUOTE_ME( LWHAL_IMPL_LW4C ),
    QUOTE_ME( LWHAL_IMPL_LW4E ),
    QUOTE_ME( LWHAL_IMPL_LW63 ),
    QUOTE_ME( LWHAL_IMPL_LW67 ),
    QUOTE_ME( LWHAL_IMPL_LW50 ),
    QUOTE_ME( LWHAL_IMPL_G80 ),
    QUOTE_ME( LWHAL_IMPL_G82 ),
    QUOTE_ME( LWHAL_IMPL_G84 ),
    QUOTE_ME( LWHAL_IMPL_G86 ),
    QUOTE_ME( LWHAL_IMPL_G92 ),
    QUOTE_ME( LWHAL_IMPL_G94 ),
    QUOTE_ME( LWHAL_IMPL_G96 ),
    QUOTE_ME( LWHAL_IMPL_G98 ),
    QUOTE_ME( LWHAL_IMPL_GT200 ),
    QUOTE_ME( LWHAL_IMPL_dGT206 ),
    QUOTE_ME( LWHAL_IMPL_iGT206 ),
    QUOTE_ME( LWHAL_IMPL_MCP77 ),
    QUOTE_ME( LWHAL_IMPL_iGT209 ),
    QUOTE_ME( LWHAL_IMPL_MCP79 ),
    QUOTE_ME( LWHAL_IMPL_GT214 ),
    QUOTE_ME( LWHAL_IMPL_GT215 ),
    QUOTE_ME( LWHAL_IMPL_GT216 ),
    QUOTE_ME( LWHAL_IMPL_GT218 ),
    QUOTE_ME( LWHAL_IMPL_iGT21A ),
    QUOTE_ME( LWHAL_IMPL_MCP89 ),
    QUOTE_ME( LWHAL_IMPL_GF100 ),
    QUOTE_ME( LWHAL_IMPL_GF100B ),
    QUOTE_ME( LWHAL_IMPL_GF104 ),
    QUOTE_ME( LWHAL_IMPL_GF106 ),
    QUOTE_ME( LWHAL_IMPL_GF108 ),
    QUOTE_ME( LWHAL_IMPL_GF110D ),
    QUOTE_ME( LWHAL_IMPL_GF110F ),
    QUOTE_ME( LWHAL_IMPL_GF110F2 ),
    QUOTE_ME( LWHAL_IMPL_GF110F3 ),
    QUOTE_ME( LWHAL_IMPL_GF117 ),
    QUOTE_ME( LWHAL_IMPL_GF119 ),
    QUOTE_ME( LWHAL_IMPL_GK104 ),
    QUOTE_ME( LWHAL_IMPL_GK106 ),
    QUOTE_ME( LWHAL_IMPL_GK107 ),
    QUOTE_ME( LWHAL_IMPL_GK107B ),
    QUOTE_ME( LWHAL_IMPL_GK110 ),
    QUOTE_ME( LWHAL_IMPL_GK208 ),
    QUOTE_ME( LWHAL_IMPL_GM107 ),
    QUOTE_ME( LWHAL_IMPL_GM200 ),
    QUOTE_ME( LWHAL_IMPL_GM204 ),
    QUOTE_ME( LWHAL_IMPL_GM206 ),
    QUOTE_ME( LWHAL_IMPL_GP100 ),
    QUOTE_ME( LWHAL_IMPL_GP107 ),
    QUOTE_ME( LWHAL_IMPL_GP108 ),
    QUOTE_ME( LWHAL_IMPL_GK20A ),
};
 
// static function prototypes
//
static void findTegraHal(LwU32);
static void findLW4XHal(LwU32);
static void findLW6XHal(LwU32);
static void findG8XHal(LwU32);
static void findG9XHal(LwU32);
static void findGT2XXHal(LwU32);
static void findGF10XHal(LwU32);
static void findGF10XFHal(LwU32);
static void findGF11XHal(LwU32);
static void findLwWatchHal(LwU32);
static void findGK10XHal(LwU32);
static void findGK11XHal(LwU32);
static void findGK20XHal(LwU32);
static void findGM10XHal(LwU32);
static void findGM20XHal(LwU32);
static void findGP10XHal(LwU32);
//
// static function definitions 
//
static void
findTegraHal
(
 LwU32 thisPmcBoot0
)
{
#if (LWWATCHCFG_CHIP_ENABLED(T30))
    if (IsT30())
    {
        if (verboseLevel)
        {
            dprintf("flcndbg: Wiring up T30 routines.\n");
        }
        hal.pHal = &lwhalIface_T30;
        hal.halImpl = LWHAL_IMPL_T30;
        hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_0; 
        return;
    }
#endif

#if (LWWATCHCFG_CHIP_ENABLED(T114))
    if (IsT114())
    {
        if (verboseLevel)
        {
            dprintf("flcndbg: Wiring up T114 routines.\n");
        }
        hal.pHal = &lwhalIface_T114;
        hal.halImpl = LWHAL_IMPL_T114;
        hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_0; 
        return;
    }
#endif

#if (LWWATCHCFG_CHIP_ENABLED(T124))
    if (IsT124())
    {
        if (verboseLevel)
        {
            dprintf("flcndbg: Wiring up T124 routines.\n");
        }
        hal.pHal = &lwhalIface_T124;
        hal.halImpl = LWHAL_IMPL_T124;
        hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_0; 
        return;
    }
#endif

#if (LWWATCHCFG_CHIP_ENABLED(T148))
    if (IsT148())
    {
        if (verboseLevel)
        {
            dprintf("flcndbg: Wiring up T148 routines.\n");
        }
        hal.pHal = &lwhalIface_T148;
        hal.halImpl = LWHAL_IMPL_T148;
        hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_0; 
        return;
    }
#endif

#if (LWWATCHCFG_CHIP_ENABLED(T210))
    if (IsT210())
    {
        if (verboseLevel)
        {
            dprintf("flcndbg: Wiring up T210 routines.\n");
        }
        hal.pHal = &lwhalIface_T210;
        hal.halImpl = LWHAL_IMPL_T210;
        hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_0; 
        return;
    }
#endif
}

static void
findLW4XHal
(
 LwU32 thisPmcBoot0
)
{
    switch(DRF_VAL(_PMC, _BOOT_0, _IMPLEMENTATION, thisPmcBoot0))
    {
#if (LWWATCHCFG_CHIP_ENABLED(LW40))
        case LW_PMC_BOOT_0_IMPLEMENTATION_0: 
        case LW_PMC_BOOT_0_IMPLEMENTATION_8: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up LW40 routines.\n");
            }

            hal.pHal = &lwhalIface_LW40;
            hal.halImpl = LWHAL_IMPL_LW40;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_0; 
            return;
#endif
#if (LWWATCHCFG_CHIP_ENABLED(LW41))
        case LW_PMC_BOOT_0_IMPLEMENTATION_1: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up LW41 routines.\n");
            }

            hal.pHal = &lwhalIface_LW41;
            hal.halImpl = LWHAL_IMPL_LW41;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_1; 
            return;
#endif
#if (LWWATCHCFG_CHIP_ENABLED(LW42))
        case LW_PMC_BOOT_0_IMPLEMENTATION_2: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up LW42 routines.\n");
            }

            hal.pHal = &lwhalIface_LW42;
            hal.halImpl = LWHAL_IMPL_LW42;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_2; 
            return;
#endif
#if (LWWATCHCFG_CHIP_ENABLED(LW43))
        case LW_PMC_BOOT_0_IMPLEMENTATION_3: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up LW43 routines.\n");
            }

            hal.pHal = &lwhalIface_LW43;
            hal.halImpl = LWHAL_IMPL_LW43;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_3; 
            return;
#endif
#if (LWWATCHCFG_CHIP_ENABLED(LW44))
        case LW_PMC_BOOT_0_IMPLEMENTATION_4: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up LW44 routines.\n");
            }

            hal.pHal = &lwhalIface_LW44;
            hal.halImpl = LWHAL_IMPL_LW44;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_4; 
            return;
#endif
#if (LWWATCHCFG_CHIP_ENABLED(LW46))
        case LW_PMC_BOOT_0_IMPLEMENTATION_6: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up LW40 routines.\n");
            }

            hal.pHal = &lwhalIface_LW46;
            hal.halImpl = LWHAL_IMPL_LW46;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_6; 
            return;
#endif
#if (LWWATCHCFG_CHIP_ENABLED(LW47))
        case LW_PMC_BOOT_0_IMPLEMENTATION_7: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up LW47 routines.\n");
            }

            hal.pHal = &lwhalIface_LW47;
            hal.halImpl = LWHAL_IMPL_LW47;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_7; 
            return;
#endif
#if (LWWATCHCFG_CHIP_ENABLED(LW49))
        case LW_PMC_BOOT_0_IMPLEMENTATION_9: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up LW49 routines.\n");
            }

            hal.pHal = &lwhalIface_LW49;
            hal.halImpl = LWHAL_IMPL_LW49;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_9; 
            return;
#endif
#if (LWWATCHCFG_CHIP_ENABLED(LW44A))
        case LW_PMC_BOOT_0_IMPLEMENTATION_A: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up LW4A routines.\n");
            }

            hal.pHal = &lwhalIface_LW44A;
            hal.halImpl = LWHAL_IMPL_LW44A;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_A; 
            return;
#endif
#if (LWWATCHCFG_CHIP_ENABLED(LW4B))
        case LW_PMC_BOOT_0_IMPLEMENTATION_B: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up LW4B routines.\n");
            }

            hal.pHal = &lwhalIface_LW4B;
            hal.halImpl = LWHAL_IMPL_LW4B;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_B; 
            return;
#endif
#if (LWWATCHCFG_CHIP_ENABLED(LW4C))
        case LW_PMC_BOOT_0_IMPLEMENTATION_C: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up LW4C routines.\n");
            }

            hal.pHal = &lwhalIface_LW4C;
            hal.halImpl = LWHAL_IMPL_LW4C;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_C; 
            return;
#endif
#if (LWWATCHCFG_CHIP_ENABLED(LW4E))
        case LW_PMC_BOOT_0_IMPLEMENTATION_E: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up LW4E routines.\n");
            }

            hal.pHal = &lwhalIface_LW4E;
            hal.halImpl = LWHAL_IMPL_LW4E;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_E; 
            return;
#endif
#if (LWWATCHCFG_CHIP_ENABLED(LW40))
        default:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up LW40 routines.\n");
            }

            hal.pHal = &lwhalIface_LW40;
            hal.halImpl = LWHAL_IMPL_LW40;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_0; 
            return;
#endif
    }
}

static void
findLW6XHal
(
 LwU32 thisPmcBoot0
)
{
    switch(DRF_VAL(_PMC, _BOOT_0, _IMPLEMENTATION, thisPmcBoot0))
    {
#if (LWWATCHCFG_CHIP_ENABLED(LW63))
        case LW_PMC_BOOT_0_IMPLEMENTATION_3: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up LW63 routines.\n");
            }

            hal.pHal = &lwhalIface_LW63;
            hal.halImpl = LWHAL_IMPL_LW63;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_3; 
            return;
#endif
#if (LWWATCHCFG_CHIP_ENABLED(LW67))
        case LW_PMC_BOOT_0_IMPLEMENTATION_7: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up LW67 routines.\n");
            }

            hal.pHal = &lwhalIface_LW67;
            hal.halImpl = LWHAL_IMPL_LW67;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_7; 
            return;
#endif
#if (LWWATCHCFG_CHIP_ENABLED(LW63))
        default:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up LW63 routines.\n");
            }

            hal.pHal = &lwhalIface_LW63;
            hal.halImpl = LWHAL_IMPL_LW63;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_3; 
            return;
#endif
    }
}

#if (LWWATCHCFG_CHIP_ENABLED(G8X))
static void
findG8XHal
(
 LwU32 thisPmcBoot0
)
{
    switch(DRF_VAL(_PMC, _BOOT_0, _IMPLEMENTATION, thisPmcBoot0))
    {
        case LW_PMC_BOOT_0_IMPLEMENTATION_4:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up G84 routines.\n");
            }

            hal.pHal = &lwhalIface_G84;
            hal.halImpl = LWHAL_IMPL_G84;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_4; 
            return;
        case LW_PMC_BOOT_0_IMPLEMENTATION_6:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up G86 routines.\n");
            }

            hal.pHal = &lwhalIface_G86;
            hal.halImpl = LWHAL_IMPL_G86;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_6; 
            return;
        default:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up G84 routines.\n");
            }
            hal.pHal = &lwhalIface_G84;
            hal.halImpl = LWHAL_IMPL_G84;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_4; 
            return;
    }

}
#endif

#if (LWWATCHCFG_CHIP_ENABLED(G9X))
static void
findG9XHal
(
 LwU32 thisPmcBoot0
)
{
    switch(DRF_VAL(_PMC, _BOOT_0, _IMPLEMENTATION, thisPmcBoot0))
    {
        case LW_PMC_BOOT_0_IMPLEMENTATION_2: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up G92 routines.\n");
            }


            hal.pHal = &lwhalIface_G92;
            hal.halImpl = LWHAL_IMPL_G92;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_2; 
            return;
        case LW_PMC_BOOT_0_IMPLEMENTATION_4:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up G94 routines.\n");
            }


            hal.pHal = &lwhalIface_G94;
            hal.halImpl = LWHAL_IMPL_G94;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_4; 
            return;
        case LW_PMC_BOOT_0_IMPLEMENTATION_6:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up G96 routines.\n");
            }


            hal.pHal = &lwhalIface_G96;
            hal.halImpl = LWHAL_IMPL_G96;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_6; 
            return;
        case LW_PMC_BOOT_0_IMPLEMENTATION_8:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up G98 routines.\n");
            }


            hal.pHal = &lwhalIface_G98;
            hal.halImpl = LWHAL_IMPL_G98;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_8; 
            return;
        default:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up G92 routines.\n");
            }


            hal.pHal = &lwhalIface_G92;
            hal.halImpl = LWHAL_IMPL_G92;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_2; 
            return;
    }
}
#endif

#if LWWATCHCFG_CHIP_ENABLED(GT20X) || LWWATCHCFG_CHIP_ENABLED(GT21X)
static void
findGT2XXHal
(
 LwU32 thisPmcBoot0
)
{
    switch(DRF_VAL(_PMC, _BOOT_0, _IMPLEMENTATION, thisPmcBoot0))
    {
        case LW_PMC_BOOT_0_IMPLEMENTATION_0: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GT200 routines.\n");
            }


            hal.pHal = &lwhalIface_GT200;
            hal.halImpl = LWHAL_IMPL_GT200;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_0; 
            return;
        case LW_PMC_BOOT_0_IMPLEMENTATION_3:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GT215 routines.\n");
            }


            hal.pHal = &lwhalIface_GT215;
            hal.halImpl = LWHAL_IMPL_GT215;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_3; 
            return;
        case LW_PMC_BOOT_0_IMPLEMENTATION_5:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GT216 routines.\n");
            }


            hal.pHal = &lwhalIface_GT216;
            hal.halImpl = LWHAL_IMPL_GT216;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_5;
            return;
        case LW_PMC_BOOT_0_IMPLEMENTATION_8:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GT218 routines.\n");
            }


            hal.pHal = &lwhalIface_GT218;
            hal.halImpl = LWHAL_IMPL_GT218;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_8; 
            return;
        case LW_PMC_BOOT_0_IMPLEMENTATION_A:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up MCP77 routines.\n");
            }


            hal.pHal = &lwhalIface_MCP77;
            hal.halImpl = LWHAL_IMPL_MCP77;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_A; 
            return;
        case LW_PMC_BOOT_0_IMPLEMENTATION_C:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up MCP79 routines.\n");
            }


            hal.pHal = &lwhalIface_MCP79;
            hal.halImpl = LWHAL_IMPL_MCP79;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_C; 
            return;
        case LW_PMC_BOOT_0_IMPLEMENTATION_D:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up iGT21A routines.\n");
            }


            hal.pHal = &lwhalIface_iGT21A;
            hal.halImpl = LWHAL_IMPL_iGT21A;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_D; 
            return;
        case LW_PMC_BOOT_0_IMPLEMENTATION_F:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up MCP89 routines.\n");
            }


            hal.pHal = &lwhalIface_MCP89;
            hal.halImpl = LWHAL_IMPL_MCP89;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_F; 
            return;
        default:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GT200 routines.\n");
            }


            hal.pHal = &lwhalIface_GT200;
            hal.halImpl = LWHAL_IMPL_GT200;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_0; 
            return;
    }
}
#endif

#if LWWATCHCFG_CHIP_ENABLED(GF10X)
static void
findGF10XHal
(
 LwU32 thisPmcBoot0
)
{
    switch(DRF_VAL(_PMC, _BOOT_0, _IMPLEMENTATION, thisPmcBoot0))
    {
        case LW_PMC_BOOT_0_IMPLEMENTATION_0: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GF100 routines.\n");
            }

            hal.pHal = &lwhalIface_GF100;
            hal.halImpl = LWHAL_IMPL_GF100;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_0;
            return;
        case LW_PMC_BOOT_0_IMPLEMENTATION_4:
        case LW_PMC_BOOT_0_IMPLEMENTATION_E:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GF104 routines.\n");
            }

            hal.pHal = &lwhalIface_GF104;
            hal.halImpl = LWHAL_IMPL_GF104;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_4;    
            return;
        case LW_PMC_BOOT_0_IMPLEMENTATION_3: 
        case LW_PMC_BOOT_0_IMPLEMENTATION_F:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GF106 routines.\n");
            }

            hal.pHal = &lwhalIface_GF106;
            hal.halImpl = LWHAL_IMPL_GF106;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_3;    
            return;
        case LW_PMC_BOOT_0_IMPLEMENTATION_1: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GF108 routines.\n");
            }

            hal.pHal = &lwhalIface_GF108;
            hal.halImpl = LWHAL_IMPL_GF108;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_1;    
            return;
        case LW_PMC_BOOT_0_IMPLEMENTATION_8:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GF110 routines.\n");
            }

            hal.pHal = &lwhalIface_GF100B;
            hal.halImpl = LWHAL_IMPL_GF100B;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_8;
            return;
        default:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GF100 routines.\n");
            }

            hal.pHal = &lwhalIface_GF100;
            hal.halImpl = LWHAL_IMPL_GF100;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_0; 
            return;
    }
}
#endif


#if LWWATCHCFG_CHIP_ENABLED(GF110F) || LWWATCHCFG_CHIP_ENABLED(GF110F2) || LWWATCHCFG_CHIP_ENABLED(GF110F3)

static void
findGF10XFHal
(
 LwU32 thisPmcBoot0
)
{
    LwU32 impl = DRF_VAL(_PMC, _BOOT_0, _IMPLEMENTATION, thisPmcBoot0);

    switch (impl)
    {
        case LW_PMC_BOOT_0_IMPLEMENTATION_B: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GF110F2 routines.\n");
            }

            hal.pHal = &lwhalIface_GF110F2;
            hal.halImpl = LWHAL_IMPL_GF110F2;
            hal.chipInfo.Implementation = impl;
            return;

        case LW_PMC_BOOT_0_IMPLEMENTATION_C: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GF110F3 routines.\n");
            }

            hal.pHal = &lwhalIface_GF110F3;
            hal.halImpl = LWHAL_IMPL_GF110F3;
            hal.chipInfo.Implementation = impl;
            return;
    }
}
#endif

#if LWWATCHCFG_CHIP_ENABLED(GF11X)

static void
findGF11XHal
(
    LwU32 thisPmcBoot0
)
{
    LwU32 impl = DRF_VAL(_PMC, _BOOT_0, _IMPLEMENTATION, thisPmcBoot0);

    switch (impl)
    {
        default:
            impl = LW_PMC_BOOT_0_IMPLEMENTATION_7;

        case LW_PMC_BOOT_0_IMPLEMENTATION_7: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GF117 routines.\n");
            }

            hal.pHal = &lwhalIface_GF117;
            hal.halImpl = LWHAL_IMPL_GF117;
            break;

        case LW_PMC_BOOT_0_IMPLEMENTATION_9: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GF119 routines.\n");
            }

            hal.pHal = &lwhalIface_GF119;
            hal.halImpl = LWHAL_IMPL_GF119;
            break;
    }
    hal.chipInfo.Implementation = impl; 
}
#endif


#if LWWATCHCFG_CHIP_ENABLED(GK10X) || LWWATCHCFG_CHIP_ENABLED(GK20A)

static void
findGK10XHal
(
    LwU32 thisPmcBoot0
)
{
    LwU32 impl = DRF_VAL(_PMC, _BOOT_0, _IMPLEMENTATION, thisPmcBoot0);

    switch (impl)
    {
        default:
            impl = LW_PMC_BOOT_0_IMPLEMENTATION_4;

        case LW_PMC_BOOT_0_IMPLEMENTATION_4: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GK104 routines.\n");
            }

            hal.pHal = &lwhalIface_GK104;
            hal.halImpl = LWHAL_IMPL_GK104;
            break;
        case LW_PMC_BOOT_0_IMPLEMENTATION_6:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GK106 routines.\n");
            }
            hal.pHal = &lwhalIface_GK106;
            hal.halImpl = LWHAL_IMPL_GK106;
            break;
        case LW_PMC_BOOT_0_IMPLEMENTATION_7:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GK107 routines.\n");
            }

            hal.pHal = &lwhalIface_GK107;
            hal.halImpl = LWHAL_IMPL_GK107;
            break;
#if LWWATCHCFG_CHIP_ENABLED(GK20A)
        case LW_PMC_BOOT_0_IMPLEMENTATION_A:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GK20A/GK20A routines.\n");
            }

            hal.pHal = &lwhalIface_GK20A;
            hal.halImpl = LWHAL_IMPL_GK20A;
            break;
#endif
    }
    hal.chipInfo.Implementation = impl; 
}
#endif

/*!
 * @brief Find the display hal setup function for FPGA based on implementation.
 *        Checks if ARCHITECTURE is GF100 and IMPLEMENTATION is either _7, _B
 *        or _C to find if the platform is FPGA.
 * 
 * @param[in] thisPmcBoot0 PMC_BOOT_0 register value
 */
static void
findDispFpgaHal
(
    LwU32 thisPmcBoot0
)
{
    LwU32 impl = DRF_VAL(_PMC, _BOOT_0, _IMPLEMENTATION, thisPmcBoot0);
    LwU32 arch = DRF_VAL(_PMC, _BOOT_0, _ARCHITECTURE, thisPmcBoot0);

    if (arch != LW_PMC_BOOT_0_ARCHITECTURE_GF100)
    {
        return;
    }

    switch (impl)
    {
#if (LWWATCHCFG_HAL_SETUP_GF119 || LWWATCHCFG_HAL_SETUP_GF11X || LWWATCHCFG_HAL_SETUP_ALL)
        case LW_PMC_BOOT_0_IMPLEMENTATION_7:

           // GF110F implementation
            return;
#endif         
#if (LWWATCHCFG_HAL_SETUP_GK104 || LWWATCHCFG_HAL_SETUP_GK10X || LWWATCHCFG_HAL_SETUP_ALL)   
        case LW_PMC_BOOT_0_IMPLEMENTATION_B:

            // GF110F2 implementation
            return;
#endif   
#if (LWWATCHCFG_HAL_SETUP_GK110 || LWWATCHCFG_HAL_SETUP_GK11X || LWWATCHCFG_HAL_SETUP_ALL)
        case LW_PMC_BOOT_0_IMPLEMENTATION_C:

            // GF110F3 implementation
            return;
#endif         
        default:
            return;
    }
}


#if LWWATCHCFG_CHIP_ENABLED(GK110)

static void
findGK11XHal
(
    LwU32 thisPmcBoot0
)
{
    LwU32 impl = DRF_VAL(_PMC, _BOOT_0, _IMPLEMENTATION, thisPmcBoot0);

    switch (impl)
    {
        default:
            impl = LW_PMC_BOOT_0_IMPLEMENTATION_0;

        case LW_PMC_BOOT_0_IMPLEMENTATION_0: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GK110 routines.\n");
            }

            hal.pHal = &lwhalIface_GK110;
            hal.halImpl = LWHAL_IMPL_GK110;
            break;

    }
    hal.chipInfo.Implementation = impl; 
}
#endif

#if LWWATCHCFG_CHIP_ENABLED(GK208)

static void
findGK20XHal
(
    LwU32 thisPmcBoot0
)
{
    LwU32 impl = DRF_VAL(_PMC, _BOOT_0, _IMPLEMENTATION, thisPmcBoot0);

    switch (impl)
    {
        default:
            impl = LW_PMC_BOOT_0_IMPLEMENTATION_8;
        case LW_PMC_BOOT_0_IMPLEMENTATION_8: 
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GK208 routines.\n");
            }

            hal.halImpl = LWHAL_IMPL_GK208;
            hal.pHal = &lwhalIface_GK208;
            break;

    }
    hal.chipInfo.Implementation = impl; 
}
#endif

#if LWWATCHCFG_CHIP_ENABLED(GM107)
static void
findGM10XHal
(
    LwU32 thisPmcBoot0
)
{
    LwU32 impl = DRF_VAL(_PMC, _BOOT_0, _IMPLEMENTATION, thisPmcBoot0);

    switch (impl)
    {
        default:
            impl = LW_PMC_BOOT_0_IMPLEMENTATION_7;

        case LW_PMC_BOOT_0_IMPLEMENTATION_7:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GM107 routines.\n");
            }

            hal.pHal = &lwhalIface_GM107;
            hal.halImpl = LWHAL_IMPL_GM107;
            break;

    }
    hal.chipInfo.Implementation = impl; 
}
#endif

#if LWWATCHCFG_CHIP_ENABLED(GM200) || LWWATCHCFG_CHIP_ENABLED(GM204) || LWWATCHCFG_CHIP_ENABLED(GM206)
static void
findGM20XHal
(
    LwU32 thisPmcBoot0
)
{
    LwU32 impl = DRF_VAL(_PMC, _BOOT_0, _IMPLEMENTATION, thisPmcBoot0);

    switch (impl)
    {
        default:
            impl = LW_PMC_BOOT_0_IMPLEMENTATION_0;

        case LW_PMC_BOOT_0_IMPLEMENTATION_0:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GM200 routines.\n");
            }

            hal.pHal = &lwhalIface_GM200;
            hal.halImpl = LWHAL_IMPL_GM200;
            break;

        case LW_PMC_BOOT_0_IMPLEMENTATION_4:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GM204 routines.\n");
            }

            hal.pHal = &lwhalIface_GM204;
            hal.halImpl = LWHAL_IMPL_GM204;
            break;

        case LW_PMC_BOOT_0_IMPLEMENTATION_6:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GM206 routines.\n");
            }
            
            hal.pHal = &lwhalIface_GM206;
            hal.halImpl = LWHAL_IMPL_GM206;
            break;
    }
    hal.chipInfo.Implementation = impl; 
}
#endif

#if LWWATCHCFG_CHIP_ENABLED(GP100) || LWWATCHCFG_CHIP_ENABLED(GP102) || LWWATCHCFG_CHIP_ENABLED(GP104) || LWWATCHCFG_CHIP_ENABLED(GP106) || LWWATCHCFG_CHIP_ENABLED(GP107) || LWWATCHCFG_CHIP_ENABLED(GP108)
static void
findGP10XHal
(
    LwU32 thisPmcBoot0
)
{
    LwU32 impl = DRF_VAL(_PMC, _BOOT_0, _IMPLEMENTATION, thisPmcBoot0);

    switch (impl)
    {
        default:
            impl = LW_PMC_BOOT_0_IMPLEMENTATION_0;

        case LW_PMC_BOOT_0_IMPLEMENTATION_0:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GP100 routines.\n");
            }

            hal.pHal = &lwhalIface_GP100;
            hal.halImpl = LWHAL_IMPL_GP100;
            break;

         case LW_PMC_BOOT_0_IMPLEMENTATION_2:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GP102 routines.\n");
            }

            hal.pHal = &lwhalIface_GP102;
            hal.halImpl = LWHAL_IMPL_GP102;
            break;

         case LW_PMC_BOOT_0_IMPLEMENTATION_4:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GP104 routines.\n");
            }

            hal.pHal = &lwhalIface_GP104;
            hal.halImpl = LWHAL_IMPL_GP104;
            break;

        case LW_PMC_BOOT_0_IMPLEMENTATION_6:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GP106 routines.\n");
            }

            hal.pHal = &lwhalIface_GP106;
            hal.halImpl = LWHAL_IMPL_GP106;
            break;

        case LW_PMC_BOOT_0_IMPLEMENTATION_7:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GP107 routines.\n");
            }

            hal.pHal = &lwhalIface_GP107;
            hal.halImpl = LWHAL_IMPL_GP107;
            break;

#if LWWATCHCFG_CHIP_ENABLED(GP108)
        case LW_PMC_BOOT_0_IMPLEMENTATION_8:
            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up GP108 routines.\n");
            }

            hal.pHal = &lwhalIface_GP108;
            hal.halImpl = LWHAL_IMPL_GP108;
            break;
#endif
    }
    hal.chipInfo.Implementation = impl; 
}
#endif

//-----------------------------------------------------
// findLwWatchHal
// + This finds the chip that we are running on and
// the hal for this GPU. It takes the PMC_BOOT0 as input
//-----------------------------------------------------
static void
findLwWatchHal
(
 LwU32 thisPmcBoot0
)
{
    hal.chipInfo.MaskRevision = thisPmcBoot0 & 0x000000FF;

    if (IsTegra())
    {
        findTegraHal(thisPmcBoot0);
        return;
    }

    switch (DRF_VAL(_PMC, _BOOT_0, _ARCHITECTURE, thisPmcBoot0))
    {
        case LW_PMC_BOOT_0_ARCHITECTURE_LW40:
            hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_LW40;
            findLW4XHal(thisPmcBoot0);
            break;
        case LW_PMC_BOOT_0_ARCHITECTURE_LW60:
            hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_LW60;
            findLW6XHal(thisPmcBoot0);
            break;
#if (LWWATCHCFG_CHIP_ENABLED(LW50))
        case LW_PMC_BOOT_0_ARCHITECTURE_LW50:
            hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_LW50;
            hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_0;

            if (verboseLevel)
            {
                dprintf("flcndbg: Wiring up LW50 routines.\n");
            }

            hal.pHal = &lwhalIface_LW50;
            hal.halImpl = LWHAL_IMPL_LW50;
            break;
#endif
#if (LWWATCHCFG_CHIP_ENABLED(G8X))
        case LW_PMC_BOOT_0_ARCHITECTURE_LW80:

            hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_LW80;
            findG8XHal(thisPmcBoot0);
            break;
#endif
#if (LWWATCHCFG_CHIP_ENABLED(G9X))
        case LW_PMC_BOOT_0_ARCHITECTURE_LW90:
            hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_LW90;
            findG9XHal(thisPmcBoot0);
            break;
#endif
#if LWWATCHCFG_CHIP_ENABLED(GT20X) || LWWATCHCFG_CHIP_ENABLED(GT21X)
        case LW_PMC_BOOT_0_ARCHITECTURE_G100:
            hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_G100;
            findGT2XXHal(thisPmcBoot0);
            break;
#endif
#if LWWATCHCFG_CHIP_ENABLED(GF10X)
        case LW_PMC_BOOT_0_ARCHITECTURE_GF100:
            hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_GF100;
#if LWWATCHCFG_CHIP_ENABLED(GF10X)
            findGF10XHal(thisPmcBoot0);
#endif
#if (LWWATCHCFG_CHIP_ENABLED(GF110F) || LWWATCHCFG_CHIP_ENABLED(GF110F2) || LWWATCHCFG_CHIP_ENABLED(GF110F3))
            findGF10XFHal(thisPmcBoot0);
#endif
            break;
#endif
#if LWWATCHCFG_CHIP_ENABLED(GF11X)
        case LW_PMC_BOOT_0_ARCHITECTURE_GF110:
            hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_GF110;
            findGF11XHal(thisPmcBoot0);
            break;
#endif
#if LWWATCHCFG_CHIP_ENABLED(GK104) || LWWATCHCFG_CHIP_ENABLED(GK20A)
        case LW_PMC_BOOT_0_ARCHITECTURE_GK100:
             hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_GK100;
             findGK10XHal(thisPmcBoot0);
             break;
#endif
#if LWWATCHCFG_CHIP_ENABLED(GK110)
        case LW_PMC_BOOT_0_ARCHITECTURE_GK110:
             hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_GK110;
             findGK11XHal(thisPmcBoot0);
             break;
#endif
#if LWWATCHCFG_CHIP_ENABLED(GK208)
        case LW_PMC_BOOT_0_ARCHITECTURE_GK200:
             hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_GK200;
             findGK20XHal(thisPmcBoot0);
             break;
#endif
#if LWWATCHCFG_CHIP_ENABLED(GM107)
        case LW_PMC_BOOT_0_ARCHITECTURE_GM100:
             hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_GM100;
             findGM10XHal(thisPmcBoot0);
             break;
#endif
#if LWWATCHCFG_CHIP_ENABLED(GM200) || LWWATCHCFG_CHIP_ENABLED(GM204) || LWWATCHCFG_CHIP_ENABLED(GM206)
        case LW_PMC_BOOT_0_ARCHITECTURE_GM200:
             hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_GM200;
             findGM20XHal(thisPmcBoot0);
             break;
#endif
#if LWWATCHCFG_CHIP_ENABLED(GP100) || LWWATCHCFG_CHIP_ENABLED(GP102) || LWWATCHCFG_CHIP_ENABLED(GP104) || LWWATCHCFG_CHIP_ENABLED(GP106) || LWWATCHCFG_CHIP_ENABLED(GP107) || LWWATCHCFG_CHIP_ENABLED(GP108)
        case LW_PMC_BOOT_0_ARCHITECTURE_GP100:
             hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_GP100;
             findGP10XHal(thisPmcBoot0);
             break;
#endif
#if LWWATCHCFG_CHIP_ENABLED(LW40)
        default: hal.pHal = &lwhalIface_LW40;
                 if (verboseLevel)
                 {
                     dprintf("flcndbg: Wiring up LW40 routines.\n");
                 }

                 hal.halImpl = LWHAL_IMPL_LW40;
                 hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_LW40;
                 hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_0; 
                 return;    
#endif
    }

}

//
// general public routines
//

//----------------------------------------------------------------------------
// initLwWatchHal
// + This finds the chip that we are running on and the hal for this GPU.
// It takes the PMC_BOOT0 as input initializes the hal by calling the halIface
// function for that GPU
//----------------------------------------------------------------------------
LW_STATUS
initLwWatchHal
(
 LwU32 thisPmcBoot0
)
{
    findLwWatchHal(thisPmcBoot0);
    if (!(hal.pHal)) {
        dprintf ("Error initializing hal, no GPU is enabled!!!!\n");
        return LW_ERR_GENERIC;
    }

    //
    // Find the displ hal for FPGA. 
    // The display class used in FPGAs are not same as the one present
    // in architecture/implementation given in pmcboot.
    //
    findDispFpgaHal(thisPmcBoot0);

    //
    // numGPU keeps track of the number of GPUs initialized so far
    // It's not being used now, however in the future, for hybrid gpus,
    // it will be used for indexing the engine objects instead of the hardcoded
    // '0' index used for now
    //
    hal.numGPU++;
    indexGpu = 0;

    hal.pHal->pmuLwHalIfacesSetupFn(&(pPmu[indexGpu]));
    hal.pHal->dpuLwHalIfacesSetupFn(&(pDpu[indexGpu]));
    hal.pHal->socbrdgLwHalIfacesSetupFn(&(pSocbrdg[indexGpu]));
    hal.pHal->tegrasysLwHalIfacesSetupFn(&(pTegrasys[indexGpu]));
    return LW_OK;
}

// Map LWHAL implementation ID to a string name that diagnostic code can print out
char*   getLwhalImplName( LWHAL_IMPLEMENTATION implId )
{
    char *retImplName = "Unknown chip implementation";

    if ((implId >= 0) && (implId < LWHAL_IMPL_MAXIMUM))
    {
        retImplName = gpuImplementationNames[implId];
    }

    return retImplName;
}
