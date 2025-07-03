/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2006-2021 by LWPU Corporation.  All rights reserved.  All
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
#include "disp.h"
#include "pmu.h"
#include "mmu.h"
#include "sig.h"
#include "fb.h"
#include "fifo.h"
#include "inst.h"
#include "falcon.h"
#include "acr.h"
#include "vpr.h"
#include "falcphys.h"
#include "clk.h"
#include "riscv.h"
#include "smbpbi.h"
#include "tegrasys.h"
#include "intr.h"
#include "g00x/g000/dev_master.h" // Include only the latest chip's dev_master

#include "g_hal_private.h"          // (rmconfig) hal/obj setup


//
// global halObject
//
Hal hal;

int indexGpu;

FB_LWHAL_IFACES pFb[MAX_GPUS];
CLK_LWHAL_IFACES pClk[MAX_GPUS];
DISP_LWHAL_IFACES pDisp[MAX_GPUS];
FIFO_LWHAL_IFACES pFifo[MAX_GPUS];
GR_LWHAL_IFACES pGr[MAX_GPUS];
MSDEC_LWHAL_IFACES pMsdec[MAX_GPUS];
PMU_LWHAL_IFACES pPmu[MAX_GPUS];
DPU_LWHAL_IFACES pDpu[MAX_GPUS];
PMGR_LWHAL_IFACES pPmgr[MAX_GPUS];
FECS_LWHAL_IFACES pFecs[MAX_GPUS];
SMBPBI_LWHAL_IFACES pSmbpbi[MAX_GPUS];
DPAUX_LWHAL_IFACES pDpaux[MAX_GPUS];
INSTMEM_LWHAL_IFACES pInstmem[MAX_GPUS];
CIPHER_LWHAL_IFACES pCipher[MAX_GPUS];
SIG_LWHAL_IFACES pSig[MAX_GPUS];
BUS_LWHAL_IFACES pBus[MAX_GPUS];
BIF_LWHAL_IFACES pBif[MAX_GPUS];
CE_LWHAL_IFACES pCe[MAX_GPUS];
FALCON_LWHAL_IFACES pFalcon[MAX_GPUS];
VIC_LWHAL_IFACES pVic[MAX_GPUS];
MSENC_LWHAL_IFACES pMsenc[MAX_GPUS];
HDA_LWHAL_IFACES pHda[MAX_GPUS];
MC_LWHAL_IFACES pMc[MAX_GPUS];
PRIV_LWHAL_IFACES pPriv[MAX_GPUS];
ELPG_LWHAL_IFACES pElpg[MAX_GPUS];
CE_LWHAL_IFACES pCe[MAX_GPUS];
TEGRASYS_LWHAL_IFACES pTegrasys[MAX_GPUS];
LWDEC_LWHAL_IFACES pLwdec[MAX_GPUS];
LWJPG_LWHAL_IFACES pLwjpg[MAX_GPUS];
ACR_LWHAL_IFACES pAcr[MAX_GPUS];
FALCPHYS_LWHAL_IFACES pFalcphys[MAX_GPUS];
PSDL_LWHAL_IFACES pPsdl[MAX_GPUS];
SEC2_LWHAL_IFACES pSec2[MAX_GPUS];
GSP_LWHAL_IFACES pGsp[MAX_GPUS];
OFA_LWHAL_IFACES pOfa[MAX_GPUS];
FBFLCN_LWHAL_IFACES pFbflcn[MAX_GPUS];
LWLINK_LWHAL_IFACES pLwlink[MAX_GPUS];
RISCV_LWHAL_IFACES pRiscv[MAX_GPUS];
INTR_LWHAL_IFACES pIntr[MAX_GPUS];

MMU_LWHAL_IFACES pMmu[MAX_GPUS];
VMEM_LWHAL_IFACES pVmem[MAX_GPUS];

HWPROD_LWHAL_IFACES pHwprod[MAX_GPUS];
VIRT_LWHAL_IFACES pVirt[MAX_GPUS];

VPR_LWHAL_IFACES pVpr[MAX_GPUS];

TEGRASYS TegraSysObj[MAX_GPUS];

// Mapping of chip implementations to strings.  From enum LWHAL_IMPLEMENTATION in hal.f
// TODO: check this into hal.h or hal.c so it's maintained centrally.
#ifndef QUOTE_ME
#define QUOTE_ME(x) #x
#endif

static char* gpuImplementationNames[LWHAL_IMPL_MAXIMUM] =
{
    QUOTE_ME( LWHAL_IMPL_T124 ),
    QUOTE_ME( LWHAL_IMPL_T210 ),
    QUOTE_ME( LWHAL_IMPL_T186 ),
    QUOTE_ME( LWHAL_IMPL_T194 ),
    QUOTE_ME( LWHAL_IMPL_T234 ),
    QUOTE_ME( LWHAL_IMPL_GM107 ),
    QUOTE_ME( LWHAL_IMPL_GM200 ),
    QUOTE_ME( LWHAL_IMPL_GM204 ),
    QUOTE_ME( LWHAL_IMPL_GM206 ),
    QUOTE_ME( LWHAL_IMPL_GP100 ),
    QUOTE_ME( LWHAL_IMPL_GP102 ),
    QUOTE_ME( LWHAL_IMPL_GP104 ),
    QUOTE_ME( LWHAL_IMPL_GP106 ),
    QUOTE_ME( LWHAL_IMPL_GP107 ),
    QUOTE_ME( LWHAL_IMPL_GP108 ),
    QUOTE_ME( LWHAL_IMPL_GV100 ),
    QUOTE_ME( LWHAL_IMPL_TU102 ),
    QUOTE_ME( LWHAL_IMPL_TU104 ),
    QUOTE_ME( LWHAL_IMPL_TU106 ),
    QUOTE_ME( LWHAL_IMPL_TU116 ),
    QUOTE_ME( LWHAL_IMPL_TU117 ),
    QUOTE_ME( LWHAL_IMPL_GA100 ),
    QUOTE_ME( LWHAL_IMPL_GA102 ),
    QUOTE_ME( LWHAL_IMPL_GA103 ),
    QUOTE_ME( LWHAL_IMPL_GA104 ),
    QUOTE_ME( LWHAL_IMPL_GA106 ),
    QUOTE_ME( LWHAL_IMPL_GA107 ),
    QUOTE_ME( LWHAL_IMPL_AD102 ),
    QUOTE_ME( LWHAL_IMPL_AD103 ),
    QUOTE_ME( LWHAL_IMPL_AD104 ),
    QUOTE_ME( LWHAL_IMPL_AD106 ),
    QUOTE_ME( LWHAL_IMPL_AD107 ),
    QUOTE_ME( LWHAL_IMPL_GH100 ),
    QUOTE_ME( LWHAL_IMPL_GH202 ),
    QUOTE_ME( LWHAL_IMPL_GB100 ),
    QUOTE_ME( LWHAL_IMPL_G000 ),
};


// static function prototypes
//
static void findTegraHal(LwU32);
static void findLwWatchHal(LwU32);
static void findGM10XHal(LwU32);
static void findGM20XHal(LwU32);
static void findGP10XHal(LwU32);
static void findGV10XHal(LwU32);
static void findTU10XHal(LwU32);
static void findGA10XHal(LwU32);
static void findAD10XHal(LwU32);
static void findG00XHal(LwU32);
//
// static function definitions
//
static void
findTegraHal
(
 LwU32 thisPmcBoot0
)
{
#if (LWWATCHCFG_CHIP_ENABLED(T124))
    if (IsT124())
    {
        if (verboseLevel)
        {
            dprintf("lw: Wiring up T124 routines.\n");
        }
        hal.pHal = &lwhalIface_T124;
        hal.halImpl = LWHAL_IMPL_T124;
        hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_0;
        return;
    }
#endif

#if (LWWATCHCFG_CHIP_ENABLED(T210))
    if (IsT210())
    {
        if (verboseLevel)
        {
            dprintf("lw: Wiring up T210 routines.\n");
        }
        hal.pHal = &lwhalIface_T210;
        hal.halImpl = LWHAL_IMPL_T210;
        hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_0;
        return;
    }
#endif

#if (LWWATCHCFG_CHIP_ENABLED(T186))
    if (IsT186())
    {
        if (verboseLevel)
        {
            dprintf("lw: Wiring up T186 routines.\n");
        }
        hal.pHal = &lwhalIface_T186;
        hal.halImpl = LWHAL_IMPL_T186;
        hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_0;
        return;
    }
#endif

#if (LWWATCHCFG_CHIP_ENABLED(T194))
    if (IsT194())
    {
        if (verboseLevel)
        {
            dprintf("lw: Wiring up T194 routines.\n");
        }
        hal.pHal = &lwhalIface_T194;
        hal.halImpl = LWHAL_IMPL_T194;
        hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_0;
        return;
    }
#endif

#if (LWWATCHCFG_CHIP_ENABLED(T234))
    if (IsT234())
    {
        if (verboseLevel)
        {
            dprintf("lw: Wiring up T234 routines.\n");
        }
        hal.pHal = &lwhalIface_T234;
        hal.halImpl = LWHAL_IMPL_T234;
        hal.chipInfo.Implementation = LW_PMC_BOOT_0_IMPLEMENTATION_0;
        return;
    }
#endif
}

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
                dprintf("lw: Wiring up GM107 routines.\n");
            }

            hal.pHal = &lwhalIface_GM107;
            hal.halImpl = LWHAL_IMPL_GM107;
            initializeDisp_v02_01("gm107");
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
                dprintf("lw: Wiring up GM200 routines.\n");
            }

            hal.pHal = &lwhalIface_GM200;
            hal.halImpl = LWHAL_IMPL_GM200;
            initializeDisp_v02_01("gm200");
            break;

        case LW_PMC_BOOT_0_IMPLEMENTATION_4:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up GM204 routines.\n");
            }

            hal.pHal = &lwhalIface_GM204;
            hal.halImpl = LWHAL_IMPL_GM204;
            initializeDisp_v02_01("gm204");
            break;

        case LW_PMC_BOOT_0_IMPLEMENTATION_6:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up GM206 routines.\n");
            }

            hal.pHal = &lwhalIface_GM206;
            hal.halImpl = LWHAL_IMPL_GM206;
            initializeDisp_v02_01("gm206");
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
                dprintf("lw: Wiring up GP100 routines.\n");
            }

            hal.pHal = &lwhalIface_GP100;
            hal.halImpl = LWHAL_IMPL_GP100;
            initializeDisp_v02_01("gp100");
            break;

        case LW_PMC_BOOT_0_IMPLEMENTATION_2:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up GP102 routines.\n");
            }

            hal.pHal = &lwhalIface_GP102;
            hal.halImpl = LWHAL_IMPL_GP102;
            initializeDisp_v02_01("gp102");
            break;

        case LW_PMC_BOOT_0_IMPLEMENTATION_4:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up GP104 routines.\n");
            }

            hal.pHal = &lwhalIface_GP104;
            hal.halImpl = LWHAL_IMPL_GP104;
            initializeDisp_v02_01("gp104");
            break;

        case LW_PMC_BOOT_0_IMPLEMENTATION_6:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up GP106 routines.\n");
            }

            hal.pHal = &lwhalIface_GP106;
            hal.halImpl = LWHAL_IMPL_GP106;
            initializeDisp_v02_01("gp106");
            break;

        case LW_PMC_BOOT_0_IMPLEMENTATION_7:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up GP107 routines.\n");
            }

            hal.pHal = &lwhalIface_GP107;
            hal.halImpl = LWHAL_IMPL_GP107;
            initializeDisp_v02_01("gp107");
            break;

#if LWWATCHCFG_CHIP_ENABLED(GP108)
        case LW_PMC_BOOT_0_IMPLEMENTATION_8:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up GP108 routines.\n");
            }

            hal.pHal = &lwhalIface_GP108;
            hal.halImpl = LWHAL_IMPL_GP108;
            initializeDisp_v02_01("gp108");
            break;
#endif
    }
    hal.chipInfo.Implementation = impl;
}
#endif

#if LWWATCHCFG_CHIP_ENABLED(GV100)
static void
findGV10XHal
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
                dprintf("lw: Wiring up GV100 routines.\n");
            }

            hal.pHal = &lwhalIface_GV100;
            hal.halImpl = LWHAL_IMPL_GV100;
            initializeDisp_v03_00("gv100");
            break;
    }
    hal.chipInfo.Implementation = impl;
}
#endif

#if LWWATCHCFG_CHIP_ENABLED(TU102) || LWWATCHCFG_CHIP_ENABLED(TU104) || LWWATCHCFG_CHIP_ENABLED(TU106) || LWWATCHCFG_CHIP_ENABLED(TU116) || LWWATCHCFG_CHIP_ENABLED(TU117)
static void
findTU10XHal
(
    LwU32 thisPmcBoot0
)
{
    LwU32 impl = DRF_VAL(_PMC, _BOOT_0, _IMPLEMENTATION, thisPmcBoot0);

    switch (impl)
    {
        default:
            impl = LW_PMC_BOOT_0_IMPLEMENTATION_0;

        case LW_PMC_BOOT_0_IMPLEMENTATION_2:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up TU102 routines.\n");
            }

            hal.pHal = &lwhalIface_TU102;
            hal.halImpl = LWHAL_IMPL_TU102;
            break;

        case LW_PMC_BOOT_0_IMPLEMENTATION_4:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up TU104 routines.\n");
            }

            hal.pHal = &lwhalIface_TU104;
            hal.halImpl = LWHAL_IMPL_TU104;
            break;

        case LW_PMC_BOOT_0_IMPLEMENTATION_6:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up TU106 routines.\n");
            }

            hal.pHal = &lwhalIface_TU106;
            hal.halImpl = LWHAL_IMPL_TU106;
            break;

        case LW_PMC_BOOT_0_IMPLEMENTATION_8:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up TU116 routines.\n");
            }

            hal.pHal = &lwhalIface_TU116;
            hal.halImpl = LWHAL_IMPL_TU116;
            break;

        case LW_PMC_BOOT_0_IMPLEMENTATION_7:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up TU117 routines.\n");
            }

            hal.pHal = &lwhalIface_TU117;
            hal.halImpl = LWHAL_IMPL_TU117;
            break;

    }
    hal.chipInfo.Implementation = impl;
}
#endif

#if LWWATCHCFG_CHIP_ENABLED(GA100)
static void
findGA10XHal
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
                dprintf("lw: Wiring up GA100 routines.\n");
            }

            hal.pHal = &lwhalIface_GA100;
            hal.halImpl = LWHAL_IMPL_GA100;
            break;

        case LW_PMC_BOOT_0_IMPLEMENTATION_2:
        case LW_PMC_BOOT_0_IMPLEMENTATION_F:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up GA102 routines.\n");
            }

            hal.pHal = &lwhalIface_GA102;
            hal.halImpl = LWHAL_IMPL_GA102;
            break;

        case LW_PMC_BOOT_0_IMPLEMENTATION_3:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up GA103 routines.\n");
            }

            hal.pHal = &lwhalIface_GA103;
            hal.halImpl = LWHAL_IMPL_GA103;
            break;

        case LW_PMC_BOOT_0_IMPLEMENTATION_4:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up GA104 routines.\n");
            }

            hal.pHal = &lwhalIface_GA104;
            hal.halImpl = LWHAL_IMPL_GA104;
            break;

        case LW_PMC_BOOT_0_IMPLEMENTATION_6:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up GA106 routines.\n");
            }

            hal.pHal = &lwhalIface_GA106;
            hal.halImpl = LWHAL_IMPL_GA106;
            break;

        case LW_PMC_BOOT_0_IMPLEMENTATION_7:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up GA107 routines.\n");
            }

            hal.pHal = &lwhalIface_GA107;
            hal.halImpl = LWHAL_IMPL_GA107;
            break;                                              
    }
    hal.chipInfo.Implementation = impl;
}
#endif

#if LWWATCHCFG_CHIP_ENABLED(AD102)
static void
findAD10XHal
(
    LwU32 thisPmcBoot0
)
{
    LwU32 impl = DRF_VAL(_PMC, _BOOT_0, _IMPLEMENTATION, thisPmcBoot0);

    switch (impl)
    {
        default:
            impl = LW_PMC_BOOT_0_IMPLEMENTATION_0;

        case LW_PMC_BOOT_0_IMPLEMENTATION_2:
        case LW_PMC_BOOT_0_IMPLEMENTATION_F:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up AD102 routines.\n");
            }

            hal.pHal = &lwhalIface_AD102;
            hal.halImpl = LWHAL_IMPL_AD102;
            break;
        case LW_PMC_BOOT_0_IMPLEMENTATION_3:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up AD103 routines.\n");
            }

            hal.pHal = &lwhalIface_AD103;
            hal.halImpl = LWHAL_IMPL_AD103;
            break;
        case LW_PMC_BOOT_0_IMPLEMENTATION_4:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up AD104 routines.\n");
            }

            hal.pHal = &lwhalIface_AD104;
            hal.halImpl = LWHAL_IMPL_AD104;
            break;
        case LW_PMC_BOOT_0_IMPLEMENTATION_6:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up AD106 routines.\n");
            }

            hal.pHal = &lwhalIface_AD106;
            hal.halImpl = LWHAL_IMPL_AD106;
            break;
        case LW_PMC_BOOT_0_IMPLEMENTATION_7:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up AD107 routines.\n");
            }

            hal.pHal = &lwhalIface_AD107;
            hal.halImpl = LWHAL_IMPL_AD107;
            break;
    }
    hal.chipInfo.Implementation = impl;
}
#endif


#if LWWATCHCFG_CHIP_ENABLED(GH100)
static void
findGH10XHal
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
                dprintf("lw: Wiring up GH100 routines.\n");
            }

            hal.pHal = &lwhalIface_GH100;
            hal.halImpl = LWHAL_IMPL_GH100;
            break;
        case LW_PMC_BOOT_0_IMPLEMENTATION_2:
            if (verboseLevel)
            {
                dprintf("lw: Wiring up GH202 routines.\n");
            }

            hal.pHal = &lwhalIface_GH202;
            hal.halImpl = LWHAL_IMPL_GH202;
            break;

    }
    hal.chipInfo.Implementation = impl;
}
#endif


#if LWWATCHCFG_CHIP_ENABLED(GB100)
static void
findGB10XHal
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
                dprintf("lw: Wiring up GB100 routines.\n");
            }

            hal.pHal = &lwhalIface_GB100;
            hal.halImpl = LWHAL_IMPL_GB100;
            break;

    }
    hal.chipInfo.Implementation = impl;
}
#endif


#if LWWATCHCFG_CHIP_ENABLED(G000)
static void
findG00XHal
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
                dprintf("lw: Wiring up G000 routines.\n");
            }

            hal.pHal = &lwhalIface_G000;
            hal.halImpl = LWHAL_IMPL_G000;
            break;
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
        case LW_PMC_BOOT_0_ARCHITECTURE_GM100:
#if LWWATCHCFG_CHIP_ENABLED(GM107)
            hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_GM100;
            findGM10XHal(thisPmcBoot0);
#else
            dprintf("lw: GM10x not supported by this LwWatch!\n");
#endif
            break;
        case LW_PMC_BOOT_0_ARCHITECTURE_GM200:
#if LWWATCHCFG_CHIP_ENABLED(GM200) || LWWATCHCFG_CHIP_ENABLED(GM204) || LWWATCHCFG_CHIP_ENABLED(GM206)
            hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_GM200;
            findGM20XHal(thisPmcBoot0);
#else
            printf("lw: GM20x not supported by this LwWatch!\n");
#endif
            break;
        case LW_PMC_BOOT_0_ARCHITECTURE_GP100:
#if LWWATCHCFG_CHIP_ENABLED(GP100) || LWWATCHCFG_CHIP_ENABLED(GP102) || LWWATCHCFG_CHIP_ENABLED(GP104) || LWWATCHCFG_CHIP_ENABLED(GP106) ||LWWATCHCFG_CHIP_ENABLED(GP107) || LWWATCHCFG_CHIP_ENABLED(GP108)
            hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_GP100;
            findGP10XHal(thisPmcBoot0);
#else
            dprintf("lw: GP10x not supported by this LwWatch!\n");
#endif
            break;
        case LW_PMC_BOOT_0_ARCHITECTURE_GV100:
#if LWWATCHCFG_CHIP_ENABLED(GV100)
            hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_GV100;
            findGV10XHal(thisPmcBoot0);
#else
            dprintf("lw: GV10x not supported by this LwWatch!\n");
#endif
            break;
        case LW_PMC_BOOT_0_ARCHITECTURE_TU100:
#if LWWATCHCFG_CHIP_ENABLED(TU102) || LWWATCHCFG_CHIP_ENABLED(TU104) || LWWATCHCFG_CHIP_ENABLED(TU106) || LWWATCHCFG_CHIP_ENABLED(TU116) || LWWATCHCFG_CHIP_ENABLED(TU117)

            hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_TU100;
            findTU10XHal(thisPmcBoot0);
#else
            dprintf("lw: TU10x not supported by this LwWatch!\n");
#endif
            break;
        case LW_PMC_BOOT_0_ARCHITECTURE_GA100:
#if LWWATCHCFG_CHIP_ENABLED(GA100) || LWWATCHCFG_CHIP_ENABLED(GA102)

            hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_GA100;
            findGA10XHal(thisPmcBoot0);
#else
            dprintf("lw: GA10x not supported by this LwWatch!\n");
#endif
            break;
        case LW_PMC_BOOT_0_ARCHITECTURE_GH100:
#if LWWATCHCFG_CHIP_ENABLED(GH100)

            hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_GH100;
            findGH10XHal(thisPmcBoot0);
#else
            dprintf("lw: GH10x not supported by this LwWatch!\n");
#endif
            break;
        case LW_PMC_BOOT_0_ARCHITECTURE_GB100:
#if LWWATCHCFG_CHIP_ENABLED(GB100)

            hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_GB100;
            findGB10XHal(thisPmcBoot0);
#else
            dprintf("lw: GB10x not supported by this LwWatch!\n");
#endif
            break;
        case LW_PMC_BOOT_0_ARCHITECTURE_AD100:
#if LWWATCHCFG_CHIP_ENABLED(AD102)

            hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_AD100;
            findAD10XHal(thisPmcBoot0);
#else
            dprintf("lw: AD10x not supported by this LwWatch!\n");
#endif
            break;
        case LW_PMC_BOOT_0_ARCHITECTURE_G000:
#if LWWATCHCFG_CHIP_ENABLED(G000)

            hal.chipInfo.Architecture = LW_PMC_BOOT_0_ARCHITECTURE_G000;
            findG00XHal(thisPmcBoot0);
#else
            dprintf("lw: G00x not supported by this LwWatch!\n");
#endif
            break;
        default:
            if (hal.pHal == NULL)
            {
                dprintf("lw: Unknown GPU architecture not supported by this LwWatch!\n");
            }
            break;
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
        dprintf ("Error initializing hal!!!!\n");
        return LW_ERR_GENERIC;
    }

    //
    // numGPU keeps track of the number of GPUs initialized so far
    // It's not being used now, however in the future, for hybrid gpus,
    // it will be used for indexing the engine objects instead of the hardcoded
    // '0' index used for now
    //
    hal.numGPU++;
    indexGpu = 0;

    hal.pHal->bifLwHalIfacesSetupFn(&(pBif[indexGpu]));
    hal.pHal->cipherLwHalIfacesSetupFn(&(pCipher[indexGpu]));
    hal.pHal->clkLwHalIfacesSetupFn(&(pClk[indexGpu]));
    hal.pHal->dpauxLwHalIfacesSetupFn(&(pDpaux[indexGpu]));
    hal.pHal->fbLwHalIfacesSetupFn(&(pFb[indexGpu]));
    hal.pHal->fifoLwHalIfacesSetupFn(&(pFifo[indexGpu]));
    hal.pHal->grLwHalIfacesSetupFn(&(pGr[indexGpu]));
    hal.pHal->instmemLwHalIfacesSetupFn(&(pInstmem[indexGpu]));
    hal.pHal->msdecLwHalIfacesSetupFn(&(pMsdec[indexGpu]));
    hal.pHal->pmgrLwHalIfacesSetupFn(&(pPmgr[indexGpu]));
    hal.pHal->pmuLwHalIfacesSetupFn(&(pPmu[indexGpu]));
    hal.pHal->dpuLwHalIfacesSetupFn(&(pDpu[indexGpu]));
    hal.pHal->fecsLwHalIfacesSetupFn(&(pFecs[indexGpu]));
    hal.pHal->smbpbiLwHalIfacesSetupFn(&(pSmbpbi[indexGpu]));
    hal.pHal->sigLwHalIfacesSetupFn(&(pSig[indexGpu]));
    hal.pHal->busLwHalIfacesSetupFn(&(pBus[indexGpu]));
    hal.pHal->ceLwHalIfacesSetupFn(&(pCe[indexGpu]));
    hal.pHal->falconLwHalIfacesSetupFn(&(pFalcon[indexGpu]));
    hal.pHal->vicLwHalIfacesSetupFn(&(pVic[indexGpu]));
    hal.pHal->msencLwHalIfacesSetupFn(&(pMsenc[indexGpu]));
    hal.pHal->hdaLwHalIfacesSetupFn(&(pHda[indexGpu]));
    hal.pHal->mcLwHalIfacesSetupFn(&(pMc[indexGpu]));
    hal.pHal->privLwHalIfacesSetupFn(&(pPriv[indexGpu]));
    hal.pHal->elpgLwHalIfacesSetupFn(&(pElpg[indexGpu]));
    hal.pHal->ceLwHalIfacesSetupFn(&(pCe[indexGpu]));
    hal.pHal->tegrasysLwHalIfacesSetupFn(&(pTegrasys[indexGpu]));
    hal.pHal->lwdecLwHalIfacesSetupFn(&(pLwdec[indexGpu]));
    hal.pHal->lwjpgLwHalIfacesSetupFn(&(pLwjpg[indexGpu]));
    hal.pHal->acrLwHalIfacesSetupFn(&(pAcr[indexGpu]));
    hal.pHal->falcphysLwHalIfacesSetupFn(&(pFalcphys[indexGpu]));
    hal.pHal->psdlLwHalIfacesSetupFn(&(pPsdl[indexGpu]));
    hal.pHal->sec2LwHalIfacesSetupFn(&(pSec2[indexGpu]));
    hal.pHal->gspLwHalIfacesSetupFn(&(pGsp[indexGpu]));
    hal.pHal->ofaLwHalIfacesSetupFn(&(pOfa[indexGpu]));
    hal.pHal->fbflcnLwHalIfacesSetupFn(&(pFbflcn[indexGpu]));
    hal.pHal->lwlinkLwHalIfacesSetupFn(&(pLwlink[indexGpu]));
    hal.pHal->vprLwHalIfacesSetupFn(&(pVpr[indexGpu]));
    hal.pHal->riscvLwHalIfacesSetupFn(&(pRiscv[indexGpu]));
    hal.pHal->intrLwHalIfacesSetupFn(&(pIntr[indexGpu]));

    hal.pHal->dispLwHalIfacesSetupFn(&(pDisp[indexGpu]));

    hal.pHal->vmemLwHalIfacesSetupFn(&(pVmem[indexGpu]));
    hal.pHal->mmuLwHalIfacesSetupFn(&(pMmu[indexGpu]));

    hal.pHal->hwprodLwHalIfacesSetupFn(&(pHwprod[indexGpu]));
    hal.pHal->virtLwHalIfacesSetupFn(&(pVirt[0]));

    pInstmem[indexGpu].instmemSetStartAddress();
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
