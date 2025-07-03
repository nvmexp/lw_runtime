/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2022 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
//
// Includes
//
#include "fifo.h"
#include "fb.h"
#include "vgpu.h"

#include "mmu/mmu_fmt.h"
#include "hopper/gh100/dev_bus.h"
#include "hopper/gh100/pri_lw_xal_ep_func.h"
#include "hopper/gh100/dev_vm.h"
#include "hopper/gh100/dev_mmu.h"
#include "hopper/gh100/dev_ram.h"

#include "g_vmem_private.h"

#define VALID_SHIFT(s) ((s) == 16 || (s) == 17)

static void  _gmmuFmtInitPdeMulti_GH10X(GMMU_FMT_PDE_MULTI *pPdeMulti, LW_FIELD_ENUM_ENTRY *pdeApertures);
static void  _gmmuFmtInitPde_GH10X(GMMU_FMT_PDE *pPde, LW_FIELD_ENUM_ENTRY *pdeApertures);
static void  _gmmuInitMmuFmt_GH10X(MMU_FMT_LEVEL *pLevels, LwU32 bigPageShift);
static LW_STATUS _gmmuTranslatePdePcfFromHw(LwU32 pdePcfHw, GMMU_APERTURE aperture, LwU32 *pPdePcfSw);
static LW_STATUS _gmmuTranslatePtePcfFromHw(LwU32    ptePcfHw, LwBool   bPteValid, LwU32   *pPtePcfSw);

//-----------------------------------------------------
// vmemGetInstanceMemoryAddrForBAR1_GH100
//
// Returns the instance memory base address for BAR1.
//-----------------------------------------------------
LwU64
vmemGetInstanceMemoryAddrForBAR1_GH100(readFn_t* readFn, writeFn_t* writeFn, MEM_TYPE* pMemType)
{
    LwU32 regHigh;
    LwU32 regLow;
    LwU64 instMemAddr;

    regHigh = isVirtualWithSriov() ?
              GPU_REG_RD32_DIRECT(LW_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_HIGH_ADDR) :
              GPU_REG_RD32(LW_XAL_EP_FUNC_BAR1_BLOCK_HIGH_ADDR(0));

    regLow = isVirtualWithSriov() ?
             GPU_REG_RD32_DIRECT(LW_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR) :
             GPU_REG_RD32(LW_XAL_EP_FUNC_BAR1_BLOCK_LOW_ADDR(0));

    instMemAddr = ((LwU64)SF_VAL(_XAL_EP_FUNC, _BAR1_BLOCK_LOW_ADDR_PTR, regLow) << LW_XAL_EP_FUNC_VBAR1_BLOCK_PTR_SHIFT) |
                  ((LwU64)SF_VAL(_XAL_EP_FUNC, _BAR1_BLOCK_HIGH_ADDR_PTR, regHigh) << 32);

    if (SF_VAL(_XAL_EP_FUNC, _BAR1_BLOCK_LOW_ADDR_TARGET, regLow) == LW_XAL_EP_FUNC_BAR1_BLOCK_LOW_ADDR_TARGET_VID_MEM)
    {
        if (readFn)
            *readFn = pFb[indexGpu].fbRead;
        if (writeFn)
            *writeFn = pFb[indexGpu].fbWrite;
        if (pMemType)
            *pMemType = FRAMEBUFFER;
    }
    else
    {
        if (readFn)
            *readFn = readSystem;
        if (writeFn)
            *writeFn = writeSystem;
        if (pMemType)
            *pMemType = SYSTEM_PHYS;
    }

    return instMemAddr;
}

//-----------------------------------------------------
// vmemGetInstanceMemoryAddrForBAR2_GH100
//
// Returns the instance memory base address for BAR2.
//-----------------------------------------------------
LwU64
vmemGetInstanceMemoryAddrForBAR2_GH100(readFn_t* readFn, writeFn_t* writeFn, MEM_TYPE* pMemType)
{
    LwU32 regHigh;
    LwU32 regLow;
    LwU64 instMemAddr;

    regHigh = isVirtualWithSriov() ?
              GPU_REG_RD32_DIRECT(LW_VIRTUAL_FUNCTION_PRIV_FUNC_BAR2_BLOCK_HIGH_ADDR) :
              GPU_REG_RD32(LW_XAL_EP_FUNC_BAR2_BLOCK_HIGH_ADDR(0));

    regLow = isVirtualWithSriov() ?
             GPU_REG_RD32_DIRECT(LW_VIRTUAL_FUNCTION_PRIV_FUNC_BAR2_BLOCK_LOW_ADDR) :
             GPU_REG_RD32(LW_XAL_EP_FUNC_BAR2_BLOCK_LOW_ADDR(0));

    instMemAddr = ((LwU64)SF_VAL(_XAL_EP_FUNC, _BAR2_BLOCK_LOW_ADDR_PTR, regLow) << LW_XAL_EP_FUNC_VBAR2_BLOCK_PTR_SHIFT) |
                  ((LwU64)SF_VAL(_XAL_EP_FUNC, _BAR2_BLOCK_HIGH_ADDR_PTR, regHigh) << 32);

    if (SF_VAL(_XAL_EP_FUNC, _BAR2_BLOCK_LOW_ADDR_TARGET, regLow) == LW_XAL_EP_FUNC_BAR2_BLOCK_LOW_ADDR_TARGET_VID_MEM)
    {
        if (readFn)
            *readFn = pFb[indexGpu].fbRead;
        if (writeFn)
            *writeFn = pFb[indexGpu].fbWrite;
        if (pMemType)
            *pMemType = FRAMEBUFFER;
    }
    else
    {
        if (readFn)
            *readFn = readSystem;
        if (writeFn)
            *writeFn = writeSystem;
        if (pMemType)
            *pMemType = SYSTEM_PHYS;
    }

    return instMemAddr;
}

void
vmemRebindBAR1Block_GH100(void)
{
    LwU32 regHigh;
    LwU32 regLow;

    //
    // For BAR1 and BAR2 binds, LOW_ADDR should be written first followed with the HIGH_ADDR.
    // Writing to the HIGH_ADDR register triggers the bind.
    //
    if (isVirtualWithSriov())
    {
        regLow = GPU_REG_RD32_DIRECT(LW_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR);
        regHigh = GPU_REG_RD32_DIRECT(LW_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_HIGH_ADDR);

        GPU_REG_WR32_DIRECT(LW_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR, regLow);
        GPU_REG_WR32_DIRECT(LW_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_HIGH_ADDR, regHigh);
    }
    else
    {
        regLow = GPU_REG_RD32(LW_XAL_EP_FUNC_BAR1_BLOCK_LOW_ADDR(0));
        regHigh = GPU_REG_RD32(LW_XAL_EP_FUNC_BAR1_BLOCK_HIGH_ADDR(0));

        GPU_REG_WR32(LW_XAL_EP_FUNC_BAR1_BLOCK_LOW_ADDR(0), regLow);
        GPU_REG_WR32(LW_XAL_EP_FUNC_BAR1_BLOCK_HIGH_ADDR(0), regHigh);
    }
}

LW_STATUS
vmemInitLayout_GH100(VMemSpace *pVMemSpace)
{
#if LWCFG(GLOBAL_ARCH_HOPPER)
    VMemLayout *pVMemLayout;
    LwU32       bigPageShift;

    if (!pVMemSpace)
    {
        return LW_ERR_GENERIC;
    }

    bigPageShift = BIT_IDX_32(pVMemSpace->bigPageSize);
    if (!VALID_SHIFT(bigPageShift))
    {
        return LW_ERR_GENERIC;
    }

    pVMemLayout = &pVMemSpace->layout;

    memset(pVMemLayout, 0, sizeof(VMemLayout));

    pVMemLayout->fmt.gmmu.version          = GMMU_FMT_VERSION_3;
    pVMemLayout->fmt.gmmu.pPdeMulti        = &pVMemLayout->pdeMulti.gmmu;
    pVMemLayout->fmt.gmmu.pPde             = &pVMemLayout->pde.gmmu;
    pVMemLayout->fmt.gmmu.pPte             = &pVMemLayout->pte.gmmu;
    pVMemLayout->fmt.gmmu.bSparseHwSupport = LW_TRUE;

    pMmu[indexGpu].mmuFmtInitPdeApertures(pVMemLayout->pdeApertures);
    pMmu[indexGpu].mmuFmtInitPteApertures(pVMemLayout->pteApertures);

    _gmmuFmtInitPdeMulti_GH10X(&pVMemLayout->pdeMulti.gmmu, pVMemLayout->pdeApertures);
    _gmmuFmtInitPde_GH10X(&pVMemLayout->pde.gmmu, pVMemLayout->pdeApertures);
    pVmem[indexGpu].vmemGmmuFmtInitPte(&pVMemLayout->pte.gmmu, pVMemLayout->pteApertures);
    _gmmuInitMmuFmt_GH10X(pVMemLayout->fmtLevels, bigPageShift);

    return LW_OK;
#else
    return LW_ERR_NOT_SUPPORTED;
#endif
}

LW_STATUS
vmemGetPDEFmt_GH100(VMemSpace *pVMemSpace, VMemFmtPde *pFmtPde, LwU32 level)
{
    if (!pVMemSpace || !pFmtPde)
    {
        return LW_ERR_GENERIC;
    }

    if (level > 4)
    {
        return LW_ERR_GENERIC;
    }

    if (level == 4)
    {
        pFmtPde->bMulti = TRUE;
        _gmmuFmtInitPdeMulti_GH10X(&pFmtPde->fmts.multi.gmmu, pVMemSpace->layout.pdeApertures);
    }
    else
    {
        pFmtPde->bMulti = FALSE;
        _gmmuFmtInitPde_GH10X(&pFmtPde->fmts.single.gmmu, pVMemSpace->layout.pdeApertures);
    }

    return LW_OK;
}

const MMU_FMT_PTE*
vmemGetPTEFmt_GH100(VMemSpace *pVMemSpace)
{
    if (!pVMemSpace)
    {
        return NULL;
    }

    return &pVMemSpace->layout.pte;
}

//
// Used by dumpPdeInfo to determine HW level number.
// SW level 0 == HW PD4
// SW level 1 == HW PD3
// SW level 2 == HW PD2
// SW level 3 == HW PD1
// SW level 4 == HW PD0
//
// PTEs don't have a HW numbering, so we return 0.
//
LwU32
vmemSWToHWLevel_GH100(VMemSpace *pVMemSpace, LwU32 level)
{
    if (!pVMemSpace)
    {
        return 0;
    }

    switch (level)
    {
    case 0:
    case 1:
    case 2:
    case 3:
        return 4 - level;
    default:
        return 0;
    }
}

/**********************************************************************************
 *
 *
 * Internal helper functions
 *
 *
 *********************************************************************************/

static void
_gmmuFmtInitPdeMulti_GH10X(GMMU_FMT_PDE_MULTI *pPdeMulti, LW_FIELD_ENUM_ENTRY *pdeApertures)
{
#if LWCFG(GLOBAL_ARCH_HOPPER)
    GMMU_FMT_PDE *pPdeBig   = &pPdeMulti->subLevels[PDE_MULTI_BIG_INDEX];
    GMMU_FMT_PDE *pPdeSmall = &pPdeMulti->subLevels[PDE_MULTI_SMALL_INDEX];

    pPdeBig->version   = GMMU_FMT_VERSION_3;
    pPdeSmall->version = GMMU_FMT_VERSION_3;

    // Common PDE fields.
    INIT_FIELD_DESC32(&pPdeMulti->fldSizeRecipExp, LW_MMU_PDE_SIZE);

    // Dual PDE - big part.
    INIT_FIELD_APERTURE(&pPdeBig->fldAperture, LW_MMU_VER3_DUAL_PDE_APERTURE_BIG,
                        pdeApertures);
    INIT_FIELD_ADDRESS(&pPdeBig->fldAddr, LW_MMU_VER3_DUAL_PDE_ADDRESS_BIG,
                       LW_MMU_VER3_DUAL_PDE_ADDRESS_BIG_SHIFT);
    INIT_FIELD_DESC32(&pPdeBig->fldPdePcf, LW_MMU_VER3_DUAL_PDE_PCF_BIG);

    // Dual PDE - small part.
    INIT_FIELD_APERTURE(&pPdeSmall->fldAperture, LW_MMU_VER3_DUAL_PDE_APERTURE_SMALL,
                        pdeApertures);
    INIT_FIELD_ADDRESS(&pPdeSmall->fldAddr, LW_MMU_VER3_DUAL_PDE_ADDRESS_SMALL,
                       LW_MMU_VER3_DUAL_PDE_ADDRESS_SHIFT);
    INIT_FIELD_DESC32(&pPdeSmall->fldPdePcf, LW_MMU_VER3_DUAL_PDE_PCF_SMALL);
#endif
}

static void
_gmmuFmtInitPde_GH10X(GMMU_FMT_PDE *pPde, LW_FIELD_ENUM_ENTRY *pdeApertures)
{
#if LWCFG(GLOBAL_ARCH_HOPPER)
    pPde->version = GMMU_FMT_VERSION_3;

    INIT_FIELD_APERTURE(&pPde->fldAperture, LW_MMU_VER3_PDE_APERTURE,
                        pdeApertures);
    INIT_FIELD_ADDRESS(&pPde->fldAddr, LW_MMU_VER3_PDE_ADDRESS,
                       LW_MMU_VER3_PDE_ADDRESS_SHIFT);
    INIT_FIELD_DESC32(&pPde->fldPdePcf, LW_MMU_VER3_PDE_PCF);
#endif
}

void
vmemGmmuFmtInitPte_GH100(GMMU_FMT_PTE *pPte, LW_FIELD_ENUM_ENTRY *pteApertures)
{
#if LWCFG(GLOBAL_ARCH_HOPPER)
    pPte->version = GMMU_FMT_VERSION_3;

    INIT_FIELD_BOOL(&pPte->fldValid, LW_MMU_VER3_PTE_VALID);
    INIT_FIELD_APERTURE(&pPte->fldAperture, LW_MMU_VER3_PTE_APERTURE, pteApertures);
    INIT_FIELD_ADDRESS(&pPte->fldAddrSysmem, LW_MMU_VER3_PTE_ADDRESS_SYS,
                       LW_MMU_VER3_PTE_ADDRESS_SHIFT);
    INIT_FIELD_ADDRESS(&pPte->fldAddrVidmem, LW_MMU_VER3_PTE_ADDRESS_VID,
                       LW_MMU_VER3_PTE_ADDRESS_SHIFT);
    INIT_FIELD_ADDRESS(&pPte->fldAddrPeer, LW_MMU_VER3_PTE_ADDRESS_PEER,
                       LW_MMU_VER3_PTE_ADDRESS_SHIFT);
    INIT_FIELD_DESC32(&pPte->fldPeerIndex, LW_MMU_VER3_PTE_PEER_ID);
    INIT_FIELD_DESC32(&pPte->fldKind, LW_MMU_VER3_PTE_KIND);
    INIT_FIELD_DESC32(&pPte->fldPtePcf, LW_MMU_VER3_PTE_PCF);
#endif
}

static void
_gmmuInitMmuFmt_GH10X(MMU_FMT_LEVEL *pLevels, LwU32 bigPageShift)
{
    // Page directory 4 (root).
    pLevels[0].virtAddrBitHi = 56;
    pLevels[0].virtAddrBitLo = 56;
    pLevels[0].entrySize     = LW_MMU_VER3_PDE__SIZE;
    pLevels[0].numSubLevels  = 1;
    pLevels[0].subLevels     = pLevels + 1;

    // Page directory 3.
    pLevels[1].virtAddrBitHi = 55;
    pLevels[1].virtAddrBitLo = 47;
    pLevels[1].entrySize     = LW_MMU_VER3_PDE__SIZE;
    pLevels[1].numSubLevels  = 1;
    pLevels[1].subLevels     = pLevels + 2;

    // Page directory 2.
    pLevels[2].virtAddrBitHi = 46;
    pLevels[2].virtAddrBitLo = 38;
    pLevels[2].entrySize     = LW_MMU_VER3_PDE__SIZE;
    pLevels[2].numSubLevels  = 1;
    pLevels[2].subLevels     = pLevels + 3;

    // Page directory 1.
    pLevels[3].virtAddrBitHi = 37;
    pLevels[3].virtAddrBitLo = 29;
    pLevels[3].entrySize     = LW_MMU_VER3_PDE__SIZE;
    pLevels[3].numSubLevels  = 1;
    pLevels[3].subLevels     = pLevels + 4;
    // Page directory 1 can hold a PTE pointing to a 512MB Page
    pLevels[3].bPageTable    = LW_TRUE;

    // Page directory 0.
    pLevels[4].virtAddrBitHi = 28;
    pLevels[4].virtAddrBitLo = 21;
    pLevels[4].entrySize     = LW_MMU_VER3_DUAL_PDE__SIZE;
    pLevels[4].numSubLevels  = 2;
    pLevels[4].bPageTable    = LW_TRUE;
    pLevels[4].subLevels     = pLevels + 5;

    // Big page table.
    pLevels[5].virtAddrBitHi = 20;
    pLevels[5].virtAddrBitLo = (LwU8)bigPageShift;
    pLevels[5].entrySize     = LW_MMU_VER3_PTE__SIZE;
    pLevels[5].bPageTable    = LW_TRUE;

    // Small page table.
    pLevels[6].virtAddrBitHi = 20;
    pLevels[6].virtAddrBitLo = 12;
    pLevels[6].entrySize     = LW_MMU_VER3_PTE__SIZE;
    pLevels[6].bPageTable    = LW_TRUE;
}

void
vmemDumpPdeFlags_GH100(const GMMU_FMT_PDE *pFmt, const GMMU_ENTRY_VALUE *pPde)
{
    LwU32 pdePcfHw = lwFieldGet32(&pFmt->fldPdePcf, pPde->v8);
    LwU32 pdePcfSw;
    LW_STATUS status;

    dprintf("PdePcf=0x%x ", pdePcfHw);

    status = _gmmuTranslatePdePcfFromHw(pdePcfHw,
                                        gmmuFieldGetAperture(&pFmt->fldAperture, pPde->v8),
                                        &pdePcfSw);

    if (status == LW_OK)
    {
        dprintf("(Sparse=%d, Vol=%d, ATS=%d)",
               (pdePcfSw >> SW_MMU_PCF_SPARSE_IDX)      & 1,
               (pdePcfSw >> SW_MMU_PCF_UNCACHED_IDX)    & 1,
               (pdePcfSw >> SW_MMU_PCF_ATS_ALLOWED_IDX) & 1);
    }
}

void
vmemDumpPtePcf_GH100(const GMMU_FMT_PTE *pFmt, const GMMU_ENTRY_VALUE *pPte)
{
    LwU32 ptePcfHw = lwFieldGet32(&pFmt->fldPtePcf, pPte->v8);
    LwU32 ptePcfSw;
    LW_STATUS status;


    dprintf("PtePcf=0x%x ", ptePcfHw);

    status = _gmmuTranslatePtePcfFromHw(ptePcfHw,
                                        lwFieldGetBool(&pFmt->fldValid, pPte->v8),
                                        &ptePcfSw);

    if (status == LW_OK)
    {
        dprintf("(Vol=%d, Priv=%d, RO=%d, Atomic=%d, ACE=%d)",
                 (ptePcfSw  >> SW_MMU_PCF_UNCACHED_IDX) & 1,
                 !(ptePcfSw >> SW_MMU_PCF_REGULAR_IDX)  & 1,
                 (ptePcfSw  >> SW_MMU_PCF_RO_IDX)       & 1,
                 !(ptePcfSw >> SW_MMU_PCF_NOATOMIC_IDX) & 1,
                 (ptePcfSw  >> SW_MMU_PCF_ACE_IDX)      & 1);
    }
}

LwBool
vmemIsPdeVolatile_GH100(const GMMU_FMT_PDE *pFmt, const GMMU_ENTRY_VALUE *pPde)
{
    LwU32 pdePcfHw = lwFieldGet32(&pFmt->fldPdePcf, pPde->v8);
    LwU32 pdePcfSw;
    LW_STATUS status;

    status = _gmmuTranslatePdePcfFromHw(pdePcfHw,
                                        gmmuFieldGetAperture(&pFmt->fldAperture, pPde->v8),
                                        &pdePcfSw);

    if (status != LW_OK)
    {
        return LW_FALSE;
    }

    return ((pdePcfSw  >> SW_MMU_PCF_UNCACHED_IDX) & 1);
}

BOOL
vmemIsGvpteDeprecated_GH100()
{
    return LW_TRUE;
}

static LW_STATUS
_gmmuTranslatePdePcfFromHw
(
    LwU32    pdePcfHw,
    GMMU_APERTURE aperture,
    LwU32   *pPdePcfSw
)
{
    if (!aperture)
    {
        switch (pdePcfHw)
        {
            case (LW_MMU_VER3_PDE_PCF_ILWALID_ATS_ALLOWED):
            {
                *pPdePcfSw = SW_MMU_PDE_PCF_ILWALID_ATS_ALLOWED;
                break;
            }
            case (LW_MMU_VER3_PDE_PCF_ILWALID_ATS_NOT_ALLOWED):
            {
                *pPdePcfSw = SW_MMU_PDE_PCF_ILWALID_ATS_NOT_ALLOWED;
                break;
            }
            case (LW_MMU_VER3_PDE_PCF_SPARSE_ATS_ALLOWED):
            {
                *pPdePcfSw = SW_MMU_PDE_PCF_SPARSE_ATS_ALLOWED;
                break;
            }
            case (LW_MMU_VER3_PDE_PCF_SPARSE_ATS_NOT_ALLOWED):
            {
                *pPdePcfSw = SW_MMU_PDE_PCF_SPARSE_ATS_NOT_ALLOWED;
                break;
            }
            default:
                dprintf("Invalid HW PCF! ");
                return LW_ERR_GENERIC;
        }
    }
    else
    {
        switch (pdePcfHw)
        {
            case (LW_MMU_VER3_PDE_PCF_VALID_CACHED_ATS_ALLOWED):
            {
                *pPdePcfSw = SW_MMU_PDE_PCF_VALID_CACHED_ATS_ALLOWED;
                break;
            }
            case (LW_MMU_VER3_PDE_PCF_VALID_UNCACHED_ATS_ALLOWED):
            {
                *pPdePcfSw = SW_MMU_PDE_PCF_VALID_UNCACHED_ATS_ALLOWED;
                break;
            }
            case (LW_MMU_VER3_PDE_PCF_VALID_CACHED_ATS_NOT_ALLOWED):
            {
                *pPdePcfSw = SW_MMU_PDE_PCF_VALID_CACHED_ATS_NOT_ALLOWED;
                break;
            }
            case (LW_MMU_VER3_PDE_PCF_VALID_UNCACHED_ATS_NOT_ALLOWED):
            {
                *pPdePcfSw = SW_MMU_PDE_PCF_VALID_UNCACHED_ATS_NOT_ALLOWED;
                break;
            }
            default:
                dprintf("Invalid HW PCF! ");
                return LW_ERR_GENERIC;
        }
    }

    return LW_OK;
}



static LW_STATUS
_gmmuTranslatePtePcfFromHw
(
    LwU32    ptePcfHw,
    LwBool   bPteValid,
    LwU32   *pPtePcfSw
)
{
    if (!bPteValid)
    {
        switch (ptePcfHw)
        {
            case (LW_MMU_VER3_PTE_PCF_ILWALID):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_ILWALID;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_NO_VALID_4KB_PAGE):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_NO_VALID_4KB_PAGE;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_SPARSE):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_SPARSE;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_MAPPING_NOWHERE):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_MAPPING_NOWHERE;
                break;
            }
            default:
                dprintf("Invalid HW PCF!");
                return LW_ERR_GENERIC;
        }
    }
    else
    {
        switch (ptePcfHw)
        {
            case (LW_MMU_VER3_PTE_PCF_PRIVILEGE_RW_ATOMIC_CACHED_ACD):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_PRIVILEGE_RW_ATOMIC_CACHED_ACD;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_PRIVILEGE_RW_ATOMIC_CACHED_ACE):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_PRIVILEGE_RW_ATOMIC_CACHED_ACE;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_PRIVILEGE_RW_ATOMIC_UNCACHED_ACD):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_PRIVILEGE_RW_ATOMIC_UNCACHED_ACD;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_PRIVILEGE_RW_ATOMIC_UNCACHED_ACE):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_PRIVILEGE_RW_ATOMIC_UNCACHED_ACE;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_PRIVILEGE_RW_NO_ATOMIC_UNCACHED_ACE):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_PRIVILEGE_RW_NO_ATOMIC_UNCACHED_ACE;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_PRIVILEGE_RW_NO_ATOMIC_CACHED_ACE):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_PRIVILEGE_RW_NO_ATOMIC_CACHED_ACE;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_PRIVILEGE_RO_ATOMIC_UNCACHED_ACE):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_PRIVILEGE_RO_ATOMIC_UNCACHED_ACE;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_PRIVILEGE_RO_NO_ATOMIC_UNCACHED_ACE):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_PRIVILEGE_RO_NO_ATOMIC_UNCACHED_ACE;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_PRIVILEGE_RO_NO_ATOMIC_CACHED_ACE):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_PRIVILEGE_RO_NO_ATOMIC_CACHED_ACE;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_REGULAR_RW_ATOMIC_CACHED_ACD):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_REGULAR_RW_ATOMIC_CACHED_ACD;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_REGULAR_RW_ATOMIC_CACHED_ACE):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_REGULAR_RW_ATOMIC_CACHED_ACE;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_REGULAR_RW_ATOMIC_UNCACHED_ACD):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_REGULAR_RW_ATOMIC_UNCACHED_ACD;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_REGULAR_RW_ATOMIC_UNCACHED_ACE):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_REGULAR_RW_ATOMIC_UNCACHED_ACE;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_REGULAR_RW_NO_ATOMIC_CACHED_ACD):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_REGULAR_RW_NO_ATOMIC_CACHED_ACD;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_REGULAR_RW_NO_ATOMIC_CACHED_ACE):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_REGULAR_RW_NO_ATOMIC_CACHED_ACE;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_REGULAR_RW_NO_ATOMIC_UNCACHED_ACD):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_REGULAR_RW_NO_ATOMIC_UNCACHED_ACD;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_REGULAR_RW_NO_ATOMIC_UNCACHED_ACE):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_REGULAR_RW_NO_ATOMIC_UNCACHED_ACE;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_REGULAR_RO_ATOMIC_CACHED_ACD):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_REGULAR_RO_ATOMIC_CACHED_ACD;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_REGULAR_RO_ATOMIC_CACHED_ACE):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_REGULAR_RO_ATOMIC_CACHED_ACE;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_REGULAR_RO_ATOMIC_UNCACHED_ACD):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_REGULAR_RO_ATOMIC_UNCACHED_ACD;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_REGULAR_RO_ATOMIC_UNCACHED_ACE):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_REGULAR_RO_ATOMIC_UNCACHED_ACE;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_REGULAR_RO_NO_ATOMIC_CACHED_ACD):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_REGULAR_RO_NO_ATOMIC_CACHED_ACD;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_REGULAR_RO_NO_ATOMIC_CACHED_ACE):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_REGULAR_RO_NO_ATOMIC_CACHED_ACE;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_REGULAR_RO_NO_ATOMIC_UNCACHED_ACD):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_REGULAR_RO_NO_ATOMIC_UNCACHED_ACD;
                break;
            }
            case (LW_MMU_VER3_PTE_PCF_REGULAR_RO_NO_ATOMIC_UNCACHED_ACE):
            {
                *pPtePcfSw = SW_MMU_PTE_PCF_REGULAR_RO_NO_ATOMIC_UNCACHED_ACE;
                break;
            }
            default:
                dprintf("Invalid HW PCF!");
                return LW_ERR_GENERIC;
        }
    }

    return LW_OK;
}
