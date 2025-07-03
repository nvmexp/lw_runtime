/* _lw_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _lw_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// vmemt124.c
//
//*****************************************************

//
// includes
//
#include "vmem.h"
#include "tegrasys.h"

#include "t12x/t124/dev_armc.h"
#include "cheetah/tegra_access.h"

#include "g_vmem_private.h"
#include "class/cl90f1.h"      // FERMI_VASPACE_A
#include "class/cle3f1.h"      // TEGRA_VASPACE_A


//
// getGpuAsid
//
static LwU32
getGpuAsid(void)
{
    LwU32 regData, devIndex = pTegrasys[indexGpu].tegrasysGetDeviceBroadcastIndex(&TegraSysObj[indexGpu], "MC");

    regData = MC_REG_RD32(LW_PMC_SMMU_GPUB_ASID, devIndex);

    if (!DRF_VAL(_PMC, _SMMU_GPUB_ASID, _GPUB_SMMU_ENABLE, regData))
        dprintf("lw: expected GPUB_ASID to be enabled.\n");

    return DRF_VAL(_PMC, _SMMU_GPUB_ASID, _GPUB_ASID, regData);
}

/*!
 *  Colwert a virtual address into a physical address and what type of memory
 *  that physical address resides.
 *
 *  @param[in]  pVMemSpace  Virtual memory space.
 *  @param[in]  va          Virtual address.
 *  @param[out] pPa         Physical address.
 *  @param[out] pMemDesc    Type of memory physical address resides.
 *
 *  @return LW_ERR_GENERIC on error, LW_OK on success.
 */
LW_STATUS
vmemVToP_T124
(
    VMemSpace     *pVMemSpace,
    LwU64          va,
    LwU64         *pPa,
    GMMU_APERTURE *pMemDesc
)
{
    VMemSpace smmuMemSpace;
    VMEM_INPUT_TYPE Id;
    memset(&Id, 0, sizeof(Id));

    switch (pVMemSpace->class)
    {
        case FERMI_VASPACE_A:
            Id.iommu.asId = getGpuAsid();
            pVMemSpace = &smmuMemSpace;
            if (vmemGet(pVMemSpace, VMEM_TYPE_IOMMU, &Id) != LW_OK)
            {
                return LW_ERR_GENERIC;
            }
            /*FALLTHROUGH*/

        case TEGRA_VASPACE_A:
            return vmemVToP_T30(pVMemSpace, va, pPa, pMemDesc);
    }
    dprintf("Unexpected memory class 0x%08x\n", (LwU32)pVMemSpace->class);
    return LW_ERR_GENERIC;
}

/*!
 *  Dump the GPU virtual address of phys address. If the
 *  colwersion failed, error message will be printed. The final virtual
 *  address is printed along with the type of virtual memory space it belongs to.
 *
 *  @see gptov
 *
 *  @param[in]  vMemType    Virtual memory type.
 *  @param[in]  pId         Pointer to union VMEM_INPUT_TYPE
 *  @param[in]  physAddr    Physical address.
 *  @param[in]  vidMem      If the physical address resides in vidmem
 *
 * vidMem is used for the carveout scenario, where there are two
 * physical addresses (1. the sysmem address 2. vidmem address
 * The parameter tells whether we need interpret the phys address
 * is the sysmem or vidmem address
 */
void
vmemPToV_T124(VMemTypes vMemType, VMEM_INPUT_TYPE *pId, LwU64 physAddr, BOOL vidMem)
{
    if (vMemType == VMEM_TYPE_IOMMU)
        vmemPToV_T30(vMemType, pId, physAddr, vidMem);
    else
        vmemPToV_STUB(vMemType, pId, physAddr, vidMem);
}

/*!
 *  Read PDE corresponding to a virtual address from @a pVMemSpace.
 *
 *  @param[in]  pVMemSpace  Virtual memory space.
 *  @param[in]  va          Virtual address.
 *  @param[out] pPde        PDE structure to populate.
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS
vmemPdeGetByVa_T124
(
    VMemSpace  *pVMemSpace,
    LwU64       va,
    PdeEntry   *pPde
)
{
    switch (pVMemSpace->class)
    {
    case FERMI_VASPACE_A: return vmemPdeGetByVa_GK104(pVMemSpace, va, pPde);
    case TEGRA_VASPACE_A: return vmemPdeGetByVa_T30(pVMemSpace, va, pPde);
    }
    dprintf("Unexpected memory class 0x%08x\n", (LwU32)pVMemSpace->class);
    return LW_ERR_GENERIC;
}

/*!
 *  Read PDE at @a index from @a pVMemSpace. If @a index is not in the valid
 *  allowed range for PDE indices, an error message is printed and the call
 *  returns failure. Any fatal error will print an error message.
 *
 *  @param[in]  pVMemSpace  Virtual memory space.
 *  @param[in]  index       PDE index to read.
 *  @param[out] pPde        PDE structure to populate.
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS
vmemPdeGetByIndex_T124
(
    VMemSpace        *pVMemSpace,
    LwU32             index,
    GMMU_ENTRY_VALUE *pPde
)
{
    return vmemPdeGetByIndex_T30(pVMemSpace, index, pPde);
}

/*!
 *  Get a PTE and the PDE which it resides in from a virtual address.
 *
 *  @param[in]  pVMemSpace  Virtual memory space.
 *  @param[in]  va          Virtual address.
 *  @param[out] pPde        PDE which PTE is in.
 *  @param[out] pPte        PTE for @a va.
 *
 *  @return LW_OK on success, LW_ERR_GENERIC otherwise.
 */
LW_STATUS
vmemPteGetByVa_T124
(
    VMemSpace  *pVMemSpace,
    LwU64       va,
    PdeEntry   *pPde,
    PteEntry   *pPte
)
{
    switch (pVMemSpace->class)
    {
    case FERMI_VASPACE_A: return vmemPteGetByVa_GK104(pVMemSpace, va, pPde, pPte);
    case TEGRA_VASPACE_A: return vmemPteGetByVa_T30(pVMemSpace, va, pPde, pPte);
    }
    dprintf("Unexpected memory class 0x%08x\n", (LwU32)pVMemSpace->class);
    return LW_ERR_GENERIC;
}

/*!
 *  Read PTE at @a index from @a PDE in @a pVMemSpace. If the PTE index is
 *  invalid or any other failure oclwrs, an error message will be printed and
 *  the function returns failure. @a index is checked for it's range in the PDE
 *  and an error is printed if not valid.
 *
 *  @param[in]  pVmemSpace  Virtual memory space.
 *  @param[in]  pteIndex    PTE index.
 *  @param[in]  pPde        PDE where PTE resides.
 *  @param[out] pPte        PTE structure to populate.
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS
vmemPteGetByIndex_T124
(
    VMemSpace  *pVMemSpace,
    LwU32       pteIndex,
    PdeEntry   *pPde,
    PteEntry   *pPte
)
{
    switch (pVMemSpace->class)
    {
    case FERMI_VASPACE_A: return vmemPteGetByIndex_GK104(pVMemSpace, pteIndex, pPde, pPte);
    case TEGRA_VASPACE_A: return vmemPteGetByIndex_T30(pVMemSpace, pteIndex, pPde, pPte);
    }
    dprintf("Unexpected memory class 0x%08x\n", (LwU32)pVMemSpace->class);
    return LW_ERR_GENERIC;
}

/*!
 *  Dump the physical address mapping of a virtual address of a channel. If the
 *  colwersion failed, error message will be printed. The final physical
 *  address is printed along with the type of memory it points to.
 *
 *  @param[in]  pVmemspace    vmem space for target channel/BAR etc.
 *  @param[in]  va      Virtual address.
 */
void
vmemDoVToPDump_T124
(
    VMemSpace  *pVMemSpace,
    LwU64       va
)
{
    VMemSpace smmuMemSpace;
    VMEM_INPUT_TYPE Id;
    memset(&Id, 0, sizeof(Id));

    switch (pVMemSpace->class)
    {
    case FERMI_VASPACE_A:
        Id.iommu.asId = getGpuAsid();
        pVMemSpace = &smmuMemSpace;
        if (vmemGet(pVMemSpace, VMEM_TYPE_IOMMU, &Id) != LW_OK)
        {
            return;
        }

        /*FALLTHROUGH*/

    case TEGRA_VASPACE_A:
        vmemDoVToPDump_T30(pVMemSpace, va);
        return;
    }
    dprintf("Unexpected memory class 0x%08x\n", (LwU32)pVMemSpace->class);
}
