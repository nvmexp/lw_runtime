/* _lw_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _lw_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// vmemt30.c
//
//*****************************************************

//
// includes
//
#include "hal.h"
#include "vmem.h"

#include "t3x/t30/dev_armc_addendum.h"

#include "class/cle3f1.h"       // TEGRA_VASPACE_A
#include <limits.h>             // CHAR_BIT


// This file contains both vmem and mmu abstractions.

//
// defines
//
#define VASPACE_BYTEOFFSET_INDEX_FROM_VADDR_T30(V)      ( (LwU32)(V) & 0xfff )
#define VASPACE_PDE_INDEX_FROM_VADDR_T30(V)             ( (LwU32)(V >> 22) & 0x3ff )
#define VASPACE_PTE_INDEX_FROM_VADDR_T30(V)             ( (LwU32)(V >> 12) & 0x3ff )
#define SMMU_BIT(b)                                     (1<<(b))

#define LW_MMU_PTE_APERTURE_VIDEO_MEMORY                0
#define LW_MMU_PTE_APERTURE_PEER_MEMORY                 1
#define LW_MMU_PTE_APERTURE_SYSTEM_COHERENT_MEMORY      2
#define LW_MMU_PTE_APERTURE_SYSTEM_NON_COHERENT_MEMORY  3

#define TEGRA_PDE_COUNT                                 1024
#define TEGRA_BIG_PAGE_SIZE                             0x1000
#define TEGRA_PDE_SIZE                                  (TEGRA_PDE_COUNT * TEGRA_BIG_PAGE_SIZE)
#define LW_MMU_PDE_SIZE                                 (SMMU_PDE_SIZE / CHAR_BIT)
#define LW_MMU_PTE_SIZE                                 (SMMU_PTE_SIZE / CHAR_BIT)

//Used for compacting various memory flags into a single word
#define LW_WATCH_MEMORY_DESCRIPTION_APERTURE            1:0
#define LW_WATCH_MEMORY_DESCRIPTION_PEER                3:2

// All CheetAh memory is system non-coherent memmory
#define TEGRA_DEFAULT_MEM_DESCRIPTION DRF_NUM(_WATCH, _MEMORY_DESCRIPTION, _APERTURE, LW_MMU_PTE_APERTURE_SYSTEM_NON_COHERENT_MEMORY);



/*!
 *  Get the @a VMemSpace for a given ASID.
 *
 *  @param[out] pVMemSpace  Virtual memory space structure to populate.
 *  @param[in]  pIommu      Pointer to IOMMU struct
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS
vmemGetByAsId_T194
(
    VMemSpace  *pVMemSpace,
    VMEM_INPUT_TYPE_IOMMU *pIommu
)
{
    // Get PDE base for the requested ASID
    pVMemSpace->PdeBase = ILWALID_ADDRESS;
    pMmu[indexGpu].mmuGetIommuPdb(pIommu->asId, &pVMemSpace->PdeBase);

    pVMemSpace->readFn  = readSystem;
    pVMemSpace->writeFn = writeSystem;

    pVMemSpace->instBlock.memType = SYSTEM_PHYS;
    pVMemSpace->instBlock.readFn  = readSystem;
    pVMemSpace->instBlock.writeFn = writeSystem;

    // Setup the CheetAh PDE values
    pVMemSpace->pdeCount          = TEGRA_PDE_COUNT;
    pVMemSpace->bigPageSize       = TEGRA_BIG_PAGE_SIZE;
    pVMemSpace->class             = TEGRA_VASPACE_A;

    return LW_OK;
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
vmemPdeGetByVa_T30
(
    VMemSpace  *pVMemSpace,
    LwU64       va,
    PdeEntry   *pPde
)
{
    LwU32       pdeIndex;
    LW_STATUS status;

    pdeIndex = VASPACE_PDE_INDEX_FROM_VADDR_T30(va);
    status = vmemPdeGetByIndex(pVMemSpace, pdeIndex, (GMMU_ENTRY_VALUE*)pPde);
    if (status != LW_OK)
    {
        dprintf("lw:%s: PDE #%d is not present\n", __FUNCTION__, pdeIndex);
    }

    return status;
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
vmemPdeGetByIndex_T30
(
    VMemSpace        *pVMemSpace,
    LwU32             index,
    GMMU_ENTRY_VALUE *pPde
)
{
    LwU64       pdeBase;
    LW_STATUS       status;

    if (index >= pVMemSpace->pdeCount)
    {
        dprintf("lw: %s: PDE #%d is greater than max PDE table size (%d)\n",
                __FUNCTION__, index, pVMemSpace->pdeCount);
        return LW_ERR_GENERIC;
    }

    memset(pPde, 0, LW_MMU_PDE_SIZE);
    pdeBase = pVMemSpace->PdeBase + (index * LW_MMU_PDE_SIZE);
    if (pVMemSpace->readFn == NULL)
    {
        dprintf("**ERROR: NULL value of readFn.\n");
        return LW_ERR_NOT_SUPPORTED;
    }
    CHECK(pVMemSpace->readFn(pdeBase, pPde, LW_MMU_PDE_SIZE));

    if ((((PdeEntry*)pPde)->w0 & SMMU_PDE_NEXT_FIELD) == 0)
    {
        return LW_ERR_GENERIC;
    }

    return LW_OK;
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
vmemPteGetByVa_T30
(
    VMemSpace  *pVMemSpace,
    LwU64       va,
    PdeEntry   *pPde,
    PteEntry   *pPte
)
{
    LwU32     pteIndex = VASPACE_PTE_INDEX_FROM_VADDR_T30(va);
    LW_STATUS status;

    CHECK(pVmem[indexGpu].vmemPdeGetByVa(pVMemSpace, va, pPde))
    CHECK(pVmem[indexGpu].vmemPteGetByIndex(pVMemSpace, pteIndex, pPde, pPte));

    return LW_OK;
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
vmemPteGetByIndex_T30
(
    VMemSpace  *pVMemSpace,
    LwU32       pteIndex,
    PdeEntry   *pPde,
    PteEntry   *pPte
)
{
    LwU64       pteBase;
    LW_STATUS       status;

    if (pteIndex >= 1024)
    {
        dprintf("lw: %s: PTE index #%d is greater than PDE size (%d)\n",
                __FUNCTION__, pteIndex, 1024);
        return LW_ERR_GENERIC;
    }

    memset(pPte, 0, LW_MMU_PTE_SIZE);
    pteBase = ((pPde->w0 & SMMU_PDE_PTE_BASE_FIELD) << 12) + (pteIndex * LW_MMU_PTE_SIZE);
    if (pVMemSpace->readFn == NULL)
    {
        dprintf("**ERROR: NULL value of readFn.\n");
        return LW_ERR_NOT_SUPPORTED;
    }
    CHECK(pVMemSpace->readFn(pteBase, pPte, LW_MMU_PTE_SIZE));

    return LW_OK;
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
vmemVToP_T30
(
    VMemSpace       *pVMemSpace,
    LwU64            va,
    LwU64           *pPa,
    GMMU_APERTURE   *pMemDesc
)
{
    LW_STATUS       status;
    PdeEntry    pde;
    PteEntry    pte;

    CHECK(pVmem[indexGpu].vmemPdeGetByVa(pVMemSpace, va, &pde));
    CHECK(pVmem[indexGpu].vmemPteGetByVa(pVMemSpace, va, &pde, &pte));

    // Check for a valid PDE entry
    if ((pde.w0 & SMMU_PDE_NEXT_FIELD))
    {
        // Check for a valid PTE entry
        if ((pte.w0 & SMMU_PTE_READABLE_FIELD) || (pte.w0 & SMMU_PTE_WRITABLE_FIELD))
        {
            // Compute the physical address value
            *pPa = ((pte.w0 & SMMU_PTE_PA_FIELD) << 12) + VASPACE_BYTEOFFSET_INDEX_FROM_VADDR_T30(va);

            if (pMemDesc)
            {
                // All CheetAh memory is system non-coherent memmory
                *pMemDesc = TEGRA_DEFAULT_MEM_DESCRIPTION;
            }
            return LW_OK;
        }
    }
    return LW_ERR_GENERIC;
}

/*!
 *  Read @a length of data from a virtual address into a buffer.
 *
 *  @param[in]  pVMemSpace  Virtual memory space.
 *  @param[in]  va          Virtual address.
 *  @param[in]  length      Number of bytes to read.
 *  @param[out] pData       Buffer to store the bytes.
 */
LW_STATUS
vmemRead_T194
(
    VMemSpace  *pVMemSpace,
    LwU64       va,
    LwU32       length,
    void       *pData
)
{
    LW_STATUS status;
    LwU8 *pByteData = (LwU8*) pData;
    LwU32 virtAddr = (LwU32) va;
    LwU64 physAddr;
    LwU32 readSize = min(length, LW_PAGE_SIZE - VASPACE_BYTEOFFSET_INDEX_FROM_VADDR_T30(virtAddr));

    while (length)
    {
        // Read a page at a time
        CHECK(pVmem[indexGpu].vmemVToP(pVMemSpace, va, &physAddr, NULL));
        if (pVMemSpace->readFn == NULL)
        {
            dprintf("**ERROR: NULL value of readFn.\n");
            return LW_ERR_NOT_SUPPORTED;
        }
        CHECK(pVMemSpace->readFn(physAddr, pByteData, readSize));

        pByteData += readSize;
        va      += readSize;
        length  -= readSize;

        readSize   = min(length, LW_PAGE_SIZE);
        }

    return LW_OK;
}

/*!
 *  Write @a length of data to a virtual address from a buffer.
 *
 *  @param[in]  pVMemSpace  Virtual memory space.
 *  @param[in]  va          Virtual address.
 *  @param[in]  length      Number of bytes to write.
 *  @param[out] pData       Buffer of data to write.
 */
LW_STATUS
vmemWrite_T194
(
    VMemSpace  *pVMemSpace,
    LwU64       va,
    LwU32        length,
    void       *pData
)
{
    LW_STATUS status;
    LwU8 *pByteData = (LwU8*) pData;
    LwU32 virtAddr = (LwU32) va;
    LwU64 physAddr;
    LwU32 writeSize = min(length, LW_PAGE_SIZE - VASPACE_BYTEOFFSET_INDEX_FROM_VADDR_T30(virtAddr));

    while (length)
        {
        // Write a page at a time
        CHECK(pVmem[indexGpu].vmemVToP(pVMemSpace, va, &physAddr, NULL));
        if (pVMemSpace->writeFn == NULL)
        {
            dprintf("**ERROR: NULL value of writeFn.\n");
            return LW_ERR_NOT_SUPPORTED;
        }
        CHECK(pVMemSpace->writeFn(physAddr, pByteData, writeSize));

        pByteData += writeSize;
        va        += writeSize;
        length    -= writeSize;

        writeSize  = min(length, LW_PAGE_SIZE);
    }

    return LW_OK;
}

/*!
 *  Fill @a data to a virtual address.
 *
 *  @param[in]  pVMemSpace  Virtual memory space.
 *  @param[in]  va          Virtual address.
 *  @param[in]  length      Length to fill.
 *  @param[in]  data        Data to write.
 *
 *  @return LW_ERR_GENERIC on error, LW_OK on success.
 */
LW_STATUS
vmemFill_T194
(
    VMemSpace  *pVMemSpace,
    LwU64       va,
    LwU32        length,
    LwU32        data
)
{
    LW_STATUS status;
    LwU32 virtAddr = (LwU32) va;
    LwU64 physAddr;
    LwU32 fillSize = min(length, LW_PAGE_SIZE - VASPACE_BYTEOFFSET_INDEX_FROM_VADDR_T30(virtAddr));
    LwU8 pageData[LW_PAGE_SIZE];

    // Fill the page data with the data avlue
    memset(pageData, data, LW_PAGE_SIZE);

    while (length)
    {
        // Fill a page at a time
        CHECK(pVmem[indexGpu].vmemVToP(pVMemSpace, va, &physAddr, NULL));
        if (pVMemSpace->writeFn == NULL)
        {
            dprintf("**ERROR: NULL value of writeFn.\n");
            return LW_ERR_NOT_SUPPORTED;
        }
        CHECK(pVMemSpace->writeFn(physAddr, &pageData, fillSize));

        va     += fillSize;
        length -= fillSize;

        fillSize = min(length, LW_PAGE_SIZE);
    }

    return LW_OK;
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
vmemDoVToPDump_T30(VMemSpace *pVMemSpace, LwU64 va)
{
    LW_STATUS status;
    LwU32 pdeIndex = VASPACE_PDE_INDEX_FROM_VADDR_T30(va);
    LwU32 pteIndex = VASPACE_PTE_INDEX_FROM_VADDR_T30(va);
    LwU64 pageDirPhysAddr;
    LwU64 pageTablePhysAddr;
    LwU64 physAddr;
    LwU64 physAddrToRead;
    LwU32 pte;
    LwU64 bytesRead;

    // Get the physical address of the Page Directory
    pageDirPhysAddr = pVMemSpace->PdeBase;

    dprintf("lw: va:                    " LwU64_FMT "\n", va);
    dprintf("lw: pdeIndex:              0x%08x\n", pdeIndex);
    dprintf("lw: pteIndex:              0x%08x\n", pteIndex);
    dprintf("lw: pageDirPhysAddr:       " LwU64_FMT "\n", pageDirPhysAddr);

    // Read the PDE
    physAddrToRead = pageDirPhysAddr + (pdeIndex * LW_MMU_PDE_SIZE);
    status = readPhysicalMem(physAddrToRead, &pageTablePhysAddr, sizeof(LwU32), &bytesRead);
    if (status != LW_OK)
    {
        dprintf("lw: %s - readPhysicalMem failed of " LwU64_FMT "!\n", __FUNCTION__, physAddrToRead);
        return;
    }
    dprintf("lw: pde:                   " LwU64_FMT "\n", pageTablePhysAddr);

    // Support for lazy pagetable allocation scheme.
    if (pageTablePhysAddr == 0)
    {
        dprintf("lw: - SMMU PDE is not yet populated!; maybe lazy PT allocation \n");
        return;
    }

    // Check if the PDE is valid
    if ((pageTablePhysAddr & SMMU_BIT(SMMU_PDE_NEXT_SHIFT)) == 0)
    {
        dprintf("lw: - SMMU_PDE_NEXT is not set!\n");
    }
    if ((pageTablePhysAddr & SMMU_BIT(SMMU_PDE_READABLE_SHIFT)) == 0)
    {
        dprintf("lw: - SMMU_PDE_READABLE is not set!\n");
    }
    if ((pageTablePhysAddr & SMMU_BIT(SMMU_PDE_WRITABLE_SHIFT)) == 0)
    {
        dprintf("lw: - SMMU_PDE_WRITABLE is not set!\n");
    }

    pageTablePhysAddr &= 0xfffff;
    pageTablePhysAddr <<= 12;
    dprintf("lw: pageTablePhysAddr:     " LwU64_FMT "\n", pageTablePhysAddr);

    // Read the PTE
    physAddrToRead = pageTablePhysAddr + (pteIndex * 4);
    status = readPhysicalMem(physAddrToRead, &pte, sizeof(LwU32), &bytesRead);
    if (status != LW_OK)
    {
        dprintf("lw: %s - readPhysicalMem failed of " LwU64_FMT "!\n", __FUNCTION__, physAddrToRead);
        return;
    }
    dprintf("lw: pte:                   0x%08x\n", pte);

    // Check if the PTE is valid
    if ((pte & SMMU_BIT(SMMU_PTE_READABLE_SHIFT)) == 0)
    {
        dprintf("lw: - SMMU_PTE_READABLE is not set!\n");
    }
    if ((pte & SMMU_BIT(SMMU_PTE_WRITABLE_SHIFT)) == 0)
    {
        dprintf("lw: - SMMU_PTE_WRITABLE is not set!\n");
    }

    pte &= 0xfffff;
    pte <<= 12;
    physAddr = pte + VASPACE_BYTEOFFSET_INDEX_FROM_VADDR_T30(va);
    dprintf("lw: physAddr:              " LwU64_FMT "\n", physAddr);
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
vmemPToV_T30(VMemTypes vMemType, VMEM_INPUT_TYPE *pId, LwU64 physAddr, BOOL vidMem)
{
    LW_STATUS status;
    VMemSpace   vMemSpace;
    LwU32       pdeIndex = 0;
    LwU32       pteIndex = 0;
    LwU64       virtAddr = 0;
    PdeEntry    pde;
    PteEntry    pte;

    // Try to get the right memory space for the search
    CHECK_EXIT(vmemGet(&vMemSpace, vMemType, pId));

    // Loop thru all the PDE's check them for a matching physical address
    for (pdeIndex = 0; pdeIndex < 1024; pdeIndex++)
    {
        // See if the user requested a break
        if (osCheckControlC())
            return;

        // Get the next PDE to check
        CHECK_EXIT(vmemPdeGetByIndex(&vMemSpace, pdeIndex, (GMMU_ENTRY_VALUE*)&pde));

        // Check for a valid PDE entry
        if ((pde.w0 & SMMU_PDE_NEXT_FIELD))
        {
            // Loop thru all the PTE's for the current PDE checking for a match
            for (pteIndex = 0; pteIndex < 1024; pteIndex++)
            {
                // See if the user requested a break
                if (osCheckControlC())
                    return;

                // Get the next PTE to check
                CHECK_EXIT(pVmem[indexGpu].vmemPteGetByIndex(&vMemSpace, pteIndex, &pde, &pte));

                // Check for a valid PTE entry
                if ((pte.w0 & SMMU_PTE_READABLE_FIELD) || (pte.w0 & SMMU_PTE_WRITABLE_FIELD))
                {
                    // Check to see if this PTE entry maps to the requested physical page
                    if ((pte.w0 & SMMU_PTE_PA_FIELD) == (physAddr >> 12))
                    {
                        // Callwlate the matching virtual address value
                        virtAddr = (pdeIndex * TEGRA_PDE_SIZE) + (pteIndex * LW_PAGE_SIZE) + (physAddr % LW_PAGE_SIZE);

                        // Address is found terminate the search loop
                        break;
                    }
                }
            }
        }
    }
    // Check to see if no matching virtual address was found
    if ((pdeIndex == 1024) && (pteIndex == 1024))
    {
        dprintf("lw: physAddr: " LwU64_FMT "\n", physAddr);
        dprintf("lw: virtAddr: No matching address\n");
    }
    else    // Matching virtual address found
    {
        dprintf("lw: physAddr: " LwU64_FMT "\n", physAddr);
        dprintf("lw: virtAddr: " LwU64_FMT "\n", virtAddr);
    }
exit:
    return;
}
