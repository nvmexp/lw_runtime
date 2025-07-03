 /****************************************************************************\
|*                                                                            *|
|*      Copyright 2016-2017 LWPU Corporation.  All rights reserved.         *|
|*                                                                            *|
|*  NOTICE TO USER:                                                           *|
|*                                                                            *|
|*  This source code is subject to LWPU ownership rights under U.S. and     *|
|*  international Copyright laws.                                             *|
|*                                                                            *|
|*  This software and the information contained herein is PROPRIETARY and     *|
|*  CONFIDENTIAL to LWPU and is being provided under the terms and          *|
|*  conditions of a Non-Disclosure Agreement. Any reproduction or             *|
|*  disclosure to any third party without the express written consent of      *|
|*  LWPU is prohibited.                                                     *|
|*                                                                            *|
|*  LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE       *|
|*  CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR           *|
|*  IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH       *|
|*  REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF           *|
|*  MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR            *|
|*  PURPOSE. IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL,              *|
|*  INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES            *|
|*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN        *|
|*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING       *|
|*  OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE        *|
|*  CODE.                                                                     *|
|*                                                                            *|
|*  U.S. Government End Users. This source code is a "commercial item"        *|
|*  as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting         *|
|*  of "commercial computer software" and "commercial computer software       *|
|*  documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)     *|
|*  and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through          *|
|*  227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the         *|
|*  source code with only those rights set forth herein.                      *|
|*                                                                            *|
|*  Module: osmdl.cpp                                                         *|
|*                                                                            *|
 \****************************************************************************/
#include "osprecomp.h"

//******************************************************************************
//
//  os namespace
//
//******************************************************************************
namespace os
{

//******************************************************************************
//
// Locals
//
//******************************************************************************
// MDL Type Helpers
CMemberType     CMdl::m_MdlType                             (&osKernel(), "MDL", "_MDL");

// MDL Field Helpers
CMemberField    CMdl::m_NextField                           (&MdlType(), false, NULL, "Next");
CMemberField    CMdl::m_SizeField                           (&MdlType(), false, NULL, "Size");
CMemberField    CMdl::m_MdlFlagsField                       (&MdlType(), false, NULL, "MdlFlags");
CMemberField    CMdl::m_ProcessField                        (&MdlType(), false, NULL, "Process");
CMemberField    CMdl::m_MappedSystemVaField                 (&MdlType(), false, NULL, "MappedSystemVa");
CMemberField    CMdl::m_StartVaField                        (&MdlType(), false, NULL, "StartVa");
CMemberField    CMdl::m_ByteCountField                      (&MdlType(), false, NULL, "ByteCount");
CMemberField    CMdl::m_ByteOffsetField                     (&MdlType(), false, NULL, "ByteOffset");

//******************************************************************************

CMdl::CMdl
(
    POINTER             ptrMdl
)
:   m_ptrMdl(ptrMdl),
    INIT(Next),
    INIT(Size),
    INIT(MdlFlags),
    INIT(Process),
    INIT(MappedSystemVa),
    INIT(StartVa),
    INIT(ByteCount),
    INIT(ByteOffset),
    m_pNextMdl(NULL),
    m_pProcess(NULL),
    m_aPages(NULL),
    m_ulTotalCount(0),
    m_ulTotalPages(0),
    m_aAllPages(NULL),
    m_aContents(NULL)
{
    // Get the MDL information
    READ(Next,           ptrMdl);
    READ(Size,           ptrMdl);
    READ(MdlFlags,       ptrMdl);
    READ(Process,        ptrMdl);
    READ(MappedSystemVa, ptrMdl);
    READ(StartVa,        ptrMdl);
    READ(ByteCount,      ptrMdl);
    READ(ByteOffset,     ptrMdl);

    // Set total size and pages to the initial MDL values
    m_ulTotalCount = ByteCount();
    m_ulTotalPages = pageCount();

    // Check for a chained MDL
    if (isChained())
    {
        // Try to create the next MDL in the chain (Relwrsive create)
        m_pNextMdl = new CMdl(Next());
        if (m_pNextMdl != NULL)
        {
            // Add the size of this MDL into the totals
            m_ulTotalCount += m_pNextMdl->totalCount();
            m_ulTotalPages += static_cast<ULONG>(m_pNextMdl->totalPages());
        }
    }

} // CMdl

//******************************************************************************

CMdl::~CMdl()
{

} // ~CMdl

//******************************************************************************

CMdlPtr
CMdl::nextMdl() const
{
    // Check to see if the next MDL hasn't been created yet
    if (m_pNextMdl == NULL)
    {
        // Check to see if there *is* a next MDL
        if (Next() != NULL)
        {
            // Try to create the next MDL
            m_pNextMdl = new CMdl(Next());
        }
    }
    return m_pNextMdl;

} // nextMdl

//******************************************************************************

CEProcessPtr
CMdl::process() const
{
    PROCESS             ptrProcess;

    // Check for MDL process object already created
    if (m_pProcess == NULL)
    {
        // Check to see if MDL process is available
        ptrProcess = Process();
        if (ptrProcess != NULL)
        {
            // Try to create the MDL process object
            m_pProcess = createEProcess(ptrProcess);
        }
    }
    return m_pProcess;

} // process

//******************************************************************************

DWORD*
CMdl::pages() const
{
    ULONG               ulPageCount;
    CPU_VIRTUAL         vaPages;
    
    // Check to see if the pages need to be loaded
    if (m_aPages == NULL)
    {
        // Get the page count for this MDL (Don't need to load anything if no pages)
        ulPageCount = pageCount();
        if (ulPageCount != 0)
        {
            // Try to allocate the array for the page values
            m_aPages = new DWORD[ulPageCount];
            if (m_aPages != NULL)
            {
                // Callwlate a pointer to where the page values are (right after the MDL)
                vaPages = static_cast<CPU_VIRTUAL>(ptrMdl()) + type().size();

                // Try to read the MDL pages into the array
                readCpuVirtual(vaPages, m_aPages.ptr(), ulPageCount * sizeof(DWORD));
            }
        }
    }
    return m_aPages;

} // pages

//******************************************************************************

DWORD*
CMdl::allPages() const
{
    CMdlPtr             pMdl = this;
    ULONG64             ulPageCount;
    DWORD              *pPages;
    ULONG64             ulPageOffset = 0;

    // Check to see if all pages need to be created
    if (m_aAllPages == NULL)
    {
        // Get the total page count for this MDL (Don't need to create anything if no pages)
        ulPageCount = totalPages();
        if (ulPageCount != 0)
        {
            // Try to allocate the array for the page values
            m_aAllPages = new DWORD[static_cast<unsigned int>(ulPageCount)];
            if (m_aAllPages != NULL)
            {
                // Loop over all the MDL's in the chain copying the pages
                while (pMdl != NULL)
                {
                    // Get the pages for this MDL
                    pPages = pMdl->pages();
                    if (pPages != NULL)
                    {
                        // Copy the pages into the all pages array at the proper offset
                        memcpy(m_aAllPages.ptr() + ulPageOffset, pPages, pMdl->pageCount() * sizeof(DWORD));

                        // Update the page offset and move to the next MDL in the chain
                        ulPageOffset += pMdl->pageCount();
                        pMdl          = pMdl->nextMdl();
                    }
                    else    // Ran out of MDL pages (Shouldn't happen)
                    {
                        // Fill the rest of the pages with 0
                        memset(m_aAllPages.ptr() + ulPageOffset, 0, static_cast<size_t>((totalPages() - ulPageOffset) * sizeof(DWORD)));

                        break;
                    }
                }
            }
        }
    }
    return m_aAllPages;

} // allPages

//******************************************************************************

DWORD
CMdl::page
(
    ULONG64             ulPage
) const
{
    CMdlPtr             pMdl = this;
    ULONG64             ulPageOffset = ulPage;
    DWORD              *pPages;
    DWORD               dwPage = 0;

    // Try to find the right MDL for the requested page (Could be in another MDL if MDL is chained)
    while((pMdl != NULL) && (ulPageOffset >= pMdl->pageCount()))
    {
        // Adjust the page offset for the current MDL
        ulPageOffset -= pMdl->pageCount();

        // Move to the next MDL in the chain (if there is one)
        pMdl = pMdl->nextMdl();
    }
    // Check for an invalid page (Can't find the MDL this page is part of)
    if (pMdl == NULL)
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Invalid MDL page index %I64d (>= %I64d)",
                         ulPage, totalPages());
    }
    // Try to get the pages for the MDL found
    pPages = pMdl->pages();
    if (pPages != NULL)    
    {
        // Get the requested page
        dwPage = pPages[ulPageOffset];
    }
    return dwPage;

} // page

//******************************************************************************

CString
CMdl::flagsString() const
{
    CString             sFlagsString("");

    // Add the selected flag strings
    if (isMappedToSystemVa())     addFlag(sFlagsString, "MAPPED_TO_SYSTEM_VA");
    if (hasPagesLocked())         addFlag(sFlagsString, "PAGES_LOCKED");
    if (isSourceNonpagedPool())   addFlag(sFlagsString, "SOURCE_IS_NONPAGED_POOL");
    if (isAllocatedFixedSize())   addFlag(sFlagsString, "ALLOCATED_FIXED_SIZE");
    if (isPartial())              addFlag(sFlagsString, "PARTIAL");
    if (hasPartialBeenMapped())   addFlag(sFlagsString, "PARTIAL_HAS_BEEN_MAPPED");
    if (isIoPageRead())           addFlag(sFlagsString, "IO_PAGE_READ");
    if (isWriteOperation())       addFlag(sFlagsString, "WRITE_OPERATION");
    if (isParentMappedSystemVa()) addFlag(sFlagsString, "PARENT_MAPPED_SYSTEM_VA");
    if (hasFreeExtraPtes())       addFlag(sFlagsString, "FREE_EXTRA_PTES");
    if (describesAwe())           addFlag(sFlagsString, "DESCRIBES_AWE");
    if (isIoSpace())              addFlag(sFlagsString, "IO_SPACE");
    if (isNetworkHeader())        addFlag(sFlagsString, "NETWORK_HEADER");
    if (mappingCanFail())         addFlag(sFlagsString, "MAPPING_CAN_FAIL");
    if (allocatedMustSucceed())   addFlag(sFlagsString, "ALLOCATED_MUST_SUCCEED");
    if (isInternal())             addFlag(sFlagsString, "INTERNAL");

    return sFlagsString;

} // flagsString

//******************************************************************************

ULONG
CMdl::readPhysical
(
    ULONG64             ulOffset,
    PVOID               pBuffer,
    ULONG               ulBufferSize,
    ULONG               ulFlags
) const
{
    const CMdlPtr       pMdl = this;
    CPU_PHYSICAL        paCpuAddress;
    ULONG               ulTotalSize;
    ULONG               ulMdlSize;
    ULONG               ulPartialSize;
    float               fPercentage;
    char               *pLwrrentBuffer = reinterpret_cast<char*>(pBuffer);
    ULONG               ulReadSize = 0;

    assert(pBuffer != NULL);

    // Make sure we have a valid offset and size (Doesn't exceed MDL)
    if ((ulOffset + ulBufferSize) > totalCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Requested offset/size exceeds MDL size (0x%0I64x >= 0x%0x)",
                         (ulOffset + ulBufferSize), totalCount());
    }
    // Search for the correct MDL containing the requested offset (in MDL chain)
    while (ulOffset > pMdl->ByteCount())
    {
        // Update the offset value and move to next MDL in chain
        ulOffset -= pMdl->ByteCount();
        pMdl      = pMdl->nextMdl();
    }
    // Initialize total size to the buffer size
    ulTotalSize = ulBufferSize;

    // Set percentage to amount read
    fPercentage = static_cast<float>(ulTotalSize - ulBufferSize) / static_cast<float>(ulTotalSize);
    progressPercentage(fPercentage * 100.0f);

    // Loop until requested data is read
    while (ulBufferSize != 0)
    {
        // Compute the MDL size (In case chained MDL load)
        ulMdlSize = min(ulBufferSize, pMdl->ByteCount());

        // Check for initial partial page read required
        if (pMdl->ByteOffset() != 0)
        {
            // Compute the partial read size
            ulPartialSize = min((PAGE_SIZE - pMdl->ByteOffset()), ulMdlSize);

            // Call routine to get the current physical address
            paCpuAddress = pMdl->physicalAddress(ulOffset);

            // Try to read the partial initial page
            ulReadSize = readCpuPhysical(paCpuAddress, pLwrrentBuffer, ulPartialSize, ulFlags);
            if (ulReadSize == ulPartialSize)
            {
                // Update the offset, sizes, and buffer pointer
                ulOffset       += ulReadSize;
                ulMdlSize      -= ulReadSize;
                ulBufferSize   -= ulReadSize;
                pLwrrentBuffer += ulReadSize;
            }
            else    // Unable to read partial initial page
            {
                throw CTargetException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unable to read 0x%0x (0x%0x) bytes at offset 0x%0x (0x%0I64x)",
                                       ulPartialSize, ulReadSize, ulOffset, paCpuAddress.addr());
            }
            // Update percentage read
            fPercentage = static_cast<float>(ulTotalSize - ulBufferSize) / static_cast<float>(ulTotalSize);
            progressPercentage(fPercentage * 100.0f);
        }
        // Loop reading full MDL pages
        while (ulMdlSize >= PAGE_SIZE)
        {
            // Call routine to get the current physical address
            paCpuAddress = pMdl->physicalAddress(ulOffset);

            // Try to read the full page
            ulReadSize = readCpuPhysical(paCpuAddress, pLwrrentBuffer, PAGE_SIZE, ulFlags);
            if (ulReadSize == PAGE_SIZE)
            {
                // Update the offset, sizes, and buffer pointer
                ulOffset       += ulReadSize;
                ulMdlSize      -= ulReadSize;
                ulBufferSize   -= ulReadSize;
                pLwrrentBuffer += ulReadSize;
            }
            else    // Unable to read partial initial page
            {
                throw CTargetException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unable to read 0x%0x (0x%0x) bytes at offset 0x%0x (0x%0I64x)",
                                       PAGE_SIZE, ulReadSize, ulOffset, paCpuAddress.addr());
            }
            // Update percentage read
            fPercentage = static_cast<float>(ulTotalSize - ulBufferSize) / static_cast<float>(ulTotalSize);
            progressPercentage(fPercentage * 100.0f);
        }
        // Check for final partial page read required
        if (ulMdlSize != 0)
        {
            // Call routine to get the current physical address
            paCpuAddress = pMdl->physicalAddress(ulOffset);

            // Try to read the final partial page
            ulReadSize = readCpuPhysical(paCpuAddress, pLwrrentBuffer, ulMdlSize, ulFlags);
            if (ulReadSize == ulMdlSize)
            {
                // Update the offset, sizes, and buffer pointer
                ulOffset       += ulReadSize;
                ulMdlSize      -= ulReadSize;
                ulBufferSize   -= ulReadSize;
                pLwrrentBuffer += ulReadSize;
            }
            else    // Unable to read partial initial page
            {
                throw CTargetException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unable to read 0x%0x (0x%0x) bytes at offset 0x%0x (0x%0I64x)",
                                       ulMdlSize, ulReadSize, ulOffset, paCpuAddress.addr());
            }
            // Update percentage read
            fPercentage = static_cast<float>(ulTotalSize - ulBufferSize) / static_cast<float>(ulTotalSize);
            progressPercentage(fPercentage * 100.0f);
        }
        // Move to the next MDL in the chain (if there is one) and reset the offset
        pMdl     = pMdl->nextMdl();
        ulOffset = 0;
    }
    return ulTotalSize;

} // readPhysical

//******************************************************************************

ULONG
CMdl::writePhysical
(
    ULONG64             ulOffset,
    PVOID               pBuffer,
    ULONG               ulBufferSize,
    ULONG               ulFlags
) const
{
    const CMdlPtr       pMdl = this;
    CPU_PHYSICAL        paCpuAddress;
    ULONG               ulTotalSize;
    ULONG               ulMdlSize;
    ULONG               ulPartialSize;
    float               fPercentage;
    char               *pLwrrentBuffer = reinterpret_cast<char*>(pBuffer);
    ULONG               ulWriteSize = 0;

    assert(pBuffer != NULL);

    // Make sure we have a valid offset and size (Doesn't exceed MDL)
    if ((ulOffset + ulBufferSize) > totalCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Requested offset/size exceeds MDL size (0x%0I64x >= 0x%0x)",
                         (ulOffset + ulBufferSize), totalCount());
    }
    // Search for the correct MDL containing the requested offset (in MDL chain)
    while (ulOffset > pMdl->ByteCount())
    {
        // Update the offset value and move to next MDL in chain
        ulOffset -= pMdl->ByteCount();
        pMdl      = pMdl->nextMdl();
    }
    // Initialize total size to the buffer size
    ulTotalSize = ulBufferSize;

    // Set percentage to amount written
    fPercentage = static_cast<float>(ulTotalSize - ulBufferSize) / static_cast<float>(ulTotalSize);
    progressPercentage(fPercentage * 100.0f);

    // Loop until requested data is written
    while (ulBufferSize != 0)
    {
        // Compute the MDL size (In case chained MDL load)
        ulMdlSize = min(ulBufferSize, pMdl->ByteCount());

        // Check for initial partial page write required
        if (pMdl->ByteOffset() != 0)
        {
            // Compute the partial write size
            ulPartialSize = min((PAGE_SIZE - pMdl->ByteOffset()), ulMdlSize);

            // Call routine to get the current physical address
            paCpuAddress = pMdl->physicalAddress(ulOffset);

            // Try to write the partial initial page
            ulWriteSize = writeCpuPhysical(paCpuAddress, pLwrrentBuffer, ulPartialSize, ulFlags);
            if (ulWriteSize == ulPartialSize)
            {
                // Update the offset, sizes, and buffer pointer
                ulOffset       += ulWriteSize;
                ulMdlSize      -= ulWriteSize;
                ulBufferSize   -= ulWriteSize;
                pLwrrentBuffer += ulWriteSize;
            }
            else    // Unable to write partial initial page
            {
                throw CTargetException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unable to write 0x%0x (0x%0x) bytes at offset 0x%0x (0x%0I64x)",
                                       ulPartialSize, ulWriteSize, ulOffset, paCpuAddress.addr());
            }
            // Update percentage written
            fPercentage = static_cast<float>(ulTotalSize - ulBufferSize) / static_cast<float>(ulTotalSize);
            progressPercentage(fPercentage * 100.0f);
        }
        // Loop writing full MDL pages
        while (ulMdlSize >= PAGE_SIZE)
        {
            // Call routine to get the current physical address
            paCpuAddress = pMdl->physicalAddress(ulOffset);

            // Try to write the full page
            ulWriteSize = writeCpuPhysical(paCpuAddress, pLwrrentBuffer, PAGE_SIZE, ulFlags);
            if (ulWriteSize == PAGE_SIZE)
            {
                // Update the offset, sizes, and buffer pointer
                ulOffset       += ulWriteSize;
                ulMdlSize      -= ulWriteSize;
                ulBufferSize   -= ulWriteSize;
                pLwrrentBuffer += ulWriteSize;
            }
            else    // Unable to write partial initial page
            {
                throw CTargetException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unable to write 0x%0x (0x%0x) bytes at offset 0x%0x (0x%0I64x)",
                                       PAGE_SIZE, ulWriteSize, ulOffset, paCpuAddress.addr());
            }
            // Update percentage written
            fPercentage = static_cast<float>(ulTotalSize - ulBufferSize) / static_cast<float>(ulTotalSize);
            progressPercentage(fPercentage * 100.0f);
        }
        // Check for final partial page write required
        if (ulMdlSize != 0)
        {
            // Call routine to get the current physical address
            paCpuAddress = pMdl->physicalAddress(ulOffset);

            // Try to write the final partial page
            ulWriteSize = readCpuPhysical(paCpuAddress, pLwrrentBuffer, ulMdlSize, ulFlags);
            if (ulWriteSize == ulMdlSize)
            {
                // Update the offset, sizes, and buffer pointer
                ulOffset       += ulWriteSize;
                ulMdlSize      -= ulWriteSize;
                ulBufferSize   -= ulWriteSize;
                pLwrrentBuffer += ulWriteSize;
            }
            else    // Unable to write partial initial page
            {
                throw CTargetException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unable to write 0x%0x (0x%0x) bytes at offset 0x%0x (0x%0I64x)",
                                       ulMdlSize, ulWriteSize, ulOffset, paCpuAddress.addr());
            }
            // Update percentage written
            fPercentage = static_cast<float>(ulTotalSize - ulBufferSize) / static_cast<float>(ulTotalSize);
            progressPercentage(fPercentage * 100.0f);
        }
        // Move to the next MDL in the chain (if there is one) and reset the offset
        pMdl     = pMdl->nextMdl();
        ulOffset = 0;
    }
    return ulTotalSize;

} // writePhysical

//******************************************************************************

ULONG
CMdl::readVirtual
(
    ULONG64             ulOffset,
    PVOID               pBuffer,
    ULONG               ulBufferSize,
    bool                bUncached
) const
{
    const CMdlPtr       pMdl = this;
    CPU_VIRTUAL         vaCpuAddress;
    ULONG               ulTotalSize;
    ULONG               ulMdlSize;
    float               fPercentage;
    char               *pLwrrentBuffer = reinterpret_cast<char*>(pBuffer);
    ULONG               ulReadSize = 0;

    assert(pBuffer != NULL);

    // Make sure we have a valid offset and size (Doesn't exceed MDL)
    if ((ulOffset + ulBufferSize) > totalCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Requested offset/size exceeds MDL size (0x%0I64x >= 0x%0x)",
                         (ulOffset + ulBufferSize), totalCount());
    }
    // Search for the correct MDL containing the requested offset (in MDL chain)
    while (ulOffset > pMdl->ByteCount())
    {
        // Update the offset value and move to next MDL in chain
        ulOffset -= pMdl->ByteCount();
        pMdl      = pMdl->nextMdl();
    }
    // Initialize total size to the buffer size
    ulTotalSize = ulBufferSize;

    // Set percentage to amount read
    fPercentage = static_cast<float>(ulTotalSize - ulBufferSize) / static_cast<float>(ulTotalSize);
    progressPercentage(fPercentage * 100.0f);

    // Loop until requested data is read
    while (ulBufferSize != 0)
    {
        // Compute the MDL size (In case chained MDL load)
        ulMdlSize = min(ulBufferSize, pMdl->ByteCount());

        // Get the CPU virtual address for the given offset
        vaCpuAddress = pMdl->virtualAddress(ulOffset);
        if (vaCpuAddress != NULL)
        {
            // Try to read the requested MDL data
            ulReadSize = readCpuVirtual(vaCpuAddress, pLwrrentBuffer, ulMdlSize, bUncached);
            if (ulReadSize != ulMdlSize)
            {
                throw CTargetException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unable to read 0x%0x (0x%0x) bytes at offset 0x%0x (0x%0I64x)",
                                       ulMdlSize, ulReadSize, ulOffset, vaCpuAddress.addr());
            }
        }
        else    // No CPU virtual address
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             ": MDL does not have a system CPU virtual address");
        }
        // Update the offset, sizes, and buffer pointer
        ulOffset       += ulReadSize;
        ulMdlSize      -= ulReadSize;
        ulBufferSize   -= ulReadSize;
        pLwrrentBuffer += ulReadSize;

        // Update percentage read
        fPercentage = static_cast<float>(ulTotalSize - ulBufferSize) / static_cast<float>(ulTotalSize);
        progressPercentage(fPercentage * 100.0f);

        // Move to the next MDL in the chain (if there is one) and reset the offset
        pMdl     = pMdl->nextMdl();
        ulOffset = 0;
    }
    return ulTotalSize;

} // readVirtual

//******************************************************************************

ULONG
CMdl::writeVirtual
(
    ULONG64             ulOffset,
    PVOID               pBuffer,
    ULONG               ulBufferSize,
    bool                bUncached
) const
{
    const CMdlPtr       pMdl = this;
    CPU_VIRTUAL         vaCpuAddress;
    ULONG               ulTotalSize;
    ULONG               ulMdlSize;
    float               fPercentage;
    char               *pLwrrentBuffer = reinterpret_cast<char*>(pBuffer);
    ULONG               ulWriteSize = 0;

    assert(pBuffer != NULL);

    // Make sure we have a valid offset and size (Doesn't exceed MDL)
    if ((ulOffset + ulBufferSize) > totalCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Requested offset/size exceeds MDL size (0x%0I64x >= 0x%0x)",
                         (ulOffset + ulBufferSize), totalCount());
    }
    // Search for the correct MDL containing the requested offset (in MDL chain)
    while (ulOffset > pMdl->ByteCount())
    {
        // Update the offset value and move to next MDL in chain
        ulOffset -= pMdl->ByteCount();
        pMdl      = pMdl->nextMdl();
    }
    // Initialize total size to the buffer size
    ulTotalSize = ulBufferSize;

    // Set percentage to amount written
    fPercentage = static_cast<float>(ulTotalSize - ulBufferSize) / static_cast<float>(ulTotalSize);
    progressPercentage(fPercentage * 100.0f);

    // Loop until requested data is written
    while (ulBufferSize != 0)
    {
        // Compute the MDL size (In case chained MDL load)
        ulMdlSize = min(ulBufferSize, pMdl->ByteCount());

        // Get the CPU virtual address for the given offset
        vaCpuAddress = pMdl->virtualAddress(ulOffset);
        if (vaCpuAddress != NULL)
        {
            // Try to write the requested MDL data
            ulWriteSize = writeCpuVirtual(vaCpuAddress, pLwrrentBuffer, ulMdlSize, bUncached);
            if (ulWriteSize != ulMdlSize)
            {
                throw CTargetException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unable to write 0x%0x (0x%0x) bytes at offset 0x%0x (0x%0I64x)",
                                       ulMdlSize, ulWriteSize, ulOffset, vaCpuAddress.addr());
            }
        }
        else    // No CPU virtual address
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             ": MDL does not have a system CPU virtual address");
        }
        // Update the offset, sizes, and buffer pointer
        ulOffset       += ulWriteSize;
        ulMdlSize      -= ulWriteSize;
        ulBufferSize   -= ulWriteSize;
        pLwrrentBuffer += ulWriteSize;

        // Update percentage written
        fPercentage = static_cast<float>(ulTotalSize - ulBufferSize) / static_cast<float>(ulTotalSize);
        progressPercentage(fPercentage * 100.0f);

        // Move to the next MDL in the chain (if there is one) and reset the offset
        pMdl     = pMdl->nextMdl();
        ulOffset = 0;
    }
    return ulTotalSize;

} // writeVirtual

//******************************************************************************

ULONG
CMdl::readProcess
(
    ULONG64             ulOffset,
    PVOID               pBuffer,
    ULONG               ulBufferSize,
    bool                bUncached
) const
{
    const CMdlPtr       pMdl = this;
    CPU_VIRTUAL         vaCpuAddress;
    ULONG               ulTotalSize;
    ULONG               ulMdlSize;
    float               fPercentage;
    char               *pLwrrentBuffer = reinterpret_cast<char*>(pBuffer);
    ULONG               ulReadSize = 0;

    assert(pBuffer != NULL);

    // Make sure we have a valid offset and size (Doesn't exceed MDL)
    if ((ulOffset + ulBufferSize) > totalCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Requested offset/size exceeds MDL size (0x%0I64x >= 0x%0x)",
                         (ulOffset + ulBufferSize), totalCount());
    }
    // Search for the correct MDL containing the requested offset (in MDL chain)
    while (ulOffset > pMdl->ByteCount())
    {
        // Update the offset value and move to next MDL in chain
        ulOffset -= pMdl->ByteCount();
        pMdl      = pMdl->nextMdl();
    }
    // Initialize total size to the buffer size
    ulTotalSize = ulBufferSize;

    // Set percentage to amount read
    fPercentage = static_cast<float>(ulTotalSize - ulBufferSize) / static_cast<float>(ulTotalSize);
    progressPercentage(fPercentage * 100.0f);

    // Loop until requested data is read
    while (ulBufferSize != 0)
    {
        // Compute the MDL size (In case chained MDL load)
        ulMdlSize = min(ulBufferSize, pMdl->ByteCount());

        // Get the CPU process address for the given offset
        vaCpuAddress = pMdl->processAddress(ulOffset);
        if (vaCpuAddress != NULL)
        {
            // Try to read the requested MDL data
            ulReadSize = readCpuVirtual(vaCpuAddress, pLwrrentBuffer, ulMdlSize, bUncached);
            if (ulReadSize != ulMdlSize)
            {
                throw CTargetException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unable to read 0x%0x (0x%0x) bytes at offset 0x%0x (0x%0I64x)",
                                       ulMdlSize, ulReadSize, ulOffset, vaCpuAddress.addr());
            }
        }
        else    // No CPU virtual address
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             ": MDL does not have a system CPU virtual address");
        }
        // Update the offset, sizes, and buffer pointer
        ulOffset       += ulReadSize;
        ulMdlSize      -= ulReadSize;
        ulBufferSize   -= ulReadSize;
        pLwrrentBuffer += ulReadSize;

        // Update percentage read
        fPercentage = static_cast<float>(ulTotalSize - ulBufferSize) / static_cast<float>(ulTotalSize);
        progressPercentage(fPercentage * 100.0f);

        // Move to the next MDL in the chain (if there is one) and reset the offset
        pMdl     = pMdl->nextMdl();
        ulOffset = 0;
    }
    return ulTotalSize;

} // readProcess

//******************************************************************************

ULONG
CMdl::writeProcess
(
    ULONG64             ulOffset,
    PVOID               pBuffer,
    ULONG               ulBufferSize,
    bool                bUncached
) const
{
    const CMdlPtr       pMdl = this;
    CPU_VIRTUAL         vaCpuAddress;
    ULONG               ulTotalSize;
    ULONG               ulMdlSize;
    float               fPercentage;
    char               *pLwrrentBuffer = reinterpret_cast<char*>(pBuffer);
    ULONG               ulWriteSize = 0;

    assert(pBuffer != NULL);

    // Make sure we have a valid offset and size (Doesn't exceed MDL)
    if ((ulOffset + ulBufferSize) > totalCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Requested offset/size exceeds MDL size (0x%0I64x >= 0x%0x)",
                         (ulOffset + ulBufferSize), totalCount());
    }
    // Search for the correct MDL containing the requested offset (in MDL chain)
    while (ulOffset > pMdl->ByteCount())
    {
        // Update the offset value and move to next MDL in chain
        ulOffset -= pMdl->ByteCount();
        pMdl      = pMdl->nextMdl();
    }
    // Initialize total size to the buffer size
    ulTotalSize = ulBufferSize;

    // Set percentage to amount written
    fPercentage = static_cast<float>(ulTotalSize - ulBufferSize) / static_cast<float>(ulTotalSize);
    progressPercentage(fPercentage * 100.0f);

    // Loop until requested data is written
    while (ulBufferSize != 0)
    {
        // Compute the MDL size (In case chained MDL load)
        ulMdlSize = min(ulBufferSize, pMdl->ByteCount());

        // Get the CPU process address for the given offset
        vaCpuAddress = pMdl->processAddress(ulOffset);
        if (vaCpuAddress != NULL)
        {
            // Try to write the requested MDL data
            ulWriteSize = writeCpuVirtual(vaCpuAddress, pLwrrentBuffer, ulMdlSize, bUncached);
            if (ulWriteSize != ulMdlSize)
            {
                throw CTargetException(E_FAIL, __FILE__, __FUNCTION__, __LINE__,
                                       ": Unable to write 0x%0x (0x%0x) bytes at offset 0x%0x (0x%0I64x)",
                                       ulMdlSize, ulWriteSize, ulOffset, vaCpuAddress.addr());
            }
        }
        else    // No CPU virtual address
        {
            throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                             ": MDL does not have a system CPU virtual address");
        }
        // Update the offset, sizes, and buffer pointer
        ulOffset       += ulWriteSize;
        ulMdlSize      -= ulWriteSize;
        ulBufferSize   -= ulWriteSize;
        pLwrrentBuffer += ulWriteSize;

        // Update percentage written
        fPercentage = static_cast<float>(ulTotalSize - ulBufferSize) / static_cast<float>(ulTotalSize);
        progressPercentage(fPercentage * 100.0f);

        // Move to the next MDL in the chain (if there is one) and reset the offset
        pMdl     = pMdl->nextMdl();
        ulOffset = 0;
    }
    return ulTotalSize;

} // writeProcess

//******************************************************************************

bool
CMdl::loadPhysical
(
    ULONG64             ulOffset,
    bool               *pPage,
    PVOID               pBuffer,
    ULONG               ulBufferSize,
    ULONG               ulFlags
) const
{
    CPU_PHYSICAL        paCpuAddress;
    ULONG64             ulLwrrentOffset;
    ULONG               ulLwrrentSize;
    ULONG               ulPartialSize;
    ULONG               ulReadSize;
    ULONG               ulLwrrentPage = 0;
    float               fPercentage;
    char               *pLwrrentBuffer;
    bool                bPage;
    bool                bValid = true;

    assert(pBuffer != NULL);
    assert(ulBufferSize != 0);

    // Make sure we have a valid offset and size (Doesn't exceed MDL)
    if ((ulOffset + ulBufferSize) >= totalCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Requested offset/size exceeds MDL size (0x%0I64x >= 0x%0x)",
                         (ulOffset + ulBufferSize), totalCount());
    }
    // Setup to handle the load in pages
    ulLwrrentOffset = ulOffset;
    ulLwrrentSize   = ulBufferSize;
    pLwrrentBuffer  = static_cast<char *>(pBuffer);

    // Set percentage to amount loaded
    fPercentage = static_cast<float>(ulBufferSize - ulLwrrentSize) / static_cast<float>(ulBufferSize);
    progressPercentage(fPercentage * 100.0f);

    // Check for initial partial page read required
    ulPartialSize = static_cast<ULONG>(PAGE_SIZE - (ulLwrrentOffset % PAGE_SIZE));
    if (ulPartialSize != PAGE_SIZE)
    {
        // Update the partial size if larger than requested size
        if (ulPartialSize > ulLwrrentSize)
        {
            ulPartialSize = ulLwrrentSize;
        }
        // Load initial partial page (Catch any memory errors)
        try
        {
            // Get the CPU physical address for this offset
            paCpuAddress = physicalAddress(ulLwrrentOffset);

            // Try to read the initial partial page (Physical)
            ulReadSize = readCpuPhysical(paCpuAddress, pLwrrentBuffer, ulPartialSize, ulFlags);
            if (ulReadSize == ulPartialSize)
            {
                // Indicate current page is valid
                bPage = true;
            }
            else    // Unable to read initial partial page
            {
                // Indicate current page is invalid
                bPage = false;
            }
        }
        catch (CTargetException& exception)
        {
            UNREFERENCED_PARAMETER(exception);

            // Indicate current page is invalid
            bPage = false;
        }
        // Update page array (if present)
        if (pPage != NULL)
        {
            pPage[ulLwrrentPage++] = bPage;
        }
        // Update global valid result
        bValid &= bPage;

        // Update the current offset, size, and buffer pointer
        ulLwrrentOffset += ulPartialSize;
        ulLwrrentSize   -= ulPartialSize;
        pLwrrentBuffer  += ulPartialSize;

        // Update percentage loaded
        fPercentage = static_cast<float>(ulBufferSize - ulLwrrentSize) / static_cast<float>(ulBufferSize);
        progressPercentage(fPercentage * 100.0f);
    }
    // Loop reading full pages
    while (ulLwrrentSize >= PAGE_SIZE)
    {
        // Load next full page (Catch any memory errors)
        try
        {
            // Get the CPU physical address for this offset
            paCpuAddress = physicalAddress(ulLwrrentOffset);

            // Try to read the next full page (Physical)
            ulReadSize = readCpuPhysical(paCpuAddress, pLwrrentBuffer, PAGE_SIZE, ulFlags);
            if (ulReadSize == PAGE_SIZE)
            {
                // Indicate current page is valid
                bPage = true;
            }
            else    // Unable to read next full page
            {
                // Indicate current page is invalid
                bPage = false;
            }
        }
        catch (CTargetException& exception)
        {
            UNREFERENCED_PARAMETER(exception);

            // Indicate current page is invalid
            bPage = false;
        }
        // Update page array (if present)
        if (pPage != NULL)
        {
            pPage[ulLwrrentPage++] = bPage;
        }
        // Update global valid result
        bValid &= bPage;

        // Update the current offset, size, and buffer pointer
        ulLwrrentOffset += PAGE_SIZE;
        ulLwrrentSize   -= PAGE_SIZE;
        pLwrrentBuffer  += PAGE_SIZE;

        // Update percentage loaded
        fPercentage = static_cast<float>(ulBufferSize - ulLwrrentSize) / static_cast<float>(ulBufferSize);
        progressPercentage(fPercentage * 100.0f);
    }
    // Check for final partial page read required
    if (ulLwrrentSize != 0)
    {
        // Load final partial page (Catch any memory errors)
        try
        {
            // Get the CPU physical address for this offset
            paCpuAddress = physicalAddress(ulLwrrentOffset);

            // Try to read the final partial page (Physical)
            ulReadSize = readCpuPhysical(paCpuAddress, pLwrrentBuffer, ulLwrrentSize, ulFlags);
            if (ulReadSize == ulLwrrentSize)
            {
                // Indicate current page is valid
                bPage = true;
            }
            else    // Unable to read final partial page
            {
                // Indicate current page is invalid
                bPage = false;
            }
        }
        catch (CTargetException& exception)
        {
            UNREFERENCED_PARAMETER(exception);

            // Indicate current page is invalid
            bPage = false;
        }
        // Update page array (if present)
        if (pPage != NULL)
        {
            pPage[ulLwrrentPage++] = bPage;
        }
        // Update global valid result
        bValid &= bPage;

        // Update the current offset, size, and buffer pointer
        ulLwrrentOffset += ulLwrrentSize;
        ulLwrrentSize   -= ulLwrrentSize;
        pLwrrentBuffer  += ulLwrrentSize;

        // Update percentage loaded
        fPercentage = static_cast<float>(ulBufferSize - ulLwrrentSize) / static_cast<float>(ulBufferSize);
        progressPercentage(fPercentage * 100.0f);
    }
    return bValid;

} // loadPhysical

//******************************************************************************

bool
CMdl::loadVirtual
(
    ULONG64             ulOffset,
    bool               *pPage,
    PVOID               pBuffer,
    ULONG               ulBufferSize,
    bool                bUncached
) const
{
    CPU_VIRTUAL         vaCpuAddress;
    ULONG64             ulLwrrentOffset;
    ULONG               ulLwrrentSize;
    ULONG               ulPartialSize;
    ULONG               ulReadSize;
    ULONG               ulLwrrentPage = 0;
    float               fPercentage;
    char               *pLwrrentBuffer;
    bool                bPage;
    bool                bValid = true;

    assert(pBuffer != NULL);
    assert(ulBufferSize != 0);

    // Make sure we have a valid offset and size (Doesn't exceed MDL)
    if ((ulOffset + ulBufferSize) >= totalCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Requested offset/size exceeds MDL size (0x%0I64x >= 0x%0x)",
                         (ulOffset + ulBufferSize), totalCount());
    }
    // Setup to handle the load in pages
    ulLwrrentOffset = ulOffset;
    ulLwrrentSize   = ulBufferSize;
    pLwrrentBuffer  = static_cast<char *>(pBuffer);

    // Set percentage to amount loaded
    fPercentage = static_cast<float>(ulBufferSize - ulLwrrentSize) / static_cast<float>(ulBufferSize);
    progressPercentage(fPercentage * 100.0f);

    // Check for initial partial page read required
    ulPartialSize = static_cast<ULONG>(PAGE_SIZE - (ulLwrrentOffset % PAGE_SIZE));
    if (ulPartialSize != PAGE_SIZE)
    {
        // Update the partial size if larger than requested size
        if (ulPartialSize > ulLwrrentSize)
        {
            ulPartialSize = ulLwrrentSize;
        }
        // Load initial partial page (Catch any memory errors)
        try
        {
            // Get the CPU virtual address for this offset
            vaCpuAddress = virtualAddress(ulLwrrentOffset);

            // Try to read the initial partial page (Virtual)
            ulReadSize = readCpuVirtual(vaCpuAddress, pLwrrentBuffer, ulPartialSize, bUncached);
            if (ulReadSize == ulPartialSize)
            {
                // Indicate current page is valid
                bPage = true;
            }
            else    // Unable to read initial partial page
            {
                // Indicate current page is invalid
                bPage = false;
            }
        }
        catch (CTargetException& exception)
        {
            UNREFERENCED_PARAMETER(exception);

            // Indicate current page is invalid
            bPage = false;
        }
        // Update page array (if present)
        if (pPage != NULL)
        {
            pPage[ulLwrrentPage++] = bPage;
        }
        // Update global valid result
        bValid &= bPage;

        // Update the current offset, size, and buffer pointer
        ulLwrrentOffset += ulPartialSize;
        ulLwrrentSize   -= ulPartialSize;
        pLwrrentBuffer  += ulPartialSize;

        // Update percentage loaded
        fPercentage = static_cast<float>(ulBufferSize - ulLwrrentSize) / static_cast<float>(ulBufferSize);
        progressPercentage(fPercentage * 100.0f);
    }
    // Loop reading full pages
    while (ulLwrrentSize >= PAGE_SIZE)
    {
        // Load next full page (Catch any memory errors)
        try
        {
            // Get the CPU virtual address for this offset
            vaCpuAddress = virtualAddress(ulLwrrentOffset);

            // Try to read the next full page (Virtual)
            ulReadSize = readCpuVirtual(vaCpuAddress, pLwrrentBuffer, PAGE_SIZE, bUncached);
            if (ulReadSize == PAGE_SIZE)
            {
                // Indicate current page is valid
                bPage = true;
            }
            else    // Unable to read next full page
            {
                // Indicate current page is invalid
                bPage = false;
            }
        }
        catch (CTargetException& exception)
        {
            UNREFERENCED_PARAMETER(exception);

            // Indicate current page is invalid
            bPage = false;
        }
        // Update page array (if present)
        if (pPage != NULL)
        {
            pPage[ulLwrrentPage++] = bPage;
        }
        // Update global valid result
        bValid &= bPage;

        // Update the current offset, size, and buffer pointer
        ulLwrrentOffset += PAGE_SIZE;
        ulLwrrentSize   -= PAGE_SIZE;
        pLwrrentBuffer  += PAGE_SIZE;

        // Update percentage loaded
        fPercentage = static_cast<float>(ulBufferSize - ulLwrrentSize) / static_cast<float>(ulBufferSize);
        progressPercentage(fPercentage * 100.0f);
    }
    // Check for final partial page read required
    if (ulLwrrentSize != 0)
    {
        // Load final partial page (Catch any memory errors)
        try
        {
            // Get the CPU virtual address for this offset
            vaCpuAddress = virtualAddress(ulLwrrentOffset);

            // Try to read the final partial page (Virtual)
            ulReadSize = readCpuVirtual(vaCpuAddress, pLwrrentBuffer, ulLwrrentSize, bUncached);
            if (ulReadSize == ulLwrrentSize)
            {
                // Indicate current page is valid
                bPage = true;
            }
            else    // Unable to read final partial page
            {
                // Indicate current page is invalid
                bPage = false;
            }
        }
        catch (CTargetException& exception)
        {
            UNREFERENCED_PARAMETER(exception);

            // Indicate current page is invalid
            bPage = false;
        }
        // Update page array (if present)
        if (pPage != NULL)
        {
            pPage[ulLwrrentPage++] = bPage;
        }
        // Update global valid result
        bValid &= bPage;

        // Update the current offset, size, and buffer pointer
        ulLwrrentOffset += ulLwrrentSize;
        ulLwrrentSize   -= ulLwrrentSize;
        pLwrrentBuffer  += ulLwrrentSize;

        // Update percentage loaded
        fPercentage = static_cast<float>(ulBufferSize - ulLwrrentSize) / static_cast<float>(ulBufferSize);
        progressPercentage(fPercentage * 100.0f);
    }
    return bValid;

} // loadVirtual

//******************************************************************************

bool
CMdl::loadProcess
(
    ULONG64             ulOffset,
    bool               *pPage,
    PVOID               pBuffer,
    ULONG               ulBufferSize,
    bool                bUncached
) const
{
    CPU_VIRTUAL         vaCpuAddress;
    ULONG64             ulLwrrentOffset;
    ULONG               ulLwrrentSize;
    ULONG               ulPartialSize;
    ULONG               ulReadSize;
    ULONG               ulLwrrentPage = 0;
    float               fPercentage;
    char               *pLwrrentBuffer;
    bool                bPage;
    bool                bValid = true;

    assert(pBuffer != NULL);
    assert(ulBufferSize !=  0);

    // Make sure we have a valid offset and size (Doesn't exceed MDL)
    if ((ulOffset + ulBufferSize) >= totalCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Requested offset/size exceeds MDL size (0x%0I64x >= 0x%0x)",
                         (ulOffset + ulBufferSize), totalCount());
    }
    // Setup to handle the load in pages
    ulLwrrentOffset = ulOffset;
    ulLwrrentSize   = ulBufferSize;
    pLwrrentBuffer  = static_cast<char *>(pBuffer);

    // Set percentage to amount loaded
    fPercentage = static_cast<float>(ulBufferSize - ulLwrrentSize) / static_cast<float>(ulBufferSize);
    progressPercentage(fPercentage * 100.0f);

    // Check for initial partial page read required
    ulPartialSize = static_cast<ULONG>(PAGE_SIZE - (ulLwrrentOffset % PAGE_SIZE));
    if (ulPartialSize != PAGE_SIZE)
    {
        // Update the partial size if larger than requested size
        if (ulPartialSize > ulLwrrentSize)
        {
            ulPartialSize = ulLwrrentSize;
        }
        // Load initial partial page (Catch any memory errors)
        try
        {
            // Get the CPU virtual process address for this offset
            vaCpuAddress = processAddress(ulLwrrentOffset);

            // Try to read the initial partial page (Virtual)
            ulReadSize = readCpuVirtual(vaCpuAddress, pLwrrentBuffer, ulPartialSize, bUncached);
            if (ulReadSize == ulPartialSize)
            {
                // Indicate current page is valid
                bPage = true;
            }
            else    // Unable to read initial partial page
            {
                // Indicate current page is invalid
                bPage = false;
            }
        }
        catch (CTargetException& exception)
        {
            UNREFERENCED_PARAMETER(exception);

            // Indicate current page is invalid
            bPage = false;
        }
        // Update page array (if present)
        if (pPage != NULL)
        {
            pPage[ulLwrrentPage++] = bPage;
        }
        // Update global valid result
        bValid &= bPage;

        // Update the current offset, size, and buffer pointer
        ulLwrrentOffset += ulPartialSize;
        ulLwrrentSize   -= ulPartialSize;
        pLwrrentBuffer  += ulPartialSize;

        // Update percentage loaded
        fPercentage = static_cast<float>(ulBufferSize - ulLwrrentSize) / static_cast<float>(ulBufferSize);
        progressPercentage(fPercentage * 100.0f);
    }
    // Loop reading full pages
    while (ulLwrrentSize >= PAGE_SIZE)
    {
        // Load next full page (Catch any memory errors)
        try
        {
            // Get the CPU virtual process address for this offset
            vaCpuAddress = processAddress(ulLwrrentOffset);

            // Try to read the next full page (Virtual)
            ulReadSize = readCpuVirtual(vaCpuAddress, pLwrrentBuffer, PAGE_SIZE, bUncached);
            if (ulReadSize == PAGE_SIZE)
            {
                // Indicate current page is valid
                bPage = true;
            }
            else    // Unable to read next full page
            {
                // Indicate current page is invalid
                bPage = false;
            }
        }
        catch (CTargetException& exception)
        {
            UNREFERENCED_PARAMETER(exception);

            // Indicate current page is invalid
            bPage = false;
        }
        // Update page array (if present)
        if (pPage != NULL)
        {
            pPage[ulLwrrentPage++] = bPage;
        }
        // Update global valid result
        bValid &= bPage;

        // Update the current offset, size, and buffer pointer
        ulLwrrentOffset += PAGE_SIZE;
        ulLwrrentSize   -= PAGE_SIZE;
        pLwrrentBuffer  += PAGE_SIZE;

        // Update percentage loaded
        fPercentage = static_cast<float>(ulBufferSize - ulLwrrentSize) / static_cast<float>(ulBufferSize);
        progressPercentage(fPercentage * 100.0f);
    }
    // Check for final partial page read required
    if (ulLwrrentSize != 0)
    {
        // Load final partial page (Catch any memory errors)
        try
        {
            // Get the CPU virtual process address for this offset
            vaCpuAddress = processAddress(ulLwrrentOffset);

            // Try to read the final partial page (Virtual)
            ulReadSize = readCpuVirtual(vaCpuAddress, pLwrrentBuffer, ulLwrrentSize, bUncached);
            if (ulReadSize == ulLwrrentSize)
            {
                // Indicate current page is valid
                bPage = true;
            }
            else    // Unable to read final partial page
            {
                // Indicate current page is invalid
                bPage = false;
            }
        }
        catch (CTargetException& exception)
        {
            UNREFERENCED_PARAMETER(exception);

            // Indicate current page is invalid
            bPage = false;
        }
        // Update page array (if present)
        if (pPage != NULL)
        {
            pPage[ulLwrrentPage++] = bPage;
        }
        // Update global valid result
        bValid &= bPage;

        // Update the current offset, size, and buffer pointer
        ulLwrrentOffset += ulLwrrentSize;
        ulLwrrentSize   -= ulLwrrentSize;
        pLwrrentBuffer  += ulLwrrentSize;

        // Update percentage loaded
        fPercentage = static_cast<float>(ulBufferSize - ulLwrrentSize) / static_cast<float>(ulBufferSize);
        progressPercentage(fPercentage * 100.0f);
    }
    return bValid;

} // loadProcess

//******************************************************************************

CPU_PHYSICAL
CMdl::physicalAddress
(
    ULONG64             ulOffset
) const
{
    const CMdlPtr       pMdl = this;
    ULONG64             ulPage;
    ULONG64             ulPfn;
    CPU_PHYSICAL        paAddress = 0;

    // Check for a valid MDL offset
    if (ulOffset >= totalCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Offset exceeds MDL size (0x%0I64x >= 0x%0x)",
                         ulOffset, totalCount());
    }
    // Search for the correct MDL containing the requested offset (in MDL chain)
    while (ulOffset > pMdl->ByteCount())
    {
        // Update the offset value and move to next MDL in chain
        ulOffset -= pMdl->ByteCount();
        pMdl      = pMdl->nextMdl();
    }
    // Compute the page and page offset values
    ulPage   = (ulOffset + pMdl->ByteOffset()) / PAGE_SIZE;
    ulOffset = (ulOffset + pMdl->ByteOffset()) % PAGE_SIZE;

    // Get the CPU PFN for the requested page
    ulPfn = page(static_cast<ULONG>(ulPage));

    // Compute the physical address for this offset
    paAddress = (ulPfn * PAGE_SIZE) + ulOffset;

    return paAddress;

} // physicalAddress

//******************************************************************************

CPU_VIRTUAL
CMdl::virtualAddress
(
    ULONG64             ulOffset
) const
{
    const CMdlPtr       pMdl = this;
    CPU_VIRTUAL         vaMappedSystemVa;
    CPU_VIRTUAL         vaAddress = 0;

    // Check for a valid MDL offset
    if (ulOffset >= totalCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Offset exceeds MDL size (0x%0I64x >= 0x%0x)",
                         ulOffset, totalCount());
    }
    // Search for the correct MDL containing the requested offset (in MDL chain)
    while (ulOffset > pMdl->ByteCount())
    {
        // Update the offset value and move to next MDL in chain
        ulOffset -= pMdl->ByteCount();
        pMdl      = pMdl->nextMdl();
    }
    // Check for a system mapped virtual address available
    vaMappedSystemVa = pMdl->MappedSystemVa();
    if (vaMappedSystemVa != 0)
    {
        // Compute the system virtual address for this offset
        vaAddress = vaMappedSystemVa + ulOffset;
    }
    return vaAddress;

} // virtualAddress

//******************************************************************************

CPU_VIRTUAL
CMdl::processAddress
(
    ULONG64             ulOffset
) const
{
    const CMdlPtr       pMdl = this;
    CPU_VIRTUAL         vaStartVa;
    CPU_VIRTUAL         vaAddress = 0;

    // Check for a valid MDL offset
    if (ulOffset >= totalCount())
    {
        throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                         ": Offset exceeds MDL size (0x%0I64x >= 0x%0x)",
                         ulOffset, totalCount());
    }
    // Search for the correct MDL containing the requested offset (in MDL chain)
    while (ulOffset > pMdl->ByteCount())
    {
        // Update the offset value and move to next MDL in chain
        ulOffset -= pMdl->ByteCount();
        pMdl      = pMdl->nextMdl();
    }
    // Check for a process virtual address available
    vaStartVa = pMdl->StartVa();
    if (vaStartVa != 0)
    {
        // Compute the process virtual address for this offset
        vaAddress = vaStartVa + ulOffset;
    }
    return vaAddress;

} // processAddress

//******************************************************************************

HRESULT
CMdl::allocateContents
(
    bool                bTotal
) const
{
    ULONG               ulSize;
    HRESULT             hResult = S_OK;

    // Check for simple MDL contents or total contents
    if (bTotal)
    {
        // Check for contents not yet allocated
        if (m_aTotalContents == NULL)
        {
            // Check to make sure this MDL has a size value (Force to page alignment)
            ulSize = static_cast<ULONG>(alignceil(totalCount(), PAGE_SIZE));
            if (ulSize != 0)
            {
                // Try to allocate memory to hold the total MDL contents
                m_aTotalContents = new BYTE[ulSize];
            }
            else    // MDL has no valid size value
            {
                throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                 ": MDL has no valid total size!");
            }
        }
    }
    else    // Simple MDL contents
    {
        // Check for contents not yet allocated
        if (m_aContents == NULL)
        {
            // Check to make sure this MDL has a size value (Force to page alignment)
            ulSize = alignceil(ByteCount(), PAGE_SIZE);
            if (ulSize != 0)
            {
                // Try to allocate memory to hold the MDL contents
                m_aContents = new BYTE[ulSize];
            }
            else    // MDL has no valid size value
            {
                throw CException(E_ILWALIDARG, __FILE__, __FUNCTION__, __LINE__,
                                 ": MDL has no valid size!");
            }
        }
    }
    return hResult;

} // allocateContents

//******************************************************************************

HRESULT
CMdl::loadContents
(
    bool                bVirtual,
    bool                bTotal
) const
{
    CPU_VIRTUAL         vaCpuAddress;
    CPU_PHYSICAL        paCpuAddress;
    const CMdlPtr       pMdl = this;
    ULONG64             ulOffset = 0;
    ULONG               ulSize;
    ULONG               ulTotal;
    ULONG               ulPartial;
    ULONG               ulRead;
    float               fPercentage = 0.0;
    char               *pBuffer;
    HRESULT             hResult = S_OK;

    // Setup correct size and buffer pointer based on the request
    if (bTotal)
    {
        // Setup the total MDL size and buffer pointer
        ulSize  = static_cast<ULONG>(totalCount());
        pBuffer = reinterpret_cast<char*>(m_aTotalContents.ptr());
    }
    else    // Local MDL only
    {
        // Setup the local MDL size and buffer pointer
        ulSize  = ByteCount();
        pBuffer = reinterpret_cast<char*>(m_aContents.ptr());
    }
    // Initialize total size to the requested size
    ulTotal = ulSize;

    // Set percentage to amount loaded
    if (ulTotal != 0)
    {
        fPercentage = static_cast<float>(ulTotal - ulSize) / ulTotal;
    }
    progressPercentage(fPercentage * 100.0f);

    // Loop until entire contents is loaded
    while (ulSize != 0)
    {
        // Compute the partial size (In case chained MDL load)
        ulPartial = min(ulSize, pMdl->ByteCount());

        // Check for initial partial page read required
        if (pMdl->ByteOffset() != 0)
        {
            // Compute the partial read size
            ulRead = min((PAGE_SIZE - pMdl->ByteOffset()), ulPartial);

            // Load next partial page (Catch any memory errors)
            try
            {
                // Check for a virtual or physical request
                if (bVirtual)
                {
                    // Callwlate the virtual address for this offset
                    vaCpuAddress = pMdl->virtualAddress(ulOffset);

                    // Try to read the next full page (Virtual)
                    readCpuVirtual(vaCpuAddress, pBuffer, ulRead);
                }
                else    // Physical request
                {
                    // Callwlate the physical address for this offset
                    paCpuAddress = pMdl->physicalAddress(ulOffset);

                    // Try to read the next full page (Physical)
                    readCpuPhysical(paCpuAddress, pBuffer, ulRead);
                }
            }
            catch (CTargetException& exception)
            {
                UNREFERENCED_PARAMETER(exception);
            }
            // Update the partial, offset, size, and buffer pointer
            ulPartial -= ulRead;
            ulOffset  += ulRead;
            ulSize    -= ulRead;
            pBuffer   += ulRead;

            // Update percentage loaded
            fPercentage = static_cast<float>(ulTotal - ulSize) / static_cast<float>(ulTotal);
            progressPercentage(fPercentage * 100.0f);
        }
        // Loop reading full pages
        while (ulPartial >= PAGE_SIZE)
        {
            // Load next full page (Catch any memory errors)
            try
            {
                // Check for a virtual or physical request
                if (bVirtual)
                {
                    // Callwlate the virtual address for this offset
                    vaCpuAddress = pMdl->virtualAddress(ulOffset);

                    // Try to read the next full page (Virtual)
                    readCpuVirtual(vaCpuAddress, pBuffer, PAGE_SIZE);
                }
                else    // Physical request
                {
                    // Callwlate the physical address for this offset
                    paCpuAddress = pMdl->physicalAddress(ulOffset);

                    // Try to read the next full page (Physical)
                    readCpuPhysical(paCpuAddress, pBuffer, PAGE_SIZE);
                }
            }
            catch (CTargetException& exception)
            {
                UNREFERENCED_PARAMETER(exception);
            }
            // Update the partial, offset, size, and buffer pointer
            ulPartial -= PAGE_SIZE;
            ulOffset  += PAGE_SIZE;
            ulSize    -= PAGE_SIZE;
            pBuffer   += PAGE_SIZE;

            // Update percentage loaded
            fPercentage = static_cast<float>(ulTotal - ulSize) / static_cast<float>(ulTotal);
            progressPercentage(fPercentage * 100.0f);
        }
        // Check for final partial page read required
        if (ulPartial != 0)
        {
            // Load final partial page (Catch any memory errors)
            try
            {
                // Check for a virtual or physical request
                if (bVirtual)
                {
                    // Callwlate the virtual address for this offset
                    vaCpuAddress = pMdl->virtualAddress(ulOffset);

                    // Try to read the next full page (Virtual)
                    readCpuVirtual(vaCpuAddress, pBuffer, ulPartial);
                }
                else    // Physical request
                {
                    // Callwlate the physical address for this offset
                    paCpuAddress = pMdl->physicalAddress(ulOffset);

                    // Try to read the next full page (Physical)
                    readCpuPhysical(paCpuAddress, pBuffer, ulPartial);
                }
            }
            catch (CTargetException& exception)
            {
                UNREFERENCED_PARAMETER(exception);
            }
            // Update the partial, offset, size, and buffer pointer
            ulPartial -= ulPartial;
            ulOffset  += ulPartial;
            ulSize    -= ulPartial;
            pBuffer   += ulPartial;

            // Update percentage loaded
            fPercentage = static_cast<float>(ulTotal - ulSize) / static_cast<float>(ulTotal);
            progressPercentage(fPercentage * 100.0f);
        }
        // Move to the next MDL in the chain (if there is one) and reset the offset
        pMdl     = pMdl->nextMdl();
        ulOffset = 0;
    }
    return hResult;

} // loadContents

//******************************************************************************

bool
CMdl::hasContents
(
    bool                bVirtual,
    bool                bTotal
) const
{
    const CMdlPtr       pMdl = this;
    bool                bHasContents = true;

    // Do the contents check on all the required MDL's
    do
    {
        // Check for virtual or physical request
        if (bVirtual)
        {
            // Check to see if the MDL has a virtual address
            if (!pMdl->isMappedToSystemVa())
            {
                // No virtual address, indicate no virtual contents and stop loop
                bHasContents = false;

                break;
            }
        }
        else    // Physical request
        {
            // Check to see if the MDL has a physical address (Non-paged pages or page locked)
            if (!(pMdl->isSourceNonpagedPool() || pMdl->hasPagesLocked()))
            {
                // No physical pages, indicate no physical contents and stop loop
                bHasContents = false;

                break;
            }
        }
        // Move to the next MDL in case of chained and total request
        pMdl = pMdl->nextMdl();
    }
    while ((pMdl != NULL) && bTotal);

    return bHasContents;

} // hasContents

//******************************************************************************

void*
CMdl::getContents
(
    bool                bVirtual,
    bool                bTotal
) const
{
    HRESULT             hResult = S_OK;
    void               *pContents = NULL;

    // Get the correct contents to check (Local MDL or total)
    if (bTotal)
    {
        // Get the total contents array
        pContents = m_aTotalContents;
    }
    else    // Local MDL only
    {
        // Get the local MDL contents array
        pContents = m_aContents;
    }
    // Check for correct contents not yet created (Will create if not)
    if (pContents == NULL)
    {
        // Allocate the memory for the MDL contents
        hResult = allocateContents(bTotal);
        if (SUCCEEDED(hResult))
        {
            // Update the contents pointer (Should be allocated now)
            if (bTotal)
            {
                // Get the total contents array
                pContents = m_aTotalContents;
            }
            else    // Local MDL only
            {
                // Get the local MDL contents array
                pContents = m_aContents;
            }
            // Call routine to update the MDL contents
            hResult = updateContents(bVirtual, bTotal);
        }
    }
    return pContents;

} // getContents

//******************************************************************************

HRESULT
CMdl::updateContents
(
    bool                bVirtual,
    bool                bTotal
) const
{
    void               *pContents = NULL;
    HRESULT             hResult = S_OK;

    // Get the correct contents to check (Local MDL or total)
    if (bTotal)
    {
        // Get the total contents array
        pContents = m_aTotalContents;
    }
    else    // Local MDL only
    {
        // Get the local MDL contents array
        pContents = m_aContents;
    }
    // Check for correct contents not yet created (Will create if not)
    if (pContents == NULL)
    {
        // Allocate the memory for the MDL contents
        hResult = allocateContents(bTotal);
    }
    // If no errors then load the MDL contents
    if (SUCCEEDED(hResult))
    {
        // Call routine to load the MDL contents
        hResult = loadContents(bVirtual, bTotal);
    }
    return hResult;

} // updateContents

//******************************************************************************

void
CMdl::addFlag
(
    CString&            sFlagsString,
    const char         *pString
) const
{
    // Check for previous flags in string
    if (!sFlagsString.empty())
    {
        sFlagsString += " ";
    }
    // Add this flag string to the flags string
    sFlagsString += pString;

} // addFlag

} // os namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
