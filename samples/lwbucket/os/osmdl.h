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
|*  Module: osmdl.h                                                           *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _OSMDL_H
#define _OSMDL_H

//******************************************************************************
//
//  os namespace
//
//******************************************************************************
namespace os
{

//******************************************************************************
//
//  Constants
//
//******************************************************************************
// The following #defines were copied from WDM.h
#define MDL_MAPPED_TO_SYSTEM_VA     0x0001
#define MDL_PAGES_LOCKED            0x0002
#define MDL_SOURCE_IS_NONPAGED_POOL 0x0004
#define MDL_ALLOCATED_FIXED_SIZE    0x0008
#define MDL_PARTIAL                 0x0010
#define MDL_PARTIAL_HAS_BEEN_MAPPED 0x0020
#define MDL_IO_PAGE_READ            0x0040
#define MDL_WRITE_OPERATION         0x0080
#define MDL_PARENT_MAPPED_SYSTEM_VA 0x0100
#define MDL_FREE_EXTRA_PTES         0x0200
#define MDL_DESCRIBES_AWE           0x0400
#define MDL_IO_SPACE                0x0800
#define MDL_NETWORK_HEADER          0x1000
#define MDL_MAPPING_CAN_FAIL        0x2000
#define MDL_ALLOCATED_MUST_SUCCEED  0x4000
#define MDL_INTERNAL                0x8000

#define MDL_PHYSICAL_CONTENTS       false
#define MDL_VIRTUAL_CONTENTS        true

#define MDL_ONLY_CONTENTS           false
#define MDL_TOTAL_CONTENTS          true

//******************************************************************************
//
// class CMdl
//
//******************************************************************************
class CMdl : public CRefObj
{
// MDL Type Helpers
TYPE(Mdl)

// MDL Field Helpers
FIELD(Next)
FIELD(Size)
FIELD(MdlFlags)
FIELD(Process)
FIELD(MappedSystemVa)
FIELD(StartVa)
FIELD(ByteCount)
FIELD(ByteOffset)

// MDL Members
MEMBER(Next,            CPU_VIRTUAL,    NULL,   public)
MEMBER(Size,            WORD,           0,      public)
MEMBER(MdlFlags,        WORD,           0,      public)
MEMBER(Process,         POINTER,        NULL,   public)
MEMBER(MappedSystemVa,  CPU_VIRTUAL,    NULL,   public)
MEMBER(StartVa,         CPU_VIRTUAL,    NULL,   public)
MEMBER(ByteCount,       DWORD,          0,      public)
MEMBER(ByteOffset,      DWORD,          0,      public)

private:
        POINTER         m_ptrMdl;

mutable CMdlPtr         m_pNextMdl;

mutable CEProcessPtr    m_pProcess;
mutable DwordArray      m_aPages;

mutable ULONG64         m_ulTotalCount;
mutable ULONG           m_ulTotalPages;
mutable DwordArray      m_aAllPages;

mutable ByteArray       m_aContents;
mutable ByteArray       m_aTotalContents;

        HRESULT         allocateContents(bool bTotal = false) const;

        HRESULT         loadContents(bool bVirtual = false, bool bTotal = false) const;

        void            addFlag(CString& sFlagsString, const char* pString) const;

public:
                        CMdl(POINTER ptrMdl);
virtual                ~CMdl();

        POINTER         ptrMdl() const              { return m_ptrMdl; }

        bool            isChained() const           { return (Next() != NULL); }

        bool            isMappedToSystemVa() const  { return tobool(MdlFlags() & MDL_MAPPED_TO_SYSTEM_VA); }
        bool            hasPagesLocked() const      { return tobool(MdlFlags() & MDL_PAGES_LOCKED); }
        bool            isSourceNonpagedPool() const{ return tobool(MdlFlags() & MDL_SOURCE_IS_NONPAGED_POOL); }
        bool            isAllocatedFixedSize() const{ return tobool(MdlFlags() & MDL_ALLOCATED_FIXED_SIZE); }
        bool            isPartial() const           { return tobool(MdlFlags() & MDL_PARTIAL); }
        bool            hasPartialBeenMapped() const{ return tobool(MdlFlags() & MDL_PARTIAL_HAS_BEEN_MAPPED); }
        bool            isIoPageRead() const        { return tobool(MdlFlags() & MDL_IO_PAGE_READ); }
        bool            isWriteOperation() const    { return tobool(MdlFlags() & MDL_WRITE_OPERATION); }
        bool            isParentMappedSystemVa() const
                            { return tobool(MdlFlags() & MDL_PARENT_MAPPED_SYSTEM_VA); }
        bool            hasFreeExtraPtes() const    { return tobool(MdlFlags() & MDL_FREE_EXTRA_PTES); }
        bool            describesAwe() const        { return tobool(MdlFlags() & MDL_DESCRIBES_AWE); }
        bool            isIoSpace() const           { return tobool(MdlFlags() & MDL_IO_SPACE); }
        bool            isNetworkHeader() const     { return tobool(MdlFlags() & MDL_NETWORK_HEADER); }
        bool            mappingCanFail() const      { return tobool(MdlFlags() & MDL_MAPPING_CAN_FAIL); }
        bool            allocatedMustSucceed() const{ return tobool(MdlFlags() & MDL_ALLOCATED_MUST_SUCCEED); }
        bool            isInternal() const          { return tobool(MdlFlags() & MDL_INTERNAL); }

        ULONG           pageCount() const           { return (alignceil(ByteCount() + ByteOffset(), PAGE_SIZE) / PAGE_SIZE); }
        ULONG64         totalCount() const          { return m_ulTotalCount; }
        ULONG64         totalPages() const          { return m_ulTotalPages; }

        CMdlPtr         nextMdl() const;

        CEProcessPtr    process() const;

        DWORD*          pages() const;
        DWORD*          allPages() const;
        DWORD           page(ULONG64 ulPage) const;

        CString         flagsString() const;

        ULONG           readPhysical(ULONG64 ulOffset, PVOID pBuffer, ULONG ulBufferSize, ULONG ulFlags = DEBUG_PHYSICAL_CACHED) const;
        ULONG           writePhysical(ULONG64 ulOffset, PVOID pBuffer, ULONG ulBufferSize, ULONG ulFlags = DEBUG_PHYSICAL_CACHED) const;

        ULONG           readVirtual(ULONG64 ulOffset, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false) const;
        ULONG           writeVirtual(ULONG64 ulOffset, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false) const;

        ULONG           readProcess(ULONG64 ulOffset, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false) const;
        ULONG           writeProcess(ULONG64 ulOffset, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false) const;

        bool            loadPhysical(ULONG64 ulOffset, bool* pPage, PVOID pBuffer, ULONG ulBufferSize, ULONG ulFlags = DEBUG_PHYSICAL_CACHED) const;
        bool            loadVirtual(ULONG64 ulOffset, bool* pPage, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false) const;
        bool            loadProcess(ULONG64 ulOffset, bool* pPage, PVOID pBuffer, ULONG ulBufferSize, bool bUncached = false) const;

        CPU_PHYSICAL    physicalAddress(ULONG64 ulOffset = 0) const;
        CPU_VIRTUAL     virtualAddress(ULONG64 ulOffset = 0) const;
        CPU_VIRTUAL     processAddress(ULONG64 ulOffset = 0) const;

        bool            hasContents(bool bVirtual = false, bool bTotal = false) const;

        void*           getContents(bool bVirtual = false, bool bTotal = false) const;

        HRESULT         updateContents(bool bVirtual = false, bool bTotal = false) const;

const   CMemberType&    type() const                { return m_MdlType; }

}; // class CMdl

} // os namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _OSMDL_H
