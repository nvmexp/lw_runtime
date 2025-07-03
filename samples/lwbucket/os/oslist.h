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
|*  Module: oslist.h                                                          *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _OSLIST_H
#define _OSLIST_H

//******************************************************************************
//
//  os namespace
//
//******************************************************************************
namespace os
{

//******************************************************************************
//
// class CSingleListEntry
//
//******************************************************************************
class CSingleListEntry : public CRefObj
{
// Single List Entry Type Helpers
TYPE(singleListEntry)

// Single List Entry Field Helpers
FIELD(ptrNext)

// Single List Entry Members
MEMBER(ptrNext,         POINTER,    NULL,   public)

private:
const   CField*         m_pListField;
const   CField*         m_pEntryField;

        POINTER         m_ptrListEntry;
mutable POINTER         m_ptrEntry;

public:
                        CSingleListEntry(const CField* pListField, const CField* pEntryField, POINTER ptrListEntry);
                        CSingleListEntry(const CField* pEntryField, POINTER ptrListEntry);
                        CSingleListEntry(ULONG64 ptrListEntry);
                        CSingleListEntry(POINTER ptrListEntry);
                        CSingleListEntry(const CSingleListEntry& singleListEntry);
virtual                ~CSingleListEntry();

const   CField*         listField() const           { return m_pListField; }
const   CField*         entryField() const          { return m_pEntryField; }

        POINTER         ptrListEntry() const        { return m_ptrListEntry; }
        POINTER         ptrEntry() const            { return m_ptrEntry; }

        const char*     name() const                { return ((m_pListField != NULL) ? m_pListField->name() : NULL); }
        ULONG           size() const                { return ((m_pEntryField != NULL) ? m_pEntryField->type()->size() : 0); }
        ULONG           length() const              { return ((m_pEntryField != NULL) ? m_pEntryField->type()->length() : 0); }
        ULONG           offset() const              { return ((m_pEntryField != NULL) ? m_pEntryField->offset() : 0); }
        UINT            dimensions() const          { return ((m_pListField != NULL) ? m_pListField->dimensions() : 0); }
        UINT            dimension(UINT uDimension) const
                            { return ((m_pListField != NULL) ? m_pListField->dimension(uDimension) : 0); }

        bool            isPresent() const;

        POINTER         ptrHeadEntry() const;

        POINTER         ptrNextEntry(POINTER ptrEntry = static_cast<ULONG64>(0)) const;

const   CMemberType&    type() const                { return m_singleListEntryType; }

}; // class CSingleListEntry

//******************************************************************************
//
// class CListEntry
//
//******************************************************************************
class CListEntry : public CRefObj
{
// List Entry Type Helpers
TYPE(listEntry)

// List Entry Field Helpers
FIELD(ptrFlink)
FIELD(ptrBlink)

// List Entry Members
MEMBER(ptrFlink,        POINTER,    NULL,   public)
MEMBER(ptrBlink,        POINTER,    NULL,   public)

private:
const   CField*         m_pListField;
const   CField*         m_pEntryField;

        POINTER         m_ptrListEntry;
mutable POINTER         m_ptrEntry;

public:
                        CListEntry(const CField* pListField, const CField* pEntryField, POINTER ptrListEntry);
                        CListEntry(const CField* pEntryField, POINTER ptrListEntry);
                        CListEntry(ULONG64 ptrListEntry);
                        CListEntry(POINTER ptrListEntry);
                        CListEntry(const CListEntry& listEntry);
virtual                ~CListEntry();

const   CField*         listField() const           { return m_pListField; }
const   CField*         entryField() const          { return m_pEntryField; }

        POINTER         ptrListEntry() const        { return m_ptrListEntry; }
        POINTER         ptrEntry() const            { return m_ptrEntry; }

        const char*     name() const                { return ((m_pListField != NULL) ? m_pListField->name() : NULL); }
        ULONG           size() const                { return ((m_pEntryField != NULL) ? m_pEntryField->type()->size() : 0); }
        ULONG           length() const              { return ((m_pEntryField != NULL) ? m_pEntryField->type()->length() : 0); }
        ULONG           offset() const              { return ((m_pEntryField != NULL) ? m_pEntryField->offset() : 0); }
        UINT            dimensions() const          { return ((m_pListField != NULL) ? m_pListField->dimensions() : 0); }
        UINT            dimension(UINT uDimension) const
                            { return ((m_pListField != NULL) ? m_pListField->dimension(uDimension) : 0); }

        bool            isPresent() const;

        POINTER         ptrHeadEntry() const;
        POINTER         ptrTailEntry() const;

        POINTER         ptrPrevEntry(POINTER ptrEntry = static_cast<ULONG64>(0)) const;
        POINTER         ptrNextEntry(POINTER ptrEntry = static_cast<ULONG64>(0)) const;

const   CMemberType&    type() const                { return m_listEntryType; }

}; // class CListEntry

//******************************************************************************
//
// Functions
//
//******************************************************************************
const   CGlobal&        psActiveProcessHead();
const   CGlobal&        mmProcessList();

} // os namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _OSLIST_H
