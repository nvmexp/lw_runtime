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
|*  Module: oslist.cpp                                                        *|
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
// Single List Entry Type Helpers
CMemberType     CSingleListEntry::m_singleListEntryType     (&osKernel(), "SINGLE_LIST_ENTRY", "_SINGLE_LIST_ENTRY");

// Single List Entry Field Helpers
CMemberField    CSingleListEntry::m_ptrNextField            (&singleListEntryType(), false, NULL, "Next");

// List Entry Type Helpers
CMemberType     CListEntry::m_listEntryType                 (&osKernel(), "LIST_ENTRY", "_LIST_ENTRY");

// List Entry Field Helpers
CMemberField    CListEntry::m_ptrFlinkField                 (&listEntryType(), false, NULL, "Flink");
CMemberField    CListEntry::m_ptrBlinkField                 (&listEntryType(), false, NULL, "Blink");

// Active Process Head 
static CGlobal  s_PsActiveProcessHead                       (&CListEntry::listEntryType(), "PsActiveProcessHead");

// Process List
static CGlobal  s_MmProcessList                             (&CListEntry::listEntryType(), "MmProcessList");

//******************************************************************************

CSingleListEntry::CSingleListEntry
(
    const CField       *pListField,
    const CField       *pEntryField,
    POINTER             ptrListEntry
)
:   m_pListField(pListField),
    m_pEntryField(pEntryField),
    m_ptrListEntry(ptrListEntry),
    INIT(ptrNext),
    m_ptrEntry(NULL)
{
    assert(pListField != NULL);
    assert(pEntryField != NULL);

    // Get the single list entry information
    READ(ptrNext, ptrListEntry);

} // CSingleListEntry

//******************************************************************************

CSingleListEntry::CSingleListEntry
(
    const CField       *pEntryField,
    POINTER             ptrListEntry
)
:   m_pListField(NULL),
    m_pEntryField(pEntryField),
    m_ptrListEntry(ptrListEntry),
    INIT(ptrNext),
    m_ptrEntry(NULL)
{
    assert(pEntryField != NULL);

    // Get the single list entry information
    READ(ptrNext, ptrListEntry);

} // CSingleListEntry

//******************************************************************************

CSingleListEntry::CSingleListEntry
(
    ULONG64             ptrListEntry
)
:   m_pListField(NULL),
    m_pEntryField(NULL),
    m_ptrListEntry(ptrListEntry),
    INIT(ptrNext),
    m_ptrEntry(NULL)
{
    // Get the single list entry information
    READ(ptrNext, ptrListEntry);

} // CSingleListEntry

//******************************************************************************

CSingleListEntry::CSingleListEntry
(
    POINTER             ptrListEntry
)
:   m_pListField(NULL),
    m_pEntryField(NULL),
    m_ptrListEntry(ptrListEntry),
    INIT(ptrNext),
    m_ptrEntry(NULL)
{
    // Get the single list entry information
    READ(ptrNext, ptrListEntry);

} // CSingleListEntry

//******************************************************************************

CSingleListEntry::CSingleListEntry
(
    const CSingleListEntry& listEntry
)
:   m_pListField(listEntry.listField()),
    m_pEntryField(listEntry.entryField()),
    m_ptrListEntry(listEntry.ptrListEntry()),
    INIT(ptrNext),
    m_ptrEntry(NULL)
{
    // Get the single list entry information
    READ(ptrNext, ptrListEntry());

} // CSingleListEntry

//******************************************************************************

CSingleListEntry::~CSingleListEntry()
{

} // ~CSingleListEntry

//******************************************************************************

bool
CSingleListEntry::isPresent() const
{
    // Check for both list and entry fields provided
    if ((m_pListField != NULL) && (m_pEntryField != NULL))
    {
        // Both list and entry fields must be present
        return (m_pListField->isPresent() && m_pEntryField->isPresent());
    }
    else if (m_pEntryField != NULL)
    {
        // List is present if entry field present
        return m_pEntryField->isPresent();
    }
    else if (m_pListField != NULL)
    {
        // List is present if list field present (Shouldn't happen)
        return m_pListField->isPresent();
    }
    else    // No list or entry field provided
    {
        // Return list not present
        return false;
    }
    
} // isPresent

//******************************************************************************

POINTER
CSingleListEntry::ptrHeadEntry() const
{
    POINTER             ptrHeadEntry = 0;

    // Can only return list head if entry field given
    if (entryField() != NULL)
    {
        // Check for list entries present
        if (ptrNext() != ptrListEntry())
        {
            // Compute the actual head entry
            ptrHeadEntry = ptrNext() - entryField()->offset();

            // Save head entry as the last entry value
            m_ptrEntry = ptrHeadEntry;
        }
    }
    return ptrHeadEntry;

} // ptrHeadEntry

//******************************************************************************

POINTER
CSingleListEntry::ptrNextEntry
(
    POINTER             ptrEntry
) const
{
    POINTER             ptrNextEntry;

    // Can only return next entry if entry field given
    if (entryField() != NULL)
    {
        // Check for an entry value given (Use last if none)
        if (ptrEntry == NULL)
        {
            ptrEntry = m_ptrEntry;
        }
        // Check for current entry value (Use head if none)
        if (ptrEntry != NULL)
        {
            // Read the entry next link value
            ptrNextEntry = readPointer(ptrEntry + entryField()->offset() + ptrNextMember().offset());

            // Check for next entry present (Not at end of list)
            if (ptrNextEntry != ptrListEntry())
            {
                // Compute the actual next entry address
                ptrNextEntry -= entryField()->offset();
            }
            else    // No next entry
            {
                ptrNextEntry = NULL;
            }
        }
        else    // No current entry (Use head entry)
        {
            ptrNextEntry = ptrHeadEntry();
        }
        // Save next entry as the last entry value (May be none)
        m_ptrEntry = ptrNextEntry;
    }
    return ptrNextEntry;

} // ptrNextEntry

//******************************************************************************

CListEntry::CListEntry
(
    const CField       *pListField,
    const CField       *pEntryField,
    POINTER             ptrListEntry
)
:   m_pListField(pListField),
    m_pEntryField(pEntryField),
    m_ptrListEntry(ptrListEntry),
    INIT(ptrFlink),
    INIT(ptrBlink),
    m_ptrEntry(NULL)
{
    assert(pListField != NULL);
    assert(pEntryField != NULL);

    // Get the list entry information
    READ(ptrFlink, ptrListEntry);
    READ(ptrBlink, ptrListEntry);

} // CListEntry

//******************************************************************************

CListEntry::CListEntry
(
    const CField       *pEntryField,
    POINTER             ptrListEntry
)
:   m_pListField(NULL),
    m_pEntryField(pEntryField),
    m_ptrListEntry(ptrListEntry),
    INIT(ptrFlink),
    INIT(ptrBlink),
    m_ptrEntry(NULL)
{
    assert(pEntryField != NULL);

    // Get the list entry information
    READ(ptrFlink, ptrListEntry);
    READ(ptrBlink, ptrListEntry);

} // CListEntry

//******************************************************************************

CListEntry::CListEntry
(
    ULONG64             ptrListEntry
)
:   m_pListField(NULL),
    m_pEntryField(NULL),
    m_ptrListEntry(ptrListEntry),
    INIT(ptrFlink),
    INIT(ptrBlink),
    m_ptrEntry(NULL)
{
    // Get the list entry information
    READ(ptrFlink, ptrListEntry);
    READ(ptrBlink, ptrListEntry);

} // CListEntry

//******************************************************************************

CListEntry::CListEntry
(
    POINTER             ptrListEntry
)
:   m_pListField(NULL),
    m_pEntryField(NULL),
    m_ptrListEntry(ptrListEntry),
    INIT(ptrFlink),
    INIT(ptrBlink),
    m_ptrEntry(NULL)
{
    // Get the list entry information
    READ(ptrFlink, ptrListEntry);
    READ(ptrBlink, ptrListEntry);

} // CListEntry

//******************************************************************************

CListEntry::CListEntry
(
    const CListEntry&   listEntry
)
:   m_pListField(listEntry.listField()),
    m_pEntryField(listEntry.entryField()),
    m_ptrListEntry(listEntry.ptrListEntry()),
    INIT(ptrFlink),
    INIT(ptrBlink),
    m_ptrEntry(NULL)
{
    // Get the list entry information
    READ(ptrFlink, ptrListEntry());
    READ(ptrBlink, ptrListEntry());

} // CListEntry

//******************************************************************************

CListEntry::~CListEntry()
{

} // ~CListEntry

//******************************************************************************

bool
CListEntry::isPresent() const
{
    // Check for both list and entry fields provided
    if ((m_pListField != NULL) && (m_pEntryField != NULL))
    {
        // Both list and entry fields must be present
        return (m_pListField->isPresent() && m_pEntryField->isPresent());
    }
    else if (m_pEntryField != NULL)
    {
        // List is present if entry field present
        return m_pEntryField->isPresent();
    }
    else if (m_pListField != NULL)
    {
        // List is present if list field present (Shouldn't happen)
        return m_pListField->isPresent();
    }
    else    // No list or entry field provided
    {
        // Return list not present
        return false;
    }
    
} // isPresent

//******************************************************************************

POINTER
CListEntry::ptrHeadEntry() const
{
    POINTER             ptrHeadEntry = 0;

    // Can only return list head if entry field given
    if (entryField() != NULL)
    {
        // Check for list entries present
        if (ptrFlink() != ptrListEntry())
        {
            // Compute the actual head entry
            ptrHeadEntry = ptrFlink() - entryField()->offset();

            // Save head entry as the last entry value
            m_ptrEntry = ptrHeadEntry;
        }
    }
    return ptrHeadEntry;

} // ptrHeadEntry

//******************************************************************************

POINTER
CListEntry::ptrTailEntry() const
{
    POINTER             ptrTailEntry = 0;

    // Can only return list head if entry field given
    if (entryField() != NULL)
    {
        // Check for list entries present
        if (ptrBlink() != ptrListEntry())
        {
            // Compute the actual tail entry
            ptrTailEntry = ptrBlink() - entryField()->offset();

            // Save tail entry as the last entry value
            m_ptrEntry = ptrTailEntry;
        }
    }
    return ptrTailEntry;

} // ptrTailEntry

//******************************************************************************

POINTER
CListEntry::ptrPrevEntry
(
    POINTER             ptrEntry
) const
{
    POINTER             ptrPrevEntry;

    // Can only return previous entry if entry field given
    if (entryField() != NULL)
    {
        // Check for an entry value given (Use last if none)
        if (ptrEntry == NULL)
        {
            ptrEntry = m_ptrEntry;
        }
        // Check for current entry value (Use tail if none)
        if (ptrEntry != NULL)
        {
            // Read the entry previous link value
            ptrPrevEntry = readPointer(ptrEntry + entryField()->offset() + ptrBlinkMember().offset());

            // Check for previous entry present (Not at head of list)
            if (ptrPrevEntry != ptrListEntry())
            {
                // Compute the actual previous entry address
                ptrPrevEntry -= entryField()->offset();
            }
            else    // No previous entry
            {
                ptrPrevEntry = NULL;
            }
        }
        else    // No current entry (Use tail entry)
        {
            ptrPrevEntry = ptrTailEntry();
        }
        // Save previous entry as the last entry value (May be none)
        m_ptrEntry = ptrPrevEntry;
    }
    return ptrPrevEntry;

} // ptrPrevEntry

//******************************************************************************

POINTER
CListEntry::ptrNextEntry
(
    POINTER             ptrEntry
) const
{
    POINTER             ptrNextEntry;

    // Can only return next entry if entry field given
    if (entryField() != NULL)
    {
        // Check for an entry value given (Use last if none)
        if (ptrEntry == NULL)
        {
            ptrEntry = m_ptrEntry;
        }
        // Check for current entry value (Use head if none)
        if (ptrEntry != NULL)
        {
            // Read the entry next link value
            ptrNextEntry = readPointer(ptrEntry + entryField()->offset() + ptrFlinkMember().offset());

            // Check for next entry present (Not at tail of list)
            if (ptrNextEntry != ptrListEntry())
            {
                // Compute the actual next entry address
                ptrNextEntry -= entryField()->offset();
            }
            else    // No next entry
            {
                ptrNextEntry = NULL;
            }
        }
        else    // No current entry (Use head entry)
        {
            ptrNextEntry = ptrHeadEntry();
        }
        // Save next entry as the last entry value (May be none)
        m_ptrEntry = ptrNextEntry;
    }
    return ptrNextEntry;

} // ptrNextEntry

//******************************************************************************

const CGlobal&
psActiveProcessHead()
{
    // Return reference to active process head global
    return s_PsActiveProcessHead;

} // psActiveProcessHead

//******************************************************************************

const CGlobal&
mmProcessList()
{
    // Return reference to process list global
    return s_MmProcessList;

} // mmProcessList

} // os namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
