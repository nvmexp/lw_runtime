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
|*  Module: osdispatcher.cpp                                                  *|
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
// Dispatcher Header Type Helpers
CMemberType     CDispatcherHeader::m_dispatcherHeaderType   (&osKernel(), "DISPATCH_HEADER", "_DISPATCHER_HEADER");

// Dispatcher Header Field Helpers
CMemberField    CDispatcherHeader::m_dispatcherTypeField    (&dispatcherHeaderType(), false, NULL, "Type");
CMemberField    CDispatcherHeader::m_abandonedField         (&dispatcherHeaderType(), false, NULL, "Abandoned");
CMemberField    CDispatcherHeader::m_absoluteField          (&dispatcherHeaderType(), false, NULL, "Absolute");
CMemberField    CDispatcherHeader::m_nxpIrqlField           (&dispatcherHeaderType(), false, NULL, "NpxIrql");
CMemberField    CDispatcherHeader::m_signallingField        (&dispatcherHeaderType(), false, NULL, "Signalling");
CMemberField    CDispatcherHeader::m_sizeField              (&dispatcherHeaderType(), false, NULL, "Size");
CMemberField    CDispatcherHeader::m_handField              (&dispatcherHeaderType(), false, NULL, "Hand");
CMemberField    CDispatcherHeader::m_insertedField          (&dispatcherHeaderType(), false, NULL, "Inserted");
CMemberField    CDispatcherHeader::m_debugActiveField       (&dispatcherHeaderType(), false, NULL, "DebugActive");
CMemberField    CDispatcherHeader::m_dpcActiveField         (&dispatcherHeaderType(), false, NULL, "DpcActive");
CMemberField    CDispatcherHeader::m_lockField              (&dispatcherHeaderType(), false, NULL, "Lock");
CMemberField    CDispatcherHeader::m_signalStateField       (&dispatcherHeaderType(), false, NULL, "SignalState");
CMemberField    CDispatcherHeader::m_waitListHeadField      (&dispatcherHeaderType(), false, NULL, "WaitListHead");

// CDispatcherHeader object tracking
CDispatcherHeaderList   CDispatcherHeader::m_DispatcherHeaderList;

//******************************************************************************

CDispatcherHeader::CDispatcherHeader
(
    CDispatcherHeaderList *pDispatcherHeaderList,
    POINTER             ptrDispatcherHeader
)
:   LWnqObj(pDispatcherHeaderList, ptrDispatcherHeader),
    m_ptrDispatcherHeader(ptrDispatcherHeader),
    INIT(dispatcherType),
    INIT(abandoned),
    INIT(absolute),
    INIT(nxpIrql),
    INIT(signalling),
    INIT(size),
    INIT(hand),
    INIT(inserted),
    INIT(debugActive),
    INIT(dpcActive),
    INIT(lock),
    INIT(signalState),
    m_WaitList(&m_waitListHeadField, &CDispatcherHeader::waitListHeadField(), ptrDispatcherHeader + (m_waitListHeadField.isPresent() ? m_waitListHeadField.offset() : 0))
{
    assert(pDispatcherHeaderList != NULL);

    // Get the dispatcher header information
    READ(dispatcherType, ptrDispatcherHeader);
    READ(abandoned,      ptrDispatcherHeader);
    READ(absolute,       ptrDispatcherHeader);
    READ(nxpIrql,        ptrDispatcherHeader);
    READ(signalling,     ptrDispatcherHeader);
    READ(size,           ptrDispatcherHeader);
    READ(hand,           ptrDispatcherHeader);
    READ(inserted,       ptrDispatcherHeader);
    READ(debugActive,    ptrDispatcherHeader);
    READ(dpcActive,      ptrDispatcherHeader);
    READ(lock,           ptrDispatcherHeader);
    READ(signalState,    ptrDispatcherHeader);

} // CDispatcherHeader

//******************************************************************************

CDispatcherHeader::~CDispatcherHeader()
{

} // ~CDispatcherHeader

//******************************************************************************

CDispatcherHeaderPtr
CDispatcherHeader::createDispatcherHeader
(
    POINTER             ptrDispatcherHeader
)
{
    CDispatcherHeaderPtr pDispatcherHeader;

    // Check for valid dispatcher header address given
    if (ptrDispatcherHeader != NULL)
    {
        // Check to see if this dispatcher header already exists
        pDispatcherHeader = findObject(dispatcherHeaderList(), ptrDispatcherHeader);
        if (pDispatcherHeader == NULL)
        {
            // Try to create the new dispatcher header object
            pDispatcherHeader = new CDispatcherHeader(dispatcherHeaderList(), ptrDispatcherHeader);
        }
    }
    return pDispatcherHeader;

} // createDispatcherHeader

} // os namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
