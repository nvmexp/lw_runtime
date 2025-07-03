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
|*  Module: osdispatcher.h                                                    *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _OSDISPATCHER_H
#define _OSDISPATCHER_H

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
// Define the common OS time frequency value (100 ns units)
#define OS_TIME_FREQUENCY   10000000                // Common OS time frequency (10 MHz - 100 ns)

// Define the OS DISPATCHER_HEADER types (Used to detect valid OS Objects)
enum DispatcherHeaderType
{
    NotificationEvent = 0,                          // Notification event (0)
    SynchronizationEvent,                           // Synchronization event (1)
    Mutant,                                         // Kernel mutant (2)
    Process,                                        // Kernel process (3)
    Queue,                                          // Kernel queue (4)
    Semaphore,                                      // Kernel semaphore (5)
    Thread,                                         // Kernel thread (6)
    NotificationTimer = 8,                          // Notification timer (8)
    SynchronizationTimer,                           // Synchronization timer (9)

}; // DispatcherHeaderType

//******************************************************************************
//
// class CDispatcherHeader
//
//******************************************************************************
class CDispatcherHeader : public CDispatcherHeaderObject
{
// Dispatcher Header Type Helpers
TYPE(dispatcherHeader)

// Dispatcher Header Field Helpers
FIELD(dispatcherType)
FIELD(abandoned)
FIELD(absolute)
FIELD(nxpIrql)
FIELD(signalling)
FIELD(size)
FIELD(hand)
FIELD(inserted)
FIELD(debugActive)
FIELD(dpcActive)
FIELD(lock)
FIELD(signalState)
FIELD(waitListHead)

// Dispatcher Header Members
MEMBER(dispatcherType,  UCHAR,  0,  public)
MEMBER(abandoned,       UCHAR,  0,  public)
MEMBER(absolute,        UCHAR,  0,  public)
MEMBER(nxpIrql,         UCHAR,  0,  public)
MEMBER(signalling,      UCHAR,  0,  public)
MEMBER(size,            UCHAR,  0,  public)
MEMBER(hand,            UCHAR,  0,  public)
MEMBER(inserted,        UCHAR,  0,  public)
MEMBER(debugActive,     UCHAR,  0,  public)
MEMBER(dpcActive,       UCHAR,  0,  public)
MEMBER(lock,            LONG,   0,  public)
MEMBER(signalState,     LONG,   0,  public)

private:
static  CDispatcherHeaderList m_DispatcherHeaderList;

        POINTER         m_ptrDispatcherHeader;

        CListEntry      m_WaitList;

protected:
static  CDispatcherHeaderList* dispatcherHeaderList()
                            { return &m_DispatcherHeaderList; }

                        CDispatcherHeader(CDispatcherHeaderList *pDispatcherHeaderList, POINTER ptrDispatcherHeader);
virtual                ~CDispatcherHeader();
public:
static  CDispatcherHeaderPtr createDispatcherHeader(POINTER ptrDispatcherHeader);

        POINTER         ptrDispatcherHeader() const { return m_ptrDispatcherHeader; }

const   CListEntry&     waitList() const            { return m_WaitList; }

const   CMemberType&    type() const                { return m_dispatcherHeaderType; }

}; // CDispatcherHeader

//******************************************************************************
//
// Inline Functions
//
//******************************************************************************
inline  CDispatcherHeaderPtr    createDispatcherHeader(POINTER ptrDispatcherHeader)
                                    { return CDispatcherHeader::createDispatcherHeader(ptrDispatcherHeader); }
} // os namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _OSDISPATCHER_H
