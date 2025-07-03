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
|*  Module: ossession.h                                                       *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _OSSESSION_H
#define _OSSESSION_H

//******************************************************************************
//
//  os namespace
//
//******************************************************************************
namespace os
{

//******************************************************************************
//
// class CSession
//
//******************************************************************************
class CSession : public CSessionObject
{
// Session Type Helpers
TYPE(Session)

// Session Field Helpers
FIELD(referenceCount)
FIELD(sessionId)
FIELD(processList)
FIELD(pagedPoolStart)
FIELD(pagedPoolEnd)
FIELD(session)
FIELD(createTime)

// Session Members
MEMBER(referenceCount,  ULONG,      0,  public)
MEMBER(sessionId,       ULONG,      0,  public)
MEMBER(pagedPoolStart,  POINTER, NULL,  public)
MEMBER(pagedPoolEnd,    POINTER, NULL,  public)
MEMBER(session,         POINTER, NULL,  public)
MEMBER(createTime,      ULONG64,    0,  public)

private:
static  CSessionList    m_SessionList;

        SESSION         m_ptrSession;

mutable CEProcessesPtr  m_pEProcesses;

        CListEntry      m_EProcessList;

protected:
static  CSessionList*   sessionList()               { return &m_SessionList; }

                        CSession(CSessionList *pSessionList, SESSION ptrSession);
virtual                ~CSession();
public:
static  CSessionPtr     createSession(SESSION ptrSession);

        SESSION         ptrSession() const          { return m_ptrSession; }

const   CListEntry&     processList() const         { return m_EProcessList; }

const   CEProcessesPtr  EProcesses() const;

        ULONG           processCount() const        { return ((EProcesses() != NULL) ? EProcesses()->EProcessCount() : 0); }

const   CMemberType&    type() const                { return m_SessionType; }

}; // CSession

//******************************************************************************
//
// Inline Functions
//
//******************************************************************************
inline  CSessionPtr     createSession(SESSION ptrSession)
                            { return CSession::createSession(ptrSession); }

} // os namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _OSSESSION_H
