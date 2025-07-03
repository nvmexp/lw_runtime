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
|*  Module: ossession.cpp                                                     *|
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
// Session Type Helpers
CMemberType     CSession::m_SessionType                     (&osKernel(), "MM_SESSION_SPACE", "_MM_SESSION_SPACE");

// Session Field Helpers
CMemberField    CSession::m_referenceCountField             (&SessionType(), false, NULL, "ReferenceCount");
CMemberField    CSession::m_sessionIdField                  (&SessionType(), false, NULL, "SessionId");
CMemberField    CSession::m_processListField                (&SessionType(), false, NULL, "ProcessList");
CMemberField    CSession::m_pagedPoolStartField             (&SessionType(), false, NULL, "PagedPoolStart");
CMemberField    CSession::m_pagedPoolEndField               (&SessionType(), false, NULL, "PagedPoolEnd");
CMemberField    CSession::m_sessionField                    (&SessionType(), false, NULL, "Session");
CMemberField    CSession::m_createTimeField                 (&SessionType(), false, NULL, "CreateTime");

// CSession object tracking
CSessionList    CSession::m_SessionList;

//******************************************************************************

CSession::CSession
(
    CSessionList       *pSessionList,
    SESSION             ptrSession
)
:   LWnqObj(pSessionList, ptrSession),
    m_ptrSession(ptrSession),
    INIT(referenceCount),
    INIT(sessionId),
    INIT(pagedPoolStart),
    INIT(pagedPoolEnd),
    INIT(session),
    INIT(createTime),
    m_EProcessList(&m_processListField, &CEProcess::sessionProcessLinksField(), ptrSession + (m_processListField.isPresent() ? m_processListField.offset() : 0)),
    m_pEProcesses(NULL)
{
    assert(pSessionList != NULL);

    // Get the session information
    READ(referenceCount, ptrSession);
    READ(sessionId,      ptrSession);
    READ(pagedPoolStart, ptrSession);
    READ(pagedPoolEnd,   ptrSession);
    READ(session,        ptrSession);
    READ(createTime,     ptrSession);

} // CSession

//******************************************************************************

CSession::~CSession()
{

} // ~CSession

//******************************************************************************

CSessionPtr
CSession::createSession
(
    SESSION             ptrSession
)
{
    CSessionPtr         pSession;

    // Check for valid session address given
    if (isKernelModeAddress(ptrSession))
    {
        // Check to see if this session already exists
        pSession = findObject(sessionList(), ptrSession);
        if (pSession == NULL)
        {
            // Try to create the new session object
            pSession = new CSession(sessionList(), ptrSession);
        }
    }
    return pSession;

} // createSession

//******************************************************************************

const CEProcessesPtr
CSession::EProcesses() const
{
    // Check for exelwtive processes already created
    if (m_pEProcesses == NULL)
    {
        // Check to see if exelwtive processes are present
        if (processList().isPresent())
        {
            // Try to create the exelwtive process threads
            m_pEProcesses = new CEProcesses(processList());
        }
    }
    return m_pEProcesses;

} // EProcesses

} // os namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
