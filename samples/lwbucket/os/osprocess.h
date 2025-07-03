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
|*  Module: osprocess.h                                                       *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _OSPROCESS_H
#define _OSPROCESS_H

//******************************************************************************
//
//  os namespace
//
//******************************************************************************
namespace os
{

//******************************************************************************
//
// class CKProcess
//
//******************************************************************************
class CKProcess : public CKProcessObject
{
// Kernel Process Type Helpers
TYPE(KProcess)

// Kernel Process Field Helpers
FIELD(header)
FIELD(threadListHead)
FIELD(processLock)
FIELD(basePriority)
FIELD(quantumReset)
FIELD(processListEntry)
FIELD(cycleTime)
FIELD(kernelTime)
FIELD(userTime)

// Kernel Process Members
MEMBER(processLock,     ULONG,      0,  public)
MEMBER(basePriority,    CHAR,       0,  public)
MEMBER(quantumReset,    CHAR,       0,  public)
MEMBER(cycleTime,       ULONG64,    0,  public)
MEMBER(kernelTime,      ULONG,      0,  public)
MEMBER(userTime,        ULONG,      0,  public)

private:
static  CKProcessList   m_KProcessList;

        PROCESS         m_ptrKProcess;

mutable CDispatcherHeaderPtr m_pDispatcherHeader;
mutable CKThreadsPtr    m_pKThreads;

        CListEntry      m_ThreadList;

protected:
static  CKProcessList*  KProcessList()              { return &m_KProcessList; }

                        CKProcess(CKProcessList *pKProcessList, PROCESS ptrKProcess);
virtual                ~CKProcess();
public:
static  CKProcessPtr    createKProcess(PROCESS ptrKProcess);

        PROCESS         ptrKProcess() const         { return m_ptrKProcess; }

const   CListEntry&     threadList() const          { return m_ThreadList; }

const   CDispatcherHeaderPtr dispatcherHeader() const;
const   CKThreadsPtr    KThreads() const;
        CEProcessPtr    EProcess() const;

        bool            isValid() const             { return ((dispatcherHeader() != NULL) ? (dispatcherHeader()->dispatcherType() == Process) : false); }

        ULONG           threadCount() const         { return ((KThreads() != NULL) ? KThreads()->KThreadCount() : 0); }

const   CMemberType&    type() const                { return m_KProcessType; }

}; // CKProcess

//******************************************************************************
//
// class CKProcesses
//
//******************************************************************************
class CKProcesses : public CRefObj
{
private:
        ULONG           m_ulKProcessCount;

        CProcessBase    m_ptrKProcesses;
        CKProcessArray  m_aKProcesses;

public:
                        CKProcesses(const CListEntry& KProcessList);
                        CKProcesses(const CEProcesses* pEProcesses);
virtual                ~CKProcesses();

        ULONG           KProcessCount() const       { return m_ulKProcessCount; }

        CKProcessPtr*   KProcesses();
const   CKProcessPtr    KProcess(ULONG ulKProcess) const;

        PROCESS         ptrKProcess(ULONG ulKProcess) const;

const   CKProcessPtr    findKProcess(PROCESS ptrKProcess) const;

}; // class CKProcesses

//******************************************************************************
//
// class CEProcess
//
//******************************************************************************
class CEProcess : public CEProcessObject
{
// Exelwtive Process Type Helpers
TYPE(EProcess)

// Exelwtive Process Field Helpers
FIELD(pcb)
FIELD(createTime)
FIELD(exitTime)
FIELD(uniqueProcessId)
FIELD(activeProcessLinks)
FIELD(commitCharge)
FIELD(peakVirtualSize)
FIELD(virtualSize)
FIELD(sessionProcessLinks)
FIELD(session)
FIELD(imageFileName)
FIELD(threadListHead)
FIELD(activeThreads)
FIELD(lastThreadExitStatus)
FIELD(peb)
FIELD(readOperationCount)
FIELD(writeOperationCount)
FIELD(otherOperationCount)
FIELD(readTransferCount)
FIELD(writeTransferCount)
FIELD(otherTransferCount)
FIELD(commitChargeLimit)
FIELD(commitChargePeak)
FIELD(mmProcessLinks)

// Exelwtive Process Members
MEMBER(createTime,              ULONG64,    0,      public)
MEMBER(exitTime,                ULONG64,    0,      public)
MEMBER(uniqueProcessId,         POINTER,    NULL,   public)
MEMBER(commitCharge,            ULONG,      0,      public)
MEMBER(peakVirtualSize,         ULONG,      0,      public)
MEMBER(virtualSize,             ULONG,      0,      public)
MEMBER(session,                 POINTER,    NULL,   public)
MEMBER(imageFileName,           PVOID,      NULL,   public)
MEMBER(activeThreads,           ULONG,      0,      public)
MEMBER(lastThreadExitStatus,    ULONG,      0,      public)
MEMBER(peb,                     POINTER,    NULL,   public)
MEMBER(readOperationCount,      ULONG64,    0,      public)
MEMBER(writeOperationCount,     ULONG64,    0,      public)
MEMBER(otherOperationCount,     ULONG64,    0,      public)
MEMBER(readTransferCount,       ULONG64,    0,      public)
MEMBER(writeTransferCount,      ULONG64,    0,      public)
MEMBER(otherTransferCount,      ULONG64,    0,      public)
MEMBER(commitChargeLimit,       ULONG,      0,      public)
MEMBER(commitChargePeak,        ULONG,      0,      public)

private:
static  CEProcessList   m_EProcessList;

        PROCESS         m_ptrEProcess;

mutable CKProcessPtr    m_pKProcess;
mutable CEThreadsPtr    m_pEThreads;

        CListEntry      m_ThreadList;

protected:
static  CEProcessList*  EProcessList()
                            { return &m_EProcessList; }

                        CEProcess(CEProcessList *pEProcessList, PROCESS ptrEProcess);
virtual                ~CEProcess();
public:
static  CEProcessPtr    createEProcess(PROCESS ptrEProcess);

        PROCESS         ptrEProcess() const         { return m_ptrEProcess; }

const   CListEntry&     threadList() const          { return m_ThreadList; }

const   CKProcessPtr    KProcess() const;
const   CEThreadsPtr    EThreads() const;

const   CDispatcherHeaderPtr dispatcherHeader() const
                            { return ((KProcess() != NULL) ? KProcess()->dispatcherHeader() : NULL); }

        ULONG           threadCount() const         { return ((EThreads() != NULL) ? EThreads()->EThreadCount() : 0); }

        CString         processName() const;

        bool            isValid() const             { return ((KProcess() != NULL) ? KProcess()->isValid() : false); }

const   CMemberType&    type() const                { return m_EProcessType; }

}; // CEProcess

//******************************************************************************
//
// class CEProcesses
//
//******************************************************************************
class CEProcesses : public CRefObj
{
private:
        ULONG           m_ulEProcessCount;

        CProcessBase    m_ptrEProcesses;
        CEProcessArray  m_aEProcesses;

mutable CKProcessesPtr  m_pKProcesses;

public:
                        CEProcesses(const CListEntry& EProcessList);
virtual                ~CEProcesses();

        ULONG           EProcessCount() const
                            { return m_ulEProcessCount; }

        CEProcessPtr*   EProcesses();
const   CEProcessPtr    EProcess(ULONG ulEProcess) const;

        PROCESS         ptrEProcess(ULONG ulEProcess) const;

const   CEProcessPtr    findEProcess(PROCESS ptrEProcess) const;
const   CEProcessPtr    findEProcess(ULONG64 ulEProcessId) const;

const   CKProcessesPtr  KProcesses() const;

}; // class CEProcesses

//******************************************************************************
//
// Inline Functions
//
//******************************************************************************
inline  CKProcessPtr    createKProcess(PROCESS ptrKProcess)
                            { return CKProcess::createKProcess(ptrKProcess); }

inline  CEProcessPtr    createEProcess(PROCESS ptrEProcess)
                            { return CEProcess::createEProcess(ptrEProcess); }

//******************************************************************************
//
// Functions
//
//******************************************************************************
const   CKProcessesPtr  getKProcesses(bool bThrow = true);
const   CKProcessesPtr  getKernelProcesses(bool bThrow = true);

const   CEProcessesPtr  getEProcesses(bool bThrow = true);
const   CEProcessesPtr  getExelwtiveProcesses(bool bThrow = true);

const   CKProcessPtr    findKProcess(PROCESS ptrKProcess);
const   CEProcessPtr    findEProcess(PROCESS ptrEProcess);

} // os namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _OSPROCESS_H
