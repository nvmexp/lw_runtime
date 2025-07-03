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
|*  Module: osthread.h                                                        *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _OSTHREAD_H
#define _OSTHREAD_H

//******************************************************************************
//
//  os namespace
//
//******************************************************************************
namespace os
{

//******************************************************************************
//
// class CKThread
//
//******************************************************************************
class CKThread : public CKThreadObject
{
// Kernel Thread Type Helpers
TYPE(KThread)

// Kernel Thread Field Helpers
FIELD(header)
FIELD(initialStack)
FIELD(stackLimit)
FIELD(stackBase)
FIELD(kernelStack)
FIELD(running)
FIELD(alerted)
FIELD(priority)
FIELD(contextSwitches)
FIELD(state)
FIELD(nextProcessor)
FIELD(idealProcessor)
FIELD(process)
FIELD(kernelTime)
FIELD(userTime)
FIELD(threadListEntry)

// Kernel Thread Members
MEMBER(initialStack,    POINTER,    NULL,   public)
MEMBER(stackLimit,      POINTER,    NULL,   public)
MEMBER(stackBase,       POINTER,    NULL,   public)
MEMBER(kernelStack,     POINTER,    NULL,   public)
MEMBER(running,         bool,       FALSE,  public)
MEMBER(alerted,         bool,       FALSE,  public)
MEMBER(priority,        BYTE,       0,      public)
MEMBER(contextSwitches, ULONG,      0,      public)
MEMBER(state,           BYTE,       0,      public)
MEMBER(nextProcessor,   ULONG,      0,      public)
MEMBER(idealProcessor,  ULONG,      0,      public)
MEMBER(process,         POINTER,    NULL,   public)
MEMBER(kernelTime,      ULONG,      0,      public)
MEMBER(userTime,        ULONG,      0,      public)

private:
static  CKThreadList    m_KThreadList;

        THREAD          m_ptrKThread;

mutable CDispatcherHeaderPtr m_pDispatcherHeader;

protected:
static  CKThreadList*   KThreadList()               { return &m_KThreadList; }

                        CKThread(CKThreadList *pKThreadList, THREAD ptrKThread);
virtual                ~CKThread();
public:
static  CKThreadPtr     createKThread(THREAD ptrKThread);

        THREAD          ptrKThread() const          { return m_ptrKThread; }

const   CDispatcherHeaderPtr dispatcherHeader() const;
        CEThreadPtr     EThread() const;
        CKProcessPtr    KProcess() const;
        CEProcessPtr    EProcess() const;

        bool            isValid() const             { return ((dispatcherHeader() != NULL) ? (dispatcherHeader()->dispatcherType() == Thread) : false); }

const   CMemberType&    type() const                { return m_KThreadType; }

}; // CKThread

//******************************************************************************
//
// class CKThreads
//
//******************************************************************************
class CKThreads : public CRefObj
{
private:
        ULONG           m_ulKThreadCount;

        CThreadBase     m_ptrKThreads;
        CKThreadArray   m_aKThreads;

public:
                        CKThreads(const CListEntry& KList);
                        CKThreads(const CEThreads* pEThreads);
virtual                ~CKThreads();

        ULONG           KThreadCount() const        { return m_ulKThreadCount; }

        CKThreadPtr*    KThreads();
const   CKThreadPtr     KThread(ULONG ulKThread) const;

        THREAD          ptrKThread(ULONG ulKThread) const;

const   CKThreadPtr     findKThread(THREAD ptrKThread) const;

}; // class CKThreads

//******************************************************************************
//
// class CEThread
//
//******************************************************************************
class CEThread : public CEThreadObject
{
// Exelwtive Thread Type Helpers
TYPE(EThread)

// Exelwtive Thread Field Helpers
FIELD(tcb)
FIELD(createTime)
FIELD(exitTime)
FIELD(exitStatus)
FIELD(irpList)
FIELD(threadListEntry)

// Exelwtive Thread Members
MEMBER(createTime,      ULONG64,    0,  public)
MEMBER(exitTime,        ULONG64,    0,  public)
MEMBER(exitStatus,      LONG,       0,  public)

private:
static  CEThreadList    m_EThreadList;

        THREAD          m_ptrEThread;

mutable CKThreadPtr     m_pKThread;

//        CListEntry      m_IrpList;

protected:
static  CEThreadList*   EThreadList()               { return &m_EThreadList; }

                        CEThread(CEThreadList *pEThreadList, THREAD ptrEThread);
virtual                ~CEThread();
public:
static  CEThreadPtr     createEThread(THREAD ptrEThread);

        THREAD          ptrEThread() const          { return m_ptrEThread; }

const   CKThreadPtr     KThread() const;

const   CDispatcherHeaderPtr dispatcherHeader() const
                            { return ((KThread() != NULL) ? KThread()->dispatcherHeader() : NULL); }
        CKProcessPtr    KProcess() const;
        CEProcessPtr    EProcess() const;

        bool            isValid() const             { return ((KThread() != NULL) ? KThread()->isValid() : false); }

const   CMemberType&    type() const                { return m_EThreadType; }

}; // CEThread

//******************************************************************************
//
// class CEThreads
//
//******************************************************************************
class CEThreads : public CRefObj
{
private:
        ULONG           m_ulEThreadCount;

        CThreadBase     m_ptrEThreads;
        CEThreadArray   m_aEThreads;

mutable CKThreadsPtr    m_pKThreads;

public:
                        CEThreads(const CListEntry& EList);
virtual                ~CEThreads();

        ULONG           EThreadCount() const        { return m_ulEThreadCount; }

        CEThreadPtr*    EThreads();
const   CEThreadPtr     EThread(ULONG ulEThread) const;

        THREAD          ptrEThread(ULONG ulEThread) const;

const   CEThreadPtr     findEThread(THREAD ptrEThread) const;
const   CEThreadPtr     findEThread(ULONG64 ulEThreadId) const;

const   CKThreadsPtr    KThreads() const;

}; // class CEThreads

//******************************************************************************
//
// Inline Functions
//
//******************************************************************************
inline  CKThreadPtr     createKThread(THREAD ptrKThread)
                            { return CKThread::createKThread(ptrKThread); }

inline  CEThreadPtr     createEThread(THREAD ptrEThread)
                            { return CEThread::createEThread(ptrEThread); }

//******************************************************************************
//
// Functions
//
//******************************************************************************
const   CKThreadsPtr    getKThreads(bool bThrow = true);
const   CKThreadsPtr    getKernelThreads(bool bThrow = true);

const   CEThreadsPtr    getEThreads(bool bThrow = true);
const   CEThreadsPtr    getExelwtiveThreads(bool bThrow = true);

const   CKThreadPtr     findKThread(THREAD ptrKThread);
const   CEThreadPtr     findEThread(THREAD ptrEThread);

} // os namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _OSTHREAD_H
